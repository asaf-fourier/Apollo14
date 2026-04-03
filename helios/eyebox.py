"""Eyebox uniformity merit function for combiner optimization.

Evaluates FOV uniformity across the eyebox by sampling at the center
and corners, tracing R/G/B rays at multiple FOV angles via the JAX
tracer, and scoring uniformity + intensity + color balance separately.

The merit decomposes into three weighted terms:

- **Spatial uniformity** — at each (angle, color), how uniform is the
  intensity across eyebox samples? Low variance = same brightness
  everywhere on the eyebox.
- **Angular uniformity** — at each (sample, color), how uniform is the
  intensity across FOV angles? Low variance = full FOV visible.
- **Intensity error** — how far is the mean intensity from the D65
  white-balanced target? Drives toward the desired brightness and
  correct color balance.

Fully differentiable — use ``jax.grad`` on ``eyebox_merit`` to get
gradients w.r.t. mirror reflectances or positions.

Usage::

    from helios.eyebox import (
        eyebox_sample_points, compute_eyebox_response,
        eyebox_merit, EyeboxConfig,
    )
    from apollo14.jax_tracer import params_from_config
    from apollo14.combiner import CombinerConfig

    config = CombinerConfig.default()
    params = params_from_config(config)
    n_glass = float(config.chassis.material.n(config.light.wavelength))

    samples = eyebox_sample_points(
        config.pupil.center, config.pupil.normal, config.pupil.radius)
    mc = EyeboxConfig()

    response, dirs = compute_eyebox_response(
        params, n_glass,
        config.light.position, config.light.direction,
        config.light.x_fov, config.light.y_fov,
        samples, mc,
    )
    loss = eyebox_merit(response, mc)
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from apollo14.geometry import compute_local_axes
from apollo14.jax_tracer import trace_batch
from apollo14.projector import scan_directions
from apollo14.units import mm

from helios.merit import D65_WEIGHTS


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class EyeboxConfig:
    """Parameters for eyebox merit evaluation."""
    target_intensity: float = 0.03   # total target intensity (split by D65)
    d65_weights: jnp.ndarray = None  # (3,) per-color target ratios
    sigma: float = 1.0 * mm         # Gaussian kernel width for spatial binning
    n_fov_x: int = 5                # FOV angular grid
    n_fov_y: int = 5
    w_uniformity: float = 2.0       # weight for uniformity (spatial + angular)
    w_intensity: float = 1.0        # weight for intensity error
    visibility_threshold: float = 0.5  # fraction of peak for "visible FOV"

    def __post_init__(self):
        if self.d65_weights is None:
            self.d65_weights = D65_WEIGHTS

    @property
    def target_per_color(self):
        """(3,) D65-weighted target intensity per color channel."""
        return self.target_intensity * self.d65_weights


# ── Eyebox sampling ─────────────────────────────────────────────────────────

def eyebox_sample_points(center, normal, radius):
    """Sample the eyebox at center + 4 corners of an inscribed square.

    Returns:
        (5, 3) array of sample positions on the pupil plane.
    """
    lx, ly = compute_local_axes(normal)
    r = radius / jnp.sqrt(2.0)  # inscribed square corner distance
    return jnp.stack([
        center,
        center + r * lx + r * ly,
        center + r * lx - r * ly,
        center - r * lx + r * ly,
        center - r * lx - r * ly,
    ])


# ── Response computation ────────────────────────────────────────────────────

def compute_eyebox_response(params, n_glass, projector_pos, projector_dir,
                            x_fov, y_fov, eyebox_points, config=None):
    """Compute intensity at each eyebox sample for each FOV angle and color.

    For each FOV angle, traces a single ray from the projector center.
    Each mirror reflects toward the pupil — the contribution to each
    eyebox sample is weighted by a Gaussian kernel (soft spatial binning)
    to keep the function smooth and differentiable.

    Args:
        params: CombinerParams with system geometry
        n_glass: refractive index of chassis glass
        projector_pos: (3,) projector position
        projector_dir: (3,) projector central direction (normalized)
        x_fov: half-angle horizontal FOV (radians)
        y_fov: half-angle vertical FOV (radians)
        eyebox_points: (S, 3) sample points on the eyebox
        config: EyeboxConfig (uses defaults if None)

    Returns:
        response: (S, A, 3) intensity per sample, per angle, per color
        scan_dirs: (A, 3) the flattened FOV directions used
    """
    if config is None:
        config = EyeboxConfig()

    dirs, _ = scan_directions(projector_dir, x_fov, y_fov,
                              config.n_fov_x, config.n_fov_y)
    flat_dirs = dirs.reshape(-1, 3)
    n_angles = flat_dirs.shape[0]
    origins = jnp.tile(projector_pos, (n_angles, 1))
    sigma_sq = config.sigma ** 2

    color_responses = []
    for ci in range(3):
        # Trace all FOV angles at once for this color
        pts, ints, valid = trace_batch(
            origins, flat_dirs, n_glass, params, ci)
        # pts: (A, M, 3), ints: (A, M), valid: (A, M)

        # For each eyebox sample, compute soft-binned intensity per angle
        def _sample_intensity(sample_pt):
            delta = pts - sample_pt[None, None, :]  # (A, M, 3)
            dist_sq = jnp.sum(delta ** 2, axis=-1)  # (A, M)
            weights = jnp.exp(-0.5 * dist_sq / sigma_sq)
            weighted = jnp.where(valid, ints * weights, 0.0)
            return jnp.sum(weighted, axis=1)  # (A,)

        per_color = jax.vmap(_sample_intensity)(eyebox_points)  # (S, A)
        color_responses.append(per_color)

    response = jnp.stack(color_responses, axis=-1)  # (S, A, 3)
    return response, flat_dirs


# ── Merit function ──────────────────────────────────────────────────────────

def eyebox_merit(response, config=None):
    """Weighted merit combining uniformity and D65 intensity targets.

    Three terms, each addressing a distinct aspect:

    1. **Spatial uniformity** — ``var(response, axis=samples)`` averaged
       over angles and colors. Penalizes brightness variation across the
       eyebox at any given angle.
    2. **Angular uniformity** — ``var(response, axis=angles)`` averaged
       over samples and colors. Penalizes FOV non-uniformity at any
       given eyebox position.
    3. **Intensity error** — ``(mean_per_color - target_per_color)^2``.
       Drives overall brightness to the D65-weighted target.

    The total merit is::

        w_uniformity * (spatial_var + angular_var) + w_intensity * intensity_err

    Args:
        response: (S, A, 3) from ``compute_eyebox_response``
        config: EyeboxConfig with target and weights (uses defaults if None)

    Returns:
        scalar merit value (lower is better)
    """
    if config is None:
        config = EyeboxConfig()

    target = config.target_per_color  # (3,)

    # Spatial uniformity: for each (angle, color), variance across samples
    spatial_var = jnp.mean(jnp.var(response, axis=0))

    # Angular uniformity: for each (sample, color), variance across angles
    angular_var = jnp.mean(jnp.var(response, axis=1))

    # Intensity error: mean per color vs D65 target
    mean_per_color = jnp.mean(response, axis=(0, 1))  # (3,)
    intensity_err = jnp.mean((mean_per_color - target) ** 2)

    return (config.w_uniformity * (spatial_var + angular_var) +
            config.w_intensity * intensity_err)


# ── Diagnostics ─────────────────────────────────────────────────────────────

def visible_fov(response, threshold_fraction=0.5):
    """Compute visible FOV fraction at each eyebox sample.

    An angle is "visible" if its intensity (summed across colors) exceeds
    ``threshold_fraction`` times the peak intensity at that sample.

    Args:
        response: (S, A, 3) from ``compute_eyebox_response``
        threshold_fraction: fraction of peak to count as visible

    Returns:
        fov_fraction: (S,) fraction of angles visible at each sample
    """
    total = jnp.sum(response, axis=-1)  # (S, A)
    peak = jnp.max(total, axis=1, keepdims=True)  # (S, 1)
    threshold = threshold_fraction * peak
    visible = total > threshold  # (S, A)
    return jnp.mean(visible.astype(jnp.float32), axis=1)  # (S,)
