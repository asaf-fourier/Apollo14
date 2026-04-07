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
    from apollo14.jax_tracer import params_from_system
    from apollo14.combiner import build_default_system, DEFAULT_WAVELENGTH

    system = build_default_system()
    params = params_from_system(system, DEFAULT_WAVELENGTH)

    from apollo14.elements.pupil import RectangularPupil
    pupil = next(e for e in system.elements if isinstance(e, RectangularPupil))

    grid = eyebox_grid_points(pupil.position, pupil.normal, radius=3.0)
    mc = EyeboxConfig()

    from apollo14.combiner import (
        DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
        DEFAULT_BEAM_WIDTH, DEFAULT_BEAM_HEIGHT,
        DEFAULT_X_FOV, DEFAULT_Y_FOV,
    )
    from apollo14.elements.glass_block import GlassBlock
    chassis = next(e for e in system.elements if isinstance(e, GlassBlock))
    n_glass = float(chassis.material.n(DEFAULT_WAVELENGTH))

    response, dirs = compute_eyebox_response(
        params, n_glass,
        DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
        DEFAULT_BEAM_WIDTH, DEFAULT_BEAM_HEIGHT,
        DEFAULT_X_FOV, DEFAULT_Y_FOV,
        grid, mc,
    )
    loss = eyebox_merit(response, mc)
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from apollo14.geometry import compute_local_axes, normalize
from apollo14.jax_tracer import trace_beam
from apollo14.projector import scan_directions

from helios.merit import D65_WEIGHTS


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class EyeboxConfig:
    """Parameters for eyebox merit evaluation."""
    target_intensity: float = 0.03   # total target intensity (split by D65)
    d65_weights: jnp.ndarray = None  # (3,) per-color target ratios
    n_fov_x: int = 5                # FOV angular grid
    n_fov_y: int = 5
    n_beam_x: int = 5              # spatial beam grid per FOV angle
    n_beam_y: int = 5
    w_uniformity: float = 2.0       # weight for uniformity (spatial + angular)
    w_intensity: float = 1.0        # weight for intensity error
    w_coverage: float = 1.0         # weight for eyebox coverage penalty
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


def eyebox_grid_points(center, normal, radius, nx, ny):
    """Generate a dense grid of sample points on the eyebox plane.

    Returns:
        (nx*ny, 3) array of sample positions on the pupil plane.
    """
    lx, ly = compute_local_axes(normal)
    xs = jnp.linspace(-radius, radius, nx)
    ys = jnp.linspace(-radius, radius, ny)
    gx, gy = jnp.meshgrid(xs, ys)  # (ny, nx)
    positions = (center[None, None, :]
                 + gx[:, :, None] * lx[None, None, :]
                 + gy[:, :, None] * ly[None, None, :])
    return positions.reshape(-1, 3)


# ── Response computation ────────────────────────────────────────────────────

def _beam_origins(projector_pos, direction, beam_width, beam_height, nx, ny):
    """Generate a grid of ray origins for a collimated beam.

    Returns:
        (nx*ny, 3) array of ray start positions.
    """
    d = normalize(direction)
    # Build a local basis perpendicular to the beam direction
    up = jnp.where(jnp.abs(d[1]) < 0.99,
                   jnp.array([0.0, 1.0, 0.0]),
                   jnp.array([1.0, 0.0, 0.0]))
    lx = normalize(jnp.cross(d, up))
    ly = normalize(jnp.cross(lx, d))

    xs = jnp.linspace(-beam_width / 2, beam_width / 2, nx)
    ys = jnp.linspace(-beam_height / 2, beam_height / 2, ny)
    gx, gy = jnp.meshgrid(xs, ys)
    offsets = gx.ravel()[:, None] * lx[None, :] + gy.ravel()[:, None] * ly[None, :]
    return projector_pos[None, :] + offsets


def compute_eyebox_response(params, n_glass, projector_pos, projector_dir,
                            beam_width, beam_height,
                            x_fov, y_fov, eyebox_points, config=None):
    """Compute intensity at each eyebox sample for each FOV angle and color.

    For each FOV angle, traces a dense beam of rays (n_beam_x × n_beam_y)
    from the projector. Each ray hits M mirrors producing hit points on
    the pupil plane. Hits are hard-binned to the nearest eyebox grid
    point. Gradients flow through intensity values; spatial assignment
    is fixed (``stop_gradient`` on argmin).

    Args:
        params: CombinerParams with system geometry
        n_glass: refractive index of chassis glass
        projector_pos: (3,) projector position
        projector_dir: (3,) projector central direction (normalized)
        beam_width: physical width of beam cross-section
        beam_height: physical height of beam cross-section
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
    S = eyebox_points.shape[0]

    color_responses = []
    for ci in range(3):
        angle_responses = []
        for ai in range(n_angles):
            d = flat_dirs[ai]
            origins = _beam_origins(projector_pos, d, beam_width, beam_height,
                                   config.n_beam_x, config.n_beam_y)
            # trace_beam: shared direction, multiple origins
            pts, ints, valid, _, _ = trace_beam(origins, d, n_glass, params, ci)
            # pts: (R, M, 3), ints: (R, M), valid: (R, M)
            # Flatten rays × mirrors → (R*M,)
            R, M = ints.shape
            pts_flat = pts.reshape(R * M, 3)
            ints_flat = ints.reshape(R * M)
            valid_flat = valid.reshape(R * M)

            # Hard binning: nearest grid point
            delta = pts_flat[:, None, :] - eyebox_points[None, :, :]  # (R*M, S, 3)
            dist_sq = jnp.sum(delta ** 2, axis=-1)  # (R*M, S)
            nearest = jax.lax.stop_gradient(jnp.argmin(dist_sq, axis=-1))  # (R*M,)

            one_hot = jax.nn.one_hot(nearest, S)  # (R*M, S)
            weighted = jnp.where(valid_flat[:, None], ints_flat[:, None] * one_hot, 0.0)
            binned = jnp.sum(weighted, axis=0)  # (S,)
            angle_responses.append(binned)

        per_color = jnp.stack(angle_responses, axis=1)  # (S, A)
        color_responses.append(per_color)

    response = jnp.stack(color_responses, axis=-1)  # (S, A, 3)
    return response, flat_dirs


# ── Merit function ──────────────────────────────────────────────────────────

def eyebox_merit(response, config=None):
    """Weighted merit combining uniformity, intensity, and coverage.

    Computes a soft mask of illuminated grid points (those receiving
    above-threshold light) and evaluates uniformity only over them.
    A coverage penalty encourages the optimizer to illuminate more
    of the eyebox rather than shrinking the lit area.

    Four terms:

    1. **Spatial uniformity** — variance across illuminated samples,
       averaged over angles and colors.
    2. **Angular uniformity** — variance across angles at each
       illuminated sample, averaged over colors.
    3. **Intensity error** — ``(mean_per_color - target)^2`` over
       illuminated points only.
    4. **Coverage penalty** — ``(1 - illuminated_fraction)^2``.

    Args:
        response: (S, A, 3) from ``compute_eyebox_response``
        config: EyeboxConfig with target and weights (uses defaults if None)

    Returns:
        scalar merit value (lower is better)
    """
    if config is None:
        config = EyeboxConfig()

    target = config.target_per_color  # (3,)

    # Soft illumination mask: per-sample total intensity
    sample_total = jnp.sum(response, axis=(1, 2))  # (S,)
    peak_total = jnp.max(sample_total)
    # Soft mask: sigmoid around 1% of peak (differentiable)
    mask = jax.nn.sigmoid(20.0 * (sample_total / (peak_total + 1e-12) - 0.01))  # (S,)
    n_lit = jnp.sum(mask) + 1e-8  # avoid division by zero

    # Spatial uniformity: weighted variance across samples per (angle, color)
    weighted_mean_spatial = jnp.sum(
        mask[:, None, None] * response, axis=0) / n_lit  # (A, 3)
    spatial_diff = response - weighted_mean_spatial[None, :, :]  # (S, A, 3)
    spatial_var = jnp.sum(
        mask[:, None, None] * spatial_diff ** 2, axis=0) / n_lit  # (A, 3)
    spatial_var = jnp.mean(spatial_var)

    # Angular uniformity: variance across angles per (sample, color), weighted
    angular_var = jnp.var(response, axis=1)  # (S, 3)
    angular_var = jnp.sum(mask[:, None] * angular_var, axis=0) / n_lit  # (3,)
    angular_var = jnp.mean(angular_var)

    # Intensity error: mean per color over illuminated points vs D65 target
    mean_per_color = jnp.sum(
        mask[:, None, None] * response, axis=(0, 1)) / n_lit  # (3,)  -- per angle already averaged via sum/n_lit
    # Normalize by number of angles to get per-angle mean
    n_angles = response.shape[1]
    mean_per_color = mean_per_color / n_angles
    intensity_err = jnp.mean((mean_per_color - target) ** 2)

    # Coverage penalty: fraction of grid that is dark
    coverage = n_lit / response.shape[0]
    coverage_penalty = (1.0 - coverage) ** 2

    return (config.w_uniformity * (spatial_var + angular_var) +
            config.w_intensity * intensity_err +
            config.w_coverage * coverage_penalty)


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
