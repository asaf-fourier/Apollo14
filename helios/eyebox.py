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

See ``tests/test_helios.py`` for usage.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from apollo14.geometry import planar_grid_points
from apollo14.trace import trace_rays
from apollo14.binning import bin_hits_to_nearest
from apollo14.projector import Projector, scan_directions

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

def eyebox_grid_points(center, normal, radius, nx, ny):
    """Dense ``(nx, ny)`` grid on the eyebox plane, spanning ±``radius``."""
    return planar_grid_points(center, normal, radius, radius, nx, ny)


# ── Response computation ────────────────────────────────────────────────────

def compute_eyebox_response(routes_per_color, projector_pos, projector_dir,
                            beam_width, beam_height,
                            x_fov, y_fov, eyebox_points, config=None):
    """Compute intensity at each eyebox sample for each FOV angle and color.

    For each FOV angle, traces a dense beam of rays (n_beam_x × n_beam_y)
    from the projector through every pupil-terminated branch route, binning
    hits to the nearest eyebox grid point. Gradients flow through intensity
    values; spatial assignment is fixed (``stop_gradient`` on argmin).

    Args:
        routes_per_color: ``(n_colors, n_branches)`` list of pupil-terminated
            ``Route``s — e.g. from ``helios.merit.build_combiner_pupil_routes``.
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

    # Use Projector for beam origin generation (consistent basis computation)
    proj = Projector.uniform(
        position=projector_pos, direction=projector_dir,
        beam_width=beam_width, beam_height=beam_height,
        wavelength=0.0,  # unused for origin generation
        nx=config.n_beam_x, ny=config.n_beam_y,
    )

    dirs, _ = scan_directions(projector_dir, x_fov, y_fov,
                              config.n_fov_x, config.n_fov_y)
    flat_dirs = dirs.reshape(-1, 3)
    n_angles = flat_dirs.shape[0]

    color_responses = []
    for ci, branch_routes in enumerate(routes_per_color):
        angle_responses = []
        for ai in range(n_angles):
            d = flat_dirs[ai]
            origins, _, _, _ = proj.generate_rays(direction=d)
            binned = jnp.zeros(eyebox_points.shape[0])
            for route in branch_routes:
                tr = trace_rays(route, origins, d, color_idx=ci)
                binned = binned + bin_hits_to_nearest(
                    tr, eyebox_points, stop_grad=True)
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

# ── Eye-box area merit ─────────────────────────────────────────────────────

@dataclass
class EyeboxAreaConfig:
    """Configuration for eye-box area maximization merit.

    The merit dissects the pupil into a grid (default 2×2 mm cells).
    Each cell's score is the worst-case intensity across all FOV angles,
    using the worst color channel (D65-normalized).  A cell is "in the
    eye-box" only if that score exceeds ``intensity_threshold``.

    The loss is ``1 - active_fraction``, so 0 = all cells active (best).
    A soft sigmoid makes the threshold differentiable.
    """
    intensity_threshold: float = 0.005  # min acceptable D65-normalized intensity
    d65_weights: jnp.ndarray = None     # (3,) per-color normalization
    sigmoid_sharpness: float = 50.0     # sigmoid steepness at threshold
    w_area: float = 1.0                 # weight for area loss
    w_margin: float = 0.1              # weight for above-threshold margin bonus

    def __post_init__(self):
        if self.d65_weights is None:
            self.d65_weights = D65_WEIGHTS


def eyebox_area_merit(response, config=None):
    """Eye-box area merit: maximize the fraction of cells with full FOV.

    For each cell, computes the worst-case intensity across all FOV
    angles and all color channels (D65-normalized).  Cells above the
    threshold are "active".  The loss penalizes inactive cells and
    gives a small bonus for pushing active cells further above threshold.

    Args:
        response: (S, A, 3) from ``compute_eyebox_response``
        config: EyeboxAreaConfig (uses defaults if None)

    Returns:
        scalar loss (lower is better; 0 = all cells active)
    """
    if config is None:
        config = EyeboxAreaConfig()

    # D65-normalize: equalize color channels so min-over-colors means
    # "worst color relative to its white-balance target"
    d65 = config.d65_weights[None, None, :]  # (1, 1, 3)
    normalized = response / (d65 + 1e-12)    # (S, A, 3)

    # Worst color per (cell, angle)
    worst_color = jnp.min(normalized, axis=-1)  # (S, A)

    # Worst angle per cell — the bottleneck that determines eye-box membership
    cell_min = jnp.min(worst_color, axis=-1)    # (S,)

    # Soft active count via sigmoid, scaled relative to threshold
    sigma = config.sigmoid_sharpness
    threshold = config.intensity_threshold
    active = jax.nn.sigmoid(sigma * (cell_min / (threshold + 1e-12) - 1.0))  # (S,)

    active_fraction = jnp.mean(active)
    area_loss = 1.0 - active_fraction

    # Margin term: encourage borderline cells to push further above threshold.
    # Uses (1 - active) weighting so it focuses on cells near/below threshold.
    shortfall = jax.nn.relu(threshold - cell_min)  # (S,) how far below threshold
    margin_loss = jnp.mean(shortfall) / (threshold + 1e-12)

    return config.w_area * area_loss + config.w_margin * margin_loss


def cell_grid_from_cell_size(center, normal, width, height, cell_size):
    """Cell-sized grid on the eyebox plane.

    Computes ``(nx, ny)`` from ``cell_size`` so each cell is approximately
    ``cell_size × cell_size`` mm. Returns ``(points, nx, ny)``.
    """
    nx = max(1, int(jnp.ceil(width / cell_size)))
    ny = max(1, int(jnp.ceil(height / cell_size)))
    points = planar_grid_points(center, normal, width / 2, height / 2, nx, ny)
    return points, nx, ny


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
