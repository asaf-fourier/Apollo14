"""Merit function for evaluating combiner performance.

Samples the pupil at multiple positions and the FOV at multiple angles,
traces R/G/B rays, and computes MSE against a D65 white-balanced uniform target.

The output array shape is: (n_pupil_y, n_pupil_x, n_angle_y, n_angle_x, 3)
where the last axis is [R, G, B] intensity arriving at each pupil sample.
"""

from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp

from apollo14.system import OpticalSystem
from apollo14.projector import Projector, scan_directions
from apollo14.tracer import trace_nonsequential
from apollo14.geometry import normalize, compute_local_axes
from apollo14.units import nm


# ── Typical microLED peak wavelengths ─────────────────────────────────────────

LED_RED = 630.0 * nm
LED_GREEN = 525.0 * nm
LED_BLUE = 460.0 * nm

DEFAULT_WAVELENGTHS = jnp.array([LED_RED, LED_GREEN, LED_BLUE])

# D65 relative power at those wavelengths (from CIE D65 standard illuminant).
# These are the ratios the projector must produce for white appearance.
# Normalized so they sum to 1.
_D65_RAW = jnp.array([78.0, 107.0, 82.0])  # approximate D65 at 630/525/460 nm
D65_WEIGHTS = _D65_RAW / _D65_RAW.sum()


# ── Pupil sampling ────────────────────────────────────────────────────────────

def pupil_sample_grid(center, normal, radius, nx, ny):
    """Generate a grid of sample points on the pupil plane.

    Returns:
        positions: (ny, nx, 3) array of 3D sample positions
    """
    lx, ly = compute_local_axes(normal)

    xs = jnp.linspace(-radius, radius, nx)
    ys = jnp.linspace(-radius, radius, ny)
    gx, gy = jnp.meshgrid(xs, ys)  # (ny, nx)

    positions = (center[None, None, :]
                 + gx[:, :, None] * lx[None, None, :]
                 + gy[:, :, None] * ly[None, None, :])
    return positions


# ── Simulation ────────────────────────────────────────────────────────────────

@dataclass
class MeritConfig:
    """Configuration for merit function evaluation."""
    wavelengths: jnp.ndarray = None         # (3,) R, G, B wavelengths
    d65_weights: jnp.ndarray = None         # (3,) D65 target ratios (sum to 1)
    target_efficiency: float = 0.10         # fraction of projector light at pupil
    pupil_nx: int = 3                       # pupil sample grid
    pupil_ny: int = 3
    angle_nx: int = 5                       # FOV angular sample grid
    angle_ny: int = 5

    def __post_init__(self):
        if self.wavelengths is None:
            self.wavelengths = DEFAULT_WAVELENGTHS
        if self.d65_weights is None:
            self.d65_weights = D65_WEIGHTS


def simulate_pupil_response(system: OpticalSystem, projector: Projector,
                            pupil_center, pupil_normal, pupil_radius,
                            config: MeritConfig,
                            x_fov: float, y_fov: float,
                            ) -> jnp.ndarray:
    """Trace R/G/B rays across pupil positions and FOV angles.

    For each (pupil_sample, angle, color), fires a ray from the projector
    and checks if it arrives at that pupil sample location.

    Returns:
        intensity: (n_pupil_y, n_pupil_x, n_angle_y, n_angle_x, 3) array
    """
    pupil_positions = pupil_sample_grid(
        pupil_center, pupil_normal, pupil_radius,
        config.pupil_nx, config.pupil_ny,
    )  # (pny, pnx, 3)

    scan_dirs, scan_angles = scan_directions(
        projector.direction, x_fov, y_fov,
        config.angle_nx, config.angle_ny,
    )  # (any, anx, 3), (any, anx, 2)

    n_wl = config.wavelengths.shape[0]
    result = jnp.zeros((config.pupil_ny, config.pupil_nx,
                         config.angle_ny, config.angle_nx, n_wl))

    for ai_y in range(config.angle_ny):
        for ai_x in range(config.angle_nx):
            direction = scan_dirs[ai_y, ai_x]
            origins, directions, intensities, _ = projector.generate_rays(direction=direction)

            for wi in range(n_wl):
                wl = config.wavelengths[wi]

                for ri in range(origins.shape[0]):
                    tr = trace_nonsequential(
                        system, origins[ri], directions[ri], wl,
                        intensity=float(intensities[ri]),
                    )
                    if tr.pupil_hit is None:
                        continue

                    hit_pt = tr.pupil_hit.point
                    hit_intensity = float(tr.pupil_hit.intensity)

                    # Assign to nearest pupil sample
                    pi_y, pi_x = _nearest_pupil_index(
                        hit_pt, pupil_positions,
                        config.pupil_ny, config.pupil_nx,
                    )
                    if pi_y >= 0:
                        result = result.at[pi_y, pi_x, ai_y, ai_x, wi].add(hit_intensity)

    return result


def _nearest_pupil_index(hit_point, pupil_positions, pny, pnx):
    """Find the nearest pupil grid sample to a hit point.

    Returns (iy, ix) or (-1, -1) if outside the grid spacing.
    """
    flat_pos = pupil_positions.reshape(-1, 3)
    dists = jnp.linalg.norm(flat_pos - hit_point[None, :], axis=1)
    idx = jnp.argmin(dists)
    min_dist = dists[idx]

    # Reject if too far from any sample (> half the grid spacing)
    if pnx > 1:
        spacing_x = jnp.linalg.norm(pupil_positions[0, 1] - pupil_positions[0, 0])
    else:
        spacing_x = 1e10
    if pny > 1:
        spacing_y = jnp.linalg.norm(pupil_positions[1, 0] - pupil_positions[0, 0])
    else:
        spacing_y = 1e10
    max_dist = jnp.maximum(spacing_x, spacing_y) * 0.75

    if min_dist > max_dist:
        return -1, -1

    iy = int(idx) // pnx
    ix = int(idx) % pnx
    return iy, ix


# ── Merit function ────────────────────────────────────────────────────────────

def build_target(config: MeritConfig) -> jnp.ndarray:
    """Build the uniform D65 target intensity array.

    Returns:
        target: (n_pupil_y, n_pupil_x, n_angle_y, n_angle_x, 3)
        Each element is target_efficiency * d65_weight_for_that_color.
    """
    target_per_color = config.target_efficiency * config.d65_weights  # (3,)
    return jnp.broadcast_to(
        target_per_color[None, None, None, None, :],
        (config.pupil_ny, config.pupil_nx,
         config.angle_ny, config.angle_nx, 3),
    )


def merit_mse(simulated: jnp.ndarray, target: jnp.ndarray) -> float:
    """Mean squared error between simulated and target intensity.

    Both arrays are (n_pupil_y, n_pupil_x, n_angle_y, n_angle_x, 3).
    """
    return float(jnp.mean((simulated - target) ** 2))


def evaluate_merit(system: OpticalSystem, projector: Projector,
                   pupil_center, pupil_normal, pupil_radius,
                   x_fov: float, y_fov: float,
                   config: MeritConfig = None) -> Tuple[float, jnp.ndarray, jnp.ndarray]:
    """Full merit evaluation: simulate + compare to D65 target.

    Returns:
        (mse, simulated, target) where simulated and target are
        (n_pupil_y, n_pupil_x, n_angle_y, n_angle_x, 3) arrays.
    """
    if config is None:
        config = MeritConfig()

    simulated = simulate_pupil_response(
        system, projector, pupil_center, pupil_normal, pupil_radius,
        config, x_fov, y_fov,
    )
    target = build_target(config)
    mse = merit_mse(simulated, target)
    return mse, simulated, target
