"""Merit function for evaluating combiner performance.

Samples the pupil at multiple positions and the FOV at multiple angles,
traces R/G/B rays, and computes MSE against a D65 white-balanced uniform target.

The output array shape is: (n_pupil_y, n_pupil_x, n_angle_y, n_angle_x, 3)
where the last axis is [R, G, B] intensity arriving at each pupil sample.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple

import jax.numpy as jnp

from apollo14.projector import Projector, scan_directions
from apollo14.trace import Beam, trace_beam, prepare_beam
from apollo14.binning import bin_hits_to_nearest
from apollo14.route import build_route, branch_path, absorb
from apollo14.geometry import planar_grid_points
from apollo14.system import OpticalSystem
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


# ── Beam construction helper ─────────────────────────────────────────────────

def build_combiner_pupil_beams(system: OpticalSystem,
                               wavelengths: Sequence[float],
                               num_mirrors: int = 6,
                               pupil_name: str = "pupil",
                               chassis_name: str = "chassis",
                               ) -> list[list[Beam]]:
    """Build reflected-branch beams that terminate on the pupil.

    One branch per mirror (reflect off it, exit the chassis, absorb at the
    pupil), prepared once per wavelength. The returned list is shaped
    ``(n_wavelengths, num_mirrors)``.
    """
    main_path: list = [
        "aperture",
        (chassis_name, "back"),
    ]
    main_path.extend(f"mirror_{i}" for i in range(num_mirrors))
    main_path.append((chassis_name, "front"))

    tail = [(chassis_name, "front"), absorb(pupil_name)]
    branch_routes = [
        build_route(system, branch_path(main_path, at=f"mirror_{i}", tail=tail))
        for i in range(num_mirrors)
    ]

    return [
        [prepare_beam(r, float(wl)) for r in branch_routes]
        for wl in wavelengths
    ]


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


def simulate_pupil_response(beams_per_color: Sequence[Sequence[Beam]],
                            projector: Projector,
                            pupil_center, pupil_normal, pupil_radius,
                            config: MeritConfig,
                            x_fov: float, y_fov: float,
                            ) -> jnp.ndarray:
    """Trace R/G/B rays across pupil positions and FOV angles.

    Args:
        beams_per_color: ``(n_colors, n_branches)`` — pupil-terminated beams
            (e.g. from ``build_combiner_pupil_beams``). Each color's inner
            list is summed over branches at the pupil.

    Returns:
        intensity: (n_pupil_y, n_pupil_x, n_angle_y, n_angle_x, 3) array
    """
    flat_positions = planar_grid_points(
        pupil_center, pupil_normal, pupil_radius, pupil_radius,
        config.pupil_nx, config.pupil_ny,
    )  # (pny*pnx, 3)

    scan_dirs, _ = scan_directions(
        projector.direction, x_fov, y_fov,
        config.angle_nx, config.angle_ny,
    )  # (any, anx, 3)

    n_wl = len(beams_per_color)
    result = jnp.zeros((config.pupil_ny, config.pupil_nx,
                        config.angle_ny, config.angle_nx, n_wl))

    for ai_y in range(config.angle_ny):
        for ai_x in range(config.angle_nx):
            direction = scan_dirs[ai_y, ai_x]
            origins, _, _, _ = projector.generate_rays(direction=direction)

            for wi, branch_beams in enumerate(beams_per_color):
                binned = jnp.zeros(flat_positions.shape[0])
                for beam in branch_beams:
                    tr = trace_beam(beam, origins, direction, color_idx=wi)
                    binned = binned + bin_hits_to_nearest(
                        tr, flat_positions, stop_grad=False)
                binned_2d = binned.reshape(config.pupil_ny, config.pupil_nx)
                result = result.at[:, :, ai_y, ai_x, wi].add(binned_2d)

    return result


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


def evaluate_merit(beams_per_color: Sequence[Sequence[Beam]],
                   projector: Projector,
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
        beams_per_color, projector, pupil_center, pupil_normal, pupil_radius,
        config, x_fov, y_fov,
    )
    target = build_target(config)
    mse = merit_mse(simulated, target)
    return mse, simulated, target
