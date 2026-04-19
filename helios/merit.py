"""Combiner route construction and spectral constants.

Provides :func:`build_combiner_pupil_routes` — the standard way to
construct wavelength-resolved, pupil-terminated branch routes for
the Talos combiner — plus shared spectral constants (LED peak
wavelengths, D65 white-balance weights).
"""

from typing import Sequence

import jax.numpy as jnp

from apollo14.route import Route, build_route, branch_path, absorb
from apollo14.trace import prepare_route
from apollo14.system import OpticalSystem
from apollo14.units import nm


# ── PlayNitride microLED peak wavelengths ─────────────────────────────────────

LED_RED = 627.0 * nm
LED_GREEN = 545.0 * nm
LED_BLUE = 446.0 * nm

DEFAULT_WAVELENGTHS = jnp.array([LED_RED, LED_GREEN, LED_BLUE])

# D65 relative power at those wavelengths (from CIE D65 standard illuminant).
# These are the ratios the projector must produce for white appearance.
# Normalized so they sum to 1.
_D65_RAW = jnp.array([81.8, 101.0, 104.0])  # D65 at 627/545/446 nm
D65_WEIGHTS = _D65_RAW / _D65_RAW.sum()


# ── CIE D65 standard illuminant (5 nm intervals, 400–700 nm) ────────────────

_D65_TABLE_NM = jnp.array([
    400, 405, 410, 415, 420, 425, 430, 435, 440, 445,
    450, 455, 460, 465, 470, 475, 480, 485, 490, 495,
    500, 505, 510, 515, 520, 525, 530, 535, 540, 545,
    550, 555, 560, 565, 570, 575, 580, 585, 590, 595,
    600, 605, 610, 615, 620, 625, 630, 635, 640, 645,
    650, 655, 660, 665, 670, 675, 680, 685, 690, 695, 700,
], dtype=jnp.float32)

_D65_TABLE_SPD = jnp.array([
    82.75, 87.12, 91.49, 92.46, 93.43, 90.06, 86.68, 95.77, 104.86, 110.94,
    117.01, 117.41, 117.81, 116.34, 114.86, 115.39, 115.92, 112.37, 108.81, 109.08,
    109.35, 108.58, 107.80, 106.30, 104.79, 106.24, 107.69, 106.05, 104.41, 104.23,
    104.05, 102.02, 100.00, 98.17, 96.33, 96.06, 95.79, 92.24, 88.69, 89.35,
    90.01, 89.80, 89.60, 88.65, 87.70, 85.49, 83.29, 83.49, 83.69, 81.86,
    80.03, 80.12, 80.21, 82.28, 84.35, 78.91, 73.47, 68.66, 64.74, 61.60, 58.89,
], dtype=jnp.float32)

_D65_TABLE_WL = _D65_TABLE_NM * nm


def d65_weights_at(wavelengths: jnp.ndarray) -> jnp.ndarray:
    """Interpolate D65 illuminant at arbitrary wavelengths, normalized to sum to 1.

    Args:
        wavelengths: (N,) wavelengths in internal length units.

    Returns:
        (N,) D65 weights normalized so they sum to 1.
    """
    raw = jnp.interp(wavelengths, _D65_TABLE_WL, _D65_TABLE_SPD)
    return raw / raw.sum()


# ── Route construction helper ────────────────────────────────────────────────

def build_combiner_pupil_routes(system: OpticalSystem,
                                wavelengths: Sequence[float],
                                num_mirrors: int = 6,
                                pupil_name: str = "pupil",
                                chassis_name: str = "chassis",
                                ) -> list[list[Route]]:
    """Build reflected-branch routes that terminate on the pupil.

    One branch per mirror (reflect off it, exit the chassis, absorb at the
    pupil), wavelength-resolved once per color. The returned list is shaped
    ``(n_wavelengths, num_mirrors)``.
    """
    main_path: list = [
        "aperture",
        (chassis_name, "back"),
    ]
    main_path.extend(f"mirror_{i}" for i in range(num_mirrors))
    main_path.append((chassis_name, "front"))

    tail = [(chassis_name, "top"), absorb(pupil_name)]
    branch_routes = [
        build_route(system, branch_path(main_path, at=f"mirror_{i}", tail=tail))
        for i in range(num_mirrors)
    ]

    return [
        [prepare_route(r, wl) for r in branch_routes]
        for wl in wavelengths
    ]
