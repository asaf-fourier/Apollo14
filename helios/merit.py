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
