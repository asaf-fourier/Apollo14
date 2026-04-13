"""Route — a sequential optical path as a JAX pytree.

A Route encodes *how light travels* through the system:
  preamble (aperture + entry refraction) → mirror scan → branch (exit + pupil)

Different routes through the same system model different light paths
(display, see-through, ghost).
"""

from typing import NamedTuple

import jax.numpy as jnp

from apollo14.elements.surface import PartialMirror, MirrorState
from apollo14.elements.refracting_surface import RefractingSurface, RefractState
from apollo14.elements.aperture import RectangularAperture, ApertureState
from apollo14.elements.pupil import RectangularPupil, DetectorState
from apollo14.elements.glass_block import GlassBlock
from apollo14.geometry import compute_local_axes
from apollo14.materials import air
from apollo14.system import OpticalSystem


class Route(NamedTuple):
    """Sequential optical path — fully JAX-compatible (NamedTuple = auto pytree).

    Fields:
        aperture: ApertureState for pre-entry clipping.
        has_aperture: bool scalar — if False, aperture check is skipped.
        entry_face: RefractState for air→glass refraction at entry.
        mirrors: MirrorState with stacked (M, ...) arrays for lax.scan.
        exit_face: RefractState tiled to (M, ...) for per-mirror branch.
        target: DetectorState tiled to (M, ...) for per-mirror branch.
        n_glass: scalar refractive index inside the chassis.
    """
    aperture: ApertureState
    has_aperture: jnp.ndarray       # bool scalar
    entry_face: RefractState
    mirrors: MirrorState            # stacked (M, ...)
    exit_face: RefractState         # tiled (M, ...)
    target: DetectorState           # tiled (M, ...)
    n_glass: jnp.ndarray            # scalar


def _stack_mirror_states(mirrors: list[PartialMirror]) -> MirrorState:
    """Stack a list of PartialMirror states into a single MirrorState with leading (M,) dim."""
    states = [m.state for m in mirrors]
    return MirrorState(
        position=jnp.stack([s.position for s in states]),
        normal=jnp.stack([s.normal for s in states]),
        half_extents=jnp.stack([s.half_extents for s in states]),
        reflectance=jnp.stack([s.reflectance for s in states]),
        local_x=jnp.stack([s.local_x for s in states]),
        local_y=jnp.stack([s.local_y for s in states]),
    )


def _tile_state(state, M):
    """Tile a state's arrays to have a leading (M,) dimension."""
    return type(state)(*(jnp.broadcast_to(f[None], (M,) + f.shape).copy()
                         for f in state))


def _dummy_aperture() -> ApertureState:
    """Create a no-op aperture state (used when system has no aperture)."""
    z = jnp.zeros(3)
    return ApertureState(
        position=z,
        normal=jnp.array([0.0, 0.0, 1.0]),
        half_extents=jnp.array([1e6, 1e6]),  # huge — passes everything
        local_x=jnp.array([1.0, 0.0, 0.0]),
        local_y=jnp.array([0.0, 1.0, 0.0]),
    )


def display_route(system: OpticalSystem, wavelength: float,
                  entry_face: str = "back", exit_face: str = "top") -> Route:
    """Build the display path: projector → mirrors → pupil.

    Light enters through ``entry_face`` of the chassis, hits M mirrors
    (partial reflection at each), reflected beams exit through ``exit_face``
    and reach the pupil.

    Args:
        system: OpticalSystem with chassis, mirrors, aperture, and pupil.
        wavelength: Wavelength for refractive index lookup.
        entry_face: Name of the chassis face where light enters (default "back").
        exit_face: Name of the chassis face where reflected light exits (default "top").

    Returns:
        Route ready for tracing with ``trace_ray``/``trace_beam``/``trace_batch``.
    """
    # Find elements by type
    chassis = next(e for e in system.elements if isinstance(e, GlassBlock))
    mirrors = [e for e in system.elements if isinstance(e, PartialMirror)]
    pupil = next(e for e in system.elements if isinstance(e, RectangularPupil))
    apertures = [e for e in system.elements if isinstance(e, RectangularAperture)]

    n_air = float(air.n(wavelength))
    n_glass = float(chassis.material.n(wavelength))
    M = len(mirrors)

    # Aperture
    if apertures:
        ap_state = apertures[0].state
        has_aperture = jnp.array(True)
    else:
        ap_state = _dummy_aperture()
        has_aperture = jnp.array(False)

    # Entry face: air → glass
    entry_surf = chassis.face(entry_face, n1=n_air, n2=n_glass)
    entry_state = entry_surf.state

    # Mirrors (stacked)
    mirror_states = _stack_mirror_states(mirrors)

    # Exit face: glass → air (tiled for per-mirror branch)
    exit_surf = chassis.face(exit_face, n1=n_glass, n2=n_air)
    exit_state = _tile_state(exit_surf.state, M)

    # Pupil (tiled for per-mirror branch)
    target_state = _tile_state(pupil.state, M)

    return Route(
        aperture=ap_state,
        has_aperture=has_aperture,
        entry_face=entry_state,
        mirrors=mirror_states,
        exit_face=exit_state,
        target=target_state,
        n_glass=jnp.asarray(n_glass),
    )
