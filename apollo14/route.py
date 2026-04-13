"""Route — a sequential optical path as a JAX pytree.

A Route encodes *how light travels* through the system using uniform
``SurfaceState`` fields throughout. All elements share the same state
shape — element-specific behavior emerges from parameter values
(n1/n2, reflectance, kill_on_miss). No type dispatch needed.

Three scans:
  1. preamble — aperture + entry face (small, fixed)
  2. mirrors  — mirror stack (variable M, produces branch origins)
  3. branch   — exit face + pupil (shared by all mirrors, vmapped)

For the Talos-specific route with typed element states, see ``talos_route.py``.
"""

from typing import NamedTuple

import jax.numpy as jnp

from apollo14.elements.surface import PartialMirror
from apollo14.elements.aperture import RectangularAperture
from apollo14.elements.pupil import RectangularPupil
from apollo14.elements.glass_block import GlassBlock
from apollo14.surface import SurfaceState
from apollo14.geometry import compute_local_axes
from apollo14.materials import air
from apollo14.system import OpticalSystem


class Route(NamedTuple):
    """Sequential optical path — fully JAX-compatible (NamedTuple = auto pytree).

    Fields:
        preamble: SurfaceState with (P, ...) stacked arrays.
        mirrors: SurfaceState with (M, ...) stacked arrays.
        branch: SurfaceState with (B, ...) stacked arrays.
    """
    preamble: SurfaceState      # (P, ...) — aperture + entry face
    mirrors: SurfaceState       # (M, ...) — mirror stack
    branch: SurfaceState        # (B, ...) — exit face + pupil


# ── Helpers ──────────────────────────────────────────────────────────────────

def _to_surface(position, normal, half_extents, local_x, local_y, *,
                n1=1.0, n2=1.0, reflectance=None, kill_on_miss=True):
    """Build a SurfaceState from raw parameters."""
    if reflectance is None:
        reflectance = jnp.zeros(3)
    return SurfaceState(
        position=jnp.asarray(position, dtype=jnp.float32),
        normal=jnp.asarray(normal, dtype=jnp.float32),
        half_extents=jnp.asarray(half_extents, dtype=jnp.float32),
        local_x=jnp.asarray(local_x, dtype=jnp.float32),
        local_y=jnp.asarray(local_y, dtype=jnp.float32),
        n1=jnp.float32(n1),
        n2=jnp.float32(n2),
        reflectance=jnp.asarray(reflectance, dtype=jnp.float32),
        kill_on_miss=jnp.bool_(kill_on_miss),
    )


def _stack_surface_states(states: list[SurfaceState]) -> SurfaceState:
    """Stack a list of SurfaceStates into one with leading (N,) dim."""
    return SurfaceState(*(jnp.stack([getattr(s, f) for s in states])
                          for f in SurfaceState._fields))


# ── Route builders ───────────────────────────────────────────────────────────

def display_route(system: OpticalSystem, wavelength: float,
                  entry_face: str = "back",
                  exit_face: str = "top") -> Route:
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
    chassis = next(e for e in system.elements if isinstance(e, GlassBlock))
    mirrors = [e for e in system.elements if isinstance(e, PartialMirror)]
    pupil = next(e for e in system.elements if isinstance(e, RectangularPupil))
    apertures = [e for e in system.elements if isinstance(e, RectangularAperture)]

    n_air = float(air.n(wavelength))
    n_glass = float(chassis.material.n(wavelength))

    # ── Preamble: aperture + entry face ──
    if apertures:
        ap = apertures[0]
        lx, ly = compute_local_axes(ap.state.normal)
        aperture_s = _to_surface(
            ap.state.position, ap.state.normal, ap.state.half_extents, lx, ly,
            kill_on_miss=True)
    else:
        # Dummy aperture: huge opening at entry face position, never blocks
        entry_surf = chassis.face(entry_face, n1=n_air, n2=n_glass)
        aperture_s = _to_surface(
            entry_surf.state.position,
            entry_surf.state.normal,
            jnp.array([1e6, 1e6]),
            entry_surf.state.local_x,
            entry_surf.state.local_y,
            kill_on_miss=False)

    entry_surf = chassis.face(entry_face, n1=n_air, n2=n_glass)
    es = entry_surf.state
    entry_s = _to_surface(
        es.position, es.normal, es.half_extents, es.local_x, es.local_y,
        n1=n_air, n2=n_glass, kill_on_miss=True)

    preamble = _stack_surface_states([aperture_s, entry_s])

    # ── Mirrors: n1=n2=n_glass, reflectance from element, kill_on_miss=False ──
    mirror_surfaces = []
    for m in mirrors:
        ms = m.state
        mirror_surfaces.append(_to_surface(
            ms.position, ms.normal, ms.half_extents, ms.local_x, ms.local_y,
            n1=n_glass, n2=n_glass, reflectance=ms.reflectance,
            kill_on_miss=False))

    mirror_stack = _stack_surface_states(mirror_surfaces)

    # ── Branch: exit face + pupil ──
    exit_surf = chassis.face(exit_face, n1=n_glass, n2=n_air)
    xs = exit_surf.state
    exit_s = _to_surface(
        xs.position, xs.normal, xs.half_extents, xs.local_x, xs.local_y,
        n1=n_glass, n2=n_air, kill_on_miss=True)

    ps = pupil.state
    pupil_s = _to_surface(
        ps.position, ps.normal, ps.half_extents, ps.local_x, ps.local_y,
        kill_on_miss=True)

    branch = _stack_surface_states([exit_s, pupil_s])

    return Route(preamble=preamble, mirrors=mirror_stack, branch=branch)
