"""Generic single-path sequential tracer.

A route is a flat stacked ``Surface`` — one ``lax.scan`` worth of surfaces,
built once, wavelength-agnostic. ``prepare_beam`` turns it into a ``Beam``
for a specific wavelength + initial intensity; ``trace_ray`` consumes the
beam.

Branching (reflected daughter rays, multi-path combiner) is out of scope
for v1 — layer it on top later.
"""

from typing import NamedTuple, Sequence, Union

import jax
import jax.numpy as jnp

from apollo14.surface import Surface, TRANSMIT, surface_step, interp_n
from apollo14.system import OpticalSystem


# An element reference is either a plain string ("mirror_0") or a
# (block, face) tuple (("chassis", "back")). A path entry is that
# reference, optionally wrapped with a mode: (ref, mode).
ElementRef = Union[str, tuple]
PathEntry = Union[ElementRef, tuple]


class Beam(NamedTuple):
    """A route prepared for a specific wavelength and initial intensity."""
    surfaces: Surface              # stacked; n1/n2 are scalar (wavelength-resolved)
    intensity: jnp.ndarray         # scalar initial intensity


class TraceResult(NamedTuple):
    """Result of tracing one ray through a beam.

    Shapes are (N, ...) for N surfaces, or with extra leading batch dims
    when produced via ``trace_beam`` / ``trace_batch``.
    """
    hits: jnp.ndarray              # (..., N, 3) raw plane intersection at each step
    valids: jnp.ndarray            # (..., N) bool — whether each step was valid
    final_pos: jnp.ndarray         # (..., 3) position after the final step
    final_dir: jnp.ndarray         # (..., 3) direction after the final step
    final_intensity: jnp.ndarray   # (...,) intensity after the final step


# ── Route building ───────────────────────────────────────────────────────────

def _stack_surfaces(states: Sequence[Surface]) -> Surface:
    """Stack per-surface states into one with a leading (N,) dim on every leaf."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)


def _parse_entry(entry: PathEntry) -> tuple:
    """Split a path entry into ``(ref, mode)``.

    Accepted forms:
        "mirror_0"                         → transmit
        ("chassis", "back")                → transmit (block+face ref)
        ("mirror_0", REFLECT)              → explicit mode on plain ref
        (("chassis", "back"), REFLECT)     → explicit mode on block+face ref
    """
    if isinstance(entry, str):
        return entry, TRANSMIT
    if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], int):
        return entry[0], int(entry[1])
    return entry, TRANSMIT


def build_route(system: OpticalSystem, path: Sequence[PathEntry]) -> Surface:
    """Build a stacked wavelength-agnostic ``Surface`` from a flat path.

    Each entry is either a plain element name (``"mirror_0"``), a
    ``(block, face)`` tuple (``("chassis", "back")``), or a ``(ref, mode)``
    pair where ``mode`` is one of ``TRANSMIT``/``REFLECT``/``ABSORB``.

    The builder tracks the current medium as a ``Material`` instance
    (starting from ``system.env_material``) and hands it to each element's
    ``to_generic_surface`` — no wavelength involved.
    """
    current = system.env_material
    surfaces: list[Surface] = []
    for entry in path:
        name, mode = _parse_entry(entry)
        elem = system.resolve(name)
        surf, current = elem.to_generic_surface(current, mode)
        surfaces.append(surf)

    return _stack_surfaces(surfaces)


# ── Beam preparation ─────────────────────────────────────────────────────────

def prepare_beam(route: Surface, wavelength, intensity=1.0) -> Beam:
    """Resolve a route's per-surface n1/n2 at ``wavelength`` and pair it
    with an initial ``intensity``.

    The route itself is wavelength-agnostic — call this once per wavelength
    (and per intensity if you vary angle/wavelength dependent power).
    """
    n1 = jax.vmap(lambda m: interp_n(m, wavelength))(route.n1)
    n2 = jax.vmap(lambda m: interp_n(m, wavelength))(route.n2)
    ready = route._replace(n1=n1, n2=n2)
    return Beam(surfaces=ready,
                intensity=jnp.asarray(intensity, dtype=jnp.float32))


# ── Tracing ──────────────────────────────────────────────────────────────────

def trace_ray(beam: Beam, origin, direction,
              color_idx: int = 0) -> TraceResult:
    """Trace one ray along the beam via a single ``lax.scan``."""
    def step(carry, state):
        pos, d, inten = carry
        out_pos, out_dir, out_inten, hit, valid = surface_step(
            state, pos, d, inten, color_idx)
        return (out_pos, out_dir, out_inten), (hit, valid)

    init = (origin, direction, beam.intensity)
    (final_pos, final_dir, final_inten), (hits, valids) = jax.lax.scan(
        step, init, beam.surfaces)

    return TraceResult(
        hits=hits,
        valids=valids,
        final_pos=final_pos,
        final_dir=final_dir,
        final_intensity=final_inten,
    )


def trace_beam(beam: Beam, origins, direction,
               color_idx: int = 0) -> TraceResult:
    """Trace N rays sharing one direction through a beam."""
    return jax.vmap(
        lambda o: trace_ray(beam, o, direction, color_idx)
    )(origins)


def trace_batch(beam: Beam, origins, directions,
                color_idx: int = 0) -> TraceResult:
    """Trace N rays with per-ray directions through a beam."""
    return jax.vmap(
        lambda o, d: trace_ray(beam, o, d, color_idx)
    )(origins, directions)


# ── Combiner helper ──────────────────────────────────────────────────────────

def combiner_main_path(system: OpticalSystem,
                       entry_face: str = "back",
                       exit_face: str = "front") -> Surface:
    """Build the straight-through main path for a Talos-style combiner.

    The path is: aperture (if present) → entry chassis face → each partial
    mirror in order (all transmit) → exit chassis face. No pupil, no
    reflected branches — branching will be layered on later.
    """
    chassis = next(e for e in system.elements if hasattr(e, "get_face"))
    mirrors = [e for e in system.elements
               if hasattr(e, "reflectance") and hasattr(e, "name")
               and getattr(e, "name", "").startswith("mirror")]
    apertures = [e for e in system.elements
                 if getattr(e, "name", None) == "aperture"]

    path: list[PathEntry] = []
    if apertures:
        path.append(apertures[0].name)
    path.append((chassis.name, entry_face))
    path.extend(m.name for m in mirrors)
    path.append((chassis.name, exit_face))

    return build_route(system, path)
