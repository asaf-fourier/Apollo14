"""Generic single-path sequential tracer.

A route (built in ``apollo14.route``) is a flat stacked ``Surface`` — one
``lax.scan`` worth of surfaces, wavelength-agnostic. ``prepare_beam`` turns
it into a ``Beam`` for a specific wavelength + initial intensity; ``trace``
consumes the beam.

Branching (reflected daughter rays, multi-path combiner) is authored as
separate routes, not as branching inside the scan.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from apollo14.surface import Surface, surface_step, interp_n


class Beam(NamedTuple):
    """A route prepared for a specific wavelength and initial intensity."""
    surfaces: Surface              # stacked; n1/n2 are scalar (wavelength-resolved)
    intensity: jnp.ndarray         # scalar initial intensity


class TraceResult(NamedTuple):
    """Result of tracing one ray through a beam.

    Shapes are ``(N, ...)`` for N surfaces, with a leading ``R`` batch dim
    when produced via ``trace_beam``.
    """
    hits: jnp.ndarray              # (..., N, 3) raw plane intersection at each step
    valids: jnp.ndarray            # (..., N) bool — whether each step was valid
    final_pos: jnp.ndarray         # (..., 3) position after the final step
    final_dir: jnp.ndarray         # (..., 3) direction after the final step
    final_intensity: jnp.ndarray   # (...,) intensity after the final step


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

def trace(beam: Beam, origin, direction,
          color_idx: int = 0) -> TraceResult:
    """Trace one ray through the beam via a single ``lax.scan``."""
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
        lambda o: trace(beam, o, direction, color_idx)
    )(origins)
