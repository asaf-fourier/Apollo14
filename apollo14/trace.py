"""Segmented single-path sequential tracer.

A ``Route`` is a tuple of typed segment pytrees (see ``apollo14.route``).
``prepare_route`` resolves wavelength-dependent face materials to scalar
indices. ``trace`` walks the segments in Python, dispatching each to the
matching element's ``jax_interact`` function; consecutive transmit
mirrors are handled by ``lax.scan`` for compile-time efficiency.

Branching (reflected daughter rays, multi-path combiner) is authored as
separate routes, not as branching inside the tracer.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from apollo14.elements.aperture import ApertureSeg, aperture_interact
from apollo14.elements.glass_block import FaceSeg, face_interact
from apollo14.elements.partial_mirror import (
    MirrorStackSeg,
    ReflectMirrorSeg,
    mirror_reflect_one,
    mirror_transmit_one,
)
from apollo14.elements.pupil import PupilSeg, pupil_interact
from apollo14.materials import MaterialData
from apollo14.ray import Ray
from apollo14.route import Route


class TraceResult(NamedTuple):
    """Result of tracing one ray (or a batch of rays) through a ``Route``.

    Shapes are ``(..., N, 3)`` / ``(..., N)`` for N interaction steps,
    with a leading batch dim when produced via ``trace_rays``.
    """
    hits: jnp.ndarray
    valids: jnp.ndarray
    final_pos: jnp.ndarray
    final_dir: jnp.ndarray
    final_intensity: jnp.ndarray


# ── Route preparation ────────────────────────────────────────────────────────

def _interp_n(mat: MaterialData, wavelength):
    return jnp.interp(wavelength, mat.wavelengths, mat.n_values)


def _resolve_face(seg: FaceSeg, wavelength) -> FaceSeg:
    return seg._replace(
        n1=_interp_n(seg.n1, wavelength),
        n2=_interp_n(seg.n2, wavelength),
    )


def prepare_route(route: Route, wavelength) -> Route:
    """Resolve every ``FaceSeg``'s ``MaterialData`` to a scalar n at
    ``wavelength``. Mirror/aperture/pupil segments pass through unchanged.
    """
    new_segs = tuple(
        _resolve_face(s, wavelength) if isinstance(s, FaceSeg) else s
        for s in route.segments
    )
    return Route(segments=new_segs)


# ── Tracing ──────────────────────────────────────────────────────────────────

def trace(route: Route, ray: Ray, wavelength: float = 0.0) -> TraceResult:
    """Trace one ``Ray`` through a wavelength-resolved ``Route``.

    ``wavelength`` is passed to each element's interact function. Elements
    that care (partial mirrors) use it to interpolate their reflectance
    curve; aperture/face/pupil ignore it at trace time — face index is
    already resolved to a scalar by ``prepare_route``.
    """
    hits_accum = []
    valids_accum = []

    def _push(hit, valid):
        hits_accum.append(hit[None, :])
        valids_accum.append(valid[None])

    for seg in route.segments:
        if isinstance(seg, ApertureSeg):
            ray, hit, valid = aperture_interact(seg, ray, wavelength)
            _push(hit, valid)

        elif isinstance(seg, FaceSeg):
            ray, hit, valid = face_interact(seg, ray, wavelength)
            _push(hit, valid)

        elif isinstance(seg, MirrorStackSeg):
            def step(r, params):
                r_out, hit, valid = mirror_transmit_one(params, r, wavelength)
                return r_out, (hit, valid)
            ray, (stack_hits, stack_valids) = jax.lax.scan(step, ray, seg)
            hits_accum.append(stack_hits)
            valids_accum.append(stack_valids)

        elif isinstance(seg, ReflectMirrorSeg):
            ray, hit, valid = mirror_reflect_one(seg, ray, wavelength)
            _push(hit, valid)

        elif isinstance(seg, PupilSeg):
            ray, hit, valid = pupil_interact(seg, ray, wavelength)
            _push(hit, valid)

        else:
            raise TypeError(f"Unknown segment type: {type(seg).__name__}")

    hits = jnp.concatenate(hits_accum, axis=0)
    valids = jnp.concatenate(valids_accum, axis=0)

    return TraceResult(
        hits=hits,
        valids=valids,
        final_pos=ray.pos,
        final_dir=ray.dir,
        final_intensity=ray.intensity,
    )


def trace_rays(route: Route, ray: Ray, wavelength: float = 0.0) -> TraceResult:
    """Trace a batched ``Ray`` through a ``Route``.

    ``ray.pos`` must be ``(N, 3)`` and ``ray.intensity`` ``(N,)``; ``ray.dir``
    is ``(3,)`` and shared across all rays (collimated beam). Returns a
    ``TraceResult`` whose fields carry a leading batch dim of ``N``.

    ``wavelength`` is a scalar (float or traced) that drives mirror
    reflectance curve evaluation. Faces must already be resolved for the
    same wavelength via ``prepare_route``.
    """
    shared_dir = jnp.asarray(ray.dir, dtype=jnp.float32)

    def one(pos, intensity):
        r = Ray(pos=pos, dir=shared_dir, intensity=intensity)
        return trace(route, r, wavelength)

    return jax.vmap(one)(ray.pos, ray.intensity)
