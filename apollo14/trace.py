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
from apollo14.elements.glass_block import FaceSeg, PreparedFaceSeg, face_interact
from apollo14.elements.partial_mirror import (
    MirrorStackSeg,
    PreparedMirrorStackSeg,
    PreparedReflectMirrorSeg,
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


def _resolve_face(seg: FaceSeg, wavelength) -> PreparedFaceSeg:
    return PreparedFaceSeg(
        position=seg.position,
        normal=seg.normal,
        local_x=seg.local_x,
        local_y=seg.local_y,
        half_extents=seg.half_extents,
        n1=_interp_n(seg.n1, wavelength),
        n2=_interp_n(seg.n2, wavelength),
        coating_reflectance=seg.coating_reflectance.sample(wavelength),
    )


def _resolve_mirror_stack(seg: MirrorStackSeg, wavelength) -> PreparedMirrorStackSeg:
    reflectance = jax.vmap(
        lambda wavelengths, samples: jnp.interp(wavelength, wavelengths, samples)
    )(seg.wavelengths, seg.reflectance)
    return PreparedMirrorStackSeg(
        position=seg.position,
        normal=seg.normal,
        local_x=seg.local_x,
        local_y=seg.local_y,
        half_extents=seg.half_extents,
        reflectance=reflectance,
    )


def _resolve_reflect_mirror(
    seg: ReflectMirrorSeg, wavelength
) -> PreparedReflectMirrorSeg:
    return PreparedReflectMirrorSeg(
        position=seg.position,
        normal=seg.normal,
        local_x=seg.local_x,
        local_y=seg.local_y,
        half_extents=seg.half_extents,
        reflectance=jnp.interp(wavelength, seg.wavelengths, seg.reflectance),
    )


def prepare_route(route: Route, wavelength) -> Route:
    """Resolve all wavelength-dependent route data before tracing."""
    new_segs = []
    for seg in route.segments:
        if isinstance(seg, FaceSeg):
            new_segs.append(_resolve_face(seg, wavelength))
        elif isinstance(seg, MirrorStackSeg):
            new_segs.append(_resolve_mirror_stack(seg, wavelength))
        elif isinstance(seg, ReflectMirrorSeg):
            new_segs.append(_resolve_reflect_mirror(seg, wavelength))
        else:
            new_segs.append(seg)
    return Route(segments=tuple(new_segs))


# ── Tracing ──────────────────────────────────────────────────────────────────

def trace(route: Route, ray: Ray) -> TraceResult:
    """Trace one ``Ray`` through a wavelength-resolved ``Route``.

    All wavelength-dependent material indices and mirror reflectances must
    already be resolved by :func:`prepare_route`.
    """
    for seg in route.segments:
        if isinstance(seg, (FaceSeg, MirrorStackSeg, ReflectMirrorSeg)):
            raise TypeError(
                "Route contains unprepared wavelength-dependent segments; "
                "call prepare_route(route, wavelength) before tracing.")

    hits_accum = []
    valids_accum = []

    def _push(hit, valid):
        hits_accum.append(hit[None, :])
        valids_accum.append(valid[None])

    for seg in route.segments:
        if isinstance(seg, ApertureSeg):
            ray, hit, valid = aperture_interact(seg, ray, 0.0)
            _push(hit, valid)

        elif isinstance(seg, PreparedFaceSeg):
            ray, hit, valid = face_interact(seg, ray, 0.0)
            _push(hit, valid)

        elif isinstance(seg, PreparedMirrorStackSeg):
            def step(r, params):
                r_out, hit, valid = mirror_transmit_one(params, r)
                return r_out, (hit, valid)
            ray, (stack_hits, stack_valids) = jax.lax.scan(step, ray, seg)
            hits_accum.append(stack_hits)
            valids_accum.append(stack_valids)

        elif isinstance(seg, PreparedReflectMirrorSeg):
            ray, hit, valid = mirror_reflect_one(seg, ray)
            _push(hit, valid)

        elif isinstance(seg, PupilSeg):
            ray, hit, valid = pupil_interact(seg, ray, 0.0)
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


def trace_rays(route: Route, ray: Ray) -> TraceResult:
    """Trace a batched ``Ray`` through a ``Route``.

    ``ray.pos`` must be ``(N, 3)`` and ``ray.intensity`` ``(N,)``; ``ray.dir``
    is ``(3,)`` and shared across all rays (collimated beam). Returns a
    ``TraceResult`` whose fields carry a leading batch dim of ``N``.

    The route must already be wavelength-resolved via :func:`prepare_route`.
    """
    shared_dir = jnp.asarray(ray.dir, dtype=jnp.float32)

    def one(pos, intensity):
        r = Ray(pos=pos, dir=shared_dir, intensity=intensity)
        return trace(route, r)

    return jax.vmap(one)(ray.pos, ray.intensity)
