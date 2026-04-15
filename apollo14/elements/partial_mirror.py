"""Partial mirror — splits light into reflected and transmitted fractions.

Reflectance is stored as a sampled spectral curve: a shared ``wavelengths``
grid and a per-mirror ``reflectance`` array. At trace time the effective
reflectance is ``jnp.interp(wavelength, wavelengths, reflectance)`` —
differentiable w.r.t. the control values and continuous in wavelength.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp

from apollo14.elements.planar import PlanarElement
from apollo14.geometry import ray_rect_intersect, reflect
from apollo14.ray import Ray
from apollo14.route import TRANSMIT, REFLECT
from apollo14.units import nm


# Default spectral sample grid: R/G/B micro-LED peaks, sorted ascending so
# ``jnp.interp`` can consume it directly. Every new mirror inherits this
# grid unless the caller overrides it.
DEFAULT_MIRROR_WAVELENGTHS = jnp.array([460.0, 525.0, 630.0],
                                        dtype=jnp.float32) * nm


class _SingleMirror(NamedTuple):
    """Unstacked transmit-mirror emitted by ``PartialMirror.build_segment``.

    The route builder's grouper fuses runs of these into one
    ``MirrorStackSeg``. It never reaches the tracer directly.
    """
    position: jnp.ndarray       # (3,)
    normal: jnp.ndarray         # (3,)
    local_x: jnp.ndarray        # (3,)
    local_y: jnp.ndarray        # (3,)
    half_extents: jnp.ndarray   # (2,)
    wavelengths: jnp.ndarray    # (K,) spectral sample grid (ascending)
    reflectance: jnp.ndarray    # (K,) reflectance at each sample


class MirrorStackSeg(NamedTuple):
    """Stacked partial mirrors, all transmit, same medium — traced via scan."""
    position: jnp.ndarray       # (M, 3)
    normal: jnp.ndarray         # (M, 3)
    local_x: jnp.ndarray        # (M, 3)
    local_y: jnp.ndarray        # (M, 3)
    half_extents: jnp.ndarray   # (M, 2)
    wavelengths: jnp.ndarray    # (M, K) — each row identical in practice
    reflectance: jnp.ndarray    # (M, K)


class ReflectMirrorSeg(NamedTuple):
    """Single reflecting mirror (the fork point of a branch route)."""
    position: jnp.ndarray       # (3,)
    normal: jnp.ndarray
    local_x: jnp.ndarray
    local_y: jnp.ndarray
    half_extents: jnp.ndarray
    wavelengths: jnp.ndarray    # (K,)
    reflectance: jnp.ndarray    # (K,)


@dataclass
class PartialMirror(PlanarElement):
    reflectance: jnp.ndarray = None   # (K,) sampled reflectance curve
    wavelengths: jnp.ndarray = None   # (K,) wavelength grid (ascending)

    def __post_init__(self):
        super().__post_init__()
        if self.wavelengths is None:
            self.wavelengths = DEFAULT_MIRROR_WAVELENGTHS
        else:
            self.wavelengths = jnp.asarray(self.wavelengths, dtype=jnp.float32)

        k = self.wavelengths.shape[0]
        if self.reflectance is None:
            self.reflectance = jnp.full((k,), 0.05, dtype=jnp.float32)
        else:
            r = jnp.asarray(self.reflectance, dtype=jnp.float32)
            self.reflectance = jnp.broadcast_to(r, (k,)).copy()

    def build_segment(self, current_material, mode):
        if mode == REFLECT:
            seg = ReflectMirrorSeg(
                position=self.position,
                normal=self.normal,
                local_x=self._local_x,
                local_y=self._local_y,
                half_extents=self.half_extents,
                wavelengths=self.wavelengths,
                reflectance=self.reflectance,
            )
            return seg, current_material
        # TRANSMIT — emit one ``_SingleMirror``; ``_group_mirror_runs``
        # fuses adjacent ones into a single ``MirrorStackSeg``.
        seg = _SingleMirror(
            position=self.position,
            normal=self.normal,
            local_x=self._local_x,
            local_y=self._local_y,
            half_extents=self.half_extents,
            wavelengths=self.wavelengths,
            reflectance=self.reflectance,
        )
        return seg, current_material


def _interp_reflectance(wavelengths, reflectance, wavelength):
    """``jnp.interp`` with a consistent float32 output."""
    return jnp.interp(wavelength, wavelengths, reflectance)


def mirror_transmit_one(mirror_params, ray: Ray, wavelength):
    """Transmit through one partial mirror (used inside ``lax.scan``).

    The mirror coating is a thin layer inside a single medium, so the
    transmitted direction is unchanged — only the intensity attenuates.
    """
    alive_in = ray.intensity > 0

    hit, _, in_bounds = ray_rect_intersect(
        ray.pos, ray.dir,
        mirror_params.position, mirror_params.normal,
        mirror_params.local_x, mirror_params.local_y,
        mirror_params.half_extents)

    r = _interp_reflectance(
        mirror_params.wavelengths, mirror_params.reflectance, wavelength)
    new_intensity = ray.intensity * (1.0 - r)

    valid = in_bounds & alive_in
    out_intensity = jnp.where(valid, new_intensity, 0.0)
    out_pos = jnp.where(valid, hit, ray.pos)
    return Ray(pos=out_pos, dir=ray.dir, intensity=out_intensity), hit, valid


def mirror_reflect_one(seg: ReflectMirrorSeg, ray: Ray, wavelength):
    """Reflect off one partial mirror (branch fork point)."""
    alive_in = ray.intensity > 0

    hit, _, in_bounds = ray_rect_intersect(
        ray.pos, ray.dir,
        seg.position, seg.normal, seg.local_x, seg.local_y, seg.half_extents)

    facing = jnp.where(jnp.dot(ray.dir, seg.normal) < 0,
                       seg.normal, -seg.normal)
    reflected = reflect(ray.dir, facing)
    r = _interp_reflectance(seg.wavelengths, seg.reflectance, wavelength)
    new_intensity = ray.intensity * r

    valid = in_bounds & alive_in
    out_intensity = jnp.where(valid, new_intensity, 0.0)
    out_pos = jnp.where(valid, hit, ray.pos)
    out_dir = jnp.where(valid, reflected, ray.dir)
    return Ray(pos=out_pos, dir=out_dir, intensity=out_intensity), hit, valid
