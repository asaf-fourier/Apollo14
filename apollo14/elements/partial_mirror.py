"""Partial mirror — splits light into reflected and transmitted fractions."""

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp

from apollo14.elements.planar import PlanarElement
from apollo14.geometry import ray_rect_intersect, reflect
from apollo14.ray import Ray
from apollo14.route import TRANSMIT, REFLECT


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
    reflectance: jnp.ndarray    # (3,)


class MirrorStackSeg(NamedTuple):
    """Stacked partial mirrors, all transmit, same medium — traced via scan."""
    position: jnp.ndarray       # (M, 3)
    normal: jnp.ndarray         # (M, 3)
    local_x: jnp.ndarray        # (M, 3)
    local_y: jnp.ndarray        # (M, 3)
    half_extents: jnp.ndarray   # (M, 2)
    reflectance: jnp.ndarray    # (M, 3)


class ReflectMirrorSeg(NamedTuple):
    """Single reflecting mirror (the fork point of a branch route)."""
    position: jnp.ndarray       # (3,)
    normal: jnp.ndarray
    local_x: jnp.ndarray
    local_y: jnp.ndarray
    half_extents: jnp.ndarray
    reflectance: jnp.ndarray    # (3,)


@dataclass
class PartialMirror(PlanarElement):
    reflectance: jnp.ndarray = None  # (3,) per-color

    def __post_init__(self):
        super().__post_init__()
        if self.reflectance is None:
            self.reflectance = jnp.array([0.05, 0.05, 0.05], dtype=jnp.float32)
        else:
            r = jnp.asarray(self.reflectance, dtype=jnp.float32)
            self.reflectance = jnp.broadcast_to(r, (3,)).copy()

    def build_segment(self, current_material, mode):
        if mode == REFLECT:
            seg = ReflectMirrorSeg(
                position=self.position,
                normal=self.normal,
                local_x=self._local_x,
                local_y=self._local_y,
                half_extents=self.half_extents,
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
            reflectance=self.reflectance,
        )
        return seg, current_material


def mirror_transmit_one(mirror_params, ray: Ray, color_idx):
    """Transmit through one partial mirror (used inside ``lax.scan``).

    The mirror coating is a thin layer inside a single medium, so the
    transmitted direction is unchanged — only the intensity attenuates.
    """
    hit, _, in_bounds = ray_rect_intersect(
        ray.pos, ray.dir,
        mirror_params.position, mirror_params.normal,
        mirror_params.local_x, mirror_params.local_y,
        mirror_params.half_extents)

    r = mirror_params.reflectance[color_idx]
    new_intensity = ray.intensity * (1.0 - r)

    out_intensity = jnp.where(in_bounds, new_intensity, 0.0)
    out_pos = jnp.where(in_bounds, hit, ray.pos)
    return Ray(pos=out_pos, dir=ray.dir, intensity=out_intensity), hit, in_bounds


def mirror_reflect_one(seg: ReflectMirrorSeg, ray: Ray, color_idx):
    """Reflect off one partial mirror (branch fork point)."""
    hit, _, in_bounds = ray_rect_intersect(
        ray.pos, ray.dir,
        seg.position, seg.normal, seg.local_x, seg.local_y, seg.half_extents)

    facing = jnp.where(jnp.dot(ray.dir, seg.normal) < 0,
                       seg.normal, -seg.normal)
    reflected = reflect(ray.dir, facing)
    r = seg.reflectance[color_idx]
    new_intensity = ray.intensity * r

    out_intensity = jnp.where(in_bounds, new_intensity, 0.0)
    out_pos = jnp.where(in_bounds, hit, ray.pos)
    out_dir = jnp.where(in_bounds, reflected, ray.dir)
    return Ray(pos=out_pos, dir=out_dir, intensity=out_intensity), hit, in_bounds
