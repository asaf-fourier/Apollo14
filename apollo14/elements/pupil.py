"""Detector element — records ray hits on a planar surface."""

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp

from apollo14.elements.planar import PlanarElement
from apollo14.geometry import ray_rect_intersect
from apollo14.ray import Ray


class PupilSeg(NamedTuple):
    """Terminal absorbing detector."""
    position: jnp.ndarray
    normal: jnp.ndarray
    local_x: jnp.ndarray
    local_y: jnp.ndarray
    half_extents: jnp.ndarray


@dataclass
class RectangularPupil(PlanarElement):
    def build_segment(self, current_material, mode):
        seg = PupilSeg(
            position=self.position,
            normal=self.normal,
            local_x=self._local_x,
            local_y=self._local_y,
            half_extents=self.half_extents,
        )
        return seg, current_material


# Backward-compat alias
Pupil = RectangularPupil


def pupil_interact(seg: PupilSeg, ray: Ray, color_idx):
    """Terminal: record the hit, mark valid if within bounds.
    Intensity and direction are preserved (the pupil doesn't alter the ray).
    """
    hit, _, in_bounds = ray_rect_intersect(
        ray.pos, ray.dir, seg.position, seg.normal,
        seg.local_x, seg.local_y, seg.half_extents)

    out_intensity = jnp.where(in_bounds, ray.intensity, 0.0)
    out_pos = jnp.where(in_bounds, hit, ray.pos)
    out_ray = Ray(pos=out_pos, dir=ray.dir, intensity=out_intensity)
    return out_ray, hit, in_bounds
