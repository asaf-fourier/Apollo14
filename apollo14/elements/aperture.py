"""Rectangular aperture — framed opaque stop with a rectangular hole.

``width``/``height`` define the outer opaque frame; ``inner_width``/
``inner_height`` define the opening that transmits light. Rays hitting
the frame are absorbed; rays passing through the inner opening are
unchanged; rays that miss the outer frame entirely are a no-op (the
stop is a finite physical screen, not an infinite occluder).

Default ``inner_width``/``inner_height`` match the outer extents, which
is a pure no-op aperture — useful when the combiner's opening is the
whole plane and the aperture is purely a bookkeeping entry.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp

from apollo14.elements.planar import PlanarElement
from apollo14.geometry import ray_rect_intersect
from apollo14.ray import Ray


class ApertureSeg(NamedTuple):
    """Framed rectangular stop. Outer = frame, inner = hole."""
    position: jnp.ndarray       # (3,)
    normal: jnp.ndarray         # (3,)
    local_x: jnp.ndarray        # (3,)
    local_y: jnp.ndarray        # (3,)
    half_extents: jnp.ndarray   # (2,) outer
    inner_half_extents: jnp.ndarray  # (2,) hole


@dataclass
class RectangularAperture(PlanarElement):
    inner_width: float = None
    inner_height: float = None

    def __post_init__(self):
        super().__post_init__()
        if self.inner_width is None:
            self.inner_width = self.width
        if self.inner_height is None:
            self.inner_height = self.height

    @property
    def inner_half_extents(self) -> jnp.ndarray:
        return jnp.array([self.inner_width / 2.0, self.inner_height / 2.0],
                         dtype=jnp.float32)

    def build_segment(self, current_material, mode):
        seg = ApertureSeg(
            position=self.position,
            normal=self.normal,
            local_x=self._local_x,
            local_y=self._local_y,
            half_extents=self.half_extents,
            inner_half_extents=self.inner_half_extents,
        )
        return seg, current_material


def aperture_interact(seg: ApertureSeg, ray: Ray, color_idx):
    """Framed aperture:
    - hit inner hole → pass unchanged
    - hit outer frame outside hole → absorbed (intensity 0)
    - miss outer frame → no-op, ray continues unchanged

    Also propagates upstream death: if the incoming ray is already dead
    (``intensity == 0``) it stays put and stays invalid, so the tracer
    doesn't resurrect corpse rays at later segments.
    """
    alive_in = ray.intensity > 0

    hit, _, in_outer = ray_rect_intersect(
        ray.pos, ray.dir, seg.position, seg.normal,
        seg.local_x, seg.local_y, seg.half_extents)
    _, _, in_inner = ray_rect_intersect(
        ray.pos, ray.dir, seg.position, seg.normal,
        seg.local_x, seg.local_y, seg.inner_half_extents)

    blocked = in_outer & (~in_inner)
    out_intensity = jnp.where(blocked, 0.0, ray.intensity)

    # Position only advances when a live ray actually engaged the stop.
    advance = in_outer & alive_in
    out_pos = jnp.where(advance, hit, ray.pos)

    out_ray = Ray(pos=out_pos, dir=ray.dir, intensity=out_intensity)
    valid = (~blocked) & alive_in
    return out_ray, hit, valid
