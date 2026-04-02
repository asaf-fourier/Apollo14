from dataclasses import dataclass

import jax.numpy as jnp

from apollo14.interaction import Interaction
from apollo14.geometry import (
    normalize, ray_plane_intersection, compute_local_axes, point_in_rect, reflect,
)


@dataclass
class PartialMirror:
    """A flat partial mirror that splits light into reflected and transmitted."""
    name: str
    position: jnp.ndarray   # (3,)
    normal: jnp.ndarray     # (3,) outward-facing
    width: float
    height: float
    transmission_ratio: float = 0.5
    reflection_ratio: float = 0.5

    def find_intersection(self, origin, direction):
        """Returns (distance, hit_point, hit_normal) or None."""
        n = normalize(self.normal)
        dist = ray_plane_intersection(origin, direction, n, self.position)
        if dist == jnp.inf:
            return None

        hit = origin + dist * direction
        local_x, local_y = compute_local_axes(n)
        delta = hit - self.position
        lx = jnp.dot(delta, local_x)
        ly = jnp.dot(delta, local_y)

        if not point_in_rect(lx, ly, self.width / 2.0, self.height / 2.0):
            return None

        # Flip normal to face the incoming ray
        hit_normal = jnp.where(jnp.dot(direction, n) < 0, n, -n)
        return dist, hit, hit_normal

    def interact(self, origin, direction, intensity):
        """Returns list of (new_origin, new_direction, new_intensity, interaction_type)."""
        result = self.find_intersection(origin, direction)
        if result is None:
            return []

        dist, hit, hit_normal = result
        children = []

        if self.transmission_ratio > 0:
            children.append((hit, direction, intensity * self.transmission_ratio, Interaction.TRANSMITTED))

        if self.reflection_ratio > 0:
            refl_dir = reflect(direction, hit_normal)
            children.append((hit, refl_dir, intensity * self.reflection_ratio, Interaction.REFLECTED))

        return children
