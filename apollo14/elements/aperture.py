from dataclasses import dataclass

import jax.numpy as jnp

from apollo14.geometry import normalize, ray_plane_intersection, compute_local_axes, point_in_rect


@dataclass
class RectangularAperture:
    """Absorbing aperture with a rectangular opening.

    Rays passing through the opening are unaffected.
    Rays hitting the opaque border (outside the opening) are absorbed.
    """
    name: str
    position: jnp.ndarray  # (3,)
    normal: jnp.ndarray    # (3,)
    width: float
    height: float

    def find_intersection(self, origin, direction):
        n = normalize(self.normal)
        dist = ray_plane_intersection(origin, direction, n, self.position)
        if dist == jnp.inf:
            return None

        hit = origin + dist * direction
        local_x, local_y = compute_local_axes(n)
        delta = hit - self.position
        lx = jnp.dot(delta, local_x)
        ly = jnp.dot(delta, local_y)

        # Rays INSIDE the opening pass through (no intersection)
        # Only rays hitting the opaque border are intercepted
        if point_in_rect(lx, ly, self.width / 2.0, self.height / 2.0):
            return None

        hit_normal = jnp.where(jnp.dot(direction, n) < 0, n, -n)
        return dist, hit, hit_normal

    def interact(self, origin, direction, intensity):
        """Aperture absorbs — returns empty list (ray stops)."""
        return []
