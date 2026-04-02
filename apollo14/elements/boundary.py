from dataclasses import dataclass

import jax.numpy as jnp

from apollo14.geometry import normalize, ray_plane_intersection


@dataclass
class BoundaryPlane:
    """Infinite absorbing plane that defines a simulation boundary.

    Rays hitting this plane are terminated. The normal points inward
    (toward the simulation volume).
    """
    name: str
    position: jnp.ndarray  # (3,) point on the plane
    normal: jnp.ndarray    # (3,) inward-pointing normal

    def find_intersection(self, origin, direction):
        n = normalize(self.normal)
        dist = ray_plane_intersection(origin, direction, n, self.position)
        if dist == jnp.inf:
            return None
        hit = origin + dist * direction
        hit_normal = jnp.where(jnp.dot(direction, n) < 0, n, -n)
        return dist, hit, hit_normal

    def interact(self, origin, direction, intensity):
        """Boundary absorbs — ray stops."""
        return []
