from dataclasses import dataclass, field
from typing import List, Optional

import jax.numpy as jnp

from apollo14.materials import Material, air


@dataclass
class OpticalSystem:
    """Container for optical elements + environment material."""
    elements: List = field(default_factory=list)
    env_material: Material = field(default_factory=lambda: air)

    def add(self, element):
        self.elements.append(element)

    def find_closest_intersection(self, origin, direction):
        """Find the nearest element hit.

        Returns (element, distance, hit_point, hit_normal) or None.
        All element-specific extra fields are stripped for a uniform interface.
        """
        best = None
        best_dist = jnp.inf

        for elem in self.elements:
            result = elem.find_intersection(origin, direction)
            if result is not None:
                dist = result[0]
                if dist < best_dist:
                    best_dist = dist
                    # Normalize to (element, dist, hit_point, hit_normal)
                    best = (elem, result[0], result[1], result[2])

        return best
