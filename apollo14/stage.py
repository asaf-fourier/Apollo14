from dataclasses import dataclass
from typing import List

import jax.numpy as jnp

from apollo14.elements.boundary import BoundaryPlane
from apollo14.system import OpticalSystem


@dataclass
class Stage:
    """Axis-aligned bounding box that defines the simulation volume.

    Adds six BoundaryPlane elements to the system so that stray rays
    terminate at the stage walls instead of vanishing.
    """
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def get_boundary_elements(self) -> List[BoundaryPlane]:
        return [
            BoundaryPlane("boundary_x_min", jnp.array([self.x_min, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.0])),
            BoundaryPlane("boundary_x_max", jnp.array([self.x_max, 0.0, 0.0]), jnp.array([-1.0, 0.0, 0.0])),
            BoundaryPlane("boundary_y_min", jnp.array([0.0, self.y_min, 0.0]), jnp.array([0.0, 1.0, 0.0])),
            BoundaryPlane("boundary_y_max", jnp.array([0.0, self.y_max, 0.0]), jnp.array([0.0, -1.0, 0.0])),
            BoundaryPlane("boundary_z_min", jnp.array([0.0, 0.0, self.z_min]), jnp.array([0.0, 0.0, 1.0])),
            BoundaryPlane("boundary_z_max", jnp.array([0.0, 0.0, self.z_max]), jnp.array([0.0, 0.0, -1.0])),
        ]

    def add_to_system(self, system: OpticalSystem):
        for boundary in self.get_boundary_elements():
            system.add(boundary)

    @classmethod
    def from_system(cls, system: OpticalSystem, margin: float = 10.0):
        """Create a stage that encloses all elements with the given margin."""
        points = []
        for elem in system.elements:
            pos = getattr(elem, 'position', None)
            if pos is not None:
                points.append(pos)
        if not points:
            return cls(-margin, margin, -margin, margin, -margin, margin)
        coords = jnp.stack(points)
        lo = jnp.min(coords, axis=0)
        hi = jnp.max(coords, axis=0)
        return cls(
            x_min=float(lo[0]) - margin, x_max=float(hi[0]) + margin,
            y_min=float(lo[1]) - margin, y_max=float(hi[1]) + margin,
            z_min=float(lo[2]) - margin, z_max=float(hi[2]) + margin,
        )
