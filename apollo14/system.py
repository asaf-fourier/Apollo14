from dataclasses import dataclass, field
from typing import List

from apollo14.materials import Material, air


@dataclass
class OpticalSystem:
    """Container for optical elements + environment material."""
    elements: List = field(default_factory=list)
    env_material: Material = field(default_factory=lambda: air)

    def add(self, element):
        self.elements.append(element)
