"""Detector element — records ray hits on a planar surface."""

from dataclasses import dataclass

from apollo14.elements.planar import PlanarElement
from apollo14.surface import ABSORB


@dataclass
class RectangularPupil(PlanarElement):
    """Rectangular detector plane. Terminal surface — uses ABSORB mode."""
    _default_mode = ABSORB


# Backward-compat alias
Pupil = RectangularPupil
