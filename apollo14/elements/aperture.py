"""Aperture element — binary intensity gate based on rectangular opening."""

from dataclasses import dataclass

from apollo14.elements.planar import PlanarElement


@dataclass
class RectangularAperture(PlanarElement):
    """Absorbing aperture with a rectangular opening.

    Rays passing through the opening keep their intensity; rays outside
    the opening are killed by the generic bounds check in ``surface_step``.
    """
