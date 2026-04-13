"""Partial mirror element — splits light into reflected and transmitted."""

from dataclasses import dataclass

import jax.numpy as jnp

from apollo14.elements.planar import PlanarElement


@dataclass
class PartialMirror(PlanarElement):
    """A flat partial mirror with per-color reflectance.

    Inherits the common planar geometry and ``to_generic_surface`` from
    ``PlanarElement``; only the reflectance plumbing is mirror-specific.
    """
    reflectance: jnp.ndarray = None  # (3,) per-color; scalar broadcast to (3,)

    def __post_init__(self):
        super().__post_init__()
        if self.reflectance is None:
            self.reflectance = jnp.array([0.05, 0.05, 0.05])
        else:
            r = jnp.asarray(self.reflectance, dtype=jnp.float32)
            self.reflectance = jnp.broadcast_to(r, (3,)).copy()

    def _reflectance(self) -> jnp.ndarray:
        return self.reflectance
