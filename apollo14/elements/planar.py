"""Shared base class for rectangular planar elements."""

from dataclasses import dataclass

import jax.numpy as jnp

from apollo14.geometry import normalize, compute_local_axes
from apollo14.surface import Surface, TRANSMIT


@dataclass
class PlanarElement:
    """Common fields + Surface construction for rectangular planar elements.

    Subclasses may override ``_default_mode`` and ``_reflectance``. Anything
    that doesn't fit this shape (e.g. ``GlassFaceRef``) should implement
    ``to_generic_surface`` directly instead of inheriting.
    """
    name: str
    position: jnp.ndarray   # (3,)
    normal: jnp.ndarray     # (3,)
    width: float
    height: float

    _default_mode = TRANSMIT

    def __post_init__(self):
        self.normal = normalize(jnp.asarray(self.normal, dtype=jnp.float32))
        self.position = jnp.asarray(self.position, dtype=jnp.float32)

    def _reflectance(self) -> jnp.ndarray:
        return jnp.zeros(3, dtype=jnp.float32)

    def to_generic_surface(self, current_material, mode=None):
        if mode is None:
            mode = self._default_mode
        lx, ly = compute_local_axes(self.normal)
        surf = Surface(
            position=self.position,
            normal=self.normal,
            half_extents=jnp.array([self.width / 2.0, self.height / 2.0]),
            local_x=lx,
            local_y=ly,
            n1=current_material.data,
            n2=current_material.data,
            reflectance=self._reflectance(),
            mode=jnp.int8(mode),
        )
        return surf, current_material
