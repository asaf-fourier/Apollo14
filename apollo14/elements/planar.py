"""Shared base for rectangular planar elements.

Holds the common geometry fields (name, position, normal, width, height)
and pre-computes the local-axis frame once at construction. Element
subclasses own their own ``build_segment`` and ``jax_interact`` — this
base class no longer produces a generic ``Surface``.
"""

from dataclasses import dataclass

import jax.numpy as jnp

from apollo14.geometry import normalize, compute_local_axes


@dataclass
class PlanarElement:
    name: str
    position: jnp.ndarray
    normal: jnp.ndarray
    width: float
    height: float

    def __post_init__(self):
        self.normal = normalize(jnp.asarray(self.normal, dtype=jnp.float32))
        self.position = jnp.asarray(self.position, dtype=jnp.float32)
        lx, ly = compute_local_axes(self.normal)
        self._local_x = lx
        self._local_y = ly

    @property
    def half_extents(self) -> jnp.ndarray:
        return jnp.array([self.width / 2.0, self.height / 2.0],
                         dtype=jnp.float32)
