"""Aperture element — binary intensity gate based on rectangular opening."""

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp

from apollo14.geometry import normalize, compute_local_axes, ray_rect_intersect


class ApertureState(NamedTuple):
    """JAX-traceable state for a rectangular aperture."""
    position: jnp.ndarray       # (3,)
    normal: jnp.ndarray         # (3,) unit normal
    half_extents: jnp.ndarray   # (2,) [half_width, half_height]
    local_x: jnp.ndarray        # (3,)
    local_y: jnp.ndarray        # (3,)


@dataclass
class RectangularAperture:
    """Absorbing aperture with a rectangular opening.

    Rays passing through the opening keep their intensity.
    Rays outside the opening get intensity zeroed.
    """
    name: str
    position: jnp.ndarray  # (3,)
    normal: jnp.ndarray    # (3,)
    width: float
    height: float

    def __post_init__(self):
        self.normal = normalize(jnp.asarray(self.normal, dtype=jnp.float32))
        self.position = jnp.asarray(self.position, dtype=jnp.float32)

    @property
    def state(self) -> ApertureState:
        lx, ly = compute_local_axes(self.normal)
        return ApertureState(
            position=self.position,
            normal=self.normal,
            half_extents=jnp.array([self.width / 2.0, self.height / 2.0]),
            local_x=lx,
            local_y=ly,
        )

    @staticmethod
    def jax_clip(state: ApertureState, origin, direction, has_aperture):
        """Pure JAX: clip intensity to zero if ray misses the aperture opening.

        Args:
            state: ApertureState for this aperture.
            origin: (3,) ray position.
            direction: (3,) ray direction (normalized).
            has_aperture: bool scalar, if False the aperture is skipped (intensity=1).

        Returns:
            intensity: scalar, 1.0 if ray passes through opening (or no aperture), else 0.0.
        """
        hit, t, in_opening = ray_rect_intersect(
            origin, direction, state.position, state.normal,
            state.local_x, state.local_y, state.half_extents)
        return jnp.where(has_aperture, jnp.where(in_opening, 1.0, 0.0), 1.0)
