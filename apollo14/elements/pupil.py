"""Detector elements — record ray hits on a planar surface."""

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp

from apollo14.geometry import normalize, compute_local_axes


class DetectorState(NamedTuple):
    """JAX-traceable state for a rectangular detector (pupil)."""
    position: jnp.ndarray       # (3,)
    normal: jnp.ndarray         # (3,) unit normal
    half_extents: jnp.ndarray   # (2,) [half_width, half_height]
    local_x: jnp.ndarray        # (3,)
    local_y: jnp.ndarray        # (3,)


@dataclass
class RectangularPupil:
    """Rectangular detector plane. Records hits, produces no child rays.

    The static ``jax_interact`` method is the pure-JAX detection kernel.
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
    def state(self) -> DetectorState:
        lx, ly = compute_local_axes(self.normal)
        return DetectorState(
            position=self.position,
            normal=self.normal,
            half_extents=jnp.array([self.width / 2.0, self.height / 2.0]),
            local_x=lx,
            local_y=ly,
        )

    @staticmethod
    def jax_interact(state: DetectorState, origin, direction, intensity):
        """Pure JAX: intersect detector plane, check bounds.

        Args:
            state: DetectorState for this detector.
            origin: (3,) ray position.
            direction: (3,) ray direction (normalized).
            intensity: scalar, current ray intensity.

        Returns:
            hit: (3,) intersection point on the detector plane.
            intensity: scalar (unchanged).
            valid: bool, whether the ray hit within detector bounds.
        """
        denom = jnp.dot(direction, state.normal)
        t = jnp.dot(state.position - origin, state.normal) / (denom + 1e-30)
        hit = origin + jnp.maximum(t, 0.0) * direction

        delta = hit - state.position
        valid = (
            (jnp.abs(jnp.dot(delta, state.local_x)) <= state.half_extents[0]) &
            (jnp.abs(jnp.dot(delta, state.local_y)) <= state.half_extents[1]) &
            (t > 0)
        )
        return hit, intensity, valid


# Keep Pupil as alias for backward compat during transition
Pupil = RectangularPupil
