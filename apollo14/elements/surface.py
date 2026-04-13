"""Partial mirror element — splits light into reflected and transmitted."""

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp

from apollo14.geometry import normalize, compute_local_axes, reflect


class MirrorState(NamedTuple):
    """JAX-traceable state for a partial mirror. Auto-pytree via NamedTuple."""
    position: jnp.ndarray       # (3,)
    normal: jnp.ndarray         # (3,) outward-facing unit normal
    half_extents: jnp.ndarray   # (2,) [half_width, half_height]
    reflectance: jnp.ndarray    # (3,) per-color [R, G, B]
    local_x: jnp.ndarray        # (3,) first local axis
    local_y: jnp.ndarray        # (3,) second local axis


@dataclass
class PartialMirror:
    """A flat partial mirror that splits light into reflected and transmitted.

    Construction helper: builds a MirrorState from user-friendly parameters.
    The static ``jax_interact`` method is the pure-JAX physics kernel.
    """
    name: str
    position: jnp.ndarray   # (3,)
    normal: jnp.ndarray     # (3,) outward-facing
    width: float
    height: float
    reflectance: jnp.ndarray = None  # (3,) per-color; scalar broadcast to (3,)

    def __post_init__(self):
        self.normal = normalize(jnp.asarray(self.normal, dtype=jnp.float32))
        self.position = jnp.asarray(self.position, dtype=jnp.float32)
        if self.reflectance is None:
            self.reflectance = jnp.array([0.05, 0.05, 0.05])
        else:
            r = jnp.asarray(self.reflectance, dtype=jnp.float32)
            self.reflectance = jnp.broadcast_to(r, (3,)).copy()

    @property
    def state(self) -> MirrorState:
        """Build the JAX-traceable state for this mirror."""
        lx, ly = compute_local_axes(self.normal)
        return MirrorState(
            position=self.position,
            normal=self.normal,
            half_extents=jnp.array([self.width / 2.0, self.height / 2.0]),
            reflectance=self.reflectance,
            local_x=lx,
            local_y=ly,
        )

    @staticmethod
    def jax_interact(state: MirrorState, origin, direction, intensity, color_idx):
        """Pure JAX: intersect mirror, split into transmitted + reflected.

        Args:
            state: MirrorState for this mirror.
            origin: (3,) ray position.
            direction: (3,) ray direction (normalized).
            intensity: scalar, current ray intensity.
            color_idx: int, selects reflectance channel [0=R, 1=G, 2=B].

        Returns:
            hit: (3,) intersection point on the mirror plane.
            trans_dir: (3,) transmitted direction (unchanged).
            trans_intensity: scalar, transmitted intensity.
            refl_dir: (3,) reflected direction.
            refl_intensity: scalar, reflected intensity.
            valid: bool, whether the ray hit within mirror bounds.
        """
        # Ray-plane intersection
        denom = jnp.dot(direction, state.normal)
        t = jnp.dot(state.position - origin, state.normal) / (denom + 1e-30)
        hit = origin + jnp.maximum(t, 0.0) * direction

        # Rectangular bounds check in local frame
        delta = hit - state.position
        in_bounds = (
            (jnp.abs(jnp.dot(delta, state.local_x)) <= state.half_extents[0]) &
            (jnp.abs(jnp.dot(delta, state.local_y)) <= state.half_extents[1]) &
            (t > 0)
        )

        # Reflect off the facing normal
        facing = jnp.where(jnp.dot(direction, state.normal) < 0,
                           state.normal, -state.normal)
        refl_dir = reflect(direction, facing)

        # Intensity split
        r = state.reflectance[color_idx]
        refl_intensity = jnp.where(in_bounds, intensity * r, 0.0)
        trans_intensity = jnp.where(in_bounds, intensity * (1.0 - r), intensity)

        return hit, direction, trans_intensity, refl_dir, refl_intensity, in_bounds
