"""Refracting surface element — Snell's law at a planar interface."""

from typing import NamedTuple

import jax.numpy as jnp

from apollo14.geometry import normalize, compute_local_axes, snell_refract, ray_rect_intersect


class RefractState(NamedTuple):
    """JAX-traceable state for a refracting planar surface."""
    position: jnp.ndarray       # (3,)
    normal: jnp.ndarray         # (3,) outward-facing unit normal
    half_extents: jnp.ndarray   # (2,) [half_width, half_height]
    local_x: jnp.ndarray        # (3,)
    local_y: jnp.ndarray        # (3,)
    n1: jnp.ndarray             # scalar, refractive index on incoming side
    n2: jnp.ndarray             # scalar, refractive index on outgoing side


class RefractingSurface:
    """A flat refracting interface between two media.

    Produced by ``GlassBlock.face()`` — not typically constructed directly.
    """

    def __init__(self, name: str, position, normal, half_width: float,
                 half_height: float, n1: float, n2: float):
        self.name = name
        n = normalize(jnp.asarray(normal, dtype=jnp.float32))
        lx, ly = compute_local_axes(n)
        self._state = RefractState(
            position=jnp.asarray(position, dtype=jnp.float32),
            normal=n,
            half_extents=jnp.array([half_width, half_height]),
            local_x=lx,
            local_y=ly,
            n1=jnp.asarray(n1, dtype=jnp.float32),
            n2=jnp.asarray(n2, dtype=jnp.float32),
        )

    @property
    def state(self) -> RefractState:
        return self._state

    @staticmethod
    def jax_interact(state: RefractState, origin, direction, intensity):
        """Pure JAX: intersect surface and apply Snell's law refraction.

        Args:
            state: RefractState for this surface.
            origin: (3,) ray position.
            direction: (3,) ray direction (normalized).
            intensity: scalar, current ray intensity.

        Returns:
            hit: (3,) intersection point.
            new_dir: (3,) refracted direction.
            intensity: scalar (unchanged; TIR zeroes via valid flag).
            valid: bool, hit within bounds and no TIR.
        """
        hit, t, in_bounds = ray_rect_intersect(
            origin, direction, state.position, state.normal,
            state.local_x, state.local_y, state.half_extents)

        # Snell refraction (normal faces incoming ray)
        facing = jnp.where(jnp.dot(direction, state.normal) < 0,
                           state.normal, -state.normal)
        new_dir, is_tir = snell_refract(direction, facing, state.n1, state.n2)

        valid = in_bounds & ~is_tir
        return hit, new_dir, intensity, valid
