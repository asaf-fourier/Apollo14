"""Generic planar surfaces for the sequential single-path tracer.

Two NamedTuples, one physics kernel:

- ``RouteSurface`` is what a ``Route`` holds — it carries ``MaterialData``
  for n1/n2, so the same route can be reused across wavelengths.
- ``Surface`` is what the scan consumes — it carries already-interpolated
  scalar indices for one specific wavelength.

``prepare_beam`` (in ``apollo14.trace``) turns a ``RouteSurface`` stack into
a ``Surface`` stack by interpolating each material once. ``surface_step``
works on ``Surface`` only — no wavelength argument at trace time.
"""

from typing import NamedTuple

import jax.numpy as jnp

from apollo14.geometry import ray_rect_intersect, snell_refract, reflect
from apollo14.materials import MaterialData


# ── Surface modes ────────────────────────────────────────────────────────────
# Stored as int8 scalars inside (Route)Surface.

TRANSMIT = 0   # refract through; mirrors attenuate by (1-r)
REFLECT = 1    # reflect off facing normal; mirrors attenuate by r
ABSORB = 2     # terminal surface (detector); keeps intensity, no direction change


class RouteSurface(NamedTuple):
    """Route-time planar surface — materials stored as ``MaterialData``."""
    position: jnp.ndarray       # (3,)
    normal: jnp.ndarray         # (3,) outward unit normal
    half_extents: jnp.ndarray   # (2,)
    local_x: jnp.ndarray        # (3,)
    local_y: jnp.ndarray        # (3,)
    n1: MaterialData            # incoming-side material
    n2: MaterialData            # outgoing-side material
    reflectance: jnp.ndarray    # (3,) per-color
    mode: jnp.ndarray           # int8 scalar — TRANSMIT / REFLECT / ABSORB


class Surface(NamedTuple):
    """Trace-time planar surface — n1/n2 are scalar refractive indices."""
    position: jnp.ndarray
    normal: jnp.ndarray
    half_extents: jnp.ndarray
    local_x: jnp.ndarray
    local_y: jnp.ndarray
    n1: jnp.ndarray             # scalar
    n2: jnp.ndarray             # scalar
    reflectance: jnp.ndarray
    mode: jnp.ndarray


def _transmit_step(state, direction, facing, intensity, color_idx):
    refracted, is_tir = snell_refract(direction, facing, state.n1, state.n2)
    out_i = intensity * (1.0 - state.reflectance[color_idx])
    return refracted, out_i, is_tir


def _reflect_step(state, direction, facing, intensity, color_idx):
    reflected = reflect(direction, facing)
    out_i = intensity * state.reflectance[color_idx]
    return reflected, out_i


def _absorb_step(direction, intensity):
    return direction, intensity


def surface_step(state: Surface, origin, direction, intensity, color_idx):
    """One generic surface interaction on a wavelength-resolved surface."""
    hit, t, in_bounds = ray_rect_intersect(
        origin, direction, state.position, state.normal,
        state.local_x, state.local_y, state.half_extents)

    facing = jnp.where(jnp.dot(direction, state.normal) < 0,
                       state.normal, -state.normal)

    t_dir, t_i, is_tir = _transmit_step(
        state, direction, facing, intensity, color_idx)
    r_dir, r_i = _reflect_step(
        state, direction, facing, intensity, color_idx)
    a_dir, a_i = _absorb_step(direction, intensity)

    is_reflect = state.mode == REFLECT
    is_absorb = state.mode == ABSORB
    is_transmit = (~is_reflect) & (~is_absorb)

    mode_intensity = jnp.where(is_absorb, a_i,
                               jnp.where(is_reflect, r_i, t_i))
    mode_dir = jnp.where(is_absorb, a_dir,
                         jnp.where(is_reflect, r_dir, t_dir))

    valid = in_bounds & ~(is_transmit & is_tir)

    out_intensity = jnp.where(valid, mode_intensity, 0.0)
    out_pos = jnp.where(valid, hit, origin)
    out_dir = jnp.where(valid, mode_dir, direction)

    return out_pos, out_dir, out_intensity, hit, valid
