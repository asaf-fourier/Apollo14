"""Generic planar surface state for the sequential single-path tracer.

All planar optical elements share this one ``Surface`` NamedTuple.
Element-specific behavior emerges from field values plus a ``mode`` flag —
no type dispatch, no branching inside the scan step.

``Surface`` has two life-stages, both using the same class:
- route-time: n1/n2 hold sampled ``MaterialData`` (wavelength-agnostic).
- beam-time: n1/n2 hold scalar refractive indices (resolved by
  ``prepare_beam`` for a specific wavelength).

``surface_step`` runs at beam-time and reads ``state.n1``/``state.n2`` as
scalars — no interpolation inside the scan.
"""

from typing import NamedTuple

import jax.numpy as jnp

from apollo14.geometry import ray_rect_intersect, snell_refract, reflect
from apollo14.materials import MaterialData


# ── Surface modes ────────────────────────────────────────────────────────────
# Stored as int8 scalars inside Surface.

TRANSMIT = 0   # refract through; mirrors attenuate by (1-r)
REFLECT = 1    # reflect off facing normal; mirrors attenuate by r
ABSORB = 2     # terminal surface (detector); keeps intensity, no direction change


class Surface(NamedTuple):
    """Universal planar surface used by the single-path tracer."""
    position: jnp.ndarray       # (3,)
    normal: jnp.ndarray         # (3,) outward unit normal
    half_extents: jnp.ndarray   # (2,)
    local_x: jnp.ndarray        # (3,)
    local_y: jnp.ndarray        # (3,)
    n1: MaterialData            # incoming-side material
    n2: MaterialData            # outgoing-side material
    reflectance: jnp.ndarray    # (3,) per-color
    mode: jnp.ndarray           # int8 scalar — TRANSMIT / REFLECT / ABSORB


def interp_n(mat: MaterialData, wavelength) -> jnp.ndarray:
    return jnp.interp(wavelength, mat.wavelengths, mat.n_values)


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
    """One generic surface interaction. Mode-driven, no Python dispatch.

    Each mode's physics lives in its own helper; this function intersects
    the plane, runs all three helpers, and selects via ``jnp.where`` on
    ``state.mode``. Same single-scan pytree, cleaner decomposition.

    Returns:
        out_pos, out_dir, out_intensity, hit, valid
    """
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
