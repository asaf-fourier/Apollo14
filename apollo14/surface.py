"""Generic surface state and interaction — uniform interface for all optical elements.

Every planar optical element (aperture, refracting face, partial mirror, detector)
is represented by the same SurfaceState. The single ``surface_step`` function
handles the transmitted path: intersection, Snell refraction, attenuation by (1-r).

Reflection is NOT computed inside the step. It's a one-time operation at the
mirror boundary, computed before launching a branch trace. This keeps the step
function simple and avoids wasted computation on non-mirror elements.

Element-specific behavior emerges from parameter values:
- Aperture: n1=n2=1, reflectance=0, kill_on_miss=True
- Refracting face: n1≠n2, reflectance=0, kill_on_miss=True
- Mirror: n1=n2 (no refraction on main path), reflectance>0, kill_on_miss=False
- Detector: n1=n2=1, reflectance=0, kill_on_miss=True
"""

from typing import NamedTuple

import jax.numpy as jnp

from apollo14.geometry import ray_rect_intersect, snell_refract, reflect


class SurfaceState(NamedTuple):
    """Universal JAX-traceable state for any planar optical surface.

    All fields are JAX arrays — auto-pytree via NamedTuple.
    """
    position: jnp.ndarray       # (3,)
    normal: jnp.ndarray         # (3,) outward-facing unit normal
    half_extents: jnp.ndarray   # (2,) [half_width, half_height]
    local_x: jnp.ndarray        # (3,)
    local_y: jnp.ndarray        # (3,)
    n1: jnp.ndarray             # scalar — refractive index on incoming side
    n2: jnp.ndarray             # scalar — refractive index on outgoing side
    reflectance: jnp.ndarray    # (3,) per-color [R, G, B]; 0 for non-mirrors
    kill_on_miss: jnp.ndarray   # bool — zero intensity when ray misses bounds


def surface_step(state: SurfaceState, origin, direction, intensity, color_idx):
    """Transmitted-path interaction: intersect, refract, attenuate.

    Computes only the transmitted ray. Reflection (direction + intensity)
    is handled separately at branch entry — see ``mirror_branch_origin``.

    Args:
        state: SurfaceState for this element.
        origin: (3,) ray position.
        direction: (3,) ray direction (normalized).
        intensity: scalar, current ray intensity.
        color_idx: int, selects reflectance channel [0=R, 1=G, 2=B].

    Returns:
        out_pos: (3,) updated position (hit point if valid, else origin).
        out_dir: (3,) updated direction (refracted if valid, else unchanged).
        out_intensity: scalar, transmitted intensity.
        hit: (3,) raw intersection point on the surface plane.
        valid: bool, whether the ray hit within bounds.
    """
    hit, t, in_bounds = ray_rect_intersect(
        origin, direction, state.position, state.normal,
        state.local_x, state.local_y, state.half_extents)

    facing = jnp.where(jnp.dot(direction, state.normal) < 0,
                       state.normal, -state.normal)

    # Refraction — identity when n1 == n2
    new_dir, is_tir = snell_refract(direction, facing, state.n1, state.n2)
    valid = in_bounds & ~is_tir

    # Attenuation: transmitted intensity = input * (1 - reflectance)
    r = state.reflectance[color_idx]
    trans_intensity = intensity * (1.0 - r)

    # Main ray continues
    out_pos = jnp.where(valid, hit, origin)
    out_dir = jnp.where(valid, new_dir, direction)
    out_intensity = jnp.where(valid, trans_intensity,
                              jnp.where(state.kill_on_miss, 0.0, intensity))

    return out_pos, out_dir, out_intensity, hit, valid


def mirror_branch_origin(state: SurfaceState, direction, intensity, hit, valid,
                         color_idx):
    """Compute reflected ray at a mirror — the starting point for a branch trace.

    Called once per mirror after the main-path step. Not part of the scan.

    Args:
        state: SurfaceState of the mirror that was hit.
        direction: (3,) incoming ray direction (before the mirror step).
        intensity: scalar, ray intensity *before* the mirror attenuated it.
        hit: (3,) intersection point from surface_step.
        valid: bool, whether the mirror was actually hit.
        color_idx: int, color channel.

    Returns:
        refl_dir: (3,) reflected direction.
        refl_intensity: scalar, reflected intensity (0 if not valid).
    """
    facing = jnp.where(jnp.dot(direction, state.normal) < 0,
                       state.normal, -state.normal)
    refl_dir = reflect(direction, facing)

    r = state.reflectance[color_idx]
    refl_intensity = jnp.where(valid, intensity * r, 0.0)

    return refl_dir, refl_intensity
