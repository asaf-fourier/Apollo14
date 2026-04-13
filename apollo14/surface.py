"""Generic surface state and interaction — uniform interface for all optical elements.

Every planar optical element (aperture, refracting face, partial mirror, detector)
is represented by the same SurfaceState. The single ``surface_interact`` function
handles all physics: intersection, Snell refraction, intensity splitting.

Element-specific behavior emerges from parameter values:
- Aperture: n1=n2=1, reflectance=0, kill_on_miss=True
- Refracting face: n1≠n2, reflectance=0, kill_on_miss=True
- Mirror: n1=n2 (no refraction on main path), reflectance>0, kill_on_miss=False
- Detector: n1=n2=1, reflectance=0, kill_on_miss=True
"""

from typing import NamedTuple

import jax
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


def surface_interact(state: SurfaceState, origin, direction, intensity, color_idx):
    """Universal surface interaction: intersect, refract, split intensity.

    Works for every element type — behavior is controlled by state parameters.
    When n1==n2, Snell's law is identity (no direction change).
    When reflectance==0, no intensity is split to reflection.
    When kill_on_miss is True, a miss zeroes intensity; otherwise the ray
    continues unchanged (for mirrors that the ray might skip past).

    Args:
        state: SurfaceState for this element.
        origin: (3,) ray position.
        direction: (3,) ray direction (normalized).
        intensity: scalar, current ray intensity.
        color_idx: int, selects reflectance channel [0=R, 1=G, 2=B].

    Returns:
        out_pos: (3,) updated ray position (hit point if valid, else origin).
        out_dir: (3,) updated ray direction (refracted if valid, else unchanged).
        out_intensity: scalar, transmitted intensity.
        hit: (3,) raw intersection point (even if out of bounds).
        refl_dir: (3,) reflected direction.
        refl_intensity: scalar, reflected intensity (0 for non-mirrors).
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

    # Intensity split
    r = state.reflectance[color_idx]
    trans_intensity = intensity * (1.0 - r)
    refl_dir = reflect(direction, facing)
    refl_intensity = intensity * r

    # Main ray continues
    out_pos = jnp.where(valid, hit, origin)
    out_dir = jnp.where(valid, new_dir, direction)
    out_intensity = jnp.where(valid, trans_intensity,
                              jnp.where(state.kill_on_miss, 0.0, intensity))

    # Branch (reflection)
    branch_intensity = jnp.where(valid, refl_intensity, 0.0)

    return out_pos, out_dir, out_intensity, hit, refl_dir, branch_intensity, valid
