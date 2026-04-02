"""JAX-native combiner tracer for differentiable optimization.

Traces rays through the mirror stack in a single vectorized pass.
At each mirror, computes the reflected ray's path through chassis exit
refraction to the pupil plane. All operations are pure JAX — no Python
control flow on array values, no dataclasses in the hot path.

Usage::

    from apollo14.jax_tracer import trace_ray, trace_batch, params_from_config
    from apollo14.combiner import CombinerConfig

    config = CombinerConfig.default()
    params = params_from_config(config)
    n_glass = float(config.chassis.material.n(config.light.wavelength))

    # Single ray
    pts, intensities, valid = trace_ray(
        config.light.position, config.light.direction, n_glass, params)

    # Batched rays (vmap over N rays)
    pts, intensities, valid = trace_batch(origins, directions, n_glass, params)
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple

from apollo14.geometry import normalize, reflect, snell_refract, compute_local_axes
from apollo14.units import EPSILON


class CombinerParams(NamedTuple):
    """JAX-compatible system parameters for the combiner tracer.

    All fields are JAX arrays. Use ``params_from_config()`` to construct
    from a ``CombinerConfig``.
    """
    mirror_positions: jnp.ndarray    # (M, 3)
    mirror_normals: jnp.ndarray      # (M, 3)
    mirror_reflectances: jnp.ndarray # (M,) per-mirror reflection ratio
    mirror_half_widths: jnp.ndarray  # (M,)
    mirror_half_heights: jnp.ndarray # (M,)
    mirror_local_x: jnp.ndarray     # (M, 3) local x-axis per mirror
    mirror_local_y: jnp.ndarray     # (M, 3) local y-axis per mirror
    chassis_min: jnp.ndarray        # (3,) AABB min corner
    chassis_max: jnp.ndarray        # (3,) AABB max corner
    pupil_center: jnp.ndarray       # (3,)
    pupil_normal: jnp.ndarray       # (3,)
    pupil_radius: jnp.ndarray       # scalar
    pupil_local_x: jnp.ndarray      # (3,)
    pupil_local_y: jnp.ndarray      # (3,)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _safe_inv(x):
    """Reciprocal avoiding division by zero."""
    return 1.0 / jnp.where(jnp.abs(x) > 1e-12, x, jnp.copysign(1e-12, x))


def _box_entry(origin, direction, box_min, box_max):
    """Ray-AABB entry from outside. Returns (t_enter, outward_normal)."""
    inv_d = _safe_inv(direction)
    t1 = (box_min - origin) * inv_d
    t2 = (box_max - origin) * inv_d
    t_near = jnp.minimum(t1, t2)
    t_enter = jnp.max(t_near)

    axis = jnp.argmax(t_near)
    # Entering through min face (dir>0) → outward normal is -axis;
    # through max face (dir<0) → outward normal is +axis.
    sign = jnp.where(direction[axis] > 0, -1.0, 1.0)
    normal = jnp.where(jnp.arange(3) == axis, sign, 0.0)
    return t_enter, normal


def _box_exit(origin, direction, box_min, box_max):
    """Ray-AABB exit from inside. Returns (t_exit, outward_normal)."""
    inv_d = _safe_inv(direction)
    t1 = (box_min - origin) * inv_d
    t2 = (box_max - origin) * inv_d
    t_far = jnp.maximum(t1, t2)
    # Avoid picking a face we're sitting on
    t_far = jnp.where(t_far > EPSILON, t_far, jnp.inf)
    t_exit = jnp.min(t_far)

    axis = jnp.argmin(t_far)
    sign = jnp.where(direction[axis] > 0, 1.0, -1.0)
    normal = jnp.where(jnp.arange(3) == axis, sign, 0.0)
    return t_exit, normal


def _plane_t(origin, direction, normal, point):
    """Signed distance from origin along direction to a plane."""
    denom = jnp.dot(direction, normal)
    return jnp.where(
        jnp.abs(denom) > 1e-12,
        jnp.dot(point - origin, normal) / denom,
        jnp.inf,
    )


# ── Main tracer ──────────────────────────────────────────────────────────────

def trace_ray(origin, direction, n_glass, params):
    """Trace one ray through the combiner mirror stack.

    The ray path through the combiner::

        projector
            |  ray in air
            v
        ┌─────────────────────────── chassis (glass) ──┐
        │   entry refraction (air → glass, Snell's law) │
        │       |                                        │
        │       v  d_glass (refracted direction)         │
        │   ┌─mirror_0─┐                                │
        │   │ split:    │──reflected──→ box exit ──→ pupil
        │   │transmitted│                                │
        │       |                                        │
        │       v                                        │
        │   ┌─mirror_1─┐                                │
        │   │ split:    │──reflected──→ box exit ──→ pupil
        │   │transmitted│                                │
        │       |                                        │
        │      ...  (repeat for all M mirrors)           │
        │       |                                        │
        │       v                                        │
        │   ┌─mirror_5─┐                                │
        │   │ split:    │──reflected──→ box exit ──→ pupil
        │   │transmitted│                                │
        │       |                                        │
        └───────|────────────────────────────────────────┘
                v  (transmitted beam exits chassis)

    Each mirror's reflected ray follows a 3-step sub-path:
      1. Travel through glass to the chassis wall (_box_exit)
      2. Refract glass → air at the wall (snell_refract)
      3. Travel through air to the pupil plane (_plane_t)

    The scan carry propagates (position, intensity) along the transmitted
    path. Intensity decreases at each mirror by the reflection ratio.
    If a ray misses a mirror (out of bounds), intensity passes through
    unchanged and no reflected contribution is recorded.

    Args:
        origin: (3,) ray start position (outside the chassis)
        direction: (3,) ray direction (normalized)
        n_glass: refractive index of chassis glass at trace wavelength
        params: CombinerParams with system geometry

    Returns:
        pupil_points: (M, 3) where each reflected ray hits the pupil plane
        pupil_intensities: (M,) reflected intensity per mirror
        pupil_valid: (M,) bool — True if the reflection reaches the pupil
    """
    # ── Step 1: Entry refraction (air → glass) ───────────────────────────
    # Find where the ray hits the chassis box (AABB intersection).
    # entry_normal points outward, toward the incoming ray — this is the
    # convention snell_refract expects (normal toward the n1 medium).
    t_entry, entry_normal = _box_entry(origin, direction,
                                       params.chassis_min, params.chassis_max)
    entry_point = origin + t_entry * direction
    # Refract into glass. d_glass is constant for the entire mirror stack
    # because the thin mirrors don't refract the transmitted beam.
    d_glass, _ = snell_refract(direction, entry_normal, 1.0, n_glass)

    # ── Step 2: Scan through mirrors (jax.lax.scan) ──────────────────────
    # carry = (position along transmitted path, remaining intensity)
    # At each mirror:
    #   - Intersect the transmitted ray with the mirror plane
    #   - Check if the hit point is within the mirror rectangle
    #   - Split intensity: reflected portion goes to pupil, rest continues
    #   - Trace the reflected ray: box exit → refraction → pupil plane
    #   - Record (pupil_point, reflected_intensity, valid) as output

    def step(carry, mirror):
        pos, intensity = carry
        m_pos, m_normal, m_refl, m_hw, m_hh, m_lx, m_ly = mirror

        # ── 2a: Intersect transmitted ray with mirror plane ──────────
        t = _plane_t(pos, d_glass, m_normal, m_pos)
        hit = pos + jnp.maximum(t, 0.0) * d_glass

        # ── 2b: Check if hit is within mirror bounds ─────────────────
        # Project hit-to-center vector onto mirror's local axes and
        # compare against half-width/height. Also reject negative t
        # (mirror is behind the ray).
        delta = hit - m_pos
        in_bounds = ((jnp.abs(jnp.dot(delta, m_lx)) <= m_hw) &
                     (jnp.abs(jnp.dot(delta, m_ly)) <= m_hh) &
                     (t > 0))

        # ── 2c: Compute reflected direction ──────────────────────────
        # Flip the mirror normal to face the incoming ray if needed.
        facing = jnp.where(jnp.dot(d_glass, m_normal) < 0, m_normal, -m_normal)
        d_refl = reflect(d_glass, facing)

        # ── 2d: Split intensity ──────────────────────────────────────
        # If the ray missed the mirror (in_bounds=False), reflected
        # intensity is 0 and transmitted intensity is unchanged.
        refl_int = jnp.where(in_bounds, intensity * m_refl, 0.0)
        trans_int = jnp.where(in_bounds, intensity * (1.0 - m_refl), intensity)

        # ── 2e: Trace reflected ray to pupil (3-step sub-path) ───────

        # Sub-step i: Find where reflected ray exits the chassis box.
        t_exit, exit_normal = _box_exit(hit, d_refl,
                                        params.chassis_min, params.chassis_max)
        exit_pt = hit + t_exit * d_refl

        # Sub-step ii: Refract glass → air at the chassis wall.
        # Negate exit_normal so it points inward (toward glass = n1 medium).
        d_air, is_tir = snell_refract(d_refl, -exit_normal, n_glass, 1.0)

        # Sub-step iii: Intersect refracted ray with pupil plane.
        t_pupil = _plane_t(exit_pt, d_air, params.pupil_normal, params.pupil_center)
        pupil_pt = exit_pt + t_pupil * d_air

        # ── 2f: Check if the reflected ray actually reaches the pupil ─
        # Must be: within mirror bounds, within pupil circle,
        # pupil is in front (t > 0), and no total internal reflection.
        p_delta = pupil_pt - params.pupil_center
        p_r2 = (jnp.dot(p_delta, params.pupil_local_x) ** 2 +
                jnp.dot(p_delta, params.pupil_local_y) ** 2)
        on_pupil = p_r2 <= params.pupil_radius ** 2

        valid = in_bounds & on_pupil & (t_pupil > 0) & ~is_tir

        # ── 2g: Advance the transmitted ray ──────────────────────────
        # If the ray hit the mirror, advance position to the hit point.
        # If it missed, keep the current position for the next mirror.
        new_pos = jnp.where(in_bounds, hit, pos)
        return (new_pos, trans_int), (pupil_pt, refl_int, valid)

    mirror_data = (
        params.mirror_positions,
        params.mirror_normals,
        params.mirror_reflectances,
        params.mirror_half_widths,
        params.mirror_half_heights,
        params.mirror_local_x,
        params.mirror_local_y,
    )

    init = (entry_point, jnp.array(1.0))
    _, (pupil_pts, pupil_ints, pupil_valids) = jax.lax.scan(step, init, mirror_data)

    return pupil_pts, pupil_ints, pupil_valids


trace_batch = jax.vmap(trace_ray, in_axes=(0, 0, None, None))
"""Batched version of trace_ray: vmap over (origin, direction).

Each ray gets its own origin and direction. The glass refractive index
and system params are shared across all rays. Useful when tracing rays
with different directions (e.g. angular scan across the FOV).

    trace_batch(origins, directions, n_glass, params)
        origins:    (N, 3)
        directions: (N, 3)
        → pupil_points (N, M, 3), intensities (N, M), valid (N, M)
"""


# ── Beam tracer (shared direction) ───────────────────────────────────────────

def trace_beam(origins, direction, n_glass, params):
    """Trace a beam of rays that share the same direction.

    Optimized for the common case of a projector beam: many parallel rays
    (different origins) all traveling in the same direction. Since the
    direction is shared, several quantities are computed once and reused::

        Precomputed once (direction-dependent):
          - d_glass:  refracted direction inside the chassis
          - d_refl_i: reflected direction at each mirror
          - d_air_i:  refracted direction after chassis exit, per mirror
          - is_tir_i: whether total internal reflection occurs, per mirror

        Computed per ray (position-dependent only):
          - Entry point on the chassis box (different per origin)
          - Hit point on each mirror plane
          - Bounds check (is the hit within the mirror rectangle?)
          - Chassis exit point for reflected ray
          - Pupil plane intersection point

    Compared to trace_batch (which recomputes all directions per ray),
    this avoids redundant Snell/reflect calls. The speedup is modest on
    CPU (~1.1x) but the API is cleaner for beam tracing.

    Args:
        origins: (N, 3) ray start positions
        direction: (3,) shared ray direction (normalized)
        n_glass: refractive index of chassis glass at trace wavelength
        params: CombinerParams with system geometry

    Returns:
        pupil_points: (N, M, 3) hit positions on pupil plane
        pupil_intensities: (N, M) reflected intensity per mirror per ray
        pupil_valid: (N, M) bool mask
    """
    # ── Precompute shared direction quantities (once for all rays) ────────

    # Entry refraction: all rays in a coherent beam enter through the same
    # chassis face, so the entry normal and refracted direction are shared.
    _, entry_normal = _box_entry(origins[0], direction,
                                 params.chassis_min, params.chassis_max)
    d_glass, _ = snell_refract(direction, entry_normal, 1.0, n_glass)

    # Per-mirror precomputation: reflected direction, chassis exit face,
    # and refracted air direction. These depend only on d_glass and the
    # mirror normals, not on ray position.
    M = params.mirror_positions.shape[0]

    def precompute_mirror(mirror):
        m_normal = mirror[1]
        facing = jnp.where(jnp.dot(d_glass, m_normal) < 0, m_normal, -m_normal)
        d_refl = reflect(d_glass, facing)

        # Exit face for this reflected direction. We approximate by using
        # the box center as origin — valid because all reflected rays from
        # one mirror travel in the same direction and exit through the
        # same face regardless of position within the chassis.
        _, exit_normal = _box_exit(params.chassis_min * 0.5 + params.chassis_max * 0.5,
                                   d_refl, params.chassis_min, params.chassis_max)
        d_air, is_tir = snell_refract(d_refl, -exit_normal, n_glass, 1.0)
        return d_refl, exit_normal, d_air, is_tir

    mirror_data = (
        params.mirror_positions,
        params.mirror_normals,
        params.mirror_reflectances,
        params.mirror_half_widths,
        params.mirror_half_heights,
        params.mirror_local_x,
        params.mirror_local_y,
    )

    # vmap precompute_mirror over the M mirrors
    d_refls, exit_normals, d_airs, is_tirs = jax.vmap(
        precompute_mirror)(mirror_data)
    # d_refls: (M, 3), d_airs: (M, 3), is_tirs: (M,)

    # ── Per-ray scan (only position-dependent math) ──────────────────────
    # This inner function is vmapped over all ray origins. It uses the
    # same scan-over-mirrors structure as trace_ray, but indexes into
    # the precomputed arrays instead of recomputing directions.

    def trace_single_origin(origin):
        t_entry, _ = _box_entry(origin, direction,
                                params.chassis_min, params.chassis_max)
        entry_point = origin + t_entry * direction

        def step(carry, idx):
            pos, intensity = carry
            # Index into per-mirror arrays (precomputed + params)
            m_pos = params.mirror_positions[idx]
            m_refl = params.mirror_reflectances[idx]
            m_hw = params.mirror_half_widths[idx]
            m_hh = params.mirror_half_heights[idx]
            m_lx = params.mirror_local_x[idx]
            m_ly = params.mirror_local_y[idx]
            m_normal = params.mirror_normals[idx]
            d_refl = d_refls[idx]       # precomputed
            d_air = d_airs[idx]         # precomputed
            is_tir = is_tirs[idx]       # precomputed

            # Intersect, bounds check, split — same as trace_ray
            t = _plane_t(pos, d_glass, m_normal, m_pos)
            hit = pos + jnp.maximum(t, 0.0) * d_glass

            delta = hit - m_pos
            in_bounds = ((jnp.abs(jnp.dot(delta, m_lx)) <= m_hw) &
                         (jnp.abs(jnp.dot(delta, m_ly)) <= m_hh) &
                         (t > 0))

            refl_int = jnp.where(in_bounds, intensity * m_refl, 0.0)
            trans_int = jnp.where(in_bounds, intensity * (1.0 - m_refl), intensity)

            # Reflected ray: exit point is position-dependent, but
            # direction (d_refl) and air refraction (d_air) are shared.
            t_exit, _ = _box_exit(hit, d_refl,
                                  params.chassis_min, params.chassis_max)
            exit_pt = hit + t_exit * d_refl

            t_pupil = _plane_t(exit_pt, d_air,
                               params.pupil_normal, params.pupil_center)
            pupil_pt = exit_pt + t_pupil * d_air

            p_delta = pupil_pt - params.pupil_center
            p_r2 = (jnp.dot(p_delta, params.pupil_local_x) ** 2 +
                    jnp.dot(p_delta, params.pupil_local_y) ** 2)
            on_pupil = p_r2 <= params.pupil_radius ** 2

            valid = in_bounds & on_pupil & (t_pupil > 0) & ~is_tir

            new_pos = jnp.where(in_bounds, hit, pos)
            return (new_pos, trans_int), (pupil_pt, refl_int, valid)

        init = (entry_point, jnp.array(1.0))
        _, (pupil_pts, pupil_ints, pupil_valids) = jax.lax.scan(
            step, init, jnp.arange(M))

        return pupil_pts, pupil_ints, pupil_valids

    return jax.vmap(trace_single_origin)(origins)


# ── Config conversion ────────────────────────────────────────────────────────

def params_from_config(config) -> CombinerParams:
    """Build CombinerParams from a CombinerConfig.

    Replicates the mirror placement logic from ``combiner.build_system()``.
    The chassis is approximated as an axis-aligned bounding box (ignoring
    the small z_skew).
    """
    # Chassis AABB
    center = config.chassis.center
    half = config.chassis.dimensions / 2
    chassis_min = center - half
    chassis_max = center + half

    # Mirror positions (same logic as build_system)
    mirror_y_width = config.mirror.y_width
    chassis_z = float(config.chassis.dimensions[2])
    mirror_edge_to_center_y = 0.5 * jnp.sqrt(mirror_y_width ** 2 - chassis_z ** 2)
    first_pos = (config.chassis.first_mirror_center -
                 jnp.array([0.0, float(mirror_edge_to_center_y), 0.0]))
    mirror_offset_y = config.chassis.distance_between_mirrors / config.mirror.normal[1]
    mirror_offset = jnp.array([0.0, float(mirror_offset_y), 0.0])

    M = config.num_mirrors
    positions = jnp.stack([first_pos - i * mirror_offset for i in range(M)])
    normals = jnp.tile(config.mirror.normal, (M, 1))

    # Per-mirror reflectances (compensated for upstream loss)
    global_refl = config.mirror.reflection_ratio
    refl_list = []
    transmitted = 1.0
    for _ in range(M):
        refl_list.append(global_refl / transmitted)
        transmitted -= global_refl
    reflectances = jnp.array(refl_list)

    half_widths = jnp.full(M, config.mirror.x_width / 2)
    half_heights = jnp.full(M, config.mirror.y_width / 2)

    # Local axes per mirror (all mirrors share the same normal)
    lx, ly = compute_local_axes(config.mirror.normal)
    local_x = jnp.tile(lx, (M, 1))
    local_y = jnp.tile(ly, (M, 1))

    # Pupil local axes
    plx, ply = compute_local_axes(config.pupil.normal)

    return CombinerParams(
        mirror_positions=positions,
        mirror_normals=normals,
        mirror_reflectances=reflectances,
        mirror_half_widths=half_widths,
        mirror_half_heights=half_heights,
        mirror_local_x=local_x,
        mirror_local_y=local_y,
        chassis_min=chassis_min,
        chassis_max=chassis_max,
        pupil_center=config.pupil.center,
        pupil_normal=config.pupil.normal,
        pupil_radius=jnp.array(config.pupil.radius),
        pupil_local_x=plx,
        pupil_local_y=ply,
    )
