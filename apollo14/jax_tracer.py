"""JAX-native path-based tracer for differentiable optimization.

Traces rays through predefined optical paths — sequences of surfaces with
specified interactions (refraction, partial reflection, target). The path
is defined as data, making the tracer general-purpose while remaining
fully JIT-compatible.

The generic entry point is ``trace_path``, which takes a sequence of
``PathStep`` surfaces. For the combiner system, ``build_combiner_paths``
constructs the appropriate paths from ``CombinerParams``, and convenience
functions (``trace_ray``, ``trace_batch``, ``trace_beam``) handle the
full workflow including chassis entry refraction.

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


# ── Interaction types ────────────────────────────────────────────────────────

REFRACT = 0   # Snell's law refraction at a surface
PARTIAL = 1   # Partial reflection: splits intensity, reflects a branch
TARGET = 2    # Target surface: record hit point, check bounds


# ── Data structures ──────────────────────────────────────────────────────────

class PathStep(NamedTuple):
    """One surface interaction in a traced optical path.

    All fields are JAX arrays. When stacked into a path, each field
    gains a leading dimension: (N,) for N steps, or (M, B) for
    M main steps x B branch steps.
    """
    position: jnp.ndarray       # (3,) surface center point
    normal: jnp.ndarray         # (3,) surface normal
    half_width: jnp.ndarray     # scalar, rectangular bounds
    half_height: jnp.ndarray    # scalar, rectangular bounds
    local_x: jnp.ndarray        # (3,) local x-axis for bounds check
    local_y: jnp.ndarray        # (3,) local y-axis for bounds check
    interaction: jnp.ndarray    # int scalar: REFRACT, PARTIAL, or TARGET
    n1: jnp.ndarray             # scalar, refractive index (incoming side)
    n2: jnp.ndarray             # scalar, refractive index (outgoing side)
    reflectance: jnp.ndarray    # (3,) per-color [R,G,B] reflectance
    use_circular: jnp.ndarray   # bool scalar, use radius instead of rect
    radius: jnp.ndarray         # scalar, circular bounds radius


class CombinerParams(NamedTuple):
    """JAX-compatible system parameters for the combiner tracer.

    All fields are JAX arrays. Use ``params_from_config()`` to construct
    from a ``CombinerConfig``.
    """
    mirror_positions: jnp.ndarray    # (M, 3)
    mirror_normals: jnp.ndarray      # (M, 3)
    mirror_reflectances: jnp.ndarray # (M, 3) per-mirror, per-color [R,G,B]
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


# ── Internal helpers ─────────────────────────────────────────────────────────

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
    # Entering through min face (dir>0) -> outward normal is -axis;
    # through max face (dir<0) -> outward normal is +axis.
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


# ── Generic path tracer ─────────────────────────────────────────────────────

def trace_path(origin, direction, intensity, main_steps, branch_steps,
               color_idx=0):
    """Trace a ray through a predefined optical path with branches.

    The ray traverses ``main_steps`` in sequence. At each step the ray
    interacts with a planar surface:

    - **PARTIAL** — splits intensity: the transmitted ray continues along
      the main path, while the reflected ray is traced through the
      corresponding ``branch_steps`` (e.g., through an exit face to a
      pupil plane).
    - **REFRACT** — applies Snell's law, changing the ray direction.
    - **TARGET** — records the hit point (no direction/intensity change).

    All operations are pure JAX — no Python control flow on array values.
    The main path uses ``jax.lax.scan``; each branch is a nested scan.

    Args:
        origin: (3,) start position (typically after entry refraction)
        direction: (3,) initial direction inside the medium
        intensity: scalar, initial intensity
        main_steps: PathStep with (M,) arrays — M surfaces in sequence
        branch_steps: PathStep with (M, B) arrays — B branch steps per
            main surface (used when a PARTIAL interaction splits the ray)
        color_idx: int, selects reflectance channel [0=R, 1=G, 2=B]

    Returns:
        branch_endpoints: (M, 3) where each branch ray ends up
        branch_intensities: (M,) reflected intensity per main step
        branch_valid: (M,) whether each branch reached its target
    """

    def _branch_step(carry, step):
        """Trace one step of a branch path (exit refraction, pupil, etc.)."""
        pos, d, inten, valid = carry

        t = _plane_t(pos, d, step.normal, step.position)
        hit = pos + jnp.maximum(t, 0.0) * d

        # Bounds check (rectangular or circular)
        delta = hit - step.position
        rect_ok = ((jnp.abs(jnp.dot(delta, step.local_x)) <= step.half_width) &
                   (jnp.abs(jnp.dot(delta, step.local_y)) <= step.half_height))
        circ_ok = (jnp.dot(delta, step.local_x) ** 2 +
                   jnp.dot(delta, step.local_y) ** 2) <= step.radius ** 2
        in_bounds = jnp.where(step.use_circular, circ_ok, rect_ok) & (t > 0)

        # Refraction (always computed; selected via interaction type)
        facing = jnp.where(jnp.dot(d, step.normal) < 0,
                           step.normal, -step.normal)
        d_refr, is_tir = snell_refract(d, facing, step.n1, step.n2)

        is_refract = step.interaction == REFRACT
        new_d = jnp.where(is_refract, d_refr, d)
        new_valid = valid & in_bounds & jnp.where(is_refract, ~is_tir, True)

        return (hit, new_d, inten, new_valid), None

    def _main_step(carry, step_and_branch):
        """Process one main-path surface (mirror / interface)."""
        step, branch = step_and_branch
        pos, d, inten = carry

        # Intersect surface
        t = _plane_t(pos, d, step.normal, step.position)
        hit = pos + jnp.maximum(t, 0.0) * d

        # Bounds check
        delta = hit - step.position
        in_bounds = ((jnp.abs(jnp.dot(delta, step.local_x)) <= step.half_width) &
                     (jnp.abs(jnp.dot(delta, step.local_y)) <= step.half_height) &
                     (t > 0))

        # Direction computations
        facing = jnp.where(jnp.dot(d, step.normal) < 0,
                           step.normal, -step.normal)
        d_refl = reflect(d, facing)
        d_refr, is_tir = snell_refract(d, facing, step.n1, step.n2)

        # Intensity split (PARTIAL only)
        refl = step.reflectance[color_idx]
        is_partial = step.interaction == PARTIAL
        refl_int = jnp.where(is_partial & in_bounds, inten * refl, 0.0)
        trans_int = jnp.where(is_partial & in_bounds,
                              inten * (1.0 - refl), inten)

        # Trace branch (reflected ray through branch steps)
        branch_init = (hit, d_refl, refl_int, in_bounds)
        (branch_end, _, _, branch_valid), _ = jax.lax.scan(
            _branch_step, branch_init, branch)

        # Update main-path carry
        is_refract = step.interaction == REFRACT
        new_d = jnp.where(is_refract & in_bounds, d_refr, d)
        new_inten = jnp.where(is_partial, trans_int, inten)
        new_pos = jnp.where(in_bounds, hit, pos)

        return (new_pos, new_d, new_inten), (branch_end, refl_int, branch_valid)

    init = (origin, direction, intensity)
    _, (endpoints, intensities, valids) = jax.lax.scan(
        _main_step, init, (main_steps, branch_steps))

    return endpoints, intensities, valids


# ── Combiner path builder ───────────────────────────────────────────────────

def build_combiner_paths(params, n_glass, direction):
    """Build main and branch optical paths for the combiner system.

    Precomputes path data from system geometry and ray direction. The
    returned paths encode the full combiner topology as data::

        Main path:  mirror_0 (PARTIAL) -> mirror_1 -> ... -> mirror_M-1
        Branch per mirror:  chassis exit (REFRACT) -> pupil (TARGET)

    Direction-dependent quantities (reflected directions, exit faces,
    air refractions) are baked into the path data. When used with
    ``trace_beam``, this precomputation happens once for the shared
    direction — the ``trace_beam`` optimization falls out naturally.

    Args:
        params: CombinerParams with system geometry
        n_glass: refractive index of chassis glass
        direction: (3,) ray direction in air (before entry refraction)

    Returns:
        entry_normal: (3,) outward normal of the chassis entry face
        d_glass: (3,) refracted direction inside the chassis
        main_steps: PathStep with (M,) arrays
        branch_steps: PathStep with (M, 2) arrays
    """
    box_center = (params.chassis_min + params.chassis_max) / 2

    # Determine entry face and refract into glass
    _, entry_normal = _box_entry(box_center, direction,
                                 params.chassis_min, params.chassis_max)
    d_glass, _ = snell_refract(direction, entry_normal, 1.0, n_glass)

    m = params.mirror_positions.shape[0]

    # ── Main steps: M partial mirrors ────────────────────────────────────
    main_steps = PathStep(
        position=params.mirror_positions,
        normal=params.mirror_normals,
        half_width=params.mirror_half_widths,
        half_height=params.mirror_half_heights,
        local_x=params.mirror_local_x,
        local_y=params.mirror_local_y,
        interaction=jnp.full(m, PARTIAL, dtype=jnp.int32),
        n1=jnp.ones(m),
        n2=jnp.ones(m),
        reflectance=params.mirror_reflectances,
        use_circular=jnp.zeros(m, dtype=bool),
        radius=jnp.zeros(m),
    )

    # ── Branch steps: [exit refraction, pupil target] per mirror ─────────

    def _compute_exit(mirror_normal):
        """Determine chassis exit face for a mirror's reflected ray."""
        facing = jnp.where(jnp.dot(d_glass, mirror_normal) < 0,
                           mirror_normal, -mirror_normal)
        d_refl = reflect(d_glass, facing)
        _, exit_normal = _box_exit(box_center, d_refl,
                                   params.chassis_min, params.chassis_max)
        # A point on the exit face plane (only the axis-aligned component
        # matters for _plane_t, so the other coords are irrelevant)
        face_point = jnp.where(exit_normal > 0, params.chassis_max,
                     jnp.where(exit_normal < 0, params.chassis_min,
                               params.chassis_max))
        exit_lx, exit_ly = compute_local_axes(exit_normal)
        return face_point, exit_normal, exit_lx, exit_ly

    face_points, exit_normals, exit_lxs, exit_lys = jax.vmap(
        _compute_exit)(params.mirror_normals)

    # Exit refraction steps (M,)
    exit_steps = PathStep(
        position=face_points,
        normal=exit_normals,
        half_width=jnp.full(m, 1e6),
        half_height=jnp.full(m, 1e6),
        local_x=exit_lxs,
        local_y=exit_lys,
        interaction=jnp.full(m, REFRACT, dtype=jnp.int32),
        n1=jnp.full(m, n_glass),
        n2=jnp.ones(m),
        reflectance=jnp.zeros((m, 3)),
        use_circular=jnp.zeros(m, dtype=bool),
        radius=jnp.zeros(m),
    )

    # Pupil target steps (M,) — same pupil geometry tiled
    pupil_steps = PathStep(
        position=jnp.tile(params.pupil_center, (m, 1)),
        normal=jnp.tile(params.pupil_normal, (m, 1)),
        half_width=jnp.full(m, 1e6),
        half_height=jnp.full(m, 1e6),
        local_x=jnp.tile(params.pupil_local_x, (m, 1)),
        local_y=jnp.tile(params.pupil_local_y, (m, 1)),
        interaction=jnp.full(m, TARGET, dtype=jnp.int32),
        n1=jnp.ones(m),
        n2=jnp.ones(m),
        reflectance=jnp.zeros((m, 3)),
        use_circular=jnp.ones(m, dtype=bool),
        radius=jnp.full(m, float(params.pupil_radius)),
    )

    # Stack into (M, 2) branch steps
    branch_steps = jax.tree.map(
        lambda a, b: jnp.stack([a, b], axis=1),
        exit_steps, pupil_steps,
    )

    return entry_normal, d_glass, main_steps, branch_steps


# ── Combiner convenience functions ──────────────────────────────────────────

def trace_ray(origin, direction, n_glass, params, color_idx=0):
    """Trace one ray through the combiner mirror stack.

    Handles chassis entry refraction, then delegates to ``trace_path``
    with combiner-specific paths built by ``build_combiner_paths``.

    Args:
        origin: (3,) ray start position (outside the chassis)
        direction: (3,) ray direction (normalized)
        n_glass: refractive index of chassis glass at trace wavelength
        params: CombinerParams with system geometry
        color_idx: color channel index (0=R, 1=G, 2=B)

    Returns:
        pupil_points: (M, 3) where each reflected ray hits the pupil plane
        pupil_intensities: (M,) reflected intensity per mirror
        pupil_valid: (M,) bool — True if the reflection reaches the pupil
    """
    _, d_glass, main_steps, branch_steps = build_combiner_paths(
        params, n_glass, direction)
    t_entry, _ = _box_entry(origin, direction,
                            params.chassis_min, params.chassis_max)
    entry_point = origin + t_entry * direction
    return trace_path(entry_point, d_glass, jnp.array(1.0),
                      main_steps, branch_steps, color_idx)


def trace_batch(origins, directions, n_glass, params, color_idx=0):
    """Batched version of trace_ray: vmap over (origin, direction).

    Each ray gets its own origin and direction. The glass refractive index,
    system params, and color_idx are shared across all rays. Useful when
    tracing rays with different directions (e.g. angular scan across the FOV).

        origins:    (N, 3)
        directions: (N, 3)
        -> pupil_points (N, M, 3), intensities (N, M), valid (N, M)
    """
    return jax.vmap(
        lambda o, d: trace_ray(o, d, n_glass, params, color_idx)
    )(origins, directions)


def trace_beam(origins, direction, n_glass, params, color_idx=0):
    """Trace a beam of rays that share the same direction.

    Optimized for the common case of a projector beam: since all rays
    share the same direction, ``build_combiner_paths`` is called once
    outside the vmap. The path data (reflected directions, exit faces,
    air refractions) is precomputed and reused for all ray origins.

    Args:
        origins: (N, 3) ray start positions
        direction: (3,) shared ray direction (normalized)
        n_glass: refractive index of chassis glass at trace wavelength
        params: CombinerParams with system geometry
        color_idx: color channel index (0=R, 1=G, 2=B)

    Returns:
        pupil_points: (N, M, 3) hit positions on pupil plane
        pupil_intensities: (N, M) reflected intensity per mirror per ray
        pupil_valid: (N, M) bool mask
    """
    # Build paths once (shared direction) — the beam optimization
    _, d_glass, main_steps, branch_steps = build_combiner_paths(
        params, n_glass, direction)

    def _trace_single(origin):
        t_entry, _ = _box_entry(origin, direction,
                                params.chassis_min, params.chassis_max)
        entry_point = origin + t_entry * direction
        return trace_path(entry_point, d_glass, jnp.array(1.0),
                          main_steps, branch_steps, color_idx)

    return jax.vmap(_trace_single)(origins)


# ── Reflectance helpers ─────────────────────────────────────────────────────

def compensated_reflectances(ratio, num_mirrors):
    """Compute per-mirror reflectances compensated for upstream losses.

    Each mirror reflects ``ratio`` of the *original* beam intensity. Because
    earlier mirrors absorb light, later mirrors need a higher local
    reflectance to achieve the same absolute reflected amount::

        r[i] = ratio / (1 - i * ratio)

    This is the JAX-differentiable equivalent of the Python loop in
    ``build_system()``.  Gradients flow from the output array back
    to ``ratio``.

    Args:
        ratio: Target fraction of original intensity reflected by each mirror.
            Either a scalar (same for all colors) or a (3,) array [R,G,B].
        num_mirrors: Number of mirrors (M).

    Returns:
        (M, 3) array of per-mirror, per-color reflectances.
        If ratio is scalar, all three color columns are identical.
    """
    ratio = jnp.atleast_1d(jnp.asarray(ratio))
    if ratio.shape == ():
        ratio = jnp.broadcast_to(ratio, (3,))
    elif ratio.shape == (1,):
        ratio = jnp.broadcast_to(ratio, (3,))
    # ratio is now (3,) — one value per color channel
    i = jnp.arange(num_mirrors)[:, None]  # (M, 1)
    return ratio[None, :] / (1.0 - i * ratio[None, :])  # (M, 3)


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

    m = config.num_mirrors
    positions = jnp.stack([first_pos - i * mirror_offset for i in range(m)])
    normals = jnp.tile(config.mirror.normal, (m, 1))

    # Per-mirror, per-color reflectances (compensated for upstream loss)
    reflectances = compensated_reflectances(config.mirror.reflection_ratio, m)

    half_widths = jnp.full(m, config.mirror.x_width / 2)
    half_heights = jnp.full(m, config.mirror.y_width / 2)

    # Local axes per mirror (all mirrors share the same normal)
    lx, ly = compute_local_axes(config.mirror.normal)
    local_x = jnp.tile(lx, (m, 1))
    local_y = jnp.tile(ly, (m, 1))

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
