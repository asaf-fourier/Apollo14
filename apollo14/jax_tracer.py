"""JAX-native path-based tracer for differentiable ray tracing.

The tracer is **system-agnostic**: it traces rays through user-defined
optical paths — sequences of ``PathStep`` surfaces with specified
interactions. You build the path, the tracer executes it. Fully
JIT-compatible and differentiable via ``jax.grad``.

Architecture
------------

The tracer separates **path definition** (what surfaces exist and how
rays interact with them) from **path execution** (propagating rays
through those surfaces). This means:

- You can trace any optical system, not just the combiner.
- You control which surfaces are included and in what order.
- You can build paths once and reuse them for many rays (``vmap``).

How it works
~~~~~~~~~~~~

A ray propagates through two nested sequences of ``PathStep`` surfaces:

1. **Main path** ``(M,)`` — the primary sequence of surfaces the ray
   visits in order. At each step, the ray intersects the surface and
   interacts according to the step's ``interaction`` type.

2. **Branch paths** ``(M, B)`` — at each main step, an optional branch
   of B surfaces is traced from the reflected ray. This is how light
   that splits off the main path (e.g., reflections from partial mirrors)
   reaches a detector.

The main-path scan uses ``jax.lax.scan`` over M steps. At each PARTIAL
step, the reflected ray is traced through B branch steps via a nested
scan. The branch endpoints, intensities, and validity flags are the
tracer's output.

Interaction types
~~~~~~~~~~~~~~~~~

Each ``PathStep`` has an ``interaction`` field (integer):

- ``REFRACT`` (0) — Snell's law refraction. Changes the ray direction
  based on ``n1`` and ``n2``. Used for glass-air boundaries.
- ``PARTIAL`` (1) — Partial mirror. Splits intensity: ``reflectance``
  fraction goes into the branch path, the rest continues along the
  main path. The branch ray direction is the specular reflection.
- ``TARGET`` (2) — Detector/target surface. Records the hit point.
  No direction or intensity change. Used as the final branch step
  to check if the ray reaches the pupil/sensor.

Building paths with step constructors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the step constructors ``partial()``, ``refract()``, and ``target()``
to create steps from your optical elements. Then ``stack_path()``
assembles the lists into the arrays ``trace_path`` expects.

Step constructors (one per interaction type):

- ``partial(mirror)`` — wraps a ``PartialMirror`` element. Extracts
  position, normal, width/height, and reflection_ratio. Accepts an
  optional ``reflectance`` override ``(3,)`` for per-color R/G/B.
- ``refract(face, n1, n2)`` — wraps a ``GlassFace`` element. You
  must provide the refractive indices explicitly.
- ``target(pupil)`` — wraps a ``Pupil`` element. Circular bounds from
  the pupil's radius.

``stack_path(main_list, branch_lists)`` takes:

- ``main_list``: ``List[PathStep]`` of length M
- ``branch_lists``: ``List[List[PathStep]]`` of length M, each inner
  list of length B (same B for all branches)
- Returns ``(main_steps, branch_steps)`` with shapes ``(M,)``/``(M,B)``

Usage — generic path
~~~~~~~~~~~~~~~~~~~~

::

    from apollo14.jax_tracer import partial, refract, target, stack_path, trace_path
    from apollo14.elements.surface import PartialMirror
    from apollo14.elements.pupil import Pupil

    # Define paths using your elements
    main = [partial(m) for m in mirrors]
    branches = [
        [refract(exit_face, n_glass, 1.0), target(pupil)]
        for exit_face in exit_faces
    ]
    main_steps, branch_steps = stack_path(main, branches)

    # Trace
    endpoints, intensities, valid = trace_path(
        origin, direction, jnp.array(1.0),
        main_steps, branch_steps, color_idx=0)

    # Batch with vmap:
    trace_many = jax.vmap(lambda o: trace_path(
        o, direction, jnp.array(1.0), main_steps, branch_steps))
    all_endpoints, all_ints, all_valid = trace_many(origins)

Usage — combiner convenience
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the AR combiner system, helper functions build the paths from
``CombinerParams``::

    from apollo14.jax_tracer import trace_beam, params_from_system
    from apollo14.combiner import build_default_system, DEFAULT_WAVELENGTH

    system = build_default_system()
    params = params_from_system(system, DEFAULT_WAVELENGTH)
    # n_glass is computed internally by params_from_system

    # Beam of rays with shared direction (most efficient)
    pts, intensities, valid = trace_beam(origins, direction, n_glass, params)

    # Single ray
    pts, intensities, valid = trace_ray(origin, direction, n_glass, params)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import List, NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
    from apollo14.elements.surface import PartialMirror
    from apollo14.elements.glass_block import GlassFace
    from apollo14.elements.pupil import Pupil, RectangularPupil

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


# ── Step constructors ───────────────────────────────────────────────────────

def partial(mirror: PartialMirror, reflectance=None) -> PathStep:
    """Create a PARTIAL step from a PartialMirror element.

    Args:
        mirror: PartialMirror with position, normal, width, height,
            and reflection_ratio.
        reflectance: optional (3,) per-color [R,G,B] reflectance override.
            If None, uses mirror.reflection_ratio for all 3 channels.
    """
    n = normalize(mirror.normal)
    lx, ly = compute_local_axes(n)
    if reflectance is None:
        r = float(mirror.reflection_ratio)
        reflectance = jnp.array([r, r, r])
    else:
        reflectance = jnp.asarray(reflectance)
    return PathStep(
        position=jnp.asarray(mirror.position),
        normal=n,
        half_width=jnp.array(mirror.width / 2),
        half_height=jnp.array(mirror.height / 2),
        local_x=lx,
        local_y=ly,
        interaction=jnp.array(PARTIAL, dtype=jnp.int32),
        n1=jnp.array(1.0),
        n2=jnp.array(1.0),
        reflectance=reflectance,
        use_circular=jnp.array(False),
        radius=jnp.array(0.0),
    )


def refract(face: GlassFace, n1: float, n2: float) -> PathStep:
    """Create a REFRACT step from a GlassFace element.

    Args:
        face: GlassFace with position, normal, and vertices.
        n1: refractive index on the incoming side.
        n2: refractive index on the outgoing side.
    """
    n = normalize(face.normal)
    lx, ly = compute_local_axes(n)
    # Compute rectangular bounds from vertex bbox in local coords
    deltas = face.vertices - face.position
    proj_x = jnp.array([float(jnp.dot(d, lx)) for d in deltas])
    proj_y = jnp.array([float(jnp.dot(d, ly)) for d in deltas])
    half_w = float(jnp.max(jnp.abs(proj_x)))
    half_h = float(jnp.max(jnp.abs(proj_y)))
    return PathStep(
        position=jnp.asarray(face.position),
        normal=n,
        half_width=jnp.array(half_w),
        half_height=jnp.array(half_h),
        local_x=lx,
        local_y=ly,
        interaction=jnp.array(REFRACT, dtype=jnp.int32),
        n1=jnp.array(float(n1)),
        n2=jnp.array(float(n2)),
        reflectance=jnp.zeros(3),
        use_circular=jnp.array(False),
        radius=jnp.array(0.0),
    )


def target(pupil: Pupil | RectangularPupil) -> PathStep:
    """Create a TARGET step from a Pupil or RectangularPupil element.

    Args:
        pupil: Pupil (circular) or RectangularPupil (rectangular).
    """
    from apollo14.elements.pupil import RectangularPupil as _RectPupil

    n = normalize(pupil.normal)
    lx, ly = compute_local_axes(n)
    is_rect = isinstance(pupil, _RectPupil)
    return PathStep(
        position=jnp.asarray(pupil.position),
        normal=n,
        half_width=jnp.array(pupil.width / 2 if is_rect else 1e6),
        half_height=jnp.array(pupil.height / 2 if is_rect else 1e6),
        local_x=lx,
        local_y=ly,
        interaction=jnp.array(TARGET, dtype=jnp.int32),
        n1=jnp.array(1.0),
        n2=jnp.array(1.0),
        reflectance=jnp.zeros(3),
        use_circular=jnp.array(not is_rect),
        radius=jnp.array(float(pupil.radius) if not is_rect else 0.0),
    )


def stack_path(main_list: List[PathStep],
               branch_lists: List[List[PathStep]]):
    """Stack lists of PathStep into batched arrays for trace_path.

    Args:
        main_list: list of M PathStep values (one per main surface)
        branch_lists: list of M lists, each containing B PathStep values
            (branch path per main surface). All inner lists must have the
            same length B.

    Returns:
        (main_steps, branch_steps) where main_steps has (M,) arrays
        and branch_steps has (M, B) arrays.
    """
    # Stack main path: list of M scalar PathSteps → one PathStep with (M,) arrays
    main_steps = jax.tree.map(lambda *args: jnp.stack(args), *main_list)

    # Stack branches: list of M lists of B PathSteps → PathStep with (M, B) arrays
    # First stack each branch list into (B,), then stack across M
    stacked_branches = [
        jax.tree.map(lambda *args: jnp.stack(args), *branch)
        for branch in branch_lists
    ]
    branch_steps = jax.tree.map(lambda *args: jnp.stack(args), *stacked_branches)

    return main_steps, branch_steps


class CombinerParams(NamedTuple):
    """JAX-compatible system parameters for the combiner tracer.

    All fields are JAX arrays. Use ``params_from_system()`` to construct
    from an ``OpticalSystem``.
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
    pupil_half_width: jnp.ndarray   # scalar
    pupil_half_height: jnp.ndarray  # scalar
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
    """Trace a ray through a sequence of surfaces with branch paths.

    This is the primary entry point. Build your ``PathStep`` arrays
    (see module docstring for how), then call this function. Use
    ``jax.vmap`` to trace many rays efficiently.

    Args:
        origin: (3,) ray start position
        direction: (3,) ray direction (normalized)
        intensity: scalar, initial intensity (typically 1.0)
        main_steps: PathStep with (M,) arrays — M surfaces in sequence
        branch_steps: PathStep with (M, B) arrays — B branch steps per
            main surface (traced when a PARTIAL step splits the ray)
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
        half_width=jnp.full(m, float(params.pupil_half_width)),
        half_height=jnp.full(m, float(params.pupil_half_height)),
        local_x=jnp.tile(params.pupil_local_x, (m, 1)),
        local_y=jnp.tile(params.pupil_local_y, (m, 1)),
        interaction=jnp.full(m, TARGET, dtype=jnp.int32),
        n1=jnp.ones(m),
        n2=jnp.ones(m),
        reflectance=jnp.zeros((m, 3)),
        use_circular=jnp.zeros(m, dtype=bool),
        radius=jnp.zeros(m),
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
    ``build_default_system()``.  Gradients flow from the output array back
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


# ── System conversion ────────────────────────────────────────────────────────

def params_from_system(system, wavelength) -> CombinerParams:
    """Build CombinerParams from an OpticalSystem.

    Extracts mirrors, chassis, and pupil from the system. The chassis is
    approximated as an axis-aligned bounding box from its face vertices.

    Args:
        system: OpticalSystem containing GlassBlock, PartialMirror, and
            RectangularPupil elements.
        wavelength: Trace wavelength (for computing glass refractive index).
    """
    from apollo14.elements.surface import PartialMirror as _PM
    from apollo14.elements.glass_block import GlassBlock as _GB
    from apollo14.elements.pupil import RectangularPupil as _RP

    mirrors = [e for e in system.elements if isinstance(e, _PM)]
    chassis = next(e for e in system.elements if isinstance(e, _GB))
    pupil = next(e for e in system.elements if isinstance(e, _RP))

    m = len(mirrors)

    # Chassis AABB from face vertices
    all_verts = jnp.concatenate([f.vertices for f in chassis.faces], axis=0)
    chassis_min = jnp.min(all_verts, axis=0)
    chassis_max = jnp.max(all_verts, axis=0)

    # Mirror geometry
    positions = jnp.stack([jnp.asarray(mir.position) for mir in mirrors])
    normals = jnp.stack([jnp.asarray(mir.normal) for mir in mirrors])

    # Per-mirror, per-color reflectances — tile scalar to (3,) per mirror
    reflectances = jnp.stack([
        jnp.full(3, float(mir.reflection_ratio)) for mir in mirrors
    ])

    half_widths = jnp.array([mir.width / 2 for mir in mirrors])
    half_heights = jnp.array([mir.height / 2 for mir in mirrors])

    # Local axes per mirror
    lx_list, ly_list = [], []
    for mir in mirrors:
        lx, ly = compute_local_axes(jnp.asarray(mir.normal))
        lx_list.append(lx)
        ly_list.append(ly)
    local_x = jnp.stack(lx_list)
    local_y = jnp.stack(ly_list)

    # Pupil local axes
    plx, ply = compute_local_axes(jnp.asarray(pupil.normal))

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
        pupil_center=jnp.asarray(pupil.position),
        pupil_normal=jnp.asarray(pupil.normal),
        pupil_half_width=jnp.array(pupil.width / 2),
        pupil_half_height=jnp.array(pupil.height / 2),
        pupil_local_x=plx,
        pupil_local_y=ply,
    )
