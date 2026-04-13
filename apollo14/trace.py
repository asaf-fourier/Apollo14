"""Generic sequential ray tracer — uniform SurfaceState throughout.

One step function for the transmitted path, three scans:
  1. preamble  — aperture + entry face
  2. mirrors   — mirror stack (each step also records branch origin)
  3. branch    — exit face + pupil, vmapped over all mirrors

Reflection is computed once per mirror between scans 2 and 3,
not inside the step function.

For the Talos-specific hard-coded tracer, see ``talos_trace.py``.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from apollo14.surface import surface_step, mirror_branch_origin
from apollo14.route import Route


class TraceResult(NamedTuple):
    """Result of tracing rays through a route.

    For a single ray, arrays have shape (M, ...) for M mirrors.
    For batched rays, they gain an additional leading (N, ...) dimension.
    """
    pupil_points: jnp.ndarray      # (..., M, 3) where reflected rays hit pupil
    intensities: jnp.ndarray       # (..., M) reflected intensity per mirror
    valid: jnp.ndarray             # (..., M) bool — reflection reached pupil
    main_hits: jnp.ndarray         # (..., M, 3) hit points on mirrors
    branch_hits: jnp.ndarray       # (..., M, B, 3) intermediate branch hits

    @property
    def total_intensity(self) -> jnp.ndarray:
        """Sum of valid intensities across all mirrors. Shape (...)."""
        return jnp.sum(jnp.where(self.valid, self.intensities, 0.0), axis=-1)


def trace_ray(origin, direction, route: Route, color_idx=0) -> TraceResult:
    """Trace a single ray through a Route.

    Args:
        origin: (3,) ray start position.
        direction: (3,) ray direction (normalized).
        route: Route with preamble, mirrors, and branch.
        color_idx: color channel index (0=R, 1=G, 2=B).

    Returns:
        TraceResult with (M,) arrays.
    """
    def step(carry, state):
        pos, d, inten = carry
        out_pos, out_dir, out_inten, hit, valid = \
            surface_step(state, pos, d, inten, color_idx)
        return (out_pos, out_dir, out_inten), (hit, d, inten, valid)

    # 1. Preamble: aperture + entry face
    init = (origin, direction, jnp.float32(1.0))
    (pos, d, inten), _ = jax.lax.scan(step, init, route.preamble)

    # 2. Mirrors: transmitted path continues, record pre-step state for branches
    (_, _, _), (mirror_hits, pre_dirs, pre_intens, mirror_valids) = \
        jax.lax.scan(step, (pos, d, inten), route.mirrors)

    # Compute reflected ray at each mirror (one-time, between scans)
    refl_dirs, refl_intens = jax.vmap(
        lambda state, d, inten, hit, valid:
            mirror_branch_origin(state, d, inten, hit, valid, color_idx)
    )(route.mirrors, pre_dirs, pre_intens, mirror_hits, mirror_valids)

    # 3. Branch per mirror: exit face + pupil
    def branch_trace(hit, refl_dir, refl_inten):
        def bstep(carry, state):
            pos, d, inten = carry
            out_pos, out_dir, out_inten, bhit, bvalid = \
                surface_step(state, pos, d, inten, color_idx)
            return (out_pos, out_dir, out_inten), (bhit, bvalid)

        _, (bhits, bvalids) = jax.lax.scan(
            bstep, (hit, refl_dir, refl_inten), route.branch)
        all_valid = jnp.all(bvalids)
        pupil_point = bhits[-1]
        return pupil_point, all_valid, bhits

    pupil_points, branch_valids, branch_hits = \
        jax.vmap(branch_trace)(mirror_hits, refl_dirs, refl_intens)

    return TraceResult(
        pupil_points=pupil_points,           # (M, 3)
        intensities=refl_intens,             # (M,)
        valid=mirror_valids & branch_valids,  # (M,)
        main_hits=mirror_hits,               # (M, 3)
        branch_hits=branch_hits,             # (M, B, 3)
    )


def trace_beam(origins, direction, route: Route, color_idx=0) -> TraceResult:
    """Trace N rays sharing one direction through a Route.

    Args:
        origins: (N, 3) ray start positions.
        direction: (3,) shared direction (normalized).
        route: Route with preamble, mirrors, and branch.
        color_idx: color channel index (0=R, 1=G, 2=B).

    Returns:
        TraceResult with (N, M, ...) arrays.
    """
    def _single(origin):
        return trace_ray(origin, direction, route, color_idx)

    return jax.vmap(_single)(origins)


def trace_batch(origins, directions, route: Route, color_idx=0) -> TraceResult:
    """Trace N rays with different directions through a Route.

    Args:
        origins: (N, 3) ray start positions.
        directions: (N, 3) per-ray directions (normalized).
        route: Route with preamble, mirrors, and branch.
        color_idx: color channel index (0=R, 1=G, 2=B).

    Returns:
        TraceResult with (N, M, ...) arrays.
    """
    def _single(origin, direction):
        return trace_ray(origin, direction, route, color_idx)

    return jax.vmap(_single)(origins, directions)
