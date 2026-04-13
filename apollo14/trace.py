"""Sequential ray tracer — operates on Routes.

Pure JAX functions: JIT-compiled, vmap-batched, and jax.grad-differentiable.
No Python control flow on traced values, no jnp.where dispatch on types.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from apollo14.elements.surface import PartialMirror, MirrorState
from apollo14.elements.refracting_surface import RefractingSurface, RefractState
from apollo14.elements.aperture import RectangularAperture, ApertureState
from apollo14.elements.pupil import RectangularPupil, DetectorState
from apollo14.geometry import snell_refract, normalize, ray_rect_intersect
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
    branch_hits: jnp.ndarray       # (..., M, 2, 3) exit face + pupil hits

    @property
    def total_intensity(self) -> jnp.ndarray:
        """Sum of valid intensities across all mirrors. Shape (...)."""
        return jnp.sum(jnp.where(self.valid, self.intensities, 0.0), axis=-1)


def trace_ray(origin, direction, route: Route, color_idx=0) -> TraceResult:
    """Trace a single ray through a route. Pure JAX, JIT-compatible.

    Args:
        origin: (3,) ray start position (outside chassis).
        direction: (3,) ray direction (normalized).
        route: Route with system geometry.
        color_idx: color channel index (0=R, 1=G, 2=B).

    Returns:
        TraceResult with (M,) arrays.
    """
    # ── Preamble: aperture clip + entry refraction (unrolled) ──
    intensity = RectangularAperture.jax_clip(
        route.aperture, origin, direction, route.has_aperture)

    entry_hit, d_glass, intensity, entry_valid = RefractingSurface.jax_interact(
        route.entry_face, origin, direction, intensity)

    # Zero intensity if entry missed
    intensity = jnp.where(entry_valid, intensity, 0.0)

    # ── Mirror scan (lax.scan, dynamic M) ──
    def mirror_step(carry, step_data):
        pos, d, inten = carry
        mirror_s, exit_s, target_s = step_data

        # Mirror interaction: split into transmitted + reflected
        hit, trans_dir, trans_int, refl_dir, refl_int, m_valid = \
            PartialMirror.jax_interact(mirror_s, pos, d, inten, color_idx)

        # Branch: exit face refraction (unrolled step 1)
        exit_hit, exit_dir, _, exit_valid = RefractingSurface.jax_interact(
            exit_s, hit, refl_dir, refl_int)

        # Branch: pupil detection (unrolled step 2)
        pupil_hit, _, pupil_valid = RectangularPupil.jax_interact(
            target_s, exit_hit, exit_dir, refl_int)

        branch_valid = m_valid & exit_valid & pupil_valid

        # Advance main ray (transmitted path)
        new_pos = jnp.where(m_valid, hit, pos)

        return (new_pos, d, trans_int), (
            pupil_hit, refl_int, branch_valid, hit,
            jnp.stack([exit_hit, pupil_hit]),
        )

    init = (entry_hit, d_glass, intensity)
    scan_data = (route.mirrors, route.exit_face, route.target)
    _, (pupil_pts, intensities, valids, main_hits, branch_hits) = \
        jax.lax.scan(mirror_step, init, scan_data)

    return TraceResult(
        pupil_points=pupil_pts,
        intensities=intensities,
        valid=valids,
        main_hits=main_hits,
        branch_hits=branch_hits,
    )


def trace_beam(origins, direction, route: Route, color_idx=0) -> TraceResult:
    """Trace N rays sharing one direction. Optimized: entry refraction computed once.

    Args:
        origins: (N, 3) ray start positions.
        direction: (3,) shared direction (normalized).
        route: Route with system geometry.
        color_idx: color channel index (0=R, 1=G, 2=B).

    Returns:
        TraceResult with (N, M, ...) arrays.
    """
    # Precompute refracted direction inside glass (shared for all rays)
    facing = jnp.where(jnp.dot(direction, route.entry_face.normal) < 0,
                       route.entry_face.normal, -route.entry_face.normal)
    d_glass, _ = snell_refract(direction, facing,
                               route.entry_face.n1, route.entry_face.n2)

    def _trace_single(origin):
        # Aperture clip
        intensity = RectangularAperture.jax_clip(
            route.aperture, origin, direction, route.has_aperture)

        # Entry face intersection + bounds check (direction already refracted)
        denom = jnp.dot(direction, route.entry_face.normal)
        t = jnp.dot(route.entry_face.position - origin, route.entry_face.normal) / (denom + 1e-30)
        entry_hit = origin + jnp.maximum(t, 0.0) * direction

        delta = entry_hit - route.entry_face.position
        entry_valid = (
            (jnp.abs(jnp.dot(delta, route.entry_face.local_x)) <= route.entry_face.half_extents[0]) &
            (jnp.abs(jnp.dot(delta, route.entry_face.local_y)) <= route.entry_face.half_extents[1]) &
            (t > 0)
        )
        intensity = jnp.where(entry_valid, intensity, 0.0)

        # Mirror scan using shared d_glass
        def mirror_step(carry, step_data):
            pos, d, inten = carry
            mirror_s, exit_s, target_s = step_data

            hit, _, trans_int, refl_dir, refl_int, m_valid = \
                PartialMirror.jax_interact(mirror_s, pos, d, inten, color_idx)

            exit_hit, exit_dir, _, exit_valid = RefractingSurface.jax_interact(
                exit_s, hit, refl_dir, refl_int)

            pupil_hit, _, pupil_valid = RectangularPupil.jax_interact(
                target_s, exit_hit, exit_dir, refl_int)

            branch_valid = m_valid & exit_valid & pupil_valid
            new_pos = jnp.where(m_valid, hit, pos)

            return (new_pos, d, trans_int), (
                pupil_hit, refl_int, branch_valid, hit,
                jnp.stack([exit_hit, pupil_hit]),
            )

        init = (entry_hit, d_glass, intensity)
        scan_data = (route.mirrors, route.exit_face, route.target)
        _, (pupil_pts, intensities, valids, main_hits, branch_hits) = \
            jax.lax.scan(mirror_step, init, scan_data)

        return TraceResult(
            pupil_points=pupil_pts,
            intensities=intensities,
            valid=valids,
            main_hits=main_hits,
            branch_hits=branch_hits,
        )

    return jax.vmap(_trace_single)(origins)


def trace_batch(origins, directions, route: Route, color_idx=0) -> TraceResult:
    """Trace N rays with different directions. vmap over (origin, direction).

    Args:
        origins: (N, 3) ray start positions.
        directions: (N, 3) per-ray directions (normalized).
        route: Route with system geometry.
        color_idx: color channel index (0=R, 1=G, 2=B).

    Returns:
        TraceResult with (N, M, ...) arrays.
    """
    def _trace_single(origin, direction):
        return trace_ray(origin, direction, route, color_idx)

    return jax.vmap(_trace_single)(origins, directions)
