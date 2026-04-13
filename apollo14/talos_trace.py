"""Talos-specific sequential ray tracer — operates on typed TalosRoute.

Hard-coded element dispatch: PartialMirror, RefractingSurface, RectangularPupil,
RectangularAperture. Optimized for the Talos combiner layout.

For the generic tracer that works with any Route, see ``trace.py``.
"""

import jax
import jax.numpy as jnp

from apollo14.elements.surface import PartialMirror, MirrorState
from apollo14.elements.refracting_surface import RefractingSurface, RefractState
from apollo14.elements.aperture import RectangularAperture, ApertureState
from apollo14.elements.pupil import RectangularPupil, DetectorState
from apollo14.geometry import snell_refract, normalize, ray_rect_intersect
from apollo14.talos_route import TalosRoute
from apollo14.trace import TraceResult


def trace_ray(origin, direction, route: TalosRoute, color_idx=0) -> TraceResult:
    """Trace a single ray through a TalosRoute. Pure JAX, JIT-compatible.

    Args:
        origin: (3,) ray start position (outside chassis).
        direction: (3,) ray direction (normalized).
        route: TalosRoute with system geometry.
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


def trace_beam(origins, direction, route: TalosRoute, color_idx=0) -> TraceResult:
    """Trace N rays sharing one direction. Optimized: entry refraction computed once.

    Args:
        origins: (N, 3) ray start positions.
        direction: (3,) shared direction (normalized).
        route: TalosRoute with system geometry.
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
        entry_hit, _, entry_valid = ray_rect_intersect(
            origin, direction, route.entry_face.position, route.entry_face.normal,
            route.entry_face.local_x, route.entry_face.local_y,
            route.entry_face.half_extents)
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


def trace_batch(origins, directions, route: TalosRoute, color_idx=0) -> TraceResult:
    """Trace N rays with different directions. vmap over (origin, direction).

    Args:
        origins: (N, 3) ray start positions.
        directions: (N, 3) per-ray directions (normalized).
        route: TalosRoute with system geometry.
        color_idx: color channel index (0=R, 1=G, 2=B).

    Returns:
        TraceResult with (N, M, ...) arrays.
    """
    def _trace_single(origin, direction):
        return trace_ray(origin, direction, route, color_idx)

    return jax.vmap(_trace_single)(origins, directions)
