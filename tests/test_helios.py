"""Smoke tests for the helios optimization layer."""

import jax
import jax.numpy as jnp

from apollo14.combiner import (
    DEFAULT_BEAM_HEIGHT,
    DEFAULT_BEAM_WIDTH,
    DEFAULT_LIGHT_DIRECTION,
    DEFAULT_LIGHT_POSITION,
    DEFAULT_X_FOV,
    DEFAULT_Y_FOV,
    build_default_system,
)
from apollo14.elements.pupil import RectangularPupil
from apollo14.projector import FovGrid, Projector
from apollo14.units import mm
from helios import compute_eyebox_response, eyebox_grid_points
from helios.merit import DEFAULT_WAVELENGTHS, build_combiner_pupil_routes


def _make_fixtures():
    system = build_default_system()
    pupil = next(e for e in system.elements if isinstance(e, RectangularPupil))
    routes = build_combiner_pupil_routes(
        system, [float(w) for w in DEFAULT_WAVELENGTHS])
    projector = Projector.uniform(
        position=DEFAULT_LIGHT_POSITION, direction=DEFAULT_LIGHT_DIRECTION,
        beam_width=DEFAULT_BEAM_WIDTH, beam_height=DEFAULT_BEAM_HEIGHT,
        nx=3, ny=3,
    )
    return system, pupil, routes, projector


class TestRoutes:

    def test_build_routes_shape(self):
        system = build_default_system()
        routes = build_combiner_pupil_routes(system, [550e-6])
        assert len(routes) == 1
        assert len(routes[0]) == 6  # six mirrors → six branches


class TestEyebox:

    def test_compute_response_shape(self):
        _, pupil, routes, projector = _make_fixtures()
        fov_grid = FovGrid(projector.direction, DEFAULT_X_FOV, DEFAULT_Y_FOV, 3, 3)
        pts = eyebox_grid_points(pupil.position, pupil.normal, 3.0 * mm,
                                 nx=3, ny=3)
        resp = compute_eyebox_response(
            routes, projector, fov_grid, pts,
        )
        assert resp.shape == (9, 9, 3)  # (S, A, 3)
        assert fov_grid.grid_shape == (3, 3)

    def test_grad_through_reflectance(self):
        """Gradients must flow through prepare_route → trace_rays."""
        from apollo14.elements.partial_mirror import MirrorStackSeg
        from apollo14.route import Route, combiner_main_path
        from apollo14.trace import prepare_route, trace_rays

        system = build_default_system()
        route = combiner_main_path(system)
        stack = next(s for s in route.segments if isinstance(s, MirrorStackSeg))

        projector = Projector.uniform(
            position=DEFAULT_LIGHT_POSITION, direction=DEFAULT_LIGHT_DIRECTION,
            beam_width=DEFAULT_BEAM_WIDTH, beam_height=DEFAULT_BEAM_HEIGHT,
            nx=3, ny=3,
        )
        ray = projector.generate_rays(direction=DEFAULT_LIGHT_DIRECTION)

        def loss(refl):
            new_segs = tuple(
                s._replace(reflectance=refl) if isinstance(s, MirrorStackSeg) else s
                for s in route.segments
            )
            prepared = prepare_route(Route(segments=new_segs), 550e-6)
            tr = trace_rays(prepared, ray)
            return jnp.sum(tr.final_intensity)

        grads = jax.grad(loss)(stack.reflectance)
        assert grads.shape == stack.reflectance.shape
        assert jnp.any(grads != 0.0)


class TestBranchGeometry:
    """``build_combiner_branch_routes`` (helios.merit) ends each pupil branch
    at ``reflect(mirror_k)`` → exit face → pupil, with **no** mirror segment
    after the bounce. That's only physically correct if the reflected ray
    truly clears every other mirror before leaving the glass — otherwise the
    model silently drops the attenuation those crossings would cause. It's a
    geometric assumption nothing enforces, so lock it in: a change to mirror
    tilt, spacing, or chassis thickness that breaks it must fail here.
    """

    def test_reflected_ray_clears_all_other_mirrors(self):
        import numpy as np

        from apollo14.elements.partial_mirror import PartialMirror
        from apollo14.projector import FovGrid, Projector
        from apollo14.route import build_route, reflect
        from apollo14.trace import prepare_route, trace_rays
        from apollo14.units import deg, nm

        system = build_default_system()
        mirrors = [e for e in system.elements if isinstance(e, PartialMirror)]
        faces = [system.resolve(("chassis", name)) for name in
                 ("bottom", "top", "left", "right", "front", "back")]

        def geom(el):
            return (np.asarray(el.position), np.asarray(el.normal),
                    np.asarray(el._local_x), np.asarray(el._local_y),
                    np.asarray(el.half_extents))

        mirror_geom = [geom(m) for m in mirrors]
        face_geom = [geom(f) for f in faces]

        def rect_hit(origins, dirs, g):
            """(R,3) rays vs one rectangle → (t, in_bounds), both (R,).

            ``t`` is finite everywhere (near-parallel rays use a placeholder
            denominator, then get ``t = inf`` on return) so no inf/nan flows
            into the in-plane dot products.
            """
            pos, normal, local_x, local_y, half = g
            denom = dirs @ normal
            parallel = np.abs(denom) < 1e-9
            t = ((pos - origins) @ normal) / np.where(parallel, 1.0, denom)
            hit = origins + t[:, None] * dirs
            delta = hit - pos
            in_bounds = (~parallel & (t > 1e-9)
                         & (np.abs(delta @ local_x) <= half[0])
                         & (np.abs(delta @ local_y) <= half[1]))
            return np.where(parallel, np.inf, t), in_bounds

        projector = Projector.uniform(
            position=DEFAULT_LIGHT_POSITION, direction=DEFAULT_LIGHT_DIRECTION,
            beam_width=10.0 * mm, beam_height=2.0 * mm, nx=9, ny=3)
        fov = FovGrid(DEFAULT_LIGHT_DIRECTION, 8 * deg, 8 * deg, num_x=3, num_y=3)
        wavelength = 545 * nm

        for k in range(len(mirrors)):
            path = (["aperture", ("chassis", "back")]
                    + [f"mirror_{j}" for j in range(k)]
                    + [reflect(f"mirror_{k}")])
            route = prepare_route(build_route(system, path), wavelength)

            def trace_dir(direction):
                ray = projector.generate_rays(direction=direction,
                                              wavelength=wavelength)
                res = trace_rays(route, ray)
                return res.final_pos, res.final_dir, res.valids[:, -1]

            pos_all, dir_all, valid_all = jax.vmap(trace_dir)(
                fov.flat_directions)
            valid = np.asarray(valid_all).reshape(-1)
            bounce_pos = np.asarray(pos_all).reshape(-1, 3)[valid]
            bounce_dir = np.asarray(dir_all).reshape(-1, 3)[valid]
            assert len(bounce_pos) > 0, f"no rays reflect off mirror_{k}"

            # Distance at which the post-bounce ray leaves the glass = the
            # nearest chassis face it hits going forward.
            exit_t = np.full(len(bounce_pos), np.inf)
            for g in face_geom:
                t, in_bounds = rect_hit(bounce_pos, bounce_dir, g)
                exit_t = np.where(in_bounds & (t > 1e-4),
                                  np.minimum(exit_t, t), exit_t)

            for j in range(len(mirrors)):
                if j == k:
                    continue
                t, in_bounds = rect_hit(bounce_pos, bounce_dir, mirror_geom[j])
                crosses = in_bounds & (t > 1e-4) & (t < exit_t - 1e-4)
                assert not crosses.any(), (
                    f"reflected branch off mirror_{k} crosses mirror_{j} "
                    f"before exiting the glass for {int(crosses.sum())} rays "
                    f"— the branch route omits that attenuation")
