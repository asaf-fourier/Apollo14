"""Smoke tests for the helios optimization layer."""

import jax
import jax.numpy as jnp

from apollo14.combiner import (
    build_default_system,
    DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
    DEFAULT_BEAM_WIDTH, DEFAULT_BEAM_HEIGHT,
    DEFAULT_X_FOV, DEFAULT_Y_FOV,
)
from apollo14.elements.pupil import RectangularPupil
from apollo14.projector import Projector
from apollo14.units import mm

from helios import (
    MeritConfig, evaluate_merit,
    EyeboxConfig, EyeboxAreaConfig,
    compute_eyebox_response, eyebox_merit, eyebox_area_merit,
    eyebox_grid_points, cell_grid_from_cell_size,
)
from helios.merit import build_combiner_pupil_routes, DEFAULT_WAVELENGTHS


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


class TestMerit:

    def test_build_routes_shape(self):
        system = build_default_system()
        routes = build_combiner_pupil_routes(system, [550e-6])
        assert len(routes) == 1
        assert len(routes[0]) == 6  # six mirrors → six branches

    def test_evaluate_merit_shapes(self):
        _, pupil, routes, projector = _make_fixtures()
        cfg = MeritConfig(pupil_nx=2, pupil_ny=2, angle_nx=3, angle_ny=3)
        mse, sim, tgt = evaluate_merit(
            routes, projector, pupil.position, pupil.normal, 3.0 * mm,
            DEFAULT_X_FOV, DEFAULT_Y_FOV, cfg,
        )
        assert sim.shape == (2, 2, 3, 3, 3)
        assert tgt.shape == sim.shape
        assert jnp.isfinite(mse)


class TestEyebox:

    def test_compute_response_shape(self):
        _, pupil, routes, _ = _make_fixtures()
        cfg = EyeboxConfig(n_fov_x=3, n_fov_y=3, n_beam_x=3, n_beam_y=3)
        pts = eyebox_grid_points(pupil.position, pupil.normal, 3.0 * mm,
                                 nx=3, ny=3)
        resp, dirs = compute_eyebox_response(
            routes, DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
            DEFAULT_BEAM_WIDTH, DEFAULT_BEAM_HEIGHT,
            DEFAULT_X_FOV, DEFAULT_Y_FOV, pts, cfg,
        )
        assert resp.shape == (9, 9, 3)  # (S, A, 3)
        assert dirs.shape == (9, 3)

    def test_eyebox_merit_scalar(self):
        _, pupil, routes, _ = _make_fixtures()
        cfg = EyeboxConfig(n_fov_x=3, n_fov_y=3, n_beam_x=3, n_beam_y=3)
        pts = eyebox_grid_points(pupil.position, pupil.normal, 3.0 * mm,
                                 nx=3, ny=3)
        resp, _ = compute_eyebox_response(
            routes, DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
            DEFAULT_BEAM_WIDTH, DEFAULT_BEAM_HEIGHT,
            DEFAULT_X_FOV, DEFAULT_Y_FOV, pts, cfg,
        )
        loss = eyebox_merit(resp, cfg)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_eyebox_area_merit_scalar(self):
        _, pupil, routes, _ = _make_fixtures()
        cfg = EyeboxConfig(n_fov_x=3, n_fov_y=3, n_beam_x=3, n_beam_y=3)
        pts, nx, ny = cell_grid_from_cell_size(
            pupil.position, pupil.normal, 6.0 * mm, 6.0 * mm, 2.0 * mm)
        resp, _ = compute_eyebox_response(
            routes, DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
            DEFAULT_BEAM_WIDTH, DEFAULT_BEAM_HEIGHT,
            DEFAULT_X_FOV, DEFAULT_Y_FOV, pts, cfg,
        )
        loss = eyebox_area_merit(resp, EyeboxAreaConfig())
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_grad_through_reflectance(self):
        """Gradients must flow through prepare_route → trace_rays."""
        from apollo14.trace import prepare_route, trace_rays
        from apollo14.route import combiner_main_path, Route
        from apollo14.elements.partial_mirror import MirrorStackSeg

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
            tr = trace_rays(prepared, ray, color_idx=1)
            return jnp.sum(tr.final_intensity)

        grads = jax.grad(loss)(stack.reflectance)
        assert grads.shape == stack.reflectance.shape
        assert jnp.any(grads != 0.0)
