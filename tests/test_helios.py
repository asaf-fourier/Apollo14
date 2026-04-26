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
            tr = trace_rays(prepared, ray, wavelength=525e-6)
            return jnp.sum(tr.final_intensity)

        grads = jax.grad(loss)(stack.reflectance)
        assert grads.shape == stack.reflectance.shape
        assert jnp.any(grads != 0.0)
