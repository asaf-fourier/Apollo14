"""Tests for the generic tracer — trace_ray, trace_beam, trace_batch, TraceResult."""

import jax
import jax.numpy as jnp
import pytest

from apollo14.combiner import (
    build_default_system,
    DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, DEFAULT_WAVELENGTH,
)
from apollo14.route import Route, display_route
from apollo14.trace import trace_ray, trace_beam, trace_batch, TraceResult
from apollo14.projector import Projector
from apollo14.units import nm, mm


WAVELENGTH = 550.0 * nm


@pytest.fixture
def system():
    return build_default_system()


@pytest.fixture
def route(system):
    return display_route(system, WAVELENGTH)


@pytest.fixture
def on_axis():
    return jnp.array([7.0, 31.0, 1.0]), jnp.array([0.0, -1.0, 0.0])


# ── TraceResult ──────────────────────────────────────────────────────────────

class TestTraceResult:

    def test_is_namedtuple(self):
        assert hasattr(TraceResult, '_fields')
        assert 'pupil_points' in TraceResult._fields

    def test_total_intensity(self):
        tr = TraceResult(
            pupil_points=jnp.zeros((3, 3)),
            intensities=jnp.array([0.1, 0.2, 0.3]),
            valid=jnp.array([True, False, True]),
            main_hits=jnp.zeros((3, 3)),
            branch_hits=jnp.zeros((3, 2, 3)),
        )
        # Only valid intensities counted: 0.1 + 0.3 = 0.4
        assert jnp.allclose(tr.total_intensity, 0.4)

    def test_total_intensity_batched(self):
        tr = TraceResult(
            pupil_points=jnp.zeros((2, 3, 3)),
            intensities=jnp.ones((2, 3)) * 0.1,
            valid=jnp.ones((2, 3), dtype=bool),
            main_hits=jnp.zeros((2, 3, 3)),
            branch_hits=jnp.zeros((2, 3, 2, 3)),
        )
        # Each ray: 0.1 * 3 = 0.3
        assert tr.total_intensity.shape == (2,)
        assert jnp.allclose(tr.total_intensity, 0.3)


# ── trace_ray ────────────────────────────────────────────────────────────────

class TestTraceRay:

    def test_result_shapes(self, route, on_axis):
        origin, direction = on_axis
        tr = trace_ray(origin, direction, route)
        M = route.mirrors.position.shape[0]
        assert tr.pupil_points.shape == (M, 3)
        assert tr.intensities.shape == (M,)
        assert tr.valid.shape == (M,)
        assert tr.main_hits.shape == (M, 3)
        B = route.branch.position.shape[0]
        assert tr.branch_hits.shape == (M, B, 3)

    def test_on_axis_has_hits(self, route, on_axis):
        origin, direction = on_axis
        tr = trace_ray(origin, direction, route)
        assert jnp.any(tr.valid)
        assert float(tr.total_intensity) > 0

    def test_off_axis_no_hits(self, route):
        """Ray far from system should miss everything."""
        origin = jnp.array([100.0, 100.0, 100.0])
        direction = jnp.array([0.0, -1.0, 0.0])
        tr = trace_ray(origin, direction, route)
        assert float(tr.total_intensity) == 0.0

    def test_color_idx_selects_channel(self, route, on_axis):
        origin, direction = on_axis
        results = []
        for ci in range(3):
            tr = trace_ray(origin, direction, route, color_idx=ci)
            results.append(float(tr.total_intensity))
        # All should have intensity but may differ due to per-color reflectance
        assert all(r > 0 for r in results)

    def test_intensities_are_reflected(self, route, on_axis):
        """Intensities in TraceResult are the reflected portion (not transmitted)."""
        origin, direction = on_axis
        tr = trace_ray(origin, direction, route, color_idx=0)
        r = float(route.mirrors.reflectance[0, 0])
        # First mirror's reflected intensity should be r * (input intensity)
        # Input is 1.0 (after preamble, approximately — depends on aperture)
        if tr.valid[0]:
            assert float(tr.intensities[0]) > 0
            assert float(tr.intensities[0]) < 1.0  # fraction, not full

    def test_mirror_hits_on_mirror_planes(self, route, on_axis):
        """Main hits should lie on the mirror planes."""
        origin, direction = on_axis
        tr = trace_ray(origin, direction, route)
        for i in range(route.mirrors.position.shape[0]):
            if tr.valid[i]:
                # Hit point projected onto mirror normal should match mirror position
                delta = tr.main_hits[i] - route.mirrors.position[i]
                dist_to_plane = jnp.abs(jnp.dot(delta, route.mirrors.normal[i]))
                assert float(dist_to_plane) < 0.01

    def test_intensity_decreases_along_path(self, route, on_axis):
        """Later mirrors receive less input, so reflected intensity decreases."""
        origin, direction = on_axis
        # Use non-uniform reflectance so the effect is clear
        new_refl = route.mirrors.reflectance.at[0].set(jnp.array([0.10, 0.10, 0.10]))
        new_route = route._replace(mirrors=route.mirrors._replace(reflectance=new_refl))
        tr = trace_ray(origin, direction, new_route)
        valid_ints = [float(tr.intensities[i])
                      for i in range(tr.valid.shape[0]) if tr.valid[i]]
        if len(valid_ints) >= 2:
            # First mirror reflects more (higher reflectance + full input)
            assert valid_ints[0] > valid_ints[1]


# ── trace_beam ───────────────────────────────────────────────────────────────

class TestTraceBeam:

    @pytest.fixture
    def beam_origins(self):
        return jnp.array([
            [6.0, 31.0, 1.0],
            [7.0, 31.0, 1.0],
            [8.0, 31.0, 1.0],
            [7.0, 31.0, 0.5],
            [7.0, 31.0, 1.5],
        ])

    def test_result_shapes(self, route, beam_origins):
        direction = jnp.array([0.0, -1.0, 0.0])
        tr = trace_beam(beam_origins, direction, route)
        N = beam_origins.shape[0]
        M = route.mirrors.position.shape[0]
        assert tr.pupil_points.shape == (N, M, 3)
        assert tr.intensities.shape == (N, M)
        assert tr.valid.shape == (N, M)

    def test_total_intensity_shape(self, route, beam_origins):
        direction = jnp.array([0.0, -1.0, 0.0])
        tr = trace_beam(beam_origins, direction, route)
        assert tr.total_intensity.shape == (beam_origins.shape[0],)

    def test_center_ray_has_intensity(self, route, beam_origins):
        direction = jnp.array([0.0, -1.0, 0.0])
        tr = trace_beam(beam_origins, direction, route)
        # Center ray (index 1) should have the most intensity
        assert float(tr.total_intensity[1]) > 0

    def test_shared_direction(self, route):
        """All rays in a beam share the same direction — results should be consistent."""
        origins = jnp.array([
            [7.0, 31.0, 1.0],
            [7.0, 31.0, 1.0],  # duplicate
        ])
        direction = jnp.array([0.0, -1.0, 0.0])
        tr = trace_beam(origins, direction, route)
        assert jnp.allclose(tr.intensities[0], tr.intensities[1])

    def test_matches_individual_rays(self, route, beam_origins):
        """trace_beam should produce same results as individual trace_ray calls."""
        direction = jnp.array([0.0, -1.0, 0.0])
        beam_tr = trace_beam(beam_origins, direction, route)
        for i in range(beam_origins.shape[0]):
            single_tr = trace_ray(beam_origins[i], direction, route)
            assert jnp.allclose(beam_tr.intensities[i], single_tr.intensities,
                                atol=1e-5)


# ── trace_batch ──────────────────────────────────────────────────────────────

class TestTraceBatch:

    def test_result_shapes(self, route):
        origins = jnp.array([
            [7.0, 31.0, 1.0],
            [7.0, 31.0, 1.0],
        ])
        directions = jnp.array([
            [0.0, -1.0, 0.0],
            [0.1, -1.0, 0.0],
        ])
        directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)
        tr = trace_batch(origins, directions, route)
        M = route.mirrors.position.shape[0]
        assert tr.pupil_points.shape == (2, M, 3)

    def test_different_directions_give_different_results(self, route):
        origins = jnp.array([
            [7.0, 31.0, 1.0],
            [7.0, 31.0, 1.0],
        ])
        directions = jnp.array([
            [0.0, -1.0, 0.0],
            [0.3, -1.0, 0.0],
        ])
        directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)
        tr = trace_batch(origins, directions, route)
        # Different directions should produce different hit points
        assert not jnp.allclose(tr.main_hits[0], tr.main_hits[1])


# ── Differentiability ────────────────────────────────────────────────────────

class TestDifferentiability:

    def test_grad_wrt_reflectances(self, route, on_axis):
        origin, direction = on_axis

        def loss(reflectances):
            new_mirrors = route.mirrors._replace(reflectance=reflectances)
            r = route._replace(mirrors=new_mirrors)
            tr = trace_ray(origin, direction, r, color_idx=0)
            return tr.total_intensity

        grads = jax.grad(loss)(route.mirrors.reflectance)
        assert grads.shape == route.mirrors.reflectance.shape
        assert jnp.any(grads != 0)

    def test_grad_wrt_mirror_positions(self, route, on_axis):
        origin, direction = on_axis

        def loss(positions):
            new_mirrors = route.mirrors._replace(position=positions)
            r = route._replace(mirrors=new_mirrors)
            tr = trace_ray(origin, direction, r, color_idx=0)
            return tr.total_intensity

        grads = jax.grad(loss)(route.mirrors.position)
        assert grads.shape == route.mirrors.position.shape

    def test_grad_wrt_branch_position(self, route, on_axis):
        """Gradients flow through the branch path too."""
        origin, direction = on_axis

        def loss(branch_pos):
            new_branch = route.branch._replace(position=branch_pos)
            r = route._replace(branch=new_branch)
            tr = trace_ray(origin, direction, r, color_idx=0)
            return tr.total_intensity

        grads = jax.grad(loss)(route.branch.position)
        assert grads.shape == route.branch.position.shape

    def test_grad_beam(self, route):
        """Gradients flow through trace_beam."""
        origins = jnp.array([
            [7.0, 31.0, 1.0],
            [7.5, 31.0, 1.0],
        ])
        direction = jnp.array([0.0, -1.0, 0.0])

        def loss(reflectances):
            new_mirrors = route.mirrors._replace(reflectance=reflectances)
            r = route._replace(mirrors=new_mirrors)
            tr = trace_beam(origins, direction, r, color_idx=0)
            return tr.total_intensity.sum()

        grads = jax.grad(loss)(route.mirrors.reflectance)
        assert jnp.any(grads != 0)


# ── Optimization pattern ─────────────────────────────────────────────────────

class TestOptimizationPattern:
    """Verify the _replace() optimization pattern works end to end."""

    def test_higher_reflectance_more_intensity(self, route, on_axis):
        origin, direction = on_axis
        tr_base = trace_ray(origin, direction, route, color_idx=0)
        base_intensity = float(tr_base.total_intensity)

        # Double reflectance
        new_refl = jnp.clip(route.mirrors.reflectance * 2, 0, 0.5)
        new_mirrors = route.mirrors._replace(reflectance=new_refl)
        new_route = route._replace(mirrors=new_mirrors)
        tr_high = trace_ray(origin, direction, new_route, color_idx=0)
        high_intensity = float(tr_high.total_intensity)

        assert high_intensity > base_intensity

    def test_zero_reflectance_zero_intensity(self, route, on_axis):
        origin, direction = on_axis
        new_refl = jnp.zeros_like(route.mirrors.reflectance)
        new_mirrors = route.mirrors._replace(reflectance=new_refl)
        new_route = route._replace(mirrors=new_mirrors)
        tr = trace_ray(origin, direction, new_route, color_idx=0)
        assert float(tr.total_intensity) == 0.0
