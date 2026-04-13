"""Tests for the Talos-specific route and tracer.

Verifies TalosRoute construction, talos trace functions, and
numerical equivalence with the generic tracer.
"""

import jax
import jax.numpy as jnp
import pytest

from apollo14.combiner import (
    build_default_system,
    DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, DEFAULT_WAVELENGTH,
)
from apollo14.route import display_route as generic_display_route
from apollo14.trace import trace_ray as generic_trace_ray
from apollo14.trace import trace_beam as generic_trace_beam
from apollo14.trace import TraceResult
from apollo14.talos_route import TalosRoute, display_route as talos_display_route
from apollo14.talos_trace import trace_ray, trace_beam, trace_batch
from apollo14.elements.surface import MirrorState
from apollo14.elements.refracting_surface import RefractState
from apollo14.elements.aperture import ApertureState
from apollo14.elements.pupil import DetectorState
from apollo14.units import nm


WAVELENGTH = 550.0 * nm


@pytest.fixture
def system():
    return build_default_system()


@pytest.fixture
def talos_route(system):
    return talos_display_route(system, WAVELENGTH)


@pytest.fixture
def generic_route(system):
    return generic_display_route(system, WAVELENGTH)


@pytest.fixture
def on_axis():
    return DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION


# ── TalosRoute structure ────────────────────────────────────────────────────

class TestTalosRouteStructure:

    def test_fields(self):
        assert 'aperture' in TalosRoute._fields
        assert 'has_aperture' in TalosRoute._fields
        assert 'entry_face' in TalosRoute._fields
        assert 'mirrors' in TalosRoute._fields
        assert 'exit_face' in TalosRoute._fields
        assert 'target' in TalosRoute._fields
        assert 'n_glass' in TalosRoute._fields

    def test_aperture_type(self, talos_route):
        assert isinstance(talos_route.aperture, ApertureState)

    def test_entry_face_type(self, talos_route):
        assert isinstance(talos_route.entry_face, RefractState)

    def test_mirrors_type(self, talos_route):
        assert isinstance(talos_route.mirrors, MirrorState)

    def test_exit_face_type(self, talos_route):
        assert isinstance(talos_route.exit_face, RefractState)

    def test_target_type(self, talos_route):
        assert isinstance(talos_route.target, DetectorState)

    def test_mirror_count(self, talos_route):
        assert talos_route.mirrors.position.shape[0] == 6

    def test_exit_face_tiled(self, talos_route):
        """Exit face is tiled to match mirror count."""
        M = talos_route.mirrors.position.shape[0]
        assert talos_route.exit_face.position.shape[0] == M

    def test_target_tiled(self, talos_route):
        M = talos_route.mirrors.position.shape[0]
        assert talos_route.target.position.shape[0] == M

    def test_n_glass_positive(self, talos_route):
        assert float(talos_route.n_glass) > 1.0

    def test_has_aperture(self, talos_route):
        assert bool(talos_route.has_aperture)

    def test_is_pytree(self, talos_route):
        leaves, treedef = jax.tree.flatten(talos_route)
        route2 = jax.tree.unflatten(treedef, leaves)
        assert isinstance(route2, TalosRoute)


# ── Talos trace_ray ──────────────────────────────────────────────────────────

class TestTalosTraceRay:

    def test_result_type(self, talos_route, on_axis):
        origin, direction = on_axis
        tr = trace_ray(origin, direction, talos_route)
        assert isinstance(tr, TraceResult)

    def test_result_shapes(self, talos_route, on_axis):
        origin, direction = on_axis
        M = talos_route.mirrors.position.shape[0]
        tr = trace_ray(origin, direction, talos_route)
        assert tr.pupil_points.shape == (M, 3)
        assert tr.intensities.shape == (M,)
        assert tr.valid.shape == (M,)
        assert tr.main_hits.shape == (M, 3)
        assert tr.branch_hits.shape == (M, 2, 3)  # exit + pupil

    def test_on_axis_has_hits(self, talos_route, on_axis):
        origin, direction = on_axis
        tr = trace_ray(origin, direction, talos_route)
        assert jnp.any(tr.valid)
        assert float(tr.total_intensity) > 0

    def test_off_axis_no_hits(self, talos_route):
        origin = jnp.array([100.0, 100.0, 100.0])
        direction = jnp.array([0.0, -1.0, 0.0])
        tr = trace_ray(origin, direction, talos_route)
        assert float(tr.total_intensity) == 0.0

    def test_all_three_colors(self, talos_route, on_axis):
        origin, direction = on_axis
        for ci in range(3):
            tr = trace_ray(origin, direction, talos_route, color_idx=ci)
            assert float(tr.total_intensity) > 0


# ── Talos trace_beam ─────────────────────────────────────────────────────────

class TestTalosTraceBeam:

    def test_result_shapes(self, talos_route, on_axis):
        origin, direction = on_axis
        origins = jnp.stack([origin, origin + jnp.array([1.0, 0, 0]),
                             origin - jnp.array([1.0, 0, 0])])
        tr = trace_beam(origins, direction, talos_route)
        M = talos_route.mirrors.position.shape[0]
        assert tr.intensities.shape == (3, M)

    def test_matches_individual_rays(self, talos_route, on_axis):
        origin, direction = on_axis
        origins = jnp.stack([origin, origin + jnp.array([0.5, 0, 0])])
        beam_tr = trace_beam(origins, direction, talos_route)
        for i in range(origins.shape[0]):
            single_tr = trace_ray(origins[i], direction, talos_route)
            assert jnp.allclose(beam_tr.intensities[i], single_tr.intensities,
                                atol=1e-5)


# ── Talos trace_batch ────────────────────────────────────────────────────────

class TestTalosTraceBatch:

    def test_result_shapes(self, talos_route, on_axis):
        origin, direction = on_axis
        origins = jnp.stack([origin, origin])
        dirs = jnp.stack([direction, direction + jnp.array([0.1, 0, 0])])
        dirs = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)
        tr = trace_batch(origins, dirs, talos_route)
        M = talos_route.mirrors.position.shape[0]
        assert tr.intensities.shape == (2, M)


# ── Numerical equivalence: Talos vs Generic ──────────────────────────────────

class TestTalosMatchesGeneric:
    """Core equivalence: both tracers produce identical physics results."""

    def test_intensities_match(self, talos_route, generic_route, on_axis):
        origin, direction = on_axis
        tr_talos = trace_ray(origin, direction, talos_route)
        tr_generic = generic_trace_ray(origin, direction, generic_route)
        assert jnp.allclose(tr_talos.intensities, tr_generic.intensities, atol=1e-5)

    def test_valid_mask_match(self, talos_route, generic_route, on_axis):
        origin, direction = on_axis
        tr_talos = trace_ray(origin, direction, talos_route)
        tr_generic = generic_trace_ray(origin, direction, generic_route)
        assert jnp.array_equal(tr_talos.valid, tr_generic.valid)

    def test_pupil_points_match(self, talos_route, generic_route, on_axis):
        origin, direction = on_axis
        tr_talos = trace_ray(origin, direction, talos_route)
        tr_generic = generic_trace_ray(origin, direction, generic_route)
        for i in range(tr_talos.valid.shape[0]):
            if tr_talos.valid[i] and tr_generic.valid[i]:
                assert jnp.allclose(tr_talos.pupil_points[i],
                                    tr_generic.pupil_points[i], atol=1e-4)

    def test_main_hits_match(self, talos_route, generic_route, on_axis):
        origin, direction = on_axis
        tr_talos = trace_ray(origin, direction, talos_route)
        tr_generic = generic_trace_ray(origin, direction, generic_route)
        for i in range(tr_talos.valid.shape[0]):
            if tr_talos.valid[i]:
                assert jnp.allclose(tr_talos.main_hits[i],
                                    tr_generic.main_hits[i], atol=1e-4)

    def test_total_intensity_match(self, talos_route, generic_route, on_axis):
        origin, direction = on_axis
        tr_talos = trace_ray(origin, direction, talos_route)
        tr_generic = generic_trace_ray(origin, direction, generic_route)
        assert jnp.allclose(tr_talos.total_intensity,
                            tr_generic.total_intensity, atol=1e-5)

    def test_all_colors_match(self, talos_route, generic_route, on_axis):
        origin, direction = on_axis
        for ci in range(3):
            tr_talos = trace_ray(origin, direction, talos_route, color_idx=ci)
            tr_generic = generic_trace_ray(origin, direction, generic_route, color_idx=ci)
            assert jnp.allclose(tr_talos.total_intensity,
                                tr_generic.total_intensity, atol=1e-5)

    def test_beam_match(self, talos_route, generic_route, on_axis):
        origin, direction = on_axis
        origins = jnp.stack([
            origin,
            origin + jnp.array([1.0, 0, 0]),
            origin - jnp.array([1.0, 0, 0]),
        ])
        tr_talos = trace_beam(origins, direction, talos_route)
        tr_generic = generic_trace_beam(origins, direction, generic_route)
        assert jnp.allclose(tr_talos.intensities, tr_generic.intensities, atol=1e-5)
        assert jnp.array_equal(tr_talos.valid, tr_generic.valid)


# ── Talos differentiability ──────────────────────────────────────────────────

class TestTalosDifferentiability:

    def test_grad_wrt_reflectances(self, talos_route, on_axis):
        origin, direction = on_axis

        def loss(reflectances):
            new_mirrors = talos_route.mirrors._replace(reflectance=reflectances)
            r = talos_route._replace(mirrors=new_mirrors)
            tr = trace_ray(origin, direction, r, color_idx=0)
            return tr.total_intensity

        grads = jax.grad(loss)(talos_route.mirrors.reflectance)
        assert grads.shape == talos_route.mirrors.reflectance.shape
        assert jnp.any(grads != 0)
