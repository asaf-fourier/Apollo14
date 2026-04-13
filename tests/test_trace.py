"""Tests for the generic single-path tracer."""

import jax
import jax.numpy as jnp

from apollo14.combiner import (
    build_default_system,
    DEFAULT_LIGHT_POSITION,
    DEFAULT_LIGHT_DIRECTION,
    DEFAULT_WAVELENGTH,
)
from apollo14.surface import TRANSMIT, REFLECT
from apollo14.trace import (
    build_route,
    combiner_main_path,
    trace_ray,
    trace_beam,
)


class TestBuildRoute:

    def test_main_path_shape(self):
        system = build_default_system()
        route = combiner_main_path(system)
        # aperture + back face + 6 mirrors + front face = 9 surfaces
        assert route.surfaces.position.shape[0] == 9
        assert route.surfaces.position.shape[1] == 3

    def test_mirrors_use_glass_medium(self):
        system = build_default_system()
        route = combiner_main_path(system)
        # Entries 2..7 are the mirrors; both n1/n2 should be the glass material.
        mirror_n1 = route.surfaces.n1.n_values[2:8]
        mirror_n2 = route.surfaces.n2.n_values[2:8]
        assert jnp.all(mirror_n1 == mirror_n2)
        # Mirror n is not air (single 1.0 sample).
        assert mirror_n1.shape[-1] > 1 or not jnp.all(mirror_n1 == 1.0)

    def test_mode_defaults_to_transmit(self):
        system = build_default_system()
        route = combiner_main_path(system)
        assert jnp.all(route.surfaces.mode == TRANSMIT)

    def test_custom_path(self):
        system = build_default_system()
        route = build_route(system, [
            "aperture",
            ("chassis", "back"),
            "mirror_0",
            "mirror_1",
            ("chassis", "front"),
        ])
        assert route.surfaces.position.shape[0] == 5


class TestTraceRay:

    def test_single_ray_shapes(self):
        system = build_default_system()
        route = combiner_main_path(system)

        result = trace_ray(route,
                           DEFAULT_LIGHT_POSITION,
                           DEFAULT_LIGHT_DIRECTION,
                           DEFAULT_WAVELENGTH)

        N = route.surfaces.position.shape[0]
        assert result.hits.shape == (N, 3)
        assert result.valids.shape == (N,)
        assert result.final_pos.shape == (3,)
        assert result.final_dir.shape == (3,)
        assert result.final_intensity.shape == ()

    def test_intensity_attenuates(self):
        system = build_default_system()
        route = combiner_main_path(system)
        result = trace_ray(route,
                           DEFAULT_LIGHT_POSITION,
                           DEFAULT_LIGHT_DIRECTION,
                           DEFAULT_WAVELENGTH)
        # Transmitting through 6 partial mirrors must reduce intensity.
        assert float(result.final_intensity) < 1.0
        assert float(result.final_intensity) > 0.0

    def test_jit_compiles(self):
        system = build_default_system()
        route = combiner_main_path(system)
        jitted = jax.jit(
            lambda o, d, w: trace_ray(route, o, d, w).final_intensity)
        val = jitted(DEFAULT_LIGHT_POSITION,
                     DEFAULT_LIGHT_DIRECTION,
                     DEFAULT_WAVELENGTH)
        assert jnp.isfinite(val)

    def test_grad_through_reflectance(self):
        system = build_default_system()
        route = combiner_main_path(system)

        def loss(reflectances):
            new_surfaces = route.surfaces._replace(reflectance=reflectances)
            new_route = route._replace(surfaces=new_surfaces)
            r = trace_ray(new_route,
                          DEFAULT_LIGHT_POSITION,
                          DEFAULT_LIGHT_DIRECTION,
                          DEFAULT_WAVELENGTH)
            return r.final_intensity

        grads = jax.grad(loss)(route.surfaces.reflectance)
        assert grads.shape == route.surfaces.reflectance.shape
        assert jnp.any(grads != 0.0)


class TestTraceBeam:

    def test_beam_shapes(self):
        system = build_default_system()
        route = combiner_main_path(system)
        origins = DEFAULT_LIGHT_POSITION + jnp.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
        ])
        result = trace_beam(route, origins,
                            DEFAULT_LIGHT_DIRECTION,
                            DEFAULT_WAVELENGTH)
        N = route.surfaces.position.shape[0]
        assert result.hits.shape == (3, N, 3)
        assert result.final_intensity.shape == (3,)
