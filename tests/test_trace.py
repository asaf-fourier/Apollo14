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
    prepare_beam,
    trace_ray,
    trace_beam,
)


class TestBuildRoute:

    def test_main_path_shape(self):
        system = build_default_system()
        route = combiner_main_path(system)
        # aperture + back face + 6 mirrors + front face = 9 surfaces
        assert route.position.shape[0] == 9
        assert route.position.shape[1] == 3

    def test_mirrors_use_glass_medium(self):
        system = build_default_system()
        route = combiner_main_path(system)
        # Entries 2..7 are the mirrors; both n1/n2 should be the glass material.
        mirror_n1 = route.n1.n_values[2:8]
        mirror_n2 = route.n2.n_values[2:8]
        assert jnp.all(mirror_n1 == mirror_n2)
        # Mirror n is not air (single 1.0 sample).
        assert mirror_n1.shape[-1] > 1 or not jnp.all(mirror_n1 == 1.0)

    def test_mode_defaults_to_transmit(self):
        system = build_default_system()
        route = combiner_main_path(system)
        assert jnp.all(route.mode == TRANSMIT)

    def test_custom_path(self):
        system = build_default_system()
        route = build_route(system, [
            "aperture",
            ("chassis", "back"),
            "mirror_0",
            "mirror_1",
            ("chassis", "front"),
        ])
        assert route.position.shape[0] == 5


class TestPrepareBeam:

    def test_beam_scalar_n(self):
        system = build_default_system()
        route = combiner_main_path(system)
        beam = prepare_beam(route, DEFAULT_WAVELENGTH)
        # After resolution, n1/n2 are (N,) scalar arrays, not MaterialData.
        N = route.position.shape[0]
        assert beam.surfaces.n1.shape == (N,)
        assert beam.surfaces.n2.shape == (N,)
        assert float(beam.intensity) == 1.0

    def test_beam_custom_intensity(self):
        system = build_default_system()
        route = combiner_main_path(system)
        beam = prepare_beam(route, DEFAULT_WAVELENGTH, intensity=0.3)
        assert jnp.isclose(beam.intensity, 0.3)


class TestTraceRay:

    def test_single_ray_shapes(self):
        system = build_default_system()
        route = combiner_main_path(system)
        beam = prepare_beam(route, DEFAULT_WAVELENGTH)

        result = trace_ray(beam,
                           DEFAULT_LIGHT_POSITION,
                           DEFAULT_LIGHT_DIRECTION)

        N = route.position.shape[0]
        assert result.hits.shape == (N, 3)
        assert result.valids.shape == (N,)
        assert result.final_pos.shape == (3,)
        assert result.final_dir.shape == (3,)
        assert result.final_intensity.shape == ()

    def test_intensity_attenuates(self):
        system = build_default_system()
        route = combiner_main_path(system)
        beam = prepare_beam(route, DEFAULT_WAVELENGTH)
        result = trace_ray(beam,
                           DEFAULT_LIGHT_POSITION,
                           DEFAULT_LIGHT_DIRECTION)
        # Transmitting through 6 partial mirrors must reduce intensity.
        assert float(result.final_intensity) < 1.0
        assert float(result.final_intensity) > 0.0

    def test_initial_intensity_respected(self):
        system = build_default_system()
        route = combiner_main_path(system)
        beam_full = prepare_beam(route, DEFAULT_WAVELENGTH, intensity=1.0)
        beam_half = prepare_beam(route, DEFAULT_WAVELENGTH, intensity=0.5)
        r_full = trace_ray(beam_full,
                           DEFAULT_LIGHT_POSITION,
                           DEFAULT_LIGHT_DIRECTION)
        r_half = trace_ray(beam_half,
                           DEFAULT_LIGHT_POSITION,
                           DEFAULT_LIGHT_DIRECTION)
        assert jnp.isclose(r_half.final_intensity, 0.5 * r_full.final_intensity)

    def test_jit_compiles(self):
        system = build_default_system()
        route = combiner_main_path(system)

        def run(wavelength, o, d):
            beam = prepare_beam(route, wavelength)
            return trace_ray(beam, o, d).final_intensity

        jitted = jax.jit(run)
        val = jitted(DEFAULT_WAVELENGTH,
                     DEFAULT_LIGHT_POSITION,
                     DEFAULT_LIGHT_DIRECTION)
        assert jnp.isfinite(val)

    def test_grad_through_reflectance(self):
        system = build_default_system()
        route = combiner_main_path(system)

        def loss(reflectances):
            new_route = route._replace(reflectance=reflectances)
            beam = prepare_beam(new_route, DEFAULT_WAVELENGTH)
            r = trace_ray(beam,
                          DEFAULT_LIGHT_POSITION,
                          DEFAULT_LIGHT_DIRECTION)
            return r.final_intensity

        grads = jax.grad(loss)(route.reflectance)
        assert grads.shape == route.reflectance.shape
        assert jnp.any(grads != 0.0)


class TestTraceBeam:

    def test_beam_shapes(self):
        system = build_default_system()
        route = combiner_main_path(system)
        beam = prepare_beam(route, DEFAULT_WAVELENGTH)
        origins = DEFAULT_LIGHT_POSITION + jnp.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
        ])
        result = trace_beam(beam, origins, DEFAULT_LIGHT_DIRECTION)
        N = route.position.shape[0]
        assert result.hits.shape == (3, N, 3)
        assert result.final_intensity.shape == (3,)
