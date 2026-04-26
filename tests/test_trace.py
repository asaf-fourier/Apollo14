"""Tests for the segmented single-path tracer."""

import jax
import jax.numpy as jnp

from apollo14.combiner import (
    DEFAULT_LIGHT_DIRECTION,
    DEFAULT_LIGHT_POSITION,
    DEFAULT_WAVELENGTH,
    build_default_system,
)
from apollo14.elements.aperture import ApertureSeg
from apollo14.elements.glass_block import FaceSeg
from apollo14.elements.partial_mirror import MirrorStackSeg
from apollo14.elements.pupil import PupilSeg
from apollo14.ray import Ray
from apollo14.route import Route, build_route, combiner_main_path
from apollo14.trace import prepare_route, trace, trace_rays


def _mirror_stack(route):
    return next(s for s in route.segments if isinstance(s, MirrorStackSeg))


def _replace_mirror_reflectance(route, new_refl):
    new_segs = tuple(
        s._replace(reflectance=new_refl) if isinstance(s, MirrorStackSeg) else s
        for s in route.segments
    )
    return Route(segments=new_segs)


class TestBuildRoute:

    def test_main_path_segments(self):
        system = build_default_system()
        route = combiner_main_path(system)
        kinds = [type(s).__name__ for s in route.segments]
        # aperture + back face + 1 mirror stack (6 mirrors fused) + front face
        assert kinds == ["ApertureSeg", "FaceSeg", "MirrorStackSeg", "FaceSeg"]

    def test_mirror_stack_length(self):
        system = build_default_system()
        route = combiner_main_path(system)
        stack = _mirror_stack(route)
        assert stack.position.shape == (6, 3)
        assert stack.reflectance.shape == (6, 3)

    def test_faces_use_glass(self):
        system = build_default_system()
        route = combiner_main_path(system)
        faces = [s for s in route.segments if isinstance(s, FaceSeg)]
        assert len(faces) == 2
        # Back face enters glass: n1=air(1), n2=glass(>1).
        back = faces[0]
        # Before prepare_route, n1/n2 are MaterialData (have n_values arrays).
        assert back.n2.n_values.shape[-1] > 1

    def test_custom_path(self):
        system = build_default_system()
        route = build_route(system, [
            "aperture",
            ("chassis", "back"),
            "mirror_0",
            "mirror_1",
            ("chassis", "front"),
        ])
        # aperture + face + (2-mirror stack) + face
        kinds = [type(s).__name__ for s in route.segments]
        assert kinds == ["ApertureSeg", "FaceSeg", "MirrorStackSeg", "FaceSeg"]
        stack = _mirror_stack(route)
        assert stack.position.shape[0] == 2


def _default_ray(intensity=1.0):
    return Ray(
        pos=jnp.asarray(DEFAULT_LIGHT_POSITION, dtype=jnp.float32),
        dir=jnp.asarray(DEFAULT_LIGHT_DIRECTION, dtype=jnp.float32),
        intensity=jnp.asarray(intensity, dtype=jnp.float32),
    )


class TestPrepareRoute:

    def test_face_scalar_n(self):
        system = build_default_system()
        route = combiner_main_path(system)
        prepared = prepare_route(route, DEFAULT_WAVELENGTH)
        faces = [s for s in prepared.segments if isinstance(s, FaceSeg)]
        assert faces[0].n1.shape == ()
        assert faces[0].n2.shape == ()


class TestTraceRay:

    def test_single_ray_shapes(self):
        system = build_default_system()
        route = prepare_route(combiner_main_path(system), DEFAULT_WAVELENGTH)
        result = trace(route, _default_ray())

        # aperture(1) + face(1) + mirrors(6) + face(1) = 9 steps
        assert result.hits.shape == (9, 3)
        assert result.valids.shape == (9,)
        assert result.final_pos.shape == (3,)
        assert result.final_dir.shape == (3,)
        assert result.final_intensity.shape == ()

    def test_intensity_attenuates(self):
        system = build_default_system()
        route = prepare_route(combiner_main_path(system), DEFAULT_WAVELENGTH)
        result = trace(route, _default_ray())
        assert float(result.final_intensity) < 1.0
        assert float(result.final_intensity) > 0.0

    def test_initial_intensity_respected(self):
        system = build_default_system()
        route = prepare_route(combiner_main_path(system), DEFAULT_WAVELENGTH)
        r_full = trace(route, _default_ray(intensity=1.0))
        r_half = trace(route, _default_ray(intensity=0.5))
        assert jnp.isclose(r_half.final_intensity, 0.5 * r_full.final_intensity)

    def test_jit_compiles(self):
        system = build_default_system()
        raw_route = combiner_main_path(system)

        def run(wavelength, o, d):
            route = prepare_route(raw_route, wavelength)
            ray = Ray(pos=o, dir=d, intensity=jnp.asarray(1.0, jnp.float32))
            return trace(route, ray).final_intensity

        jitted = jax.jit(run)
        val = jitted(DEFAULT_WAVELENGTH,
                     DEFAULT_LIGHT_POSITION,
                     DEFAULT_LIGHT_DIRECTION)
        assert jnp.isfinite(val)

    def test_grad_through_reflectance(self):
        system = build_default_system()
        route = combiner_main_path(system)
        stack = _mirror_stack(route)

        def loss(reflectances):
            new_route = _replace_mirror_reflectance(route, reflectances)
            prepared = prepare_route(new_route, DEFAULT_WAVELENGTH)
            return trace(prepared, _default_ray()).final_intensity

        grads = jax.grad(loss)(stack.reflectance)
        assert grads.shape == stack.reflectance.shape
        assert jnp.any(grads != 0.0)


class TestTraceRays:

    def test_batch_shapes(self):
        system = build_default_system()
        route = prepare_route(combiner_main_path(system), DEFAULT_WAVELENGTH)
        origins = DEFAULT_LIGHT_POSITION + jnp.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
        ])
        ray_batch = Ray(
            pos=origins,
            dir=DEFAULT_LIGHT_DIRECTION,
            intensity=jnp.ones(3),
        )
        result = trace_rays(route, ray_batch)
        assert result.hits.shape == (3, 9, 3)
        assert result.final_intensity.shape == (3,)
