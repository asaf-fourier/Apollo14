"""Tests for the segmented single-path tracer."""

import jax
import jax.numpy as jnp
import pytest

from apollo14.combiner import (
    DEFAULT_LIGHT_DIRECTION,
    DEFAULT_LIGHT_POSITION,
    DEFAULT_WAVELENGTH,
    build_default_system,
)
from apollo14.elements.aperture import ApertureSeg
from apollo14.elements.glass_block import FaceSeg, GlassBlock, PreparedFaceSeg
from apollo14.elements.partial_mirror import (
    MirrorStackSeg,
    PartialMirror,
    PreparedMirrorStackSeg,
)
from apollo14.elements.pupil import PupilSeg
from apollo14.materials import agc_m074, air
from apollo14.ray import Ray
from apollo14.route import (
    TRANSMIT,
    Route,
    _group_mirror_runs,
    build_route,
    combiner_main_path,
)
from apollo14.spectral import SpectralTable
from apollo14.system import OpticalSystem
from apollo14.trace import prepare_route, trace, trace_rays
from apollo14.units import nm


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
        faces = [s for s in prepared.segments if isinstance(s, PreparedFaceSeg)]
        assert faces[0].n1.shape == ()
        assert faces[0].n2.shape == ()

    def test_mirror_reflectance_is_resolved(self):
        system = build_default_system()
        route = combiner_main_path(system)
        raw_stack = _mirror_stack(route)

        wavelength = 500.0 * nm
        prepared = prepare_route(route, wavelength)
        prepared_stack = next(
            seg for seg in prepared.segments
            if isinstance(seg, PreparedMirrorStackSeg)
        )

        expected = jax.vmap(
            lambda wavelengths, reflectance: jnp.interp(
                wavelength, wavelengths, reflectance)
        )(raw_stack.wavelengths, raw_stack.reflectance)
        assert prepared_stack.reflectance.shape == (6,)
        assert jnp.allclose(prepared_stack.reflectance, expected)

    def test_dynamic_wavelength_jit_resolves_mirrors(self):
        system = build_default_system()
        route = combiner_main_path(system)
        stack = _mirror_stack(route)
        spectral_reflectance = jnp.broadcast_to(
            jnp.array([0.01, 0.05, 0.20]), stack.reflectance.shape)
        route = _replace_mirror_reflectance(route, spectral_reflectance)

        @jax.jit
        def resolved_reflectance(wavelength):
            prepared = prepare_route(route, wavelength)
            prepared_stack = next(
                seg for seg in prepared.segments
                if isinstance(seg, PreparedMirrorStackSeg)
            )
            return prepared_stack.reflectance

        blue = resolved_reflectance(460.0 * nm)
        red = resolved_reflectance(630.0 * nm)
        assert jnp.allclose(blue, 0.01)
        assert jnp.allclose(red, 0.20)


class TestTraceRay:

    def test_ar_coating_attenuates_each_glass_crossing(self):
        chassis = GlassBlock.create_chassis(
            name="coated", x=10.0, y=10.0, z=2.0, material=agc_m074,
            coating_reflectance=SpectralTable.constant(
                0.005, jnp.array([400.0, 700.0]) * nm))
        system = OpticalSystem(env_material=air)
        system.add(chassis)
        route = build_route(system, [("coated", "bottom"),
                                     ("coated", "top")])
        prepared = prepare_route(route, 550.0 * nm)
        faces = [seg for seg in prepared.segments
                 if isinstance(seg, PreparedFaceSeg)]
        assert all(seg.coating_reflectance.shape == () for seg in faces)

        ray = Ray(pos=jnp.array([0.0, 0.0, -2.0]),
                  dir=jnp.array([0.0, 0.0, 1.0]),
                  intensity=jnp.asarray(1.0))
        result = trace(prepared, ray)
        assert jnp.isclose(result.final_intensity, 0.995 ** 2, atol=1e-6)

    def test_unprepared_route_is_rejected(self):
        system = build_default_system()
        with pytest.raises(TypeError, match="unprepared wavelength-dependent"):
            trace(combiner_main_path(system), _default_ray())

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


class TestTransmitMissSurvives:

    def test_miss_survives_to_next_mirror(self):
        """A ray that geometrically misses mirror_0 but hits mirror_1 is
        attenuated only by mirror_1 — the upstream miss does not kill it."""
        # Two +Y mirrors on parallel planes. mirror_0 is offset in x so the
        # straight-down ray crosses its plane outside the rectangle; mirror_1
        # is centered on the ray, so the ray hits it.
        wavelengths = jnp.array([400.0, 550.0, 700.0]) * nm
        mirror_0 = PartialMirror(
            name="m0", position=jnp.array([5.0, 0.0, 0.0]),
            normal=jnp.array([0.0, 1.0, 0.0]), width=2.0, height=10.0,
            reflectance=0.5, wavelengths=wavelengths)
        mirror_1 = PartialMirror(
            name="m1", position=jnp.array([0.0, -1.0, 0.0]),
            normal=jnp.array([0.0, 1.0, 0.0]), width=2.0, height=10.0,
            reflectance=0.3, wavelengths=wavelengths)

        seg0, _ = mirror_0.build_segment(air, TRANSMIT)
        seg1, _ = mirror_1.build_segment(air, TRANSMIT)
        (stack,) = _group_mirror_runs([seg0, seg1])
        route = Route(segments=(stack,))

        ray = Ray(pos=jnp.array([0.0, 10.0, 0.0]),
                  dir=jnp.array([0.0, -1.0, 0.0]),
                  intensity=jnp.asarray(1.0))
        result = trace(prepare_route(route, 550.0 * nm), ray)

        # Only mirror_1 attenuates: 1 * (1 - 0.3) = 0.7. Not killed (0.0), and
        # not also hit by mirror_0 (which would give 0.5 * 0.7 = 0.35).
        assert abs(float(result.final_intensity) - 0.7) < 1e-5
        assert not bool(result.valids[0])   # mirror_0 missed
        assert bool(result.valids[1])       # mirror_1 hit


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
