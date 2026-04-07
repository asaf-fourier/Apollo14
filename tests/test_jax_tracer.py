import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apollo14.combiner import build_default_system, DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, DEFAULT_WAVELENGTH
from apollo14.elements.glass_block import GlassBlock
from apollo14.jax_tracer import (
    trace_ray, trace_batch, trace_beam, params_from_system,
    trace_path, trace_combiner_ray, trace_combiner_beam, trace_combiner_batch,
    combiner_path_from_system, compensated_reflectances,
    partial, refract, target, stack_path,
    build_combiner_paths, CombinerParams, CombinerPath, PathStep,
    _box_entry, _box_exit, _plane_t,
    REFRACT, PARTIAL, TARGET,
)


@pytest.fixture
def default_setup():
    system = build_default_system()
    params = params_from_system(system, DEFAULT_WAVELENGTH)
    chassis = next(e for e in system.elements if isinstance(e, GlassBlock))
    n_glass = float(chassis.material.n(DEFAULT_WAVELENGTH))
    return params, n_glass


@pytest.fixture
def default_system():
    return build_default_system()


@pytest.fixture
def combiner_path(default_system):
    return combiner_path_from_system(default_system, DEFAULT_WAVELENGTH)


# ── Box intersection helpers ────────────────────────────────────────────────

class TestBoxHelpers:

    def test_box_entry_from_above(self):
        origin = jnp.array([0.5, 0.5, 5.0])
        direction = jnp.array([0.0, 0.0, -1.0])
        box_min = jnp.array([0.0, 0.0, 0.0])
        box_max = jnp.array([1.0, 1.0, 1.0])
        t, normal = _box_entry(origin, direction, box_min, box_max)
        assert float(t) == pytest.approx(4.0)
        assert jnp.allclose(normal, jnp.array([0.0, 0.0, 1.0]))

    def test_box_exit_going_down(self):
        origin = jnp.array([0.5, 0.5, 0.5])
        direction = jnp.array([0.0, 0.0, -1.0])
        box_min = jnp.array([0.0, 0.0, 0.0])
        box_max = jnp.array([1.0, 1.0, 1.0])
        t, normal = _box_exit(origin, direction, box_min, box_max)
        assert float(t) == pytest.approx(0.5)
        assert jnp.allclose(normal, jnp.array([0.0, 0.0, -1.0]))

    def test_plane_t_normal_incidence(self):
        origin = jnp.array([0.0, 0.0, 5.0])
        direction = jnp.array([0.0, 0.0, -1.0])
        normal = jnp.array([0.0, 0.0, 1.0])
        point = jnp.array([0.0, 0.0, 0.0])
        t = _plane_t(origin, direction, normal, point)
        assert float(t) == pytest.approx(5.0)

    def test_box_entry_from_side(self):
        origin = jnp.array([-5.0, 0.5, 0.5])
        direction = jnp.array([1.0, 0.0, 0.0])
        box_min = jnp.array([0.0, 0.0, 0.0])
        box_max = jnp.array([1.0, 1.0, 1.0])
        t, normal = _box_entry(origin, direction, box_min, box_max)
        assert float(t) == pytest.approx(5.0)
        assert jnp.allclose(normal, jnp.array([-1.0, 0.0, 0.0]))

    def test_plane_t_parallel_ray(self):
        """Ray parallel to plane should return inf."""
        origin = jnp.array([0.0, 0.0, 5.0])
        direction = jnp.array([1.0, 0.0, 0.0])
        normal = jnp.array([0.0, 0.0, 1.0])
        point = jnp.array([0.0, 0.0, 0.0])
        t = _plane_t(origin, direction, normal, point)
        assert float(t) == pytest.approx(float(jnp.inf), rel=1e-3)


# ── Step constructors ──────────────────────────────────────────────────────

class TestStepConstructors:

    def test_partial_from_mirror(self, default_system):
        from apollo14.elements.surface import PartialMirror
        mirrors = [e for e in default_system.elements if isinstance(e, PartialMirror)]
        mirror = mirrors[0]

        step = partial(mirror)
        assert int(step.interaction) == PARTIAL
        assert jnp.allclose(step.position, jnp.asarray(mirror.position))
        assert float(step.half_width) == pytest.approx(mirror.width / 2)
        assert float(step.half_height) == pytest.approx(mirror.height / 2)
        # Default reflectance from mirror ratio
        assert jnp.allclose(step.reflectance,
                            jnp.full(3, mirror.reflection_ratio))

    def test_partial_with_custom_reflectance(self, default_system):
        from apollo14.elements.surface import PartialMirror
        mirror = next(e for e in default_system.elements if isinstance(e, PartialMirror))

        custom_r = jnp.array([0.1, 0.2, 0.3])
        step = partial(mirror, reflectance=custom_r)
        assert jnp.allclose(step.reflectance, custom_r)

    def test_refract_from_face(self, default_system):
        chassis = next(e for e in default_system.elements if isinstance(e, GlassBlock))
        face = chassis.faces[0]

        step = refract(face, 1.0, 1.5)
        assert int(step.interaction) == REFRACT
        assert float(step.n1) == pytest.approx(1.0)
        assert float(step.n2) == pytest.approx(1.5)
        assert jnp.allclose(step.position, jnp.asarray(face.position))

    def test_target_from_rectangular_pupil(self, default_system):
        from apollo14.elements.pupil import RectangularPupil
        pupil = next(e for e in default_system.elements if isinstance(e, RectangularPupil))

        step = target(pupil)
        assert int(step.interaction) == TARGET
        assert jnp.allclose(step.position, jnp.asarray(pupil.position))
        assert not bool(step.use_circular)
        assert float(step.half_width) == pytest.approx(pupil.width / 2)
        assert float(step.half_height) == pytest.approx(pupil.height / 2)

    def test_target_from_circular_pupil(self):
        from apollo14.elements.pupil import Pupil
        pupil = Pupil(
            name="test_pupil",
            position=jnp.array([0.0, 0.0, 0.0]),
            normal=jnp.array([0.0, 0.0, 1.0]),
            radius=5.0,
        )
        step = target(pupil)
        assert int(step.interaction) == TARGET
        assert bool(step.use_circular)
        assert float(step.radius) == pytest.approx(5.0)


# ── stack_path ──────────────────────────────────────────────────────────────

class TestStackPath:

    def test_stack_shape(self, default_system):
        from apollo14.elements.surface import PartialMirror
        from apollo14.elements.pupil import RectangularPupil

        mirrors = [e for e in default_system.elements if isinstance(e, PartialMirror)]
        pupil_elem = next(e for e in default_system.elements if isinstance(e, RectangularPupil))
        chassis = next(e for e in default_system.elements if isinstance(e, GlassBlock))
        exit_face = next(f for f in chassis.faces if f.name == "top")
        n_glass = float(chassis.material.n(DEFAULT_WAVELENGTH))

        main_list = [partial(m) for m in mirrors]
        branch_lists = [
            [refract(exit_face, n_glass, 1.0), target(pupil_elem)]
            for _ in mirrors
        ]

        main_steps, branch_steps = stack_path(main_list, branch_lists)

        M = len(mirrors)
        assert main_steps.position.shape == (M, 3)
        assert main_steps.interaction.shape == (M,)
        assert branch_steps.position.shape == (M, 2, 3)
        assert branch_steps.interaction.shape == (M, 2)


# ── trace_path (generic API) ───────────────────────────────────────────────

class TestTracePath:

    def test_trace_path_returns_five_values(self, default_setup):
        params, n_glass = default_setup
        _, d_glass, main_steps, branch_steps = build_combiner_paths(
            params, n_glass, DEFAULT_LIGHT_DIRECTION)
        t_entry, _ = _box_entry(
            DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
            params.chassis_min, params.chassis_max)
        entry_point = DEFAULT_LIGHT_POSITION + t_entry * DEFAULT_LIGHT_DIRECTION

        result = trace_path(entry_point, d_glass, jnp.array(1.0),
                            main_steps, branch_steps)
        assert len(result) == 5
        endpoints, intensities, valid, main_hits, branch_hits = result
        M = params.mirror_positions.shape[0]
        assert endpoints.shape == (M, 3)
        assert intensities.shape == (M,)
        assert valid.shape == (M,)
        assert main_hits.shape == (M, 3)
        assert branch_hits.shape == (M, 2, 3)  # B=2: exit + pupil

    def test_trace_path_color_idx(self, default_setup):
        """Different color_idx should select different reflectance channels."""
        params, n_glass = default_setup
        # Set different reflectances per color
        new_refl = jnp.array([
            [0.03, 0.05, 0.07],
        ] * 6)
        params = params._replace(mirror_reflectances=new_refl)

        _, d_glass, main_steps, branch_steps = build_combiner_paths(
            params, n_glass, DEFAULT_LIGHT_DIRECTION)
        t_entry, _ = _box_entry(
            DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
            params.chassis_min, params.chassis_max)
        entry_point = DEFAULT_LIGHT_POSITION + t_entry * DEFAULT_LIGHT_DIRECTION

        _, ints_r, _, _, _ = trace_path(
            entry_point, d_glass, jnp.array(1.0),
            main_steps, branch_steps, color_idx=0)
        _, ints_g, _, _, _ = trace_path(
            entry_point, d_glass, jnp.array(1.0),
            main_steps, branch_steps, color_idx=1)
        _, ints_b, _, _, _ = trace_path(
            entry_point, d_glass, jnp.array(1.0),
            main_steps, branch_steps, color_idx=2)

        # Mirror 0 should reflect the per-channel ratio
        assert float(ints_r[0]) == pytest.approx(0.03, abs=1e-4)
        assert float(ints_g[0]) == pytest.approx(0.05, abs=1e-4)
        assert float(ints_b[0]) == pytest.approx(0.07, abs=1e-4)

    def test_trace_path_jit_compatible(self, default_setup):
        params, n_glass = default_setup
        _, d_glass, main_steps, branch_steps = build_combiner_paths(
            params, n_glass, DEFAULT_LIGHT_DIRECTION)
        t_entry, _ = _box_entry(
            DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
            params.chassis_min, params.chassis_max)
        entry_point = DEFAULT_LIGHT_POSITION + t_entry * DEFAULT_LIGHT_DIRECTION

        @jax.jit
        def traced():
            return trace_path(entry_point, d_glass, jnp.array(1.0),
                              main_steps, branch_steps)

        endpoints, ints, valid, mh, bh = traced()
        assert jnp.all(jnp.isfinite(endpoints))
        assert jnp.all(jnp.isfinite(ints))


# ── Single ray trace ────────────────────────────────────────────────────────

class TestTraceRay:

    def test_per_mirror_reflected_intensity(self, default_setup):
        """Each mirror should reflect 0.05 of original intensity."""
        params, n_glass = default_setup
        pts, ints, valid, _, _ = trace_ray(
            DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, n_glass, params)
        for i in range(6):
            assert float(ints[i]) == pytest.approx(0.05, abs=1e-3)

    def test_total_reflected_intensity(self, default_setup):
        """6 mirrors * 0.05 = 0.30 total reflected."""
        params, n_glass = default_setup
        pts, ints, valid, _, _ = trace_ray(
            DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, n_glass, params)
        assert float(jnp.sum(ints)) == pytest.approx(0.30, abs=1e-3)

    def test_some_reflections_reach_pupil(self, default_setup):
        """On-axis ray: at least some reflected rays should hit the pupil."""
        params, n_glass = default_setup
        pts, ints, valid, _, _ = trace_ray(
            DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, n_glass, params)
        assert jnp.any(valid)
        # Upper mirrors (closer to pupil center) should hit; lower ones may miss
        assert valid[0]  # mirror_0 is closest to pupil center

    def test_pupil_points_are_on_pupil_plane(self, default_setup):
        """Hit points should lie on the pupil plane (z = pupil_center.z)."""
        params, n_glass = default_setup
        pts, ints, valid, _, _ = trace_ray(
            DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, n_glass, params)
        for i in range(6):
            if valid[i]:
                dist_to_plane = jnp.dot(
                    pts[i] - params.pupil_center, params.pupil_normal)
                assert float(jnp.abs(dist_to_plane)) < 1e-3

    def test_pupil_points_within_radius(self, default_setup):
        """Valid hits should be within the pupil radius."""
        params, n_glass = default_setup
        pts, ints, valid, _, _ = trace_ray(
            DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, n_glass, params)
        for i in range(6):
            if valid[i]:
                delta = pts[i] - params.pupil_center
                dx = jnp.abs(jnp.dot(delta, params.pupil_local_x))
                dy = jnp.abs(jnp.dot(delta, params.pupil_local_y))
                assert float(dx) <= float(params.pupil_half_width) + 1e-3
                assert float(dy) <= float(params.pupil_half_height) + 1e-3

    def test_per_color_reflectance(self, default_setup):
        """Different color channels should yield different intensities."""
        params, n_glass = default_setup
        new_refl = params.mirror_reflectances.at[:, 0].set(0.03)
        new_refl = new_refl.at[:, 2].set(0.08)
        params = params._replace(mirror_reflectances=new_refl)

        _, ints_r, _, _, _ = trace_ray(
            DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, n_glass, params, color_idx=0)
        _, ints_b, _, _, _ = trace_ray(
            DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, n_glass, params, color_idx=2)

        # Red reflectance < default < blue reflectance
        assert float(ints_r[0]) < float(ints_b[0])


# ── Intermediate hit geometry ──────────────────────────────────────────────

class TestIntermediateHits:

    def test_main_hits_shape(self, default_setup):
        params, n_glass = default_setup
        _, _, _, main_hits, branch_hits = trace_ray(
            DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, n_glass, params)
        M = params.mirror_positions.shape[0]
        assert main_hits.shape == (M, 3)
        assert branch_hits.shape == (M, 2, 3)  # B=2: exit face + pupil

    def test_main_hits_lie_on_mirror_planes(self, default_setup):
        """Each main_hit should lie on the corresponding mirror plane."""
        params, n_glass = default_setup
        _, _, _, main_hits, _ = trace_ray(
            DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, n_glass, params)

        for i in range(params.mirror_positions.shape[0]):
            delta = main_hits[i] - params.mirror_positions[i]
            dist = jnp.abs(jnp.dot(delta, params.mirror_normals[i]))
            assert float(dist) < 1e-3, f"mirror {i}: hit not on plane"

    def test_branch_endpoint_equals_pupil_hit(self, default_setup):
        """branch_hits[:, -1] should equal endpoints (pupil hit)."""
        params, n_glass = default_setup
        endpoints, _, valid, _, branch_hits = trace_ray(
            DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, n_glass, params)

        for i in range(params.mirror_positions.shape[0]):
            if valid[i]:
                assert jnp.allclose(branch_hits[i, 1], endpoints[i], atol=1e-4), \
                    f"mirror {i}: branch endpoint != pupil hit"

    def test_main_hits_within_mirror_bounds(self, default_setup):
        """Main hits should be within mirror width/height bounds."""
        params, n_glass = default_setup
        _, _, _, main_hits, _ = trace_ray(
            DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, n_glass, params)

        for i in range(params.mirror_positions.shape[0]):
            delta = main_hits[i] - params.mirror_positions[i]
            dx = jnp.abs(jnp.dot(delta, params.mirror_local_x[i]))
            dy = jnp.abs(jnp.dot(delta, params.mirror_local_y[i]))
            assert float(dx) <= float(params.mirror_half_widths[i]) + 1e-3
            assert float(dy) <= float(params.mirror_half_heights[i]) + 1e-3

    def test_beam_intermediate_hits_shape(self, default_setup):
        params, n_glass = default_setup
        N = 5
        origins = jnp.tile(DEFAULT_LIGHT_POSITION, (N, 1))
        offsets = jnp.linspace(-0.5, 0.5, N)
        origins = origins.at[:, 0].add(offsets)

        _, _, _, main_hits, branch_hits = trace_beam(
            origins, DEFAULT_LIGHT_DIRECTION, n_glass, params)
        M = params.mirror_positions.shape[0]
        assert main_hits.shape == (N, M, 3)
        assert branch_hits.shape == (N, M, 2, 3)

    def test_intermediate_hits_are_finite(self, default_setup):
        """All intermediate hit coordinates should be finite."""
        params, n_glass = default_setup
        _, _, _, main_hits, branch_hits = trace_ray(
            DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, n_glass, params)
        assert jnp.all(jnp.isfinite(main_hits))
        assert jnp.all(jnp.isfinite(branch_hits))


# ── Batched tracing ─────────────────────────────────────────────────────────

class TestTraceBatch:

    def test_batch_shape(self, default_setup):
        params, n_glass = default_setup
        N = 4
        origins = jnp.tile(DEFAULT_LIGHT_POSITION, (N, 1))
        directions = jnp.tile(DEFAULT_LIGHT_DIRECTION, (N, 1))
        pts, ints, valid, _, _ = trace_batch(origins, directions, n_glass, params)
        assert pts.shape == (N, 6, 3)
        assert ints.shape == (N, 6)
        assert valid.shape == (N, 6)

    def test_batch_matches_single(self, default_setup):
        params, n_glass = default_setup
        single_pts, single_ints, single_valid, _, _ = trace_ray(
            DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, n_glass, params)

        origins = jnp.tile(DEFAULT_LIGHT_POSITION, (3, 1))
        directions = jnp.tile(DEFAULT_LIGHT_DIRECTION, (3, 1))
        batch_pts, batch_ints, batch_valid, _, _ = trace_batch(
            origins, directions, n_glass, params)

        for i in range(3):
            assert jnp.allclose(batch_ints[i], single_ints, atol=1e-5)


# ── Beam tracing (shared direction) ─────────────────────────────────────────

class TestTraceBeam:

    def test_beam_shape(self, default_setup):
        params, n_glass = default_setup
        N = 10
        # Grid of origins with small offsets, same direction
        offsets = jnp.linspace(-1.0, 1.0, N)
        origins = jnp.tile(DEFAULT_LIGHT_POSITION, (N, 1))
        origins = origins.at[:, 0].add(offsets)

        pts, ints, valid, _, _ = trace_beam(origins, DEFAULT_LIGHT_DIRECTION, n_glass, params)
        assert pts.shape == (N, 6, 3)
        assert ints.shape == (N, 6)
        assert valid.shape == (N, 6)

    def test_beam_matches_batch(self, default_setup):
        """trace_beam should give same results as trace_batch for same inputs."""
        params, n_glass = default_setup
        N = 5
        offsets = jnp.linspace(-0.5, 0.5, N)
        origins = jnp.tile(DEFAULT_LIGHT_POSITION, (N, 1))
        origins = origins.at[:, 0].add(offsets)
        directions = jnp.tile(DEFAULT_LIGHT_DIRECTION, (N, 1))

        beam_pts, beam_ints, beam_valid, _, _ = trace_beam(
            origins, DEFAULT_LIGHT_DIRECTION, n_glass, params)
        batch_pts, batch_ints, batch_valid, _, _ = trace_batch(
            origins, directions, n_glass, params)

        assert jnp.allclose(beam_ints, batch_ints, atol=1e-4)
        # Valid flags should match
        assert jnp.array_equal(beam_valid, batch_valid)

    def test_beam_differentiable(self, default_setup):
        params, n_glass = default_setup
        N = 4
        origins = jnp.tile(DEFAULT_LIGHT_POSITION, (N, 1))

        def total_intensity(reflectances):
            p = params._replace(mirror_reflectances=reflectances)
            _, ints, valid, _, _ = trace_beam(origins, DEFAULT_LIGHT_DIRECTION, n_glass, p)
            return jnp.sum(jnp.where(valid, ints, 0.0))

        grads = jax.grad(total_intensity)(params.mirror_reflectances)
        assert grads.shape == (6, 3)
        assert jnp.all(jnp.isfinite(grads))


# ── CombinerPath API ────────────────────────────────────────────────────────

class TestCombinerPath:

    def test_combiner_path_from_system(self, combiner_path):
        """combiner_path_from_system should produce a valid CombinerPath."""
        assert isinstance(combiner_path, CombinerPath)
        M = combiner_path.mirror_positions.shape[0]
        assert M == 6
        assert combiner_path.mirror_normals.shape == (6, 3)
        assert combiner_path.mirror_reflectances.shape == (6, 3)
        assert float(combiner_path.n_glass) > 1.0

    def test_combiner_path_has_aperture(self, combiner_path):
        """Default system should have an aperture."""
        assert bool(combiner_path.has_aperture)

    def test_combiner_path_entry_exit_normals(self, combiner_path):
        """Entry and exit normals should be unit vectors."""
        assert float(jnp.linalg.norm(combiner_path.entry_normal)) == pytest.approx(1.0, abs=1e-5)
        assert float(jnp.linalg.norm(combiner_path.exit_normal)) == pytest.approx(1.0, abs=1e-5)

    def test_trace_combiner_ray(self, combiner_path):
        """trace_combiner_ray should return 5 values with correct shapes."""
        pts, ints, valid, mh, bh = trace_combiner_ray(
            DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, combiner_path)
        assert pts.shape == (6, 3)
        assert ints.shape == (6,)
        assert valid.shape == (6,)
        assert mh.shape == (6, 3)
        assert bh.shape == (6, 2, 3)
        assert jnp.any(valid)

    def test_trace_combiner_beam(self, combiner_path):
        """trace_combiner_beam should handle multiple rays."""
        N = 5
        origins = jnp.tile(DEFAULT_LIGHT_POSITION, (N, 1))
        pts, ints, valid, mh, bh = trace_combiner_beam(
            origins, DEFAULT_LIGHT_DIRECTION, combiner_path)
        assert pts.shape == (N, 6, 3)
        assert ints.shape == (N, 6)
        assert mh.shape == (N, 6, 3)
        assert bh.shape == (N, 6, 2, 3)

    def test_trace_combiner_batch(self, combiner_path):
        """trace_combiner_batch should handle per-ray directions."""
        N = 3
        origins = jnp.tile(DEFAULT_LIGHT_POSITION, (N, 1))
        directions = jnp.tile(DEFAULT_LIGHT_DIRECTION, (N, 1))
        pts, ints, valid, mh, bh = trace_combiner_batch(
            origins, directions, combiner_path)
        assert pts.shape == (N, 6, 3)
        assert ints.shape == (N, 6)

    def test_combiner_ray_matches_legacy(self, default_setup, combiner_path):
        """CombinerPath API should produce similar results to legacy API."""
        params, n_glass = default_setup
        _, legacy_ints, legacy_valid, _, _ = trace_ray(
            DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, n_glass, params)
        _, new_ints, new_valid, _, _ = trace_combiner_ray(
            DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, combiner_path)

        # Intensities should match — both APIs trace the same system
        assert jnp.allclose(legacy_ints, new_ints, atol=1e-3)

    def test_combiner_ray_per_color(self, combiner_path):
        """Different color_idx should work with CombinerPath API."""
        _, ints_r, _, _, _ = trace_combiner_ray(
            DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, combiner_path, color_idx=0)
        _, ints_g, _, _, _ = trace_combiner_ray(
            DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, combiner_path, color_idx=1)
        # With uniform reflectances, all colors should be the same
        assert jnp.allclose(ints_r, ints_g, atol=1e-5)

    def test_combiner_beam_differentiable(self, combiner_path):
        """CombinerPath API should be differentiable."""
        N = 3
        origins = jnp.tile(DEFAULT_LIGHT_POSITION, (N, 1))

        def total_intensity(reflectances):
            path = combiner_path._replace(mirror_reflectances=reflectances)
            _, ints, valid, _, _ = trace_combiner_beam(
                origins, DEFAULT_LIGHT_DIRECTION, path)
            return jnp.sum(jnp.where(valid, ints, 0.0))

        grads = jax.grad(total_intensity)(combiner_path.mirror_reflectances)
        assert grads.shape == (6, 3)
        assert jnp.all(jnp.isfinite(grads))


# ── Aperture clipping ──────────────────────────────────────────────────────

class TestApertureClipping:

    def test_ray_outside_aperture_gets_zero_intensity(self, combiner_path):
        """A ray far outside the aperture should get zero intensity."""
        # Shift origin far to the side
        far_origin = DEFAULT_LIGHT_POSITION + jnp.array([100.0, 0.0, 0.0])
        _, ints, valid, _, _ = trace_combiner_ray(
            far_origin, DEFAULT_LIGHT_DIRECTION, combiner_path)
        assert float(jnp.sum(ints)) == pytest.approx(0.0, abs=1e-6)

    def test_no_aperture_passes_all(self, combiner_path):
        """With aperture disabled, all rays should pass."""
        path_no_ap = combiner_path._replace(has_aperture=jnp.array(False))
        far_origin = DEFAULT_LIGHT_POSITION + jnp.array([100.0, 0.0, 0.0])
        _, ints_no_ap, _, _, _ = trace_combiner_ray(
            far_origin, DEFAULT_LIGHT_DIRECTION, path_no_ap)
        # Even though origin is far, with no aperture the intensity isn't
        # zeroed by the aperture check (it may still miss due to entry face)
        _, ints_ap, _, _, _ = trace_combiner_ray(
            far_origin, DEFAULT_LIGHT_DIRECTION, combiner_path)
        # Without aperture, the ray might get farther than with aperture
        # (at minimum, no worse)
        assert float(jnp.sum(ints_no_ap)) >= float(jnp.sum(ints_ap)) - 1e-6


# ── compensated_reflectances ───────────────────────────────────────────────

class TestCompensatedReflectances:

    def test_shape_scalar(self):
        r = compensated_reflectances(0.05, 6)
        assert r.shape == (6, 3)
        # All three color columns should be identical for scalar input
        assert jnp.allclose(r[:, 0], r[:, 1])
        assert jnp.allclose(r[:, 0], r[:, 2])

    def test_shape_per_color(self):
        r = compensated_reflectances(jnp.array([0.03, 0.05, 0.07]), 6)
        assert r.shape == (6, 3)
        # Columns should differ
        assert not jnp.allclose(r[:, 0], r[:, 2])

    def test_first_mirror_equals_ratio(self):
        """First mirror reflectance should equal the input ratio."""
        r = compensated_reflectances(0.05, 6)
        assert float(r[0, 0]) == pytest.approx(0.05)

    def test_later_mirrors_increase(self):
        """Later mirrors need higher reflectance to compensate."""
        r = compensated_reflectances(0.05, 6)
        for i in range(5):
            assert float(r[i + 1, 0]) > float(r[i, 0])

    def test_equal_absolute_reflection(self):
        """Each mirror should reflect the same absolute intensity."""
        ratio = 0.05
        r = compensated_reflectances(ratio, 6)
        # Simulate: beam starts at 1.0, each mirror takes r[i] of remaining
        remaining = 1.0
        for i in range(6):
            reflected = remaining * float(r[i, 0])
            assert reflected == pytest.approx(ratio, abs=1e-6)
            remaining -= reflected

    def test_differentiable(self):
        """compensated_reflectances should be differentiable."""
        def total(ratio):
            return jnp.sum(compensated_reflectances(ratio, 6))
        grad = jax.grad(total)(jnp.array(0.05))
        assert jnp.isfinite(grad)


# ── Differentiability ───────────────────────────────────────────────────────

class TestDifferentiability:

    def test_grad_wrt_reflectances(self, default_setup):
        """Gradient of total pupil intensity w.r.t. mirror reflectances."""
        params, n_glass = default_setup

        def total_intensity(reflectances):
            p = params._replace(mirror_reflectances=reflectances)
            _, ints, valid, _, _ = trace_ray(
                DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, n_glass, p)
            return jnp.sum(jnp.where(valid, ints, 0.0))

        grad_fn = jax.grad(total_intensity)
        grads = grad_fn(params.mirror_reflectances)
        assert grads.shape == (6, 3)
        assert jnp.all(jnp.isfinite(grads))

    def test_grad_wrt_positions(self, default_setup):
        """Gradient of total pupil intensity w.r.t. mirror positions."""
        params, n_glass = default_setup

        def total_intensity(positions):
            p = params._replace(mirror_positions=positions)
            _, ints, valid, _, _ = trace_ray(
                DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, n_glass, p)
            return jnp.sum(jnp.where(valid, ints, 0.0))

        grad_fn = jax.grad(total_intensity)
        grads = grad_fn(params.mirror_positions)
        assert grads.shape == (6, 3)
        assert jnp.all(jnp.isfinite(grads))

    def test_grad_combiner_path_wrt_reflectances(self, combiner_path):
        """CombinerPath API should also be differentiable w.r.t. reflectances."""
        def total_intensity(reflectances):
            path = combiner_path._replace(mirror_reflectances=reflectances)
            _, ints, valid, _, _ = trace_combiner_ray(
                DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, path)
            return jnp.sum(jnp.where(valid, ints, 0.0))

        grads = jax.grad(total_intensity)(combiner_path.mirror_reflectances)
        assert grads.shape == (6, 3)
        assert jnp.all(jnp.isfinite(grads))

    def test_intermediate_hits_differentiable(self, default_setup):
        """Gradients should flow through intermediate hit geometry."""
        params, n_glass = default_setup

        def hit_sum(positions):
            p = params._replace(mirror_positions=positions)
            _, _, _, main_hits, _ = trace_ray(
                DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, n_glass, p)
            return jnp.sum(main_hits)

        grads = jax.grad(hit_sum)(params.mirror_positions)
        assert grads.shape == (6, 3)
        assert jnp.all(jnp.isfinite(grads))


# ── params_from_system ─────────────────────────────────────────────────────

class TestParamsFromSystem:

    def test_params_fields(self, default_setup):
        params, _ = default_setup
        assert isinstance(params, CombinerParams)
        assert params.mirror_positions.shape == (6, 3)
        assert params.mirror_normals.shape == (6, 3)
        assert params.mirror_reflectances.shape == (6, 3)
        assert params.chassis_min.shape == (3,)
        assert params.chassis_max.shape == (3,)
        assert params.pupil_center.shape == (3,)
        assert params.pupil_normal.shape == (3,)

    def test_chassis_aabb_contains_mirrors(self, default_setup):
        """All mirror positions should be inside the chassis AABB."""
        params, _ = default_setup
        for i in range(6):
            pos = params.mirror_positions[i]
            assert jnp.all(pos >= params.chassis_min - 1e-3)
            assert jnp.all(pos <= params.chassis_max + 1e-3)

    def test_mirror_normals_unit(self, default_setup):
        """Mirror normals should be unit vectors."""
        params, _ = default_setup
        for i in range(6):
            norm = float(jnp.linalg.norm(params.mirror_normals[i]))
            assert norm == pytest.approx(1.0, abs=1e-5)


# ── build_combiner_paths ───────────────────────────────────────────────────

class TestBuildCombinerPaths:

    def test_returns_four_values(self, default_setup):
        params, n_glass = default_setup
        result = build_combiner_paths(params, n_glass, DEFAULT_LIGHT_DIRECTION)
        assert len(result) == 4
        entry_normal, d_glass, main_steps, branch_steps = result

    def test_entry_normal_is_unit(self, default_setup):
        params, n_glass = default_setup
        entry_normal, _, _, _ = build_combiner_paths(
            params, n_glass, DEFAULT_LIGHT_DIRECTION)
        assert float(jnp.linalg.norm(entry_normal)) == pytest.approx(1.0, abs=1e-5)

    def test_d_glass_is_unit(self, default_setup):
        params, n_glass = default_setup
        _, d_glass, _, _ = build_combiner_paths(
            params, n_glass, DEFAULT_LIGHT_DIRECTION)
        assert float(jnp.linalg.norm(d_glass)) == pytest.approx(1.0, abs=1e-3)

    def test_main_steps_are_partial(self, default_setup):
        params, n_glass = default_setup
        _, _, main_steps, _ = build_combiner_paths(
            params, n_glass, DEFAULT_LIGHT_DIRECTION)
        assert jnp.all(main_steps.interaction == PARTIAL)

    def test_branch_steps_shape(self, default_setup):
        """Branch has 2 steps per mirror: exit refraction + pupil target."""
        params, n_glass = default_setup
        _, _, _, branch_steps = build_combiner_paths(
            params, n_glass, DEFAULT_LIGHT_DIRECTION)
        M = params.mirror_positions.shape[0]
        assert branch_steps.position.shape == (M, 2, 3)
        # First branch step is REFRACT, second is TARGET
        assert jnp.all(branch_steps.interaction[:, 0] == REFRACT)
        assert jnp.all(branch_steps.interaction[:, 1] == TARGET)
