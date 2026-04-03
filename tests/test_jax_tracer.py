import jax
import jax.numpy as jnp
import pytest

from apollo14.combiner import CombinerConfig
from apollo14.jax_tracer import (
    trace_ray, trace_batch, trace_beam, params_from_config,
    _box_entry, _box_exit, _plane_t,
)


@pytest.fixture
def default_setup():
    config = CombinerConfig.default()
    params = params_from_config(config)
    n_glass = float(config.chassis.material.n(config.light.wavelength))
    return config, params, n_glass


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


# ── Single ray trace ────────────────────────────────────────────────────────

class TestTraceRay:

    def test_per_mirror_reflected_intensity(self, default_setup):
        """Each mirror should reflect 0.05 of original intensity."""
        config, params, n_glass = default_setup
        pts, ints, valid = trace_ray(
            config.light.position, config.light.direction, n_glass, params)
        for i in range(6):
            assert float(ints[i]) == pytest.approx(0.05, abs=1e-3)

    def test_total_reflected_intensity(self, default_setup):
        """6 mirrors * 0.05 = 0.30 total reflected."""
        config, params, n_glass = default_setup
        pts, ints, valid = trace_ray(
            config.light.position, config.light.direction, n_glass, params)
        assert float(jnp.sum(ints)) == pytest.approx(0.30, abs=1e-3)

    def test_some_reflections_reach_pupil(self, default_setup):
        """On-axis ray: at least some reflected rays should hit the pupil."""
        config, params, n_glass = default_setup
        pts, ints, valid = trace_ray(
            config.light.position, config.light.direction, n_glass, params)
        assert jnp.any(valid)
        # Upper mirrors (closer to pupil center) should hit; lower ones may miss
        assert valid[0]  # mirror_0 is closest to pupil center

    def test_pupil_points_are_on_pupil_plane(self, default_setup):
        """Hit points should lie on the pupil plane (z = pupil_center.z)."""
        config, params, n_glass = default_setup
        pts, ints, valid = trace_ray(
            config.light.position, config.light.direction, n_glass, params)
        for i in range(6):
            if valid[i]:
                dist_to_plane = jnp.dot(
                    pts[i] - params.pupil_center, params.pupil_normal)
                assert float(jnp.abs(dist_to_plane)) < 1e-3

    def test_pupil_points_within_radius(self, default_setup):
        """Valid hits should be within the pupil radius."""
        config, params, n_glass = default_setup
        pts, ints, valid = trace_ray(
            config.light.position, config.light.direction, n_glass, params)
        for i in range(6):
            if valid[i]:
                delta = pts[i] - params.pupil_center
                r = jnp.sqrt(
                    jnp.dot(delta, params.pupil_local_x) ** 2 +
                    jnp.dot(delta, params.pupil_local_y) ** 2)
                assert float(r) <= float(params.pupil_radius) + 1e-3



# ── Batched tracing ─────────────────────────────────────────────────────────

class TestTraceBatch:

    def test_batch_shape(self, default_setup):
        config, params, n_glass = default_setup
        N = 4
        origins = jnp.tile(config.light.position, (N, 1))
        directions = jnp.tile(config.light.direction, (N, 1))
        pts, ints, valid = trace_batch(origins, directions, n_glass, params)
        assert pts.shape == (N, 6, 3)
        assert ints.shape == (N, 6)
        assert valid.shape == (N, 6)

    def test_batch_matches_single(self, default_setup):
        config, params, n_glass = default_setup
        single_pts, single_ints, single_valid = trace_ray(
            config.light.position, config.light.direction, n_glass, params)

        origins = jnp.tile(config.light.position, (3, 1))
        directions = jnp.tile(config.light.direction, (3, 1))
        batch_pts, batch_ints, batch_valid = trace_batch(
            origins, directions, n_glass, params)

        for i in range(3):
            assert jnp.allclose(batch_ints[i], single_ints, atol=1e-5)


# ── Beam tracing (shared direction) ─────────────────────────────────────────

class TestTraceBeam:

    def test_beam_shape(self, default_setup):
        config, params, n_glass = default_setup
        N = 10
        # Grid of origins with small offsets, same direction
        offsets = jnp.linspace(-1.0, 1.0, N)
        origins = jnp.tile(config.light.position, (N, 1))
        origins = origins.at[:, 0].add(offsets)

        pts, ints, valid = trace_beam(origins, config.light.direction, n_glass, params)
        assert pts.shape == (N, 6, 3)
        assert ints.shape == (N, 6)
        assert valid.shape == (N, 6)

    def test_beam_matches_batch(self, default_setup):
        """trace_beam should give same results as trace_batch for same inputs."""
        config, params, n_glass = default_setup
        N = 5
        offsets = jnp.linspace(-0.5, 0.5, N)
        origins = jnp.tile(config.light.position, (N, 1))
        origins = origins.at[:, 0].add(offsets)
        directions = jnp.tile(config.light.direction, (N, 1))

        beam_pts, beam_ints, beam_valid = trace_beam(
            origins, config.light.direction, n_glass, params)
        batch_pts, batch_ints, batch_valid = trace_batch(
            origins, directions, n_glass, params)

        assert jnp.allclose(beam_ints, batch_ints, atol=1e-4)
        # Valid flags should match
        assert jnp.array_equal(beam_valid, batch_valid)

    def test_beam_differentiable(self, default_setup):
        config, params, n_glass = default_setup
        N = 4
        origins = jnp.tile(config.light.position, (N, 1))

        def total_intensity(reflectances):
            p = params._replace(mirror_reflectances=reflectances)
            _, ints, valid = trace_beam(origins, config.light.direction, n_glass, p)
            return jnp.sum(jnp.where(valid, ints, 0.0))

        grads = jax.grad(total_intensity)(params.mirror_reflectances)
        assert grads.shape == (6, 3)
        assert jnp.all(jnp.isfinite(grads))


# ── Differentiability ───────────────────────────────────────────────────────

class TestDifferentiability:

    def test_grad_wrt_reflectances(self, default_setup):
        """Gradient of total pupil intensity w.r.t. mirror reflectances."""
        config, params, n_glass = default_setup

        def total_intensity(reflectances):
            p = params._replace(mirror_reflectances=reflectances)
            _, ints, valid = trace_ray(
                config.light.position, config.light.direction, n_glass, p)
            return jnp.sum(jnp.where(valid, ints, 0.0))

        grad_fn = jax.grad(total_intensity)
        grads = grad_fn(params.mirror_reflectances)
        assert grads.shape == (6, 3)
        assert jnp.all(jnp.isfinite(grads))

    def test_grad_wrt_positions(self, default_setup):
        """Gradient of total pupil intensity w.r.t. mirror positions."""
        config, params, n_glass = default_setup

        def total_intensity(positions):
            p = params._replace(mirror_positions=positions)
            _, ints, valid = trace_ray(
                config.light.position, config.light.direction, n_glass, p)
            return jnp.sum(jnp.where(valid, ints, 0.0))

        grad_fn = jax.grad(total_intensity)
        grads = grad_fn(params.mirror_positions)
        assert grads.shape == (6, 3)
        assert jnp.all(jnp.isfinite(grads))
