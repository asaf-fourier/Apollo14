import jax
import jax.numpy as jnp
import pytest

from apollo14.combiner import CombinerConfig
from apollo14.jax_tracer import params_from_config
from apollo14.units import mm

from helios.eyebox import (
    EyeboxConfig, eyebox_sample_points, eyebox_grid_points,
    compute_eyebox_response, eyebox_merit, visible_fov,
)


@pytest.fixture
def default_setup():
    config = CombinerConfig.default()
    params = params_from_config(config)
    n_glass = float(config.chassis.material.n(config.light.wavelength))
    return config, params, n_glass


# ── Eyebox sampling ─────────────────────────────────────────────────────────

class TestEyeboxSampling:

    def test_sample_shape(self):
        center = jnp.array([7.0, 20.0, 17.0])
        normal = jnp.array([0.0, 0.0, -1.0])
        samples = eyebox_sample_points(center, normal, radius=4.0)
        assert samples.shape == (5, 3)

    def test_center_sample_at_center(self):
        center = jnp.array([7.0, 20.0, 17.0])
        normal = jnp.array([0.0, 0.0, -1.0])
        samples = eyebox_sample_points(center, normal, radius=4.0)
        assert jnp.allclose(samples[0], center, atol=1e-5)

    def test_corners_within_radius(self):
        center = jnp.array([0.0, 0.0, 0.0])
        normal = jnp.array([0.0, 0.0, 1.0])
        radius = 5.0
        samples = eyebox_sample_points(center, normal, radius)
        dists = jnp.linalg.norm(samples - center, axis=1)
        assert jnp.all(dists <= radius + 1e-5)

    def test_corners_symmetric(self):
        center = jnp.array([0.0, 0.0, 0.0])
        normal = jnp.array([0.0, 0.0, 1.0])
        samples = eyebox_sample_points(center, normal, radius=4.0)
        corner_dists = jnp.linalg.norm(samples[1:] - center, axis=1)
        assert jnp.allclose(corner_dists, corner_dists[0], atol=1e-5)

    def test_grid_shape(self):
        center = jnp.array([7.0, 20.0, 17.0])
        normal = jnp.array([0.0, 0.0, -1.0])
        grid = eyebox_grid_points(center, normal, radius=4.0, nx=5, ny=5)
        assert grid.shape == (25, 3)

    def test_grid_center_at_center(self):
        center = jnp.array([7.0, 20.0, 17.0])
        normal = jnp.array([0.0, 0.0, -1.0])
        grid = eyebox_grid_points(center, normal, radius=4.0, nx=3, ny=3)
        # Center of a 3x3 grid is index 4
        assert jnp.allclose(grid[4], center, atol=1e-5)

    def test_grid_within_radius(self):
        center = jnp.array([0.0, 0.0, 0.0])
        normal = jnp.array([0.0, 0.0, 1.0])
        radius = 5.0
        grid = eyebox_grid_points(center, normal, radius, nx=5, ny=5)
        dists = jnp.linalg.norm(grid - center, axis=1)
        assert jnp.all(dists <= radius * jnp.sqrt(2.0) + 1e-5)


# ── Response computation ────────────────────────────────────────────────────

class TestEyeboxResponse:

    def test_response_shape(self, default_setup):
        config, params, n_glass = default_setup
        grid = eyebox_grid_points(
            config.pupil.center, config.pupil.normal, config.pupil.radius,
            nx=3, ny=3)
        mc = EyeboxConfig(n_fov_x=3, n_fov_y=3)

        response, dirs = compute_eyebox_response(
            params, n_glass,
            config.light.position, config.light.direction,
            config.light.beam_width, config.light.beam_height,
            config.light.x_fov, config.light.y_fov,
            grid, mc,
        )
        assert response.shape == (9, 9, 3)  # 3x3 grid, 3x3 angles, 3 colors
        assert dirs.shape == (9, 3)

    def test_response_nonnegative(self, default_setup):
        config, params, n_glass = default_setup
        grid = eyebox_grid_points(
            config.pupil.center, config.pupil.normal, config.pupil.radius,
            nx=3, ny=3)
        mc = EyeboxConfig(n_fov_x=3, n_fov_y=3)

        response, _ = compute_eyebox_response(
            params, n_glass,
            config.light.position, config.light.direction,
            config.light.beam_width, config.light.beam_height,
            config.light.x_fov, config.light.y_fov,
            grid, mc,
        )
        assert jnp.all(response >= 0)

    def test_center_has_intensity(self, default_setup):
        """Center eyebox sample with on-axis ray should get some light."""
        config, params, n_glass = default_setup
        grid = eyebox_grid_points(
            config.pupil.center, config.pupil.normal, config.pupil.radius,
            nx=3, ny=3)
        mc = EyeboxConfig(n_fov_x=1, n_fov_y=1)

        response, _ = compute_eyebox_response(
            params, n_glass,
            config.light.position, config.light.direction,
            config.light.beam_width, config.light.beam_height,
            0.0, 0.0,  # on-axis only
            grid, mc,
        )
        # Center of 3x3 grid (index 4) should have nonzero intensity
        assert float(response[4].sum()) > 0


# ── Merit function ──────────────────────────────────────────────────────────

class TestEyeboxMerit:

    def test_uniform_at_d65_target_is_near_zero(self):
        """Perfectly uniform response at D65 target should give near-zero merit.
        Not exactly zero due to coverage sigmoid, but uniformity and intensity
        terms should be zero."""
        mc = EyeboxConfig(target_intensity=0.03, w_coverage=0.0)
        target = mc.target_per_color  # (3,) D65-weighted
        response = jnp.broadcast_to(target, (5, 9, 3))
        assert float(eyebox_merit(response, mc)) == pytest.approx(0.0, abs=1e-6)

    def test_merit_positive_for_wrong_intensity(self):
        """Uniform response at wrong intensity should have nonzero merit."""
        mc = EyeboxConfig(target_intensity=0.03)
        response = jnp.ones((5, 9, 3)) * 0.01  # wrong level
        assert float(eyebox_merit(response, mc)) > 0

    def test_merit_positive_for_non_uniform(self):
        """Right mean intensity but non-uniform should have nonzero merit."""
        mc = EyeboxConfig(target_intensity=0.03)
        target = mc.target_per_color
        # Mean matches target, but spatial variation
        response = jnp.broadcast_to(target, (5, 9, 3)).copy()
        response = response.at[0, :, :].multiply(2.0)  # sample 0 is 2x brighter
        response = response.at[1, :, :].multiply(0.0)  # sample 1 is dark
        assert float(eyebox_merit(response, mc)) > 0

    def test_uniformity_weighted_more_than_intensity(self):
        """With default weights (w_uniformity=2 > w_intensity=1),
        a non-uniform response should score worse than a uniformly
        offset one with similar total deviation."""
        mc = EyeboxConfig(target_intensity=0.10)
        target = mc.target_per_color  # (3,)

        # Case A: uniform but offset from target
        offset = jnp.broadcast_to(target * 1.5, (5, 9, 3))

        # Case B: right mean but spatially non-uniform
        non_uniform = jnp.broadcast_to(target, (5, 9, 3))
        # Double at center, zero at one corner — same mean deviation magnitude
        non_uniform = non_uniform.at[0].multiply(2.0)
        non_uniform = non_uniform.at[1].set(0.0)

        merit_offset = float(eyebox_merit(offset, mc))
        merit_nonuniform = float(eyebox_merit(non_uniform, mc))
        # Non-uniform should be penalized more due to higher w_uniformity
        assert merit_nonuniform > merit_offset

    def test_d65_color_balance(self):
        """Uniform-but-wrong-color response should be penalized."""
        mc = EyeboxConfig(target_intensity=0.03)
        # Same total intensity but equal across colors (not D65 balanced)
        flat_color = jnp.ones((5, 9, 3)) * (0.03 / 3.0)
        # D65 balanced
        d65_balanced = jnp.broadcast_to(mc.target_per_color, (5, 9, 3))

        assert float(eyebox_merit(d65_balanced, mc)) < float(eyebox_merit(flat_color, mc))


# ── Visible FOV diagnostic ─────────────────────────────────────────────────

class TestVisibleFOV:

    def test_all_visible(self):
        response = jnp.ones((5, 9, 3)) * 0.1
        fov = visible_fov(response, threshold_fraction=0.5)
        assert fov.shape == (5,)
        assert jnp.allclose(fov, 1.0)

    def test_none_visible(self):
        response = jnp.zeros((5, 9, 3))
        fov = visible_fov(response, threshold_fraction=0.5)
        assert jnp.allclose(fov, 0.0)


# ── Differentiability ──────────────────────────────────────────────────────

class TestEyeboxDifferentiability:

    def test_grad_wrt_reflectances(self, default_setup):
        """Gradient of eyebox merit w.r.t. mirror reflectances."""
        config, params, n_glass = default_setup
        grid = eyebox_grid_points(
            config.pupil.center, config.pupil.normal, config.pupil.radius,
            nx=3, ny=3)
        mc = EyeboxConfig(n_fov_x=3, n_fov_y=3)

        def loss(reflectances):
            p = params._replace(mirror_reflectances=reflectances)
            response, _ = compute_eyebox_response(
                p, n_glass,
                config.light.position, config.light.direction,
                config.light.beam_width, config.light.beam_height,
                config.light.x_fov, config.light.y_fov,
                grid, mc,
            )
            return eyebox_merit(response, mc)

        grads = jax.grad(loss)(params.mirror_reflectances)
        assert grads.shape == (6, 3)
        assert jnp.all(jnp.isfinite(grads))
