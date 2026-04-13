import jax
import jax.numpy as jnp
import pytest

from apollo14.combiner import (
    build_default_system,
    DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, DEFAULT_WAVELENGTH,
    DEFAULT_BEAM_WIDTH, DEFAULT_BEAM_HEIGHT, DEFAULT_X_FOV, DEFAULT_Y_FOV,
)
from apollo14.elements.pupil import RectangularPupil
from apollo14.route import display_route
from apollo14.units import mm

from helios.eyebox import (
    EyeboxConfig, eyebox_sample_points, eyebox_grid_points,
    compute_eyebox_response, eyebox_merit, visible_fov,
    EyeboxAreaConfig, eyebox_area_merit, cell_grid_from_cell_size,
)


@pytest.fixture
def default_setup():
    system = build_default_system()
    route = display_route(system, DEFAULT_WAVELENGTH)
    pupil = next(e for e in system.elements if isinstance(e, RectangularPupil))
    return route, pupil


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
        route, pupil = default_setup
        grid = eyebox_grid_points(
            pupil.position, pupil.normal, pupil.width / 2,
            nx=3, ny=3)
        mc = EyeboxConfig(n_fov_x=3, n_fov_y=3)

        response, dirs = compute_eyebox_response(
            route,
            DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
            DEFAULT_BEAM_WIDTH, DEFAULT_BEAM_HEIGHT,
            DEFAULT_X_FOV, DEFAULT_Y_FOV,
            grid, mc,
        )
        assert response.shape == (9, 9, 3)  # 3x3 grid, 3x3 angles, 3 colors
        assert dirs.shape == (9, 3)

    def test_response_nonnegative(self, default_setup):
        route, pupil = default_setup
        grid = eyebox_grid_points(
            pupil.position, pupil.normal, pupil.width / 2,
            nx=3, ny=3)
        mc = EyeboxConfig(n_fov_x=3, n_fov_y=3)

        response, _ = compute_eyebox_response(
            route,
            DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
            DEFAULT_BEAM_WIDTH, DEFAULT_BEAM_HEIGHT,
            DEFAULT_X_FOV, DEFAULT_Y_FOV,
            grid, mc,
        )
        assert jnp.all(response >= 0)

    def test_center_has_intensity(self, default_setup):
        """Center eyebox sample with on-axis ray should get some light."""
        route, pupil = default_setup
        grid = eyebox_grid_points(
            pupil.position, pupil.normal, pupil.width / 2,
            nx=3, ny=3)
        mc = EyeboxConfig(n_fov_x=1, n_fov_y=1)

        response, _ = compute_eyebox_response(
            route,
            DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
            DEFAULT_BEAM_WIDTH, DEFAULT_BEAM_HEIGHT,
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
        route, pupil = default_setup
        grid = eyebox_grid_points(
            pupil.position, pupil.normal, pupil.width / 2,
            nx=3, ny=3)
        mc = EyeboxConfig(n_fov_x=3, n_fov_y=3)

        def loss(reflectances):
            new_mirrors = route.mirrors._replace(reflectance=reflectances)
            r = route._replace(mirrors=new_mirrors)
            response, _ = compute_eyebox_response(
                r,
                DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
                DEFAULT_BEAM_WIDTH, DEFAULT_BEAM_HEIGHT,
                DEFAULT_X_FOV, DEFAULT_Y_FOV,
                grid, mc,
            )
            return eyebox_merit(response, mc)

        grads = jax.grad(loss)(route.mirrors.reflectance)
        assert grads.shape == (6, 3)
        assert jnp.all(jnp.isfinite(grads))


# ── Cell grid from cell size ──────────────────────────────────────────────

class TestCellGrid:

    def test_cell_grid_shape(self):
        center = jnp.array([7.0, 20.0, 17.0])
        normal = jnp.array([0.0, 0.0, -1.0])
        points, nx, ny = cell_grid_from_cell_size(
            center, normal, width=10.0, height=8.0, cell_size=2.0)
        assert nx == 5
        assert ny == 4
        assert points.shape == (20, 3)

    def test_cell_grid_center(self):
        center = jnp.array([7.0, 20.0, 17.0])
        normal = jnp.array([0.0, 0.0, -1.0])
        points, nx, ny = cell_grid_from_cell_size(
            center, normal, width=10.0, height=10.0, cell_size=2.0)
        # Center of a 5x5 grid is index 12
        assert jnp.allclose(points[12], center, atol=1e-5)

    def test_cell_grid_odd_size(self):
        """Non-integer division should round up."""
        center = jnp.array([0.0, 0.0, 0.0])
        normal = jnp.array([0.0, 0.0, 1.0])
        points, nx, ny = cell_grid_from_cell_size(
            center, normal, width=7.0, height=5.0, cell_size=2.0)
        assert nx == 4  # ceil(7/2)
        assert ny == 3  # ceil(5/2)


# ── Eye-box area merit ────────────────────────────────────────────────────

class TestEyeboxAreaMerit:

    def test_all_above_threshold_is_near_zero(self):
        """All cells well above threshold → loss near 0."""
        config = EyeboxAreaConfig(intensity_threshold=0.005)
        # Strong uniform response, well above threshold after D65 normalization
        response = jnp.ones((9, 9, 3)) * 0.1
        loss = float(eyebox_area_merit(response, config))
        assert loss < 0.05

    def test_all_below_threshold_is_near_one(self):
        """All cells below threshold → loss near 1."""
        config = EyeboxAreaConfig(intensity_threshold=0.01)
        response = jnp.ones((9, 9, 3)) * 1e-6  # far below threshold
        loss = float(eyebox_area_merit(response, config))
        assert loss > 0.9

    def test_zero_response(self):
        """Zero response → loss ≥ 1 (area_loss=1 + margin penalty)."""
        config = EyeboxAreaConfig(intensity_threshold=0.01)
        response = jnp.zeros((9, 9, 3))
        loss = float(eyebox_area_merit(response, config))
        assert loss >= 0.95

    def test_half_cells_active(self):
        """Half the cells above threshold → loss around 0.5."""
        config = EyeboxAreaConfig(intensity_threshold=0.005, w_margin=0.0)
        response = jnp.zeros((10, 9, 3))
        # First 5 cells bright, last 5 dark
        response = response.at[:5, :, :].set(0.1)
        loss = float(eyebox_area_merit(response, config))
        assert 0.3 < loss < 0.7

    def test_one_dark_angle_kills_cell(self):
        """A cell with one dead FOV angle should not be active."""
        config = EyeboxAreaConfig(intensity_threshold=0.005, w_margin=0.0)
        # 1 cell, 5 angles, 3 colors — all bright
        response = jnp.ones((1, 5, 3)) * 0.1
        loss_all = float(eyebox_area_merit(response, config))
        # Kill one angle
        response_dead = response.at[:, 2, :].set(0.0)
        loss_dead = float(eyebox_area_merit(response_dead, config))
        assert loss_dead > loss_all + 0.5  # should jump to ~1.0

    def test_worst_color_matters(self):
        """If one color is below threshold, cell should be inactive."""
        config = EyeboxAreaConfig(intensity_threshold=0.005, w_margin=0.0)
        # All colors strong
        response = jnp.ones((1, 5, 3)) * 0.1
        loss_good = float(eyebox_area_merit(response, config))
        # Kill blue channel (index 2) — after D65 normalization, blue weight
        # is smallest so this is the channel most easily killed
        response_no_blue = response.at[:, :, 2].set(0.0)
        loss_no_blue = float(eyebox_area_merit(response_no_blue, config))
        assert loss_no_blue > loss_good + 0.5

    def test_more_active_cells_is_better(self):
        """Adding intensity to dark cells should decrease loss."""
        config = EyeboxAreaConfig(intensity_threshold=0.005, w_margin=0.0)
        response_3 = jnp.zeros((5, 4, 3))
        response_3 = response_3.at[:3, :, :].set(0.1)  # 3/5 active
        response_5 = jnp.ones((5, 4, 3)) * 0.1          # 5/5 active
        assert float(eyebox_area_merit(response_5, config)) < float(eyebox_area_merit(response_3, config))


class TestEyeboxAreaDifferentiability:

    def test_grad_wrt_reflectances(self, default_setup):
        """Gradient of area merit w.r.t. mirror reflectances."""
        route, pupil = default_setup
        grid = eyebox_grid_points(
            pupil.position, pupil.normal, pupil.width / 2,
            nx=3, ny=3)
        mc = EyeboxConfig(n_fov_x=3, n_fov_y=3)
        area_cfg = EyeboxAreaConfig()

        def loss(reflectances):
            new_mirrors = route.mirrors._replace(reflectance=reflectances)
            r = route._replace(mirrors=new_mirrors)
            response, _ = compute_eyebox_response(
                r,
                DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
                DEFAULT_BEAM_WIDTH, DEFAULT_BEAM_HEIGHT,
                DEFAULT_X_FOV, DEFAULT_Y_FOV,
                grid, mc,
            )
            return eyebox_area_merit(response, area_cfg)

        grads = jax.grad(loss)(route.mirrors.reflectance)
        assert grads.shape == (6, 3)
        assert jnp.all(jnp.isfinite(grads))
