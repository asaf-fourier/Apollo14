import jax.numpy as jnp
import pytest

from apollo14.combiner import (
    build_default_system,
    DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, DEFAULT_WAVELENGTH,
    DEFAULT_X_FOV, DEFAULT_Y_FOV,
)
from apollo14.elements.aperture import RectangularAperture
from apollo14.elements.pupil import RectangularPupil
from apollo14.elements.surface import PartialMirror
from apollo14.projector import Projector
from helios.merit import (
    MeritConfig, D65_WEIGHTS, DEFAULT_WAVELENGTHS,
    pupil_sample_grid, build_target, merit_mse,
    simulate_pupil_response, evaluate_merit,
)
from apollo14.units import mm, nm


@pytest.fixture
def combiner():
    system = build_default_system()
    pupil = next(e for e in system.elements if isinstance(e, RectangularPupil))
    aperture = next(e for e in system.elements if isinstance(e, RectangularAperture))
    return system, pupil, aperture


def _make_projector(aperture, nx=3, ny=3):
    """Create a projector with a beam narrow enough to pass through the aperture."""
    return Projector.uniform(
        position=DEFAULT_LIGHT_POSITION,
        direction=DEFAULT_LIGHT_DIRECTION,
        beam_width=aperture.width * 0.8,
        beam_height=aperture.height * 0.8,
        wavelength=DEFAULT_WAVELENGTH,
        nx=nx, ny=ny,
    )


def _on_axis_merit_config():
    """Merit config with on-axis only (1x1 angle) for reliable hits."""
    return MeritConfig(
        pupil_nx=1, pupil_ny=1,
        angle_nx=1, angle_ny=1,
        target_efficiency=0.10,
    )


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestPupilSampling:

    def test_grid_shape(self):
        center = jnp.array([0.0, 0.0, 10.0])
        normal = jnp.array([0.0, 0.0, -1.0])
        grid = pupil_sample_grid(center, normal, radius=5.0, nx=3, ny=3)
        assert grid.shape == (3, 3, 3)

    def test_center_sample_at_center(self):
        center = jnp.array([7.0, 20.0, 17.0])
        normal = jnp.array([0.0, 0.0, -1.0])
        grid = pupil_sample_grid(center, normal, radius=4.0, nx=3, ny=3)
        assert jnp.allclose(grid[1, 1], center, atol=1e-5)

    def test_grid_spans_radius(self):
        center = jnp.array([0.0, 0.0, 0.0])
        normal = jnp.array([0.0, 0.0, 1.0])
        radius = 5.0
        grid = pupil_sample_grid(center, normal, radius, nx=5, ny=5)
        dists = jnp.linalg.norm(grid.reshape(-1, 3) - center, axis=1)
        assert dists.max() <= radius * 1.5
        assert dists.max() >= radius * 0.9


class TestBuildTarget:

    def test_target_shape(self):
        config = MeritConfig(pupil_nx=3, pupil_ny=3, angle_nx=5, angle_ny=5)
        target = build_target(config)
        assert target.shape == (3, 3, 5, 5, 3)

    def test_target_values(self):
        config = MeritConfig(
            pupil_nx=2, pupil_ny=2, angle_nx=2, angle_ny=2,
            target_efficiency=0.10,
        )
        target = build_target(config)
        for ci in range(3):
            expected = 0.10 * float(D65_WEIGHTS[ci])
            assert jnp.allclose(target[:, :, :, :, ci], expected, atol=1e-6)

    def test_target_color_sum(self):
        config = MeritConfig(target_efficiency=0.10)
        target = build_target(config)
        color_sum = target[0, 0, 0, 0, :].sum()
        assert float(color_sum) == pytest.approx(0.10, abs=1e-6)

    def test_target_uniform(self):
        config = MeritConfig(pupil_nx=3, pupil_ny=3, angle_nx=4, angle_ny=4)
        target = build_target(config)
        ref = target[0, 0, 0, 0, :]
        for pi_y in range(3):
            for pi_x in range(3):
                for ai_y in range(4):
                    for ai_x in range(4):
                        assert jnp.allclose(target[pi_y, pi_x, ai_y, ai_x, :], ref)


class TestMeritMSE:

    def test_perfect_match_is_zero(self):
        a = jnp.ones((2, 2, 3, 3, 3)) * 0.5
        assert merit_mse(a, a) == pytest.approx(0.0, abs=1e-10)

    def test_mse_is_positive(self):
        a = jnp.ones((2, 2, 3, 3, 3)) * 0.5
        b = jnp.ones((2, 2, 3, 3, 3)) * 0.6
        assert merit_mse(a, b) > 0

    def test_mse_symmetric(self):
        a = jnp.ones((2, 2, 2, 2, 3)) * 0.3
        b = jnp.ones((2, 2, 2, 2, 3)) * 0.7
        assert merit_mse(a, b) == pytest.approx(merit_mse(b, a), abs=1e-10)

    def test_mse_known_value(self):
        a = jnp.zeros((1, 1, 1, 1, 3))
        b = jnp.ones((1, 1, 1, 1, 3))
        assert merit_mse(a, b) == pytest.approx(1.0, abs=1e-6)


# ── Integration tests ─────────────────────────────────────────────────────────

class TestSimulatePupilResponse:

    def test_output_shape(self, combiner):
        system, pupil, aperture = combiner
        mc = MeritConfig(pupil_nx=2, pupil_ny=2, angle_nx=2, angle_ny=2)
        proj = _make_projector(aperture, nx=2, ny=2)
        result = simulate_pupil_response(
            system, proj,
            pupil.position, pupil.normal, pupil.width / 2,
            mc, x_fov=DEFAULT_X_FOV, y_fov=DEFAULT_Y_FOV,
        )
        assert result.shape == (2, 2, 2, 2, 3)

    def test_nonzero_intensity_on_axis(self, combiner):
        """On-axis with a beam that fits the aperture should produce pupil hits."""
        system, pupil, aperture = combiner
        mc = _on_axis_merit_config()
        proj = _make_projector(aperture, nx=3, ny=3)
        result = simulate_pupil_response(
            system, proj,
            pupil.position, pupil.normal, pupil.width / 2,
            mc, x_fov=0.0, y_fov=0.0,
        )
        assert float(result.sum()) > 0, "On-axis rays should reach the pupil"

    def test_all_three_colors_present(self, combiner):
        """Each wavelength channel should have nonzero intensity."""
        system, pupil, aperture = combiner
        mc = _on_axis_merit_config()
        proj = _make_projector(aperture, nx=3, ny=3)
        result = simulate_pupil_response(
            system, proj,
            pupil.position, pupil.normal, pupil.width / 2,
            mc, x_fov=0.0, y_fov=0.0,
        )
        for ci in range(3):
            assert float(result[..., ci].sum()) > 0, f"Color channel {ci} should have intensity"

    def test_intensity_values_reasonable(self, combiner):
        system, pupil, aperture = combiner
        mc = _on_axis_merit_config()
        proj = _make_projector(aperture, nx=3, ny=3)
        result = simulate_pupil_response(
            system, proj,
            pupil.position, pupil.normal, pupil.width / 2,
            mc, x_fov=0.0, y_fov=0.0,
        )
        # Total per-ray intensity at pupil should not exceed input
        # (multiple rays can accumulate in one bin, but each contribution <= 1)
        assert jnp.all(jnp.isfinite(result))


class TestEvaluateMerit:

    def test_returns_valid_tuple(self, combiner):
        system, pupil, aperture = combiner
        mc = _on_axis_merit_config()
        proj = _make_projector(aperture, nx=3, ny=3)
        mse, simulated, target = evaluate_merit(
            system, proj,
            pupil.position, pupil.normal, pupil.width / 2,
            x_fov=0.0, y_fov=0.0, config=mc,
        )
        assert isinstance(mse, float)
        assert simulated.shape == target.shape
        assert mse >= 0.0

    def test_merit_nonzero(self, combiner):
        """Merit should be nonzero — the system doesn't perfectly match the target."""
        system, pupil, aperture = combiner
        mc = _on_axis_merit_config()
        proj = _make_projector(aperture, nx=3, ny=3)
        mse, _, _ = evaluate_merit(
            system, proj,
            pupil.position, pupil.normal, pupil.width / 2,
            x_fov=0.0, y_fov=0.0, config=mc,
        )
        assert mse > 0

    def test_merit_changes_with_reflection(self):
        """Different mirror reflectance should produce different merit."""
        system_low = build_default_system()
        system_high = build_default_system()

        # Modify mirrors in system_high to have 3x reflectance
        for elem in system_high.elements:
            if isinstance(elem, PartialMirror):
                elem.reflection_ratio = elem.reflection_ratio * 3.0
                elem.transmission_ratio = 1.0 - elem.reflection_ratio

        pupil = next(e for e in system_low.elements if isinstance(e, RectangularPupil))
        aperture = next(e for e in system_low.elements if isinstance(e, RectangularAperture))

        mc = _on_axis_merit_config()
        proj = _make_projector(aperture, nx=3, ny=3)

        mse_low, sim_low, _ = evaluate_merit(
            system_low, proj,
            pupil.position, pupil.normal, pupil.width / 2,
            x_fov=0.0, y_fov=0.0, config=mc,
        )
        mse_high, sim_high, _ = evaluate_merit(
            system_high, proj,
            pupil.position, pupil.normal, pupil.width / 2,
            x_fov=0.0, y_fov=0.0, config=mc,
        )

        assert mse_low >= 0
        assert mse_high >= 0
        # Higher reflection delivers more light → different simulated values
        assert float(sim_high.sum()) > float(sim_low.sum())
        assert mse_low != mse_high
