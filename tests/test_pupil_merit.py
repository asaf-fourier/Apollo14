"""Per-term and end-to-end tests for the pupil merit function."""

import jax
import jax.numpy as jnp

from helios.merit import D65_WEIGHTS
from helios.pupil_merit import PupilMeritConfig, merit_breakdown, pupil_merit


def _ideal_response(num_samples=4, num_angles=3, brightness=0.2):
    """``(S, A, 3)`` D65-flat response: same brightness in every cell."""
    per_channel = brightness * D65_WEIGHTS
    return jnp.broadcast_to(per_channel, (num_samples, num_angles, 3))


def _cfg(**kwargs) -> PupilMeritConfig:
    """Config with both weights at 0 by default — caller turns one on."""
    base = dict(weight_shape=0.0, weight_target=0.0)
    base.update(kwargs)
    return PupilMeritConfig(**base)


# ── Shape term ──────────────────────────────────────────────────────────────


class TestShapeTerm:

    def test_zero_for_d65_flat_response(self):
        response = _ideal_response(brightness=0.2)
        cfg = _cfg(weight_shape=1.0)
        assert float(pupil_merit(response, 1.0, cfg)) < 1e-6

    def test_positive_when_channels_not_d65(self):
        ideal = _ideal_response(brightness=0.2)
        broken = ideal[..., [2, 1, 0]]
        cfg = _cfg(weight_shape=1.0)
        assert float(pupil_merit(broken, 1.0, cfg)) > 1e-3

    def test_positive_when_fov_not_flat(self):
        ramp = jnp.linspace(0.05, 0.4, 3)
        per_angle = ramp[:, None] * D65_WEIGHTS
        response = jnp.broadcast_to(per_angle, (4, 3, 3))
        cfg = _cfg(weight_shape=1.0)
        assert float(pupil_merit(response, 1.0, cfg)) > 1e-3


# ── Target term ─────────────────────────────────────────────────────────────


class TestTargetTerm:

    def test_zero_at_target(self):
        # mean_brightness == target_relative * input_flux exactly.
        # _ideal_response uses radiometric brightness (luminance_weights=None
        # ⇒ ones), so brightness = D65_sum * brightness_arg = brightness_arg.
        response = _ideal_response(brightness=0.05)
        cfg = _cfg(weight_target=1.0, target_relative=0.05)
        assert float(pupil_merit(response, 1.0, cfg)) < 1e-8

    def test_positive_below_target(self):
        response = _ideal_response(brightness=0.001)
        cfg = _cfg(weight_target=1.0, target_relative=0.05)
        assert float(pupil_merit(response, 1.0, cfg)) > 1e-6

    def test_positive_above_target(self):
        response = _ideal_response(brightness=0.20)
        cfg = _cfg(weight_target=1.0, target_relative=0.05)
        assert float(pupil_merit(response, 1.0, cfg)) > 1e-3

    def test_symmetric_around_target(self):
        cfg = _cfg(weight_target=1.0, target_relative=0.05)
        below = pupil_merit(_ideal_response(brightness=0.04), 1.0, cfg)
        above = pupil_merit(_ideal_response(brightness=0.06), 1.0, cfg)
        assert abs(float(below) - float(above)) < 1e-8

    def test_grows_quadratically_with_deviation(self):
        cfg = _cfg(weight_target=1.0, target_relative=0.05)
        small = pupil_merit(_ideal_response(brightness=0.04), 1.0, cfg)
        large = pupil_merit(_ideal_response(brightness=0.0), 1.0, cfg)
        # 0.05² is 25× the deviation² of 0.01.
        assert float(large) > 5.0 * float(small)


# ── Combined merit ──────────────────────────────────────────────────────────


class TestCombined:

    def test_ideal_response_near_zero(self):
        # D65-flat *and* exactly at target → both terms ~0.
        response = _ideal_response(brightness=0.05)
        cfg = PupilMeritConfig(target_relative=0.05,
                               weight_shape=1.0, weight_target=1.0)
        assert float(pupil_merit(response, 1.0, cfg)) < 1e-6

    def test_invariant_to_input_flux_scale(self):
        cfg = PupilMeritConfig(target_relative=0.05)
        base = _ideal_response(brightness=0.10)
        loss_unit = float(pupil_merit(base, 1.0, cfg))
        loss_scaled = float(pupil_merit(base * 1000.0, 1000.0, cfg))
        assert abs(loss_unit - loss_scaled) < 1e-4


# ── Cell mask ───────────────────────────────────────────────────────────────


class TestCellMask:

    def test_masked_out_cells_do_not_contribute_to_target(self):
        bright = _ideal_response(num_samples=4, brightness=0.05)
        with_dark = bright.at[3].set(0.0)
        cfg = _cfg(weight_target=1.0, target_relative=0.05)
        mask_all = jnp.ones((4,))
        mask_exclude_dark = jnp.array([1.0, 1.0, 1.0, 0.0])
        loss_all = float(pupil_merit(with_dark, 1.0, cfg, cell_mask=mask_all))
        loss_excl = float(pupil_merit(with_dark, 1.0, cfg,
                                      cell_mask=mask_exclude_dark))
        assert loss_excl < loss_all
        assert loss_excl < 1e-8


# ── Diagnostics ─────────────────────────────────────────────────────────────


class TestBreakdown:

    def test_returns_expected_keys(self):
        response = _ideal_response(brightness=0.10)
        result = merit_breakdown(response, 1.0, PupilMeritConfig())
        expected = {"shape", "target", "total",
                    "min_brightness_rel", "max_brightness_rel",
                    "mean_brightness_rel", "brightness_std_rel"}
        assert expected <= set(result)

    def test_total_matches_pupil_merit(self):
        response = _ideal_response(brightness=0.07)
        cfg = PupilMeritConfig(target_relative=0.05)
        breakdown = merit_breakdown(response, 1.0, cfg)
        scalar = pupil_merit(response, 1.0, cfg)
        assert jnp.allclose(breakdown["total"], scalar, atol=1e-6)


# ── Gradient ────────────────────────────────────────────────────────────────


class TestGradient:

    def test_grad_through_response(self):
        response = _ideal_response(brightness=0.02)
        cfg = PupilMeritConfig(target_relative=0.05)
        grads = jax.grad(lambda r: pupil_merit(r, 1.0, cfg))(response)
        assert grads.shape == response.shape
        assert jnp.any(grads != 0.0)
