"""Per-term and end-to-end tests for the pupil merit function.

Each term is exercised in isolation by zeroing out the other weights, so a
regression in one term shows up as a focused failure rather than a noisy
total-loss diff.
"""

import jax
import jax.numpy as jnp

from helios.merit import D65_WEIGHTS
from helios.pupil_merit import PupilMeritConfig, merit_breakdown, pupil_merit

# ── Fixtures ────────────────────────────────────────────────────────────────


def _ideal_response(num_samples=4, num_angles=3, brightness=0.2):
    """Build an ``(S, A, 3)`` response that is D65-flat at every cell.

    Each cell has identical brightness across angles; channels are split by
    ``D65_WEIGHTS``. With ``input_flux=1.0``, ``brightness > threshold_relative``
    means every cell is above threshold.
    """
    per_channel = brightness * D65_WEIGHTS  # (3,)
    return jnp.broadcast_to(per_channel, (num_samples, num_angles, 3))


def _flat_white_balanced_config(**kwargs) -> PupilMeritConfig:
    """PupilMeritConfig with all weights at 0 by default — caller sets one."""
    base = dict(weight_shape=0.0, weight_coverage=0.0,
                weight_warmup=0.0, weight_cap=0.0)
    base.update(kwargs)
    return PupilMeritConfig(**base)


# ── Shape term ──────────────────────────────────────────────────────────────


class TestShapeTerm:

    def test_zero_for_d65_flat_response(self):
        response = _ideal_response(brightness=0.2)
        cfg = _flat_white_balanced_config(weight_shape=1.0)
        loss = pupil_merit(response, input_flux=1.0, config=cfg)
        assert float(loss) < 1e-6

    def test_positive_when_channels_not_d65(self):
        ideal = _ideal_response(brightness=0.2)
        # Swap red and blue channels — same total brightness, wrong ratio.
        broken = ideal[..., [2, 1, 0]]
        cfg = _flat_white_balanced_config(weight_shape=1.0)
        assert float(pupil_merit(broken, 1.0, cfg)) > 1e-3

    def test_positive_when_fov_not_flat(self):
        """Same channel ratios per angle but different magnitudes per angle."""
        ramp = jnp.linspace(0.05, 0.4, 3)               # (A,)
        per_angle = ramp[:, None] * D65_WEIGHTS         # (A, 3)
        response = jnp.broadcast_to(per_angle, (4, 3, 3))
        cfg = _flat_white_balanced_config(weight_shape=1.0)
        assert float(pupil_merit(response, 1.0, cfg)) > 1e-3


# ── Coverage term ───────────────────────────────────────────────────────────


class TestCoverageTerm:

    def test_zero_when_all_cells_well_above_threshold(self):
        response = _ideal_response(brightness=0.5)
        cfg = _flat_white_balanced_config(
            weight_coverage=1.0, threshold_relative=0.05)
        assert float(pupil_merit(response, 1.0, cfg)) < 1e-3

    def test_one_when_all_cells_dark(self):
        response = jnp.zeros((4, 3, 3))
        cfg = _flat_white_balanced_config(
            weight_coverage=1.0, threshold_relative=0.05)
        loss = float(pupil_merit(response, 1.0, cfg))
        assert 0.9 < loss <= 1.0

    def test_softmin_dominated_by_weakest_cell(self):
        """Soft-min should focus on the worst cell, not the mean."""
        bright = _ideal_response(num_samples=3, brightness=0.5)
        with_dark = bright.at[0].set(0.0)  # one dark cell
        cfg = _flat_white_balanced_config(
            weight_coverage=1.0, threshold_relative=0.05)
        loss_bright = float(pupil_merit(bright, 1.0, cfg))
        loss_with_dark = float(pupil_merit(with_dark, 1.0, cfg))
        assert loss_with_dark > loss_bright + 0.1


# ── Warm-up term ────────────────────────────────────────────────────────────


class TestWarmupTerm:

    def test_zero_when_above_threshold(self):
        response = _ideal_response(brightness=0.5)
        cfg = _flat_white_balanced_config(
            weight_warmup=1.0, threshold_relative=0.05)
        assert float(pupil_merit(response, 1.0, cfg)) < 1e-6

    def test_positive_when_below_threshold(self):
        response = _ideal_response(brightness=0.001)
        cfg = _flat_white_balanced_config(
            weight_warmup=1.0, threshold_relative=0.05)
        assert float(pupil_merit(response, 1.0, cfg)) > 1e-6

    def test_grows_quadratically_with_shortfall(self):
        cfg = _flat_white_balanced_config(
            weight_warmup=1.0, threshold_relative=0.05)
        small = pupil_merit(_ideal_response(brightness=0.04), 1.0, cfg)
        large = pupil_merit(_ideal_response(brightness=0.0), 1.0, cfg)
        # 0.05² ≈ 25× the shortfall² of 0.01 — so large should be ~25× small.
        assert float(large) > 5.0 * float(small)


# ── Cap term ────────────────────────────────────────────────────────────────


class TestCapTerm:

    def test_zero_below_cap(self):
        response = _ideal_response(brightness=0.05)
        cfg = _flat_white_balanced_config(weight_cap=1.0, cap_relative=0.15)
        assert float(pupil_merit(response, 1.0, cfg)) < 1e-6

    def test_positive_above_cap(self):
        response = _ideal_response(brightness=0.30)
        cfg = _flat_white_balanced_config(weight_cap=1.0, cap_relative=0.15)
        assert float(pupil_merit(response, 1.0, cfg)) > 1e-3

    def test_disabled_when_cap_relative_none(self):
        response = _ideal_response(brightness=10.0)
        cfg = _flat_white_balanced_config(weight_cap=1.0, cap_relative=None)
        assert float(pupil_merit(response, 1.0, cfg)) == 0.0


# ── Combined merit ──────────────────────────────────────────────────────────


class TestCombined:

    def test_ideal_response_near_zero(self):
        """D65-flat, well above threshold, below cap → all four terms ~0.

        Brightness needs to be far enough above ``threshold_relative`` for
        the sigmoid (steepness 50) to saturate near 1; otherwise a small
        residual coverage loss remains.
        """
        response = _ideal_response(brightness=0.18)
        cfg = PupilMeritConfig(
            threshold_relative=0.05, cap_relative=0.30,
            weight_shape=1.0, weight_coverage=1.0,
            weight_warmup=1.0, weight_cap=1.0)
        assert float(pupil_merit(response, 1.0, cfg)) < 1e-2

    def test_invariant_to_input_flux_scale(self):
        """Scaling response and flux by the same factor must leave the loss
        unchanged — that's the whole point of relative thresholds."""
        cfg = PupilMeritConfig(threshold_relative=0.05, cap_relative=0.20)
        base = _ideal_response(brightness=0.10)
        loss_unit = float(pupil_merit(base, 1.0, cfg))
        loss_scaled = float(pupil_merit(base * 1000.0, 1000.0, cfg))
        assert abs(loss_unit - loss_scaled) < 1e-4


# ── Cell mask ───────────────────────────────────────────────────────────────


class TestCellMask:

    def test_masked_out_cells_do_not_contribute_to_warmup(self):
        """A dark cell that's masked out should not raise the warm-up term."""
        bright = _ideal_response(num_samples=4, brightness=0.5)
        with_dark = bright.at[3].set(0.0)
        cfg = _flat_white_balanced_config(
            weight_warmup=1.0, threshold_relative=0.05)

        mask_all = jnp.ones((4,))
        mask_exclude_dark = jnp.array([1.0, 1.0, 1.0, 0.0])

        loss_all = float(pupil_merit(with_dark, 1.0, cfg, cell_mask=mask_all))
        loss_excl = float(pupil_merit(with_dark, 1.0, cfg,
                                      cell_mask=mask_exclude_dark))
        assert loss_excl < loss_all
        assert loss_excl < 1e-6


# ── Diagnostics ─────────────────────────────────────────────────────────────


class TestBreakdown:

    def test_returns_expected_keys(self):
        response = _ideal_response(brightness=0.10)
        cfg = PupilMeritConfig(cap_relative=0.20, weight_cap=1.0)
        result = merit_breakdown(response, 1.0, cfg)
        expected = {"shape", "coverage", "warmup", "cap", "total",
                    "active_fraction", "min_brightness_rel"}
        assert expected <= set(result)

    def test_total_matches_pupil_merit(self):
        response = _ideal_response(brightness=0.07)
        cfg = PupilMeritConfig(cap_relative=0.20, weight_cap=1.0)
        breakdown = merit_breakdown(response, 1.0, cfg)
        scalar = pupil_merit(response, 1.0, cfg)
        assert jnp.allclose(breakdown["total"], scalar, atol=1e-6)

    def test_active_fraction_in_unit_interval(self):
        response = _ideal_response(brightness=0.10)
        result = merit_breakdown(response, 1.0)
        assert 0.0 <= float(result["active_fraction"]) <= 1.0


# ── Gradient ────────────────────────────────────────────────────────────────


class TestGradient:

    def test_grad_through_response(self):
        """The merit must be differentiable through the response tensor."""
        response = _ideal_response(brightness=0.02)  # below threshold
        cfg = PupilMeritConfig(threshold_relative=0.05)
        grads = jax.grad(lambda r: pupil_merit(r, 1.0, cfg))(response)
        assert grads.shape == response.shape
        # At least one entry should be non-zero — the loss responds to changes.
        assert jnp.any(grads != 0.0)
