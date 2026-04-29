"""Tests for the CIE 1931 photopic V(λ) and the photopic-brightness
branch of :func:`pupil_merit`.

Two layers:

1. ``photopic_v`` and helpers — known spot-checks against the CIE table.
2. ``pupil_merit`` with ``luminance_weights`` set — the brightness scale
   should weight green ~32× higher than blue per watt, and the shape
   term should keep using radiance regardless.
"""

import jax
import jax.numpy as jnp

from apollo14.units import nm
from helios.merit import D65_WEIGHTS
from helios.photometry import (
    K_M,
    luminance_weights,
    photopic_v,
    radiance_to_luminance,
)
from helios.pupil_merit import PupilMeritConfig, pupil_merit

# ── V(λ) lookups ────────────────────────────────────────────────────────────


class TestPhotopicV:

    def test_peak_at_555_nm(self):
        assert jnp.isclose(photopic_v(555.0 * nm), 1.0)

    def test_red_at_627_nm(self):
        # CIE 1931 V(627) ≈ 0.299
        assert abs(float(photopic_v(627.0 * nm)) - 0.299) < 1e-3

    def test_green_at_545_nm(self):
        assert abs(float(photopic_v(545.0 * nm)) - 0.980) < 1e-3

    def test_blue_at_446_nm(self):
        assert abs(float(photopic_v(446.0 * nm)) - 0.031) < 1e-3

    def test_far_tails_essentially_zero(self):
        for wl_nm in (380.0, 780.0):
            assert float(photopic_v(wl_nm * nm)) < 1e-3

    def test_vectorized(self):
        wls = jnp.array([446.0, 545.0, 627.0]) * nm
        v = photopic_v(wls)
        assert v.shape == (3,)
        assert v[0] < v[1]            # blue < green
        assert v[2] < v[1]            # red < green
        assert v[0] < v[2]            # blue < red (per CIE)


# ── luminance_weights ───────────────────────────────────────────────────────


class TestLuminanceWeights:

    def test_includes_k_m_and_delta(self):
        wls = jnp.array([540.0, 545.0, 550.0]) * nm  # 5 nm spacing
        weights = luminance_weights(wls)
        # weights[1] = K_m × V(545) × 5 ≈ 683 × 0.98 × 5 ≈ 3346
        assert abs(float(weights[1]) - K_M * 0.98 * 5.0) < 5.0

    def test_explicit_delta_overrides(self):
        wls = jnp.array([555.0]) * nm
        weights = luminance_weights(wls, delta_nm=10.0)
        # K_m × V(555) × 10 = 683 × 1.0 × 10 = 6830
        assert abs(float(weights[0]) - 6830.0) < 1.0

    def test_single_wavelength_without_delta_raises(self):
        try:
            luminance_weights(jnp.array([555.0]) * nm)
        except ValueError:
            return
        raise AssertionError("expected ValueError for single-λ without delta")


# ── radiance_to_luminance ──────────────────────────────────────────────────


class TestRadianceToLuminance:

    def test_monochromatic_555_nm_matches_k_m(self):
        """A 1 W/sr/m²/nm monochromatic source at 555 nm with 1 nm bandwidth
        should give K_M nits."""
        wls = jnp.array([554.0, 555.0, 556.0]) * nm
        radiance = jnp.array([0.0, 1.0, 0.0])
        luminance = radiance_to_luminance(radiance, wls)
        # Riemann sum: only the middle sample contributes K_M × V(555) × 1 nm
        assert abs(float(luminance) - K_M) < 1.0

    def test_blue_dimmer_than_green_for_equal_radiance(self):
        # For equal radiance at blue vs green, the luminance contribution
        # ratio is V(545)/V(446) ≈ 0.98 / 0.031 ≈ 32×.
        wls = jnp.array([446.0, 545.0]) * nm
        weights = luminance_weights(wls, delta_nm=1.0)
        ratio = float(weights[1] / weights[0])
        assert 25.0 < ratio < 40.0


# ── Photopic pupil_merit ────────────────────────────────────────────────────


def _response_at_wavelengths(wavelengths: jnp.ndarray, brightness: float):
    """``(S=4, A=3, K=len(wls))`` D65-flat response at the given wavelengths."""
    from helios.merit import d65_weights_at
    d65 = d65_weights_at(wavelengths)
    per_channel = brightness * d65
    return jnp.broadcast_to(per_channel, (4, 3, wavelengths.shape[0]))


class TestPhotopicMerit:

    def test_back_compat_when_luminance_weights_none(self):
        """``luminance_weights=None`` should match an explicit ones vector."""
        wls = jnp.array([446.0, 545.0, 627.0]) * nm
        response = _response_at_wavelengths(wls, brightness=0.20)
        cfg_radiometric = PupilMeritConfig(
            d65_weights=D65_WEIGHTS, luminance_weights=None,
            target_relative=0.05,
        )
        cfg_explicit_ones = PupilMeritConfig(
            d65_weights=D65_WEIGHTS, luminance_weights=jnp.ones((3,)),
            target_relative=0.05,
        )
        loss_a = float(pupil_merit(response, 1.0, cfg_radiometric))
        loss_b = float(pupil_merit(response, 1.0, cfg_explicit_ones))
        assert abs(loss_a - loss_b) < 1e-6

    def test_target_term_uses_luminance(self):
        """A green-only response is much brighter (per watt) than a
        blue-only one; with target = green's luminance, green sits at
        target, blue is far below, and the photometric merit reflects
        that asymmetry."""
        wls = jnp.array([446.0, 545.0, 627.0]) * nm
        v_weights = luminance_weights(wls, delta_nm=1.0)

        blue_only = jnp.zeros((4, 3, 3)).at[..., 0].set(1.0)
        green_only = jnp.zeros((4, 3, 3)).at[..., 1].set(1.0)

        # Pick input_flux = green's luminance so target_relative = 1.0
        # places the target exactly on green. Green hits it; blue (~3%
        # as visible per watt) sits way below.
        green_luminance = float(v_weights[1])
        cfg = PupilMeritConfig(
            d65_weights=D65_WEIGHTS, luminance_weights=v_weights,
            target_relative=1.0,
            weight_shape=0.0, weight_target=1.0,
        )
        loss_blue = float(pupil_merit(blue_only, green_luminance, cfg))
        loss_green = float(pupil_merit(green_only, green_luminance, cfg))
        assert loss_green < 1e-6
        assert loss_blue > 0.5

    def test_shape_term_unchanged_by_luminance_weights(self):
        """Shape error is in radiance space — luminance weights don't enter."""
        wls = jnp.array([446.0, 545.0, 627.0]) * nm
        ideal = _response_at_wavelengths(wls, brightness=0.5)
        wrong = ideal[..., ::-1]

        cfg_radio = PupilMeritConfig(
            luminance_weights=None,
            weight_shape=1.0, weight_target=0.0,
        )
        cfg_photo = PupilMeritConfig(
            luminance_weights=luminance_weights(wls, delta_nm=1.0),
            weight_shape=1.0, weight_target=0.0,
        )
        loss_radio = float(pupil_merit(wrong, 1.0, cfg_radio))
        loss_photo = float(pupil_merit(wrong, 1.0, cfg_photo))
        assert abs(loss_radio - loss_photo) < 1e-6

    def test_grad_through_response_with_luminance_weights(self):
        wls = jnp.array([446.0, 545.0, 627.0]) * nm
        response = _response_at_wavelengths(wls, brightness=0.02)
        cfg = PupilMeritConfig(
            luminance_weights=luminance_weights(wls, delta_nm=1.0),
            target_relative=0.5,
        )
        grads = jax.grad(lambda r: pupil_merit(r, 1.0, cfg))(response)
        assert grads.shape == response.shape
        assert jnp.any(grads != 0.0)
