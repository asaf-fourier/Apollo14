"""Tests for parametric spectral reflectance curves."""

import jax
import jax.numpy as jnp
import pytest

from apollo14.spectral import ConstantCurve, SpectralTable
from apollo14.units import nm


class TestSpectralTable:

    def test_scalar_value_builds_a_flat_table_and_interpolates(self):
        table = SpectralTable.constant(
            0.005, jnp.array([400.0, 700.0]) * nm)
        assert table.values.shape == (2,)
        assert jnp.isclose(table.sample(550.0 * nm), 0.005)

    def test_rejects_mismatched_samples(self):
        with pytest.raises(ValueError, match="must match"):
            SpectralTable.from_samples(
                jnp.array([400.0, 500.0, 600.0]) * nm,
                jnp.array([1.0, 2.0]))

    def test_is_jit_and_gradient_compatible(self):
        wavelengths = jnp.array([400.0, 700.0]) * nm

        @jax.jit
        def sampled_sum(values):
            table = SpectralTable.from_samples(wavelengths, values)
            return table.sample(550.0 * nm)

        values = jnp.array([0.0, 1.0])
        assert jnp.isclose(sampled_sum(values), 0.5)
        assert jnp.allclose(jax.grad(sampled_sum)(values), 0.5)


# ── ConstantCurve (wavelength-flat reflectance) ─────────────────────────────


class TestConstantCurve:

    def test_single_curve_sample_is_flat(self):
        curve = ConstantCurve.uniform(amplitude=0.05)
        wavelengths = jnp.linspace(450, 650, 11) * nm
        sampled = curve.sample(wavelengths)
        # One scalar curve → (K,), every value equal to the amplitude.
        assert sampled.shape == (11,)
        assert jnp.allclose(sampled, 0.05)

    def test_batched_sample_shape(self):
        curve = ConstantCurve.uniform(amplitude=0.05, num_mirrors=10)
        assert curve.amplitude.shape == (10,)
        wavelengths = jnp.linspace(450, 650, 7) * nm
        sampled = curve.sample(wavelengths)
        # (M,) curves × (K,) grid → (M, K).
        assert sampled.shape == (10, 7)
        assert jnp.allclose(sampled, 0.05)

    def test_per_mirror_amplitudes_preserved(self):
        amplitudes = jnp.array([0.05, 0.06, 0.07, 0.08])
        curve = ConstantCurve(amplitude=amplitudes)
        wavelengths = jnp.linspace(450, 650, 5) * nm
        sampled = curve.sample(wavelengths)
        assert sampled.shape == (4, 5)
        # Each row is flat at its own mirror's amplitude.
        for mirror_idx in range(4):
            assert jnp.allclose(sampled[mirror_idx], amplitudes[mirror_idx])

    def test_at_indexes_single_mirror(self):
        curve = ConstantCurve(amplitude=jnp.array([0.05, 0.06, 0.07]))
        one = curve.at(1)
        assert one.amplitude.shape == ()
        assert jnp.isclose(one.amplitude, 0.06)
        wavelengths = jnp.linspace(450, 650, 4) * nm
        assert jnp.allclose(one.sample(wavelengths), 0.06)

    def test_gradient_flows_to_amplitude(self):
        wavelengths = jnp.linspace(450, 650, 6) * nm

        def total_reflectance(amplitude):
            curve = ConstantCurve(amplitude=amplitude)
            return jnp.sum(curve.sample(wavelengths))

        amplitude = jnp.array(0.05)
        grad = jax.grad(total_reflectance)(amplitude)
        # d/da Σ_k a = K (one per sampled wavelength).
        assert jnp.isclose(grad, wavelengths.shape[0])
