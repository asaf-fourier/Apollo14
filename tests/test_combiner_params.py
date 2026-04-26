"""Tests for the parametrized Talos combiner.

Covers the four moving parts: ``CombinerParams.initial`` shape, system
build (positions + reflectance), ``ParamBounds.clip`` clamping/rescaling,
and gradient flow back to params via the GaussianMirror reflectance curve.
"""

import jax
import jax.numpy as jnp

from apollo14.elements.aperture import RectangularAperture
from apollo14.elements.glass_block import GlassBlock
from apollo14.elements.partial_mirror import GaussianMirror
from apollo14.elements.pupil import RectangularPupil
from apollo14.units import mm, nm
from helios.combiner_params import (
    CombinerParams,
    ParamBounds,
    build_parametrized_system,
)
from helios.merit import DEFAULT_WAVELENGTHS

# ── CombinerParams.initial ──────────────────────────────────────────────────


class TestInitial:

    def test_default_shapes(self):
        params = CombinerParams.initial()
        assert params.spacings.shape == (5,)
        assert params.amplitudes.shape == (6, 3)
        assert params.widths.shape == (6, 3)

    def test_custom_num_mirrors(self):
        params = CombinerParams.initial(num_mirrors=4)
        assert params.spacings.shape == (3,)
        assert params.amplitudes.shape == (4, 3)
        assert params.widths.shape == (4, 3)

    def test_uniform_initialization(self):
        params = CombinerParams.initial(
            spacing_mm=2.0, amplitude=0.07, width_nm=30.0)
        assert jnp.allclose(params.spacings, 2.0 * mm)
        assert jnp.allclose(params.amplitudes, 0.07)
        assert jnp.allclose(params.widths, 30.0 * nm)


# ── build_parametrized_system ───────────────────────────────────────────────


class TestSystemBuild:

    def test_element_inventory(self):
        system = build_parametrized_system(CombinerParams.initial())
        # 1 chassis + 1 aperture + 6 mirrors + 1 pupil
        assert len(system.elements) == 9
        assert sum(isinstance(e, GlassBlock) for e in system.elements) == 1
        assert sum(isinstance(e, RectangularAperture)
                   for e in system.elements) == 1
        assert sum(isinstance(e, GaussianMirror)
                   for e in system.elements) == 6
        assert sum(isinstance(e, RectangularPupil)
                   for e in system.elements) == 1

    def test_mirrors_named_in_order(self):
        system = build_parametrized_system(CombinerParams.initial())
        mirrors = [e for e in system.elements if isinstance(e, GaussianMirror)]
        names = [m.name for m in mirrors]
        assert names == [f"mirror_{i}" for i in range(6)]

    def test_mirror_positions_follow_spacings(self):
        """Mirrors should sit at uniformly-spaced positions when given
        uniform spacings. The 3D step magnitude scales by 1/sin(normal-angle)
        relative to the spacing parameter (mirrors are tilted), so we
        compare consecutive differences for equality rather than against
        the spacing value directly."""
        params = CombinerParams.initial(spacing_mm=1.5)
        system = build_parametrized_system(params)
        mirrors = [e for e in system.elements if isinstance(e, GaussianMirror)]
        steps = jnp.array([
            float(jnp.linalg.norm(mirrors[i + 1].position - mirrors[i].position))
            for i in range(len(mirrors) - 1)
        ])
        assert jnp.allclose(steps, steps[0], atol=1e-5)
        assert float(steps[0]) > 0.0

    def test_gaussian_reflectance_peaks_match_amplitude(self):
        """At each Gaussian's center wavelength the off-color contributions
        are nearly zero (narrow widths), so the on-color reflectance is
        essentially the amplitude."""
        params = CombinerParams.initial(amplitude=0.08, width_nm=5.0)
        system = build_parametrized_system(params)
        first_mirror = next(e for e in system.elements
                            if isinstance(e, GaussianMirror))
        # reflectance is sampled at probe_wavelengths == DEFAULT_WAVELENGTHS,
        # which are also the Gaussian centers — peak per color.
        for color_idx in range(3):
            assert abs(float(first_mirror.reflectance[color_idx]) - 0.08) < 1e-3

    def test_custom_wavelengths_used_as_probe_grid(self):
        custom = jnp.array([500.0, 550.0, 600.0]) * nm
        system = build_parametrized_system(
            CombinerParams.initial(), wavelengths=custom)
        first_mirror = next(e for e in system.elements
                            if isinstance(e, GaussianMirror))
        assert jnp.allclose(first_mirror.probe_wavelengths, custom)


# ── ParamBounds ─────────────────────────────────────────────────────────────


class TestParamBounds:

    def test_amplitude_clipped_to_range(self):
        params = CombinerParams(
            spacings=jnp.full((5,), 1.5 * mm),
            amplitudes=jnp.array([[0.001, 0.5, 0.05]] * 6),
            widths=jnp.full((6, 3), 20.0 * nm),
        )
        bounds = ParamBounds()
        clipped = bounds.clip(params)
        # Use a tiny tolerance — float32 round-trip can shave the last bit.
        eps = 1e-7
        assert float(jnp.min(clipped.amplitudes)) >= bounds.amplitude_min - eps
        assert float(jnp.max(clipped.amplitudes)) <= bounds.amplitude_max + eps

    def test_width_clipped_to_range(self):
        params = CombinerParams(
            spacings=jnp.full((5,), 1.5 * mm),
            amplitudes=jnp.full((6, 3), 0.05),
            widths=jnp.array([[5.0 * nm, 200.0 * nm, 50.0 * nm]] * 6),
        )
        bounds = ParamBounds()
        clipped = bounds.clip(params)
        eps_nm = 1e-3 * nm
        assert float(jnp.min(clipped.widths)) >= bounds.width_min_nm * nm - eps_nm
        assert float(jnp.max(clipped.widths)) <= bounds.width_max_nm * nm + eps_nm

    def test_spacing_clipped_to_range(self):
        params = CombinerParams(
            spacings=jnp.array([0.1, 5.0, 1.0, 1.0, 1.0]) * mm,
            amplitudes=jnp.full((6, 3), 0.05),
            widths=jnp.full((6, 3), 20.0 * nm),
        )
        bounds = ParamBounds()
        clipped = bounds.clip(params)
        eps_mm = 1e-5 * mm
        assert float(jnp.min(clipped.spacings)) >= bounds.spacing_min_mm * mm - eps_mm
        # After per-element clip the max is at most spacing_max_mm; subsequent
        # rescale only shrinks, so the max never grows.
        assert float(jnp.max(clipped.spacings)) <= bounds.spacing_max_mm * mm + eps_mm

    def test_total_spacing_rescaled_when_too_long(self):
        """5 spacings × 3 mm = 15 mm fits, but 5 × 4 mm = 20 mm exceeds the
        18 mm usable length. After per-element clip both cases hit 3 mm,
        and only the second triggers the rescale."""
        bounds = ParamBounds(chassis_usable_mm=18.0,
                             spacing_min_mm=0.5, spacing_max_mm=4.0)
        too_long = CombinerParams(
            spacings=jnp.full((5,), 4.0 * mm),
            amplitudes=jnp.full((6, 3), 0.05),
            widths=jnp.full((6, 3), 20.0 * nm),
        )
        clipped = bounds.clip(too_long)
        assert float(jnp.sum(clipped.spacings)) <= bounds.chassis_usable_mm * mm + 1e-6

    def test_no_rescale_when_total_within_budget(self):
        bounds = ParamBounds(chassis_usable_mm=18.0)
        ok = CombinerParams(
            spacings=jnp.full((5,), 1.5 * mm),  # total 7.5 mm
            amplitudes=jnp.full((6, 3), 0.05),
            widths=jnp.full((6, 3), 20.0 * nm),
        )
        clipped = bounds.clip(ok)
        assert jnp.allclose(clipped.spacings, ok.spacings)


# ── Gradient flow ───────────────────────────────────────────────────────────


class TestGradient:

    def test_grad_through_amplitude(self):
        """Grad of a downstream sum-of-reflectance should reach amplitudes."""
        params = CombinerParams.initial()

        def first_mirror_reflectance_sum(p):
            system = build_parametrized_system(p, wavelengths=DEFAULT_WAVELENGTHS)
            first = next(e for e in system.elements
                         if isinstance(e, GaussianMirror))
            return jnp.sum(first.reflectance)

        grads = jax.grad(first_mirror_reflectance_sum)(params)
        assert grads.amplitudes.shape == params.amplitudes.shape
        # Only mirror_0's amplitude row affects mirror_0's reflectance.
        assert jnp.any(grads.amplitudes[0] != 0.0)
        assert jnp.all(grads.amplitudes[1:] == 0.0)

    def test_grad_through_spacing(self):
        """Grad of mirror position separation should reach the spacings."""
        params = CombinerParams.initial()

        def first_to_last_mirror_distance(p):
            system = build_parametrized_system(p, wavelengths=DEFAULT_WAVELENGTHS)
            mirrors = [e for e in system.elements
                       if isinstance(e, GaussianMirror)]
            return jnp.linalg.norm(mirrors[-1].position - mirrors[0].position)

        grads = jax.grad(first_to_last_mirror_distance)(params)
        assert grads.spacings.shape == params.spacings.shape
        assert jnp.all(grads.spacings != 0.0)
