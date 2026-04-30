"""Tests for the parametrized Talos combiner.

Covers the four moving parts: ``CombinerParams.initial`` shape, system
build (positions + reflectance), ``ParamBounds.clip`` clamping/rescaling,
and gradient flow back to params via the curve's reflectance samples.
"""

import jax
import jax.numpy as jnp

from apollo14.elements.aperture import RectangularAperture
from apollo14.elements.glass_block import GlassBlock
from apollo14.elements.partial_mirror import PartialMirror
from apollo14.elements.pupil import RectangularPupil
from apollo14.spectral import SumOfGaussiansCurve
from apollo14.units import mm, nm
from helios.combiner_params import (
    CombinerParams,
    ParamBounds,
    build_parametrized_system,
    fwhm_to_sigma,
)
from helios.merit import DEFAULT_WAVELENGTHS

# ── CombinerParams.initial ──────────────────────────────────────────────────


class TestInitial:

    def test_default_shapes(self):
        params = CombinerParams.initial()
        assert params.spacings.shape == (5,)
        assert params.curves.amplitude.shape == (6, 3)
        assert params.curves.sigma.shape == (6, 3)
        assert params.curves.centers.shape == (6, 3)

    def test_custom_num_mirrors(self):
        params = CombinerParams.initial(num_mirrors=4)
        assert params.spacings.shape == (3,)
        assert params.curves.amplitude.shape == (4, 3)
        assert params.curves.sigma.shape == (4, 3)

    def test_uniform_initialization(self):
        params = CombinerParams.initial(
            spacing_mm=2.0, amplitude=0.07, width_nm=30.0)
        assert jnp.allclose(params.spacings, 2.0 * mm)
        assert jnp.allclose(params.curves.amplitude, 0.07)
        assert jnp.allclose(params.curves.sigma, 30.0 * nm)

    def test_custom_basis_size(self):
        """Centers of arbitrary length B configure a B-basis curve."""
        custom_centers = jnp.linspace(420.0, 680.0, 5) * nm
        params = CombinerParams.initial(centers=custom_centers)
        assert params.curves.amplitude.shape == (6, 5)
        assert params.curves.sigma.shape == (6, 5)
        assert jnp.allclose(params.curves.centers[0], custom_centers)


# ── build_parametrized_system ───────────────────────────────────────────────


class TestSystemBuild:

    def test_element_inventory(self):
        system = build_parametrized_system(CombinerParams.initial())
        # 1 chassis + 1 aperture + 6 mirrors + 1 pupil
        assert len(system.elements) == 9
        assert sum(isinstance(e, GlassBlock) for e in system.elements) == 1
        assert sum(isinstance(e, RectangularAperture)
                   for e in system.elements) == 1
        assert sum(isinstance(e, PartialMirror)
                   for e in system.elements) == 6
        assert sum(isinstance(e, RectangularPupil)
                   for e in system.elements) == 1

    def test_mirrors_named_in_order(self):
        system = build_parametrized_system(CombinerParams.initial())
        mirrors = [e for e in system.elements if isinstance(e, PartialMirror)]
        names = [m.name for m in mirrors]
        assert names == [f"mirror_{i}" for i in range(6)]

    def test_each_mirror_has_a_curve(self):
        system = build_parametrized_system(CombinerParams.initial())
        mirrors = [e for e in system.elements if isinstance(e, PartialMirror)]
        for m in mirrors:
            assert isinstance(m.curve, SumOfGaussiansCurve)

    def test_mirror_positions_follow_spacings(self):
        """Mirrors should sit at uniformly-spaced positions when given
        uniform spacings. The 3D step magnitude scales by 1/sin(normal-angle)
        relative to the spacing parameter (mirrors are tilted), so we
        compare consecutive differences for equality rather than against
        the spacing value directly."""
        params = CombinerParams.initial(spacing_mm=1.5)
        system = build_parametrized_system(params)
        mirrors = [e for e in system.elements if isinstance(e, PartialMirror)]
        steps = jnp.array([
            float(jnp.linalg.norm(mirrors[i + 1].position - mirrors[i].position))
            for i in range(len(mirrors) - 1)
        ])
        assert jnp.allclose(steps, steps[0], atol=1e-5)
        assert float(steps[0]) > 0.0

    def test_gaussian_reflectance_peaks_match_amplitude(self):
        """At each Gaussian's center wavelength the off-basis contributions
        are nearly zero (narrow widths), so the on-basis reflectance is
        essentially the amplitude."""
        params = CombinerParams.initial(amplitude=0.08, width_nm=5.0)
        system = build_parametrized_system(params)
        first_mirror = next(e for e in system.elements
                            if isinstance(e, PartialMirror))
        # reflectance is sampled at probe_wavelengths == DEFAULT_WAVELENGTHS,
        # which are also the Gaussian centers — peak per basis bump.
        for basis_idx in range(3):
            assert abs(float(first_mirror.reflectance[basis_idx]) - 0.08) < 1e-3

    def test_custom_centers_used(self):
        custom = jnp.array([500.0, 550.0, 600.0]) * nm
        params = CombinerParams.initial(centers=custom)
        system = build_parametrized_system(params)
        first_mirror = next(e for e in system.elements
                            if isinstance(e, PartialMirror))
        assert jnp.allclose(first_mirror.curve.centers, custom)

    def test_custom_probe_wavelengths_used(self):
        dense = jnp.linspace(400.0, 700.0, 64) * nm
        system = build_parametrized_system(
            CombinerParams.initial(), probe_wavelengths=dense)
        first_mirror = next(e for e in system.elements
                            if isinstance(e, PartialMirror))
        assert jnp.allclose(first_mirror.wavelengths, dense)
        assert first_mirror.reflectance.shape == (64,)

    def test_dense_curve_traces_three_gaussian_peaks(self):
        """With a dense probe grid, the stored reflectance should be a
        smooth sum of three Gaussians: peaks at the centers (matching
        amplitudes), troughs between centers (small at narrow σ)."""
        dense = jnp.linspace(400.0, 700.0, 301) * nm   # 1 nm spacing
        params = CombinerParams.initial(amplitude=0.10, width_nm=10.0)
        system = build_parametrized_system(
            params, probe_wavelengths=dense)
        mirror = next(e for e in system.elements
                      if isinstance(e, PartialMirror))
        # Indices closest to each Gaussian center
        for basis_idx in range(3):
            center = mirror.curve.centers[basis_idx]
            i = int(jnp.argmin(jnp.abs(dense - center)))
            assert abs(float(mirror.reflectance[i]) - 0.10) < 5e-3
        # Midpoint between blue (446) and green (545) ≈ 495 nm should be
        # near zero with σ=10 nm (peaks don't overlap there).
        i_mid = int(jnp.argmin(jnp.abs(dense - 495.5 * nm)))
        assert float(mirror.reflectance[i_mid]) < 0.01

    def test_wider_sigma_makes_peaks_overlap(self):
        """σ=150 nm should make adjacent Gaussians overlap heavily —
        the midpoint is no longer near zero."""
        dense = jnp.linspace(400.0, 700.0, 301) * nm
        params_narrow = CombinerParams.initial(amplitude=0.10, width_nm=10.0)
        params_wide = CombinerParams.initial(amplitude=0.10, width_nm=150.0)
        sys_narrow = build_parametrized_system(
            params_narrow, probe_wavelengths=dense)
        sys_wide = build_parametrized_system(
            params_wide, probe_wavelengths=dense)
        m_narrow = next(e for e in sys_narrow.elements
                        if isinstance(e, PartialMirror))
        m_wide = next(e for e in sys_wide.elements
                      if isinstance(e, PartialMirror))
        i_mid = int(jnp.argmin(jnp.abs(dense - 495.5 * nm)))
        # Wider σ ⇒ much higher reflectance at the inter-peak midpoint.
        assert float(m_wide.reflectance[i_mid]) > 5 * float(
            m_narrow.reflectance[i_mid])


# ── ParamBounds ─────────────────────────────────────────────────────────────


def _curves_with(amplitude, sigma, num_mirrors=6):
    """Build a uniform 6×3 batched curve at default centers with given
    amplitude/sigma arrays (broadcast as needed)."""
    centers = jnp.broadcast_to(DEFAULT_WAVELENGTHS, (num_mirrors, 3)).copy()
    return SumOfGaussiansCurve(
        amplitude=jnp.broadcast_to(jnp.asarray(amplitude),
                                    (num_mirrors, 3)).copy(),
        sigma=jnp.broadcast_to(jnp.asarray(sigma),
                                (num_mirrors, 3)).copy(),
        centers=centers,
    )


class TestParamBounds:

    def test_amplitude_clipped_to_range(self):
        params = CombinerParams(
            spacings=jnp.full((5,), 1.5 * mm),
            curves=_curves_with(amplitude=jnp.array([0.001, 0.5, 0.05]),
                                 sigma=20.0 * nm),
        )
        bounds = ParamBounds()
        clipped = bounds.clip(params)
        # Use a tiny tolerance — float32 round-trip can shave the last bit.
        eps = 1e-7
        assert float(jnp.min(clipped.curves.amplitude)) >= bounds.amplitude_min - eps
        assert float(jnp.max(clipped.curves.amplitude)) <= bounds.amplitude_max + eps

    def test_width_clipped_to_range(self):
        params = CombinerParams(
            spacings=jnp.full((5,), 1.5 * mm),
            curves=_curves_with(amplitude=0.05,
                                 sigma=jnp.array([5.0, 200.0, 50.0]) * nm),
        )
        bounds = ParamBounds()
        clipped = bounds.clip(params)
        eps_nm = 1e-3 * nm
        sigma_min = fwhm_to_sigma(bounds.fwhm_min_nm * nm)
        sigma_max = fwhm_to_sigma(bounds.fwhm_max_nm * nm)
        assert float(jnp.min(clipped.curves.sigma)) >= sigma_min - eps_nm
        assert float(jnp.max(clipped.curves.sigma)) <= sigma_max + eps_nm

    def test_centers_unchanged_by_clip(self):
        original = CombinerParams.initial()
        clipped = ParamBounds().clip(original)
        assert jnp.allclose(clipped.curves.centers, original.curves.centers)

    def test_spacing_clipped_to_range(self):
        params = CombinerParams(
            spacings=jnp.array([0.1, 5.0, 1.0, 1.0, 1.0]) * mm,
            curves=_curves_with(amplitude=0.05, sigma=20.0 * nm),
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
            curves=_curves_with(amplitude=0.05, sigma=20.0 * nm),
        )
        clipped = bounds.clip(too_long)
        assert float(jnp.sum(clipped.spacings)) <= bounds.chassis_usable_mm * mm + 1e-6

    def test_no_rescale_when_total_within_budget(self):
        bounds = ParamBounds(chassis_usable_mm=18.0)
        ok = CombinerParams(
            spacings=jnp.full((5,), 1.5 * mm),  # total 7.5 mm
            curves=_curves_with(amplitude=0.05, sigma=20.0 * nm),
        )
        clipped = bounds.clip(ok)
        assert jnp.allclose(clipped.spacings, ok.spacings)


# ── Gradient flow ───────────────────────────────────────────────────────────


class TestGradient:

    def test_grad_through_amplitude(self):
        """Grad of a downstream sum-of-reflectance should reach the curve's
        amplitude leaf."""
        params = CombinerParams.initial()

        def first_mirror_reflectance_sum(p):
            system = build_parametrized_system(p)
            first = next(e for e in system.elements
                         if isinstance(e, PartialMirror))
            return jnp.sum(first.reflectance)

        grads = jax.grad(first_mirror_reflectance_sum)(params)
        assert grads.curves.amplitude.shape == params.curves.amplitude.shape
        # Only mirror_0's amplitude row affects mirror_0's reflectance.
        assert jnp.any(grads.curves.amplitude[0] != 0.0)
        assert jnp.all(grads.curves.amplitude[1:] == 0.0)

    def test_grad_through_centers_is_zero(self):
        """Centers are wrapped in stop_gradient inside the curve, so no
        gradient flows back to them even though they're a pytree leaf."""
        params = CombinerParams.initial()

        def first_mirror_reflectance_sum(p):
            system = build_parametrized_system(p)
            first = next(e for e in system.elements
                         if isinstance(e, PartialMirror))
            return jnp.sum(first.reflectance)

        grads = jax.grad(first_mirror_reflectance_sum)(params)
        assert jnp.all(grads.curves.centers == 0.0)

    def test_grad_through_spacing(self):
        """Grad of mirror position separation should reach the spacings."""
        params = CombinerParams.initial()

        def first_to_last_mirror_distance(p):
            system = build_parametrized_system(p)
            mirrors = [e for e in system.elements
                       if isinstance(e, PartialMirror)]
            return jnp.linalg.norm(mirrors[-1].position - mirrors[0].position)

        grads = jax.grad(first_to_last_mirror_distance)(params)
        assert grads.spacings.shape == params.spacings.shape
        assert jnp.all(grads.spacings != 0.0)
