"""Tests for the parametrized Talos combiner.

Covers the four moving parts: ``CombinerParams.initial`` shape, system
build (positions + reflectance), ``ParamBounds.clip`` clamping/rescaling,
and gradient flow back to params via the curve's reflectance samples.
"""

import math

import jax
import jax.numpy as jnp

from apollo14.combiner import (
    DEFAULT_LIGHT_DIRECTION,
    DEFAULT_LIGHT_POSITION,
    build_default_system,
)
from apollo14.elements.aperture import RectangularAperture, aperture_interact
from apollo14.ray import Ray
from apollo14.elements.glass_block import GlassBlock
from apollo14.elements.partial_mirror import PartialMirror
from apollo14.elements.pupil import RectangularPupil
from apollo14.spectral import SumOfGaussiansCurve
from apollo14.units import mm, nm
from helios.combiner_params import (
    CHASSIS_Z,
    EYE_RELIEF,
    MIRROR_ANGLE,
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


class TestChassisThickness:
    """``chassis_z`` rebuilds every thickness-dependent piece of geometry."""

    @staticmethod
    def _mirror0(system):
        return next(e for e in system.elements if isinstance(e, PartialMirror))

    @staticmethod
    def _pupil(system):
        return next(e for e in system.elements
                    if isinstance(e, RectangularPupil))

    def test_default_matches_explicit_default(self):
        params = CombinerParams.initial()
        default = build_parametrized_system(params)
        explicit = build_parametrized_system(params, chassis_z=CHASSIS_Z)
        # Same thickness ⇒ identical mirror footprint and pupil plane.
        assert jnp.allclose(self._mirror0(default).half_extents,
                            self._mirror0(explicit).half_extents)
        assert jnp.allclose(self._pupil(default).position,
                            self._pupil(explicit).position)

    def test_mirror_height_scales_with_thickness(self):
        params = CombinerParams.initial()
        thin = build_parametrized_system(params, chassis_z=0.8 * mm)
        # Mirror height (half_extents[1]) = (chassis_z / cos θ) / 2.
        expected_half_height = (0.8 * mm / math.cos(MIRROR_ANGLE)) / 2
        assert abs(float(self._mirror0(thin).half_extents[1])
                   - expected_half_height) < 1e-6
        # Thinner glass ⇒ shorter mirrors than the default build.
        default = build_parametrized_system(params)
        assert float(self._mirror0(thin).half_extents[1]) < float(
            self._mirror0(default).half_extents[1])
        # x-extent is thickness-independent — unchanged.
        assert jnp.allclose(self._mirror0(thin).half_extents[0],
                            self._mirror0(default).half_extents[0])

    def test_pupil_keeps_constant_eye_relief(self):
        params = CombinerParams.initial()
        thin = build_parametrized_system(params, chassis_z=0.8 * mm)
        # Pupil sits EYE_RELIEF beyond the front face at z = chassis_z.
        assert abs(float(self._pupil(thin).position[2])
                   - float(EYE_RELIEF + 0.8 * mm)) < 1e-6


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
        """5 × 4 mm = 20 mm of perpendicular spacing projects to a ~29.9 mm
        y-footprint, well past the 18 mm usable budget, so the rescale kicks
        in. The enforced invariant is on the *y-footprint* (sum / sin θ), not
        the raw perpendicular sum — that's the frame the clip now uses."""
        import math
        from helios.combiner_params import _NORMAL_ANGLE
        bounds = ParamBounds(chassis_usable_mm=18.0,
                             spacing_min_mm=0.5, spacing_max_mm=4.0)
        too_long = CombinerParams(
            spacings=jnp.full((5,), 4.0 * mm),
            curves=_curves_with(amplitude=0.05, sigma=20.0 * nm),
        )
        clipped = bounds.clip(too_long)
        y_footprint = float(jnp.sum(clipped.spacings)) / math.sin(_NORMAL_ANGLE)
        assert y_footprint <= bounds.chassis_usable_mm * mm + 1e-6

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


# ── Aperture (beam-defining stop) ────────────────────────────────────────────


def _aperture(system):
    return next(e for e in system.elements
                if isinstance(e, RectangularAperture))


def _aperture_passes_ray(aperture, x_mm, z_mm):
    """Aim a ray straight at the aperture plane, offset (x, z) in-plane."""
    offset = jnp.array([x_mm * mm, 0.0, z_mm * mm])
    origin = aperture.position + offset - jnp.asarray(DEFAULT_LIGHT_DIRECTION) * 5.0 * mm
    ray = Ray(pos=origin, dir=jnp.asarray(DEFAULT_LIGHT_DIRECTION),
              intensity=jnp.asarray(1.0))
    seg, _ = aperture.build_segment(None, None)
    _, _, valid = aperture_interact(seg, ray, 550.0 * nm)
    return bool(valid)


class TestAperture:

    def test_hole_not_wider_than_frame(self):
        # The hole must fit inside the opaque frame, else the stop can never
        # block in that axis (the original bug: inner 10 mm > outer 6 mm).
        for system in (build_default_system(),
                       build_parametrized_system(CombinerParams.initial())):
            aperture = _aperture(system)
            assert aperture.inner_width <= aperture.width
            assert aperture.inner_height <= aperture.height

    def test_apertures_match_between_systems(self):
        # combiner_params lifts fixed geometry from combiner, so the aperture
        # must be identical — designs won't transfer otherwise.
        default = _aperture(build_default_system())
        param = _aperture(build_parametrized_system(CombinerParams.initial()))
        assert (float(default.width), float(default.height),
                float(default.inner_width), float(default.inner_height)) == \
               (float(param.width), float(param.height),
                float(param.inner_width), float(param.inner_height))

    def test_passes_ray_through_hole(self):
        aperture = _aperture(build_default_system())
        assert _aperture_passes_ray(aperture, x_mm=0.0, z_mm=0.0)

    def test_blocks_stray_in_frame_outside_hole(self):
        # x=6 mm is inside the 14 mm frame but outside the 10 mm hole, so it
        # must be absorbed. On the pre-fix 6 mm frame this point fell outside
        # the frame and leaked straight through — this pins the stop fix.
        aperture = _aperture(build_default_system())
        assert not _aperture_passes_ray(aperture, x_mm=6.0, z_mm=0.0)


# ── Parametrized ⇄ fixed equivalence ─────────────────────────────────────────


class TestEquivalence:

    def test_shared_fixed_geometry_matches_default(self):
        # build_parametrized_system holds chassis / aperture / mirror tilt
        # fixed, lifted from build_default_system. The pupil is intentionally
        # divergent (recentering slack), so it is excluded here.
        default = build_default_system()
        param = build_parametrized_system(CombinerParams.initial())

        default_mirror = next(e for e in default.elements
                              if isinstance(e, PartialMirror))
        param_mirror = next(e for e in param.elements
                            if isinstance(e, PartialMirror))
        # Mirror tilt (normal) and footprint match between the two builds.
        assert jnp.allclose(default_mirror.normal, param_mirror.normal,
                            atol=1e-6)
        assert jnp.allclose(default_mirror.half_extents,
                            param_mirror.half_extents, atol=1e-5)
