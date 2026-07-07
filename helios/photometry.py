"""Photopic luminous efficiency function and radiance↔luminance conversion.

The CIE 1931 V(λ) curve weights spectral radiance by the human eye's
sensitivity to give luminance — the right unit for "brightness" when the
goal is what a viewer perceives. Without this weighting, blue at 446 nm
counts the same per watt as green at 545 nm, even though the eye is
~32× more sensitive to green.

Conventions
-----------
- Wavelengths are in *internal length units* (millimeters by convention,
  so 555 nm == ``555 * apollo14.units.nm``). Lookup converts back to nm
  for the V(λ) table interpolation.
- ``V(λ)`` is unitless; peak is 1.0 at 555 nm.
- ``K_M = 683 lm/W`` — the maximum luminous efficacy (at 555 nm).

Conversion::

    L_v [cd/m²]  =  K_m · ∫ L_e,λ(λ) · V(λ) dλ

For a discrete sum, ``Δλ`` is the spacing between sampled wavelengths.
"""

import jax.numpy as jnp
import numpy as np

from apollo14.units import nm

K_M = 683.0  # lm/W — maximum luminous efficacy at 555 nm


# ── CIE 1931 photopic V(λ), 380–780 nm at 5 nm spacing ─────────────────────
#
# Source: CIE 1931 standard observer (modified by Judd 1951 / Vos 1978
# corrections are common but not used here — we use the textbook 1931
# curve to stay aligned with most colorimetry literature).

_V_TABLE_NM = jnp.arange(380.0, 781.0, 5.0)
_V_TABLE = jnp.array([
    0.000039, 0.000064, 0.000120, 0.000217, 0.000396, 0.000640, 0.001210,
    0.002180, 0.004000, 0.007300, 0.011600, 0.016840, 0.023000, 0.029800,
    0.038000, 0.048000, 0.060000, 0.073900, 0.090980, 0.112600, 0.139020,
    0.169300, 0.208020, 0.258600, 0.323000, 0.407300, 0.503000, 0.608200,
    0.710000, 0.793200, 0.862000, 0.914850, 0.954000, 0.980300, 0.994950,
    1.000000, 0.995000, 0.978600, 0.952000, 0.915400, 0.870000, 0.816300,
    0.757000, 0.694900, 0.631000, 0.566800, 0.503000, 0.441200, 0.381000,
    0.321000, 0.265000, 0.217000, 0.175000, 0.138200, 0.107000, 0.081600,
    0.061000, 0.044580, 0.032000, 0.023200, 0.017000, 0.011920, 0.008210,
    0.005723, 0.004102, 0.002929, 0.002091, 0.001484, 0.001047, 0.000740,
    0.000520, 0.000361, 0.000249, 0.000172, 0.000120, 0.000085, 0.000060,
    0.000042, 0.000030, 0.000021, 0.000015,
])


def photopic_v(wavelengths: jnp.ndarray) -> jnp.ndarray:
    """Interpolate the CIE 1931 photopic V(λ) at arbitrary wavelengths.

    Args:
        wavelengths: (K,) wavelengths in internal length units (so
            ``555 * nm`` corresponds to the V peak). Out-of-range values
            (outside 380–780 nm) clamp to the table edges, which are
            ~1e-5 — effectively zero, matching physical reality.

    Returns:
        (K,) unitless V(λ) values, peak 1.0 at 555 nm.
    """
    wavelengths_nm = jnp.asarray(wavelengths) / nm
    return jnp.interp(wavelengths_nm, _V_TABLE_NM, _V_TABLE)


def luminance_weights(wavelengths: jnp.ndarray,
                      delta_nm: float | None = None) -> jnp.ndarray:
    """Per-sample weights to convert summed spectral radiance into luminance.

    For a discrete sum ``Σ_λ L_e,λ(λ_i) · w_i`` to equal the luminance
    integral ``K_m · ∫ L_e,λ(λ) · V(λ) dλ``, set ``w_i = K_m · V(λ_i) · Δλ``.
    Pass ``delta_nm`` if you know the sample spacing; otherwise it's
    inferred from consecutive differences (uniform spacing assumed).

    Args:
        wavelengths: (K,) wavelengths in internal length units.
        delta_nm: Sample spacing in nanometers. If ``None``, inferred as
            the mean of consecutive differences (assumes uniform grid).
            For a single-wavelength input, must be supplied explicitly.

    Returns:
        (K,) weights in lm·sr⁻¹·m⁻²·(W·sr⁻¹·m⁻²·nm⁻¹)⁻¹ — i.e., scaling
        a sum over per-nm radiance values into a luminance.
    """
    wavelengths = jnp.asarray(wavelengths)
    if delta_nm is None:
        if wavelengths.shape[0] < 2:
            raise ValueError(
                "Cannot infer Δλ from a single wavelength — pass delta_nm.")
        diffs_nm = jnp.diff(wavelengths) / nm
        # abs(): Δλ is a spacing *magnitude* for the Riemann sum. Without it a
        # descending wavelength grid gives a negative Δλ and silently negates
        # every weight — turning luminance negative with no error.
        delta_nm = abs(float(jnp.mean(diffs_nm)))
    return K_M * photopic_v(wavelengths) * delta_nm


def radiance_to_luminance(radiance: jnp.ndarray,
                          wavelengths: jnp.ndarray) -> jnp.ndarray:
    """Convert a sampled spectral radiance to luminance via Riemann sum.

    Args:
        radiance: (..., K) spectral radiance values in W/sr/m²/nm at
            each wavelength.
        wavelengths: (K,) sample wavelengths in internal length units.
            Uniform spacing is assumed; Δλ is inferred from consecutive
            differences.

    Returns:
        (...,) luminance in cd/m² (nits).
    """
    weights = luminance_weights(wavelengths)
    return jnp.sum(radiance * weights, axis=-1)


# ── numpy mirror for run reports / non-JAX consumers ───────────────────────


def photopic_v_np(wavelengths_nm: np.ndarray) -> np.ndarray:
    """numpy version of :func:`photopic_v`. Input wavelengths in *nanometers*."""
    return np.interp(np.asarray(wavelengths_nm),
                     np.asarray(_V_TABLE_NM), np.asarray(_V_TABLE))


def luminance_weights_np(wavelengths_nm: np.ndarray,
                         delta_nm: float | None = None) -> np.ndarray:
    """numpy version of :func:`luminance_weights`. Wavelengths in *nanometers*.

    Multiplying a per-nm spectral-radiance sample by the corresponding
    weight and summing yields luminance in cd/m² (nits).
    """
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    if delta_nm is None:
        if wavelengths_nm.shape[0] < 2:
            raise ValueError(
                "Cannot infer Δλ from a single wavelength — pass delta_nm.")
        # abs(): see luminance_weights — a descending grid must not flip sign.
        delta_nm = abs(float(np.mean(np.diff(wavelengths_nm))))
    return K_M * photopic_v_np(wavelengths_nm) * delta_nm


# ── CIE 1931 2° standard observer x̄/ȳ/z̄, 380–780 nm at 5 nm spacing ──────
#
# Source: CIE 015:2004 reproduction of the 1931 2° observer. ȳ(λ) is
# numerically identical to ``_V_TABLE`` above (peak 1.0 at 555 nm) — we
# share the same array. Used by the report's chromaticity metric, which
# integrates per-cell spectra through these CMFs to produce CIE (x, y)
# coordinates — the right "color quality" target for a 3-band micro-LED
# whose raw spectral SPD looks nothing like a continuous D65 reference.

_X_BAR_TABLE = jnp.array([
    0.001368, 0.002236, 0.004243, 0.007650, 0.014310, 0.023190, 0.043510,
    0.077630, 0.134380, 0.214770, 0.283900, 0.328500, 0.348280, 0.348060,
    0.336200, 0.318700, 0.290800, 0.251100, 0.195360, 0.142100, 0.095640,
    0.057950, 0.032010, 0.014700, 0.004900, 0.002400, 0.009300, 0.029100,
    0.063270, 0.109600, 0.165500, 0.225750, 0.290400, 0.359700, 0.433450,
    0.512050, 0.594500, 0.678400, 0.762100, 0.842500, 0.916300, 0.978600,
    1.026300, 1.056700, 1.062200, 1.045600, 1.002600, 0.938400, 0.854450,
    0.751400, 0.642400, 0.541900, 0.447900, 0.360800, 0.283500, 0.218700,
    0.164900, 0.121200, 0.087400, 0.063600, 0.046770, 0.032900, 0.022700,
    0.015840, 0.011359, 0.008111, 0.005790, 0.004109, 0.002899, 0.002049,
    0.001440, 0.001000, 0.000690, 0.000476, 0.000332, 0.000235, 0.000166,
    0.000117, 0.000083, 0.000059, 0.000042,
])
_Y_BAR_TABLE = _V_TABLE  # ȳ(λ) ≡ V(λ) for CIE 1931 2°
_Z_BAR_TABLE = jnp.array([
    0.006450, 0.010550, 0.020050, 0.036210, 0.067850, 0.110200, 0.207400,
    0.371300, 0.645600, 1.039050, 1.385600, 1.622960, 1.747060, 1.782600,
    1.772110, 1.744100, 1.669200, 1.528100, 1.287640, 1.041900, 0.812950,
    0.616200, 0.465180, 0.353300, 0.272000, 0.212300, 0.158200, 0.111700,
    0.078250, 0.057250, 0.042160, 0.029840, 0.020300, 0.013400, 0.008750,
    0.005750, 0.003900, 0.002750, 0.002100, 0.001800, 0.001650, 0.001400,
    0.001100, 0.001000, 0.000800, 0.000600, 0.000340, 0.000240, 0.000190,
    0.000100, 0.000050, 0.000030, 0.000020, 0.000010, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000,
])

# Canonical CIE D65 illuminant white point (CIE 1931 2° observer).
D65_WHITE_POINT_XY = (0.31271, 0.32902)


def cie1931_xyz_np(spectrum: np.ndarray,
                   wavelengths_nm: np.ndarray) -> np.ndarray:
    """Integrate a spectrum through the CIE 1931 2° observer to get XYZ.

    Args:
        spectrum: ``(..., K)`` per-wavelength radiance/intensity. Trailing
            axis matches ``wavelengths_nm``.
        wavelengths_nm: ``(K,)`` sample wavelengths in nanometers.

    Returns:
        ``(..., 3)`` tristimulus values ``(X, Y, Z)``. For uniformly
        sampled wavelengths the constant Δλ cancels when normalizing to
        chromaticity (x = X/(X+Y+Z)), so we omit it here.
    """
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    x_bar = np.interp(wavelengths_nm, np.asarray(_V_TABLE_NM),
                      np.asarray(_X_BAR_TABLE))
    y_bar = np.interp(wavelengths_nm, np.asarray(_V_TABLE_NM),
                      np.asarray(_Y_BAR_TABLE))
    z_bar = np.interp(wavelengths_nm, np.asarray(_V_TABLE_NM),
                      np.asarray(_Z_BAR_TABLE))
    X = np.sum(spectrum * x_bar, axis=-1)
    Y = np.sum(spectrum * y_bar, axis=-1)
    Z = np.sum(spectrum * z_bar, axis=-1)
    return np.stack([X, Y, Z], axis=-1)


def chromaticity_xy_np(spectrum: np.ndarray,
                       wavelengths_nm: np.ndarray) -> np.ndarray:
    """CIE 1931 chromaticity coordinates ``(x, y)`` from a sampled spectrum.

    Args:
        spectrum: ``(..., K)`` per-wavelength radiance/intensity. Negative
            values are clamped to 0 (the eye can't see negative light).
        wavelengths_nm: ``(K,)`` sample wavelengths in nanometers.

    Returns:
        ``(..., 2)`` chromaticity coordinates ``(x, y)``. For a fully-zero
        spectrum, returns the D65 white point — there's no perceived hue
        and reporting it as "deep blue" or any other corner would be
        misleading.
    """
    xyz = cie1931_xyz_np(np.maximum(spectrum, 0.0), wavelengths_nm)
    total = xyz.sum(axis=-1, keepdims=True)
    # Guard zero spectra: total == 0 ⇒ return the D65 white point so a
    # dark cell shows as "neutral" rather than landing at (0, 0) which is
    # well outside the visible gamut.
    safe = total > 1e-30
    xy = np.where(safe, xyz[..., :2] / np.maximum(total, 1e-30),
                  np.array(D65_WHITE_POINT_XY))
    return xy


def d65_primary_ratios(wavelengths: np.ndarray,
                       white_point_xy: tuple = D65_WHITE_POINT_XY) -> np.ndarray:
    """Radiance ratios of N=3 narrow primaries that mix to a target white.

    For a *few-primary* display (a set of narrow R/G/B lines) the correct
    white-balance target is **not** the reference SPD's power at each primary
    wavelength — that's unrelated to how the primaries mix. What produces the
    target white is the set of per-primary *radiances* whose additive
    tristimulus sum has the target chromaticity::

        Σ_i  w_i · [x̄(λ_i), ȳ(λ_i), z̄(λ_i)]  ∝  XYZ_white

    This solves that 3×3 system for ``w`` and returns it normalized to sum 1
    (a radiance simplex), so it drops straight into the merit's ``d65_weights``
    (which are radiance-space channel fractions).

    Args:
        wavelengths: ``(3,)`` primary wavelengths in internal length units
            (so ``446 * nm`` is deep blue). Exactly three are required — the
            tristimulus matrix must be square/invertible. For a *dense* grid
            reproducing the reference SPD shape is already correct; use
            :func:`helios.merit.d65_weights_at` there instead.
        white_point_xy: target chromaticity ``(x, y)``; defaults to D65.

    Returns:
        ``(3,)`` non-negative radiance weights summing to 1, index-paired to
        ``wavelengths``.

    Raises:
        ValueError: if not exactly three primaries are given, or if the target
            white lies outside their gamut (a solved weight comes out negative
            — that white is physically unreachable with these primaries).
    """
    wavelengths_nm = np.asarray(wavelengths, dtype=float) / nm
    if wavelengths_nm.shape != (3,):
        raise ValueError(
            f"Need exactly 3 primaries to solve the tristimulus system, got "
            f"{wavelengths_nm.shape}. Use d65_weights_at for a dense grid.")
    # Column i = the tristimulus of a unit-radiance primary at λ_i. Feeding an
    # identity "spectrum" through the CMFs yields each primary's (X, Y, Z).
    primary_xyz = cie1931_xyz_np(np.eye(3), wavelengths_nm)   # (3, 3), row = primary
    tristimulus_matrix = primary_xyz.T                        # (3, 3), col = primary
    x, y = white_point_xy
    white_xyz = np.array([x / y, 1.0, (1.0 - x - y) / y])
    weights = np.linalg.solve(tristimulus_matrix, white_xyz)
    if np.any(weights < 0):
        raise ValueError(
            f"Target white {white_point_xy} is outside the gamut of primaries "
            f"{list(np.asarray(wavelengths) / nm)} nm (negative weight "
            f"{weights}); it cannot be reproduced by an additive mix.")
    return weights / weights.sum()
