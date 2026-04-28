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
        delta_nm = float(jnp.mean(diffs_nm))
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
        delta_nm = float(np.mean(np.diff(wavelengths_nm)))
    return K_M * photopic_v_np(wavelengths_nm) * delta_nm
