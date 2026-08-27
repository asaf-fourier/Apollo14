"""Write a Zemax ``.AGF`` glass catalog for an Apollo14 material.

The combiner substrate (AGC M-074) is not in any stock OpticStudio catalog, and
Apollo14 carries it as a measured ``n(λ)`` table. Zemax catalogs store a
dispersion *formula* instead, so the table is fitted to Sellmeier 1 here and the
fit residual is reported — a bad fit is then visible rather than silent.

AGF record syntax (OpticStudio "The AGF & BGF File Formats")::

    CC <catalog comment>
    NM <name> <formula#> <MIL#> <N(d)> <V(d)> <exclude sub> <status> <melt freq>
    GC <glass comment>
    CD <dispersion coefficients 1..10>
    LD <min lambda> <max lambda>

Formula 2 is Sellmeier 1::

    n² − 1 = K₁λ²/(λ² − L₁) + K₂λ²/(λ² − L₂) + K₃λ²/(λ² − L₃)

with λ in micrometres and the ``CD`` coefficients ordered K₁ L₁ K₂ L₂ K₃ L₃.
"""

from typing import NamedTuple

import numpy as np

from apollo14.units import nm

SELLMEIER_1_FORMULA = 2

# Fraunhofer lines used for the catalog's N(d) and V(d) summary fields, in nm.
D_LINE_NM = 587.5618
F_LINE_NM = 486.1327
C_LINE_NM = 656.2725

# Candidate pole positions (µm²) for the Sellmeier fit. With the poles fixed the
# fit is linear in K₁..K₃, so a small search over physically sensible pole sets
# plus a least-squares solve replaces a nonlinear optimizer — and keeps the
# exporter to numpy, without pulling scipy into apollo14's dependencies.
UV_POLE_CANDIDATES_UM2 = (0.004, 0.006, 0.008, 0.010, 0.013, 0.016, 0.020,
                          0.025, 0.032, 0.040)
IR_POLE_CANDIDATES_UM2 = (60.0, 100.0, 150.0, 220.0, 320.0)

# A visible-band glass fit should land far below this; anything worse means the
# tabulated data is not Sellmeier-like and should not be shipped as a catalog.
MAX_ACCEPTABLE_RMS_INDEX_ERROR = 5e-5


class GlassFit(NamedTuple):
    """A fitted Sellmeier 1 dispersion, with the residual it achieved."""
    coefficients: tuple[float, float, float, float, float, float]  # K₁L₁K₂L₂K₃L₃
    rms_index_error: float
    max_index_error: float
    wavelength_range_um: tuple[float, float]
    index_d: float
    abbe_d: float


def _sellmeier_index(coefficients, wavelengths_um):
    """Evaluate Sellmeier 1 at ``wavelengths_um``."""
    k1, l1, k2, l2, k3, l3 = coefficients
    squared = np.asarray(wavelengths_um, dtype=float) ** 2
    index_squared = (1.0
                     + k1 * squared / (squared - l1)
                     + k2 * squared / (squared - l2)
                     + k3 * squared / (squared - l3))
    return np.sqrt(index_squared)


def fit_sellmeier(material) -> GlassFit:
    """Fit Apollo14's tabulated ``n(λ)`` for ``material`` to Sellmeier 1.

    Searches a small grid of pole positions; for each, the amplitudes K₁..K₃
    follow from an ordinary least-squares solve because ``n² − 1`` is linear in
    them once the poles are fixed. The best residual wins.

    For AGC M-074 the residual bottoms out around 3e-5 RMS, which is the
    quantization floor of the source table (indices given to four decimals) —
    not a shortcoming of the model, so widening the pole grid buys nothing. The
    poles themselves are only weakly determined by 400–800 nm data; the fit is
    trustworthy inside the ``LD`` range it declares and nowhere else.
    """
    wavelengths_um = np.asarray(material.wavelengths, dtype=float) / (1000.0 * nm)
    measured_index = np.asarray(material.n_values, dtype=float)
    target = measured_index ** 2 - 1.0
    squared = wavelengths_um ** 2

    best_fit = None
    best_residual = np.inf

    for first_uv_pole in UV_POLE_CANDIDATES_UM2:
        for second_uv_pole in UV_POLE_CANDIDATES_UM2:
            if second_uv_pole <= first_uv_pole:
                continue
            for ir_pole in IR_POLE_CANDIDATES_UM2:
                poles = (first_uv_pole, second_uv_pole, ir_pole)
                if any(abs(squared - pole).min() < 1e-3 for pole in poles):
                    continue  # a pole inside the data range blows up the fit
                basis = np.stack(
                    [squared / (squared - pole) for pole in poles], axis=1)
                amplitudes, *_ = np.linalg.lstsq(basis, target, rcond=None)
                fitted_index = np.sqrt(1.0 + basis @ amplitudes)
                residual = float(np.sqrt(
                    np.mean((fitted_index - measured_index) ** 2)))
                if residual < best_residual:
                    best_residual = residual
                    best_fit = (amplitudes, poles, fitted_index)

    if best_fit is None:
        raise ValueError(
            f"No usable Sellmeier pole set for material {material.name!r}.")

    amplitudes, poles, fitted_index = best_fit
    coefficients = (float(amplitudes[0]), float(poles[0]),
                    float(amplitudes[1]), float(poles[1]),
                    float(amplitudes[2]), float(poles[2]))

    index_d = float(material.n(D_LINE_NM * nm))
    index_f = float(material.n(F_LINE_NM * nm))
    index_c = float(material.n(C_LINE_NM * nm))
    abbe_d = (index_d - 1.0) / (index_f - index_c)

    return GlassFit(
        coefficients=coefficients,
        rms_index_error=best_residual,
        max_index_error=float(np.max(np.abs(fitted_index - measured_index))),
        wavelength_range_um=(float(wavelengths_um.min()),
                             float(wavelengths_um.max())),
        index_d=index_d,
        abbe_d=abbe_d,
    )


def format_glass_catalog(material, fit: GlassFit, glass_name: str) -> str:
    """Render a one-glass ``.AGF`` catalog for ``material``."""
    if fit.rms_index_error > MAX_ACCEPTABLE_RMS_INDEX_ERROR:
        raise ValueError(
            f"Sellmeier fit for {material.name!r} is too poor to export: RMS "
            f"index error {fit.rms_index_error:.2e} exceeds "
            f"{MAX_ACCEPTABLE_RMS_INDEX_ERROR:.0e}. Check the tabulated data "
            "in apollo14/data before shipping it as a Zemax catalog.")

    coefficient_text = " ".join(f"{value:.12E}" for value in fit.coefficients)
    minimum_um, maximum_um = fit.wavelength_range_um

    return "\n".join([
        "CC Apollo14 combiner substrates — generated by apollo14.export.agf",
        f"NM {glass_name} {SELLMEIER_1_FORMULA} 0 {fit.index_d:.6f} "
        f"{fit.abbe_d:.6f} 0 0 0",
        f"GC Fit of {material.name} measured n(lambda) to Sellmeier 1; "
        f"RMS dn={fit.rms_index_error:.2e}, max dn={fit.max_index_error:.2e} "
        f"over {minimum_um * 1000:.0f}-{maximum_um * 1000:.0f} nm",
        f"CD {coefficient_text} 0.000000000000E+00 0.000000000000E+00 "
        "0.000000000000E+00 0.000000000000E+00",
        f"LD {minimum_um:.6f} {maximum_um:.6f}",
        "",
    ])
