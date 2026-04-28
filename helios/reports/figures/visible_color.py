"""Per-cell "what the eye actually sees" rendering.

For each pupil cell, the per-FOV-angle response is converted from
spectral radiance to displayable sRGB and shown as an image. This is
more intuitive than a ΔD65 number — you literally see the pink corner.

Spectral → sRGB pipeline:

1. Spectral radiance → CIE XYZ via the 1931 2° color-matching functions
   ``x̄(λ), ȳ(λ), z̄(λ)``. Each channel is a Riemann sum like luminance.
2. XYZ (D65 reference white) → linear sRGB via the standard matrix.
3. Linear sRGB → display sRGB via the gamma transfer curve.
4. Per-cell normalization so the cell's brightest FOV angle = (1, 1, 1)
   if it's already white, otherwise the brightest channel saturates.
   This trades absolute calibration for comparability between cells.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from helios.reports.composer import reshape_fov

# CIE 1931 2° color-matching functions, 380–780 nm at 5 nm spacing.
# Source: CIE 1931 standard observer, x̄, ȳ, z̄.
_CMF_NM = np.arange(380.0, 781.0, 5.0)
_CMF_X = np.array([
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
_CMF_Y = np.array([
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
_CMF_Z = np.array([
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

# sRGB primaries from CIE XYZ (D65 reference white) — IEC 61966-2-1.
_XYZ_TO_LINEAR_SRGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
])


def _spectral_to_xyz(response_per_angle: np.ndarray,
                     wavelengths_nm: np.ndarray) -> np.ndarray:
    """``(..., K) → (..., 3)`` via CIE 1931 color-matching Riemann sum.

    Wavelength spacing is inferred from the input grid; uniform Δλ assumed.
    """
    delta_nm = float(np.mean(np.diff(wavelengths_nm)))
    cmf_x = np.interp(wavelengths_nm, _CMF_NM, _CMF_X)
    cmf_y = np.interp(wavelengths_nm, _CMF_NM, _CMF_Y)
    cmf_z = np.interp(wavelengths_nm, _CMF_NM, _CMF_Z)
    cmf = np.stack([cmf_x, cmf_y, cmf_z], axis=-1)  # (K, 3)
    return delta_nm * (response_per_angle @ cmf)    # (..., 3)


def _linear_srgb_to_srgb(linear: np.ndarray) -> np.ndarray:
    """sRGB gamma transfer curve (per IEC 61966-2-1)."""
    a = 0.055
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        (1.0 + a) * np.power(np.maximum(linear, 0.0), 1.0 / 2.4) - a,
    )


def _xyz_to_display_srgb(xyz: np.ndarray) -> np.ndarray:
    """``(..., 3)`` XYZ → display-ready sRGB in [0, 1]."""
    linear = xyz @ _XYZ_TO_LINEAR_SRGB.T
    linear = np.clip(linear, 0.0, None)
    return np.clip(_linear_srgb_to_srgb(linear), 0.0, 1.0)


def per_cell_visible_color_figure(
    response: np.ndarray,
    scan_angles: np.ndarray,
    pupil_x_mm: np.ndarray,
    pupil_y_mm: np.ndarray,
    wavelengths_nm: np.ndarray,
) -> go.Figure:
    """Per-cell FOV image rendered in sRGB — what the eye actually sees.

    Each pupil cell becomes one image whose pixels are the FOV angles,
    colored in sRGB space derived from the cell-and-angle's spectral
    response. The cell is normalized so its brightest pixel saturates
    at white (any uniform overall scaling is a separate concern).
    """
    n_fov_y, n_fov_x = scan_angles.shape[:2]
    ax_deg = np.degrees(scan_angles[0, :, 0])
    ay_deg = np.degrees(scan_angles[:, 0, 1])
    ny, nx = len(pupil_y_mm), len(pupil_x_mm)

    fov = reshape_fov(response, n_fov_y, n_fov_x)        # (S, ny_f, nx_f, K)
    xyz = _spectral_to_xyz(fov, wavelengths_nm)          # (S, ny_f, nx_f, 3)
    srgb = _xyz_to_display_srgb(xyz)                     # in [0, 1]

    # Reshape to per-pupil-cell layout
    cells = srgb.reshape(ny, nx, n_fov_y, n_fov_x, 3)

    # Per-cell normalization: scale so the cell's brightest pixel = 1.
    cell_max = cells.max(axis=(2, 3, 4), keepdims=True) + 1e-12
    cells = cells / cell_max

    fig = go.Figure()
    n_cells = ny * nx
    steps = []
    cell_idx = 0
    for iy in range(ny):
        for ix in range(nx):
            rgb = cells[iy, ix]                          # (ny_f, nx_f, 3)
            uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)
            fig.add_trace(go.Image(
                z=uint8,
                x0=float(ax_deg[0]),
                dx=float(ax_deg[-1] - ax_deg[0]) / max(n_fov_x - 1, 1),
                y0=float(ay_deg[0]),
                dy=float(ay_deg[-1] - ay_deg[0]) / max(n_fov_y - 1, 1),
                visible=(cell_idx == 0),
                hoverinfo="skip",
            ))
            visibility = [False] * n_cells
            visibility[cell_idx] = True
            label = f"({pupil_x_mm[ix]:+.1f}, {pupil_y_mm[iy]:+.1f})"
            steps.append(dict(
                method="update",
                label=label,
                args=[{"visible": visibility},
                      {"title": f"<b>cell {label} mm</b> — visible color "
                                f"(sRGB, per-cell normalized)"}],
            ))
            cell_idx += 1

    initial = steps[0]["args"][1]["title"] if steps else "Visible color"
    fig.update_layout(
        title=initial,
        xaxis_title="FOV x (deg)",
        yaxis_title="FOV y (deg)",
        width=560, height=560,
        margin=dict(l=50, r=20, t=80, b=90),
        sliders=[dict(active=0, pad={"t": 40}, steps=steps,
                      currentvalue=dict(prefix="cell: "))],
    )
    return fig
