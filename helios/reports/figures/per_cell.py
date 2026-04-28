"""Per-cell drill-down chart — ΔD65 over FOV with brightness contours.

The user-requested view: pick a pupil cell, see its ΔD65 across the FOV
as a heatmap, with brightness contours overlaid so color drift and
brightness mountains are visible together. The title lists per-cell
metrics: nits (luminance), summed radiance, FOV brightness CV
(uniformity), and worst-FOV nits.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from helios.reports.composer import (
    coefficient_of_variation_over_angles,
    d65_distance_per_cell_per_angle,
    luminance_per_cell_per_angle,
    luminance_weights_for_response,
    mean_over_angles,
    radiance_per_cell_per_angle,
    worst_over_angles,
)


def per_cell_d65_fov_figure(
    response: np.ndarray,
    scan_angles: np.ndarray,
    pupil_x_mm: np.ndarray,
    pupil_y_mm: np.ndarray,
    wavelengths_nm: np.ndarray | None = None,
) -> go.Figure:
    """Per-cell ΔD65 over (deg_x, deg_y), with brightness contours overlaid.

    Slider steps through pupil cells. For each cell, the title reports:

    - **luminance**  (nits) — mean across FOV
    - **radiance**   (summed) — mean across FOV
    - **brightness CV** — std/mean across FOV (FOV uniformity)
    - **worst-FOV** brightness — dimmest angle the eye sees from this cell
    """
    n_fov_y, n_fov_x = scan_angles.shape[:2]
    ax_deg = np.degrees(scan_angles[0, :, 0])
    ay_deg = np.degrees(scan_angles[:, 0, 1])
    ny, nx = len(pupil_y_mm), len(pupil_x_mm)

    # Per (cell, angle) D65 distance and brightness — both reshaped (ny, nx, A)
    d65_per_angle = d65_distance_per_cell_per_angle(
        response, pupil_x_mm, pupil_y_mm, wavelengths_nm)
    weights = luminance_weights_for_response(wavelengths_nm)
    if weights is not None:
        lum_per_angle = luminance_per_cell_per_angle(
            response, pupil_x_mm, pupil_y_mm, weights)
        lum_unit = "nits"
    else:
        lum_per_angle = None
        lum_unit = "n/a (no wavelengths)"
    rad_per_angle = radiance_per_cell_per_angle(
        response, pupil_x_mm, pupil_y_mm)

    # Reshape to per-FOV-grid for plotting
    fov_rad = rad_per_angle.reshape(ny, nx, n_fov_y, n_fov_x)
    fov_d65 = d65_per_angle.reshape(ny, nx, n_fov_y, n_fov_x)

    # Per-cell scalar reductions for the title
    mean_lum = (mean_over_angles(lum_per_angle).reshape(ny, nx)
                if lum_per_angle is not None else None)
    mean_rad = mean_over_angles(rad_per_angle).reshape(ny, nx)
    cv_rad = coefficient_of_variation_over_angles(rad_per_angle).reshape(ny, nx)
    worst_lum = (worst_over_angles(lum_per_angle).reshape(ny, nx)
                 if lum_per_angle is not None else None)

    # Color scale shared across cells so heatmaps are comparable
    d65_max = float(fov_d65.max()) or 1.0

    fig = go.Figure()
    n_cells = ny * nx
    steps = []
    cell_idx = 0
    for iy in range(ny):
        for ix in range(nx):
            # Two traces per cell: D65 heatmap + brightness contour.
            d65_panel = fov_d65[iy, ix]
            rad_panel = fov_rad[iy, ix]

            fig.add_trace(go.Heatmap(
                z=d65_panel,
                x=ax_deg, y=ay_deg,
                colorscale="Hot",
                zmin=0.0, zmax=d65_max,
                visible=(cell_idx == 0),
                colorbar=dict(title="ΔD65"),
            ))
            # Brightness contours — show 50%, 75%, 100% of cell mean
            cell_mean = float(rad_panel.mean()) or 1e-12
            fig.add_trace(go.Contour(
                z=rad_panel,
                x=ax_deg, y=ay_deg,
                contours=dict(
                    start=cell_mean * 0.5, end=cell_mean * 1.5,
                    size=cell_mean * 0.25,
                    coloring="lines",
                    showlabels=True,
                    labelfont=dict(color="cyan", size=10),
                ),
                line=dict(color="cyan", width=1, dash="dot"),
                showscale=False,
                visible=(cell_idx == 0),
                name="brightness contours",
                hoverinfo="skip",
            ))

            # Visibility mask: each step shows exactly the 2 traces for this cell
            visibility = [False] * (n_cells * 2)
            visibility[cell_idx * 2] = True
            visibility[cell_idx * 2 + 1] = True

            label = (f"cell ({pupil_x_mm[ix]:+.1f}, {pupil_y_mm[iy]:+.1f}) mm")
            title = _build_title(
                label,
                mean_lum=(float(mean_lum[iy, ix]) if mean_lum is not None
                          else None),
                lum_unit=lum_unit,
                mean_rad=float(mean_rad[iy, ix]),
                cv=float(cv_rad[iy, ix]),
                worst_lum=(float(worst_lum[iy, ix]) if worst_lum is not None
                           else None),
            )
            steps.append(dict(
                method="update",
                label=f"({pupil_x_mm[ix]:+.1f}, {pupil_y_mm[iy]:+.1f})",
                args=[{"visible": visibility}, {"title": title}],
            ))
            cell_idx += 1

    initial_title = (steps[0]["args"][1]["title"] if steps
                     else "Per-cell FOV — ΔD65 + brightness contours")
    fig.update_layout(
        title=initial_title,
        xaxis_title="FOV x (deg)",
        yaxis_title="FOV y (deg)",
        yaxis_scaleanchor="x",
        width=560, height=560,
        margin=dict(l=50, r=20, t=80, b=90),
        sliders=[dict(active=0, pad={"t": 40}, steps=steps,
                      currentvalue=dict(prefix="cell: "))],
    )
    return fig


def _build_title(label, *, mean_lum, lum_unit, mean_rad, cv, worst_lum) -> str:
    parts = [f"<b>{label}</b>"]
    if mean_lum is not None:
        parts.append(f"luminance: {mean_lum:.1f} {lum_unit}")
    parts.append(f"radiance: {mean_rad:.3g}")
    parts.append(f"FOV CV: {cv * 100:.1f}%")
    if worst_lum is not None:
        parts.append(f"worst-FOV: {worst_lum:.1f} {lum_unit}")
    return "  •  ".join(parts)
