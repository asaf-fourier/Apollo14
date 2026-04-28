"""Pupil-overview figures — at-a-glance brightness and color quality.

Two heatmaps over the pupil grid, each optionally overlaid with the
"acceptance" contour (the eyebox boundary in nits or ΔD65 space).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from helios.reports.composer import (
    d65_distance_per_cell_per_angle,
    luminance_per_cell_per_angle,
    luminance_weights_for_response,
    mean_over_angles,
    radiance_per_cell_per_angle,
)
from helios.reports.primitives import pupil_heatmap_figure


def pupil_brightness_figure(
    response: np.ndarray,
    pupil_x_mm: np.ndarray,
    pupil_y_mm: np.ndarray,
    wavelengths_nm: np.ndarray | None = None,
    threshold_nits: float | None = None,
) -> go.Figure:
    """Per-cell mean brightness across the FOV.

    Photopic (nits) when ``wavelengths_nm`` is provided; otherwise falls
    back to summed radiance (the previous behavior). When
    ``threshold_nits`` is set, draws a dashed contour at that level
    marking the eyebox acceptance boundary.
    """
    weights = luminance_weights_for_response(wavelengths_nm)
    if weights is not None:
        per_angle = luminance_per_cell_per_angle(
            response, pupil_x_mm, pupil_y_mm, weights)
        unit = "cd/m² (nits)"
        title = "Pupil — luminance (FOV-averaged)"
    else:
        per_angle = radiance_per_cell_per_angle(
            response, pupil_x_mm, pupil_y_mm)
        unit = "summed radiance"
        title = "Pupil — total intensity (FOV-averaged)"

    grid = mean_over_angles(per_angle)  # (ny, nx)

    return pupil_heatmap_figure(
        pupil_x_mm, pupil_y_mm, grid,
        title=title,
        colorbar_title=unit,
        threshold=threshold_nits,
        threshold_label="eyebox threshold",
    )


def pupil_d65_distance_figure(
    response: np.ndarray,
    pupil_x_mm: np.ndarray,
    pupil_y_mm: np.ndarray,
    wavelengths_nm: np.ndarray | None = None,
    tolerance: float | None = None,
) -> go.Figure:
    """Per-cell distance from the D65 simplex (FOV-averaged).

    ``tolerance`` overlays a dashed contour marking the maximum
    acceptable color drift — cells inside the contour are "white".
    """
    per_angle = d65_distance_per_cell_per_angle(
        response, pupil_x_mm, pupil_y_mm, wavelengths_nm)
    grid = mean_over_angles(per_angle)

    return pupil_heatmap_figure(
        pupil_x_mm, pupil_y_mm, grid,
        title="Pupil — ΔD65 (FOV-averaged simplex distance)",
        colorscale="Hot",
        colorbar_title="ΔD65",
        threshold=tolerance,
        threshold_label="ΔD65 tolerance",
    )
