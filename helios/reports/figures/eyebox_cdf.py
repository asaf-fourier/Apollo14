"""Eyebox-quality cumulative distribution figure.

Reduces the whole pupil to one curve: "how many cells clear a given
brightness (and optionally a color) threshold?" — directly answers
"how big is the usable eyebox?" without the user having to count cells
on a heatmap.
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


def eyebox_quality_cdf_figure(
    response: np.ndarray,
    pupil_x_mm: np.ndarray,
    pupil_y_mm: np.ndarray,
    wavelengths_nm: np.ndarray | None = None,
    d65_tolerance: float | None = 0.05,
    num_threshold_steps: int = 80,
) -> go.Figure:
    """Cumulative fraction of pupil cells clearing a brightness threshold.

    Two curves on one axes:

    1. **Brightness only** — fraction of cells whose mean brightness is
       at least ``T``. As ``T`` rises from 0, the fraction drops from 1.
    2. **Brightness + ΔD65** (when ``d65_tolerance`` is set) — same as
       above, but additionally requires the cell's mean ΔD65 to be below
       ``d65_tolerance``. Tells you how much of the bright eyebox is
       *also* white-balanced.

    The X axis is in **nits** when wavelengths are available, otherwise
    in summed-radiance units. The two curves visually answer "what's my
    usable eyebox size at threshold T?" — read off the y-coordinate.
    """
    weights = luminance_weights_for_response(wavelengths_nm)
    if weights is not None:
        per_angle = luminance_per_cell_per_angle(
            response, pupil_x_mm, pupil_y_mm, weights)
        x_label = "brightness threshold (nits)"
    else:
        per_angle = radiance_per_cell_per_angle(
            response, pupil_x_mm, pupil_y_mm)
        x_label = "brightness threshold (summed radiance)"

    cell_brightness = mean_over_angles(per_angle).ravel()  # (ny*nx,)
    num_cells = cell_brightness.size
    max_b = float(cell_brightness.max())
    thresholds = np.linspace(0.0, max_b * 1.05 + 1e-12, num_threshold_steps)

    fraction_above = np.array([
        float((cell_brightness >= t).mean()) for t in thresholds
    ])

    traces: list = [
        go.Scatter(
            x=thresholds, y=fraction_above,
            mode="lines",
            name=f"brightness ≥ T  ({num_cells} cells)",
            line=dict(color="steelblue", width=2),
        )
    ]

    if d65_tolerance is not None:
        d65_per_angle = d65_distance_per_cell_per_angle(
            response, pupil_x_mm, pupil_y_mm, wavelengths_nm)
        cell_d65 = mean_over_angles(d65_per_angle).ravel()
        white_mask = cell_d65 <= d65_tolerance
        fraction_white_and_bright = np.array([
            float(((cell_brightness >= t) & white_mask).mean())
            for t in thresholds
        ])
        traces.append(go.Scatter(
            x=thresholds, y=fraction_white_and_bright,
            mode="lines",
            name=f"brightness ≥ T  AND  ΔD65 ≤ {d65_tolerance}",
            line=dict(color="firebrick", width=2),
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title="Eyebox quality CDF — fraction of cells clearing each threshold",
        xaxis_title=x_label,
        yaxis_title="fraction of pupil cells",
        yaxis=dict(range=[0.0, 1.05]),
        width=700, height=420,
        margin=dict(l=60, r=20, t=60, b=50),
        legend=dict(x=0.55, y=0.95),
    )
    return fig
