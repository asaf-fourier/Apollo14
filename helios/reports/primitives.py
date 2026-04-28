"""Generic Plotly heatmap primitives reused across the report figures."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def fov_heatmap_figure(
    ax_deg: np.ndarray,
    ay_deg: np.ndarray,
    intensity: np.ndarray,
    title: str,
    zmin: float | None = None,
    zmax: float | None = None,
    colorscale: str = "Viridis",
    colorbar_title: str = "intensity",
    width: int = 460,
    height: int = 420,
) -> go.Figure:
    """Heatmap on a FOV angular grid (axes in degrees)."""
    fig = go.Figure(go.Heatmap(
        z=intensity,
        x=ax_deg,
        y=ay_deg,
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        colorbar=dict(title=colorbar_title),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="FOV x (deg)",
        yaxis_title="FOV y (deg)",
        yaxis_scaleanchor="x",
        width=width, height=height,
        margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig


def pupil_heatmap_figure(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    values: np.ndarray,
    title: str,
    zmin: float | None = None,
    zmax: float | None = None,
    colorscale: str = "Viridis",
    colorbar_title: str = "intensity",
    threshold: float | None = None,
    threshold_label: str = "threshold",
    width: int = 460,
    height: int = 420,
) -> go.Figure:
    """Heatmap on a pupil spatial grid. Optionally overlays a contour line
    at ``threshold`` to mark the eyebox-acceptance boundary.

    ``values`` shape ``(len(y_mm), len(x_mm))``.
    """
    traces: list = [
        go.Heatmap(
            z=values,
            x=x_mm,
            y=y_mm,
            colorscale=colorscale,
            zmin=zmin, zmax=zmax,
            colorbar=dict(title=colorbar_title),
            name="value",
        )
    ]
    if threshold is not None:
        traces.append(go.Contour(
            z=values,
            x=x_mm,
            y=y_mm,
            contours=dict(
                start=threshold, end=threshold, size=1,
                coloring="lines",
            ),
            line=dict(color="white", width=2, dash="dash"),
            showscale=False,
            name=threshold_label,
            hoverinfo="skip",
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=title,
        xaxis_title="pupil x (mm)",
        yaxis_title="pupil y (mm)",
        yaxis_scaleanchor="x",
        width=width, height=height,
        margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig
