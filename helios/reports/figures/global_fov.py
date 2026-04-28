"""FOV-global figures: pupil-averaged and worst-cell views across the FOV.

Inverts the pupil/FOV split: instead of one pupil cell × all angles
(per_cell.py), shows all pupil cells aggregated × one FOV grid. Useful
for spotting FOV directions where the *whole* pupil drops out.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from helios.reports.composer import d65_ratios, reshape_fov
from helios.reports.primitives import fov_heatmap_figure


def fov_global_figures(
    response: np.ndarray,
    scan_angles: np.ndarray,
    wavelengths_nm: np.ndarray | None = None,
) -> list[go.Figure]:
    """Two FOV-grid heatmaps:

    - **Pupil-averaged total** — sum across channels, mean across pupil
      cells. Visualizes how much *total* light each FOV direction
      delivers (averaged over eye position).
    - **Worst pupil cell, D65-normalized** — at each FOV angle, take the
      pupil cell with the lowest D65-balanced channel min. Surfaces FOV
      angles that look bad somewhere on the pupil.
    """
    d65 = d65_ratios(wavelengths_nm)

    n_fov_y, n_fov_x = scan_angles.shape[:2]
    ax_deg = np.degrees(scan_angles[0, :, 0])
    ay_deg = np.degrees(scan_angles[:, 0, 1])

    reshaped = reshape_fov(response, n_fov_y, n_fov_x)   # (S, ny_f, nx_f, K)

    avg = reshaped.sum(axis=-1).mean(axis=0)             # (ny_f, nx_f)

    normed = reshaped / (d65[None, None, None, :] + 1e-12)
    worst_channel = normed.min(axis=-1)                  # (S, ny_f, nx_f)
    worst_cell = worst_channel.min(axis=0)               # (ny_f, nx_f)

    return [
        fov_heatmap_figure(
            ax_deg, ay_deg, avg,
            title="FOV — pupil-averaged (sum of channels)"),
        fov_heatmap_figure(
            ax_deg, ay_deg, worst_cell,
            title="FOV — worst pupil cell (D65-normalized min channel)"),
    ]
