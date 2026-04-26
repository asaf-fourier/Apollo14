"""Render a full run report from the data saved by ``helios.io.save_run``.

Reads ``manifest.json`` + ``response.npz`` from a run directory and writes
a single self-contained ``report.html`` containing:

  1. Header (run_id, git sha, timestamp)
  2. Pupil total-intensity heatmap (summed over wavelengths, FOV-averaged)
  3. White-balance error map (spectral distance from D65 per pupil cell)
  4. Global FOV maps (pupil-averaged + worst-cell)
  5. Per-cell FOV heatmap with an eyebox-cell slider

The response is ``(S, A, C)`` — samples × angles × channels — where ``S``
reshapes to the pupil grid ``(ny, nx)`` and ``A`` reshapes to the FOV
grid ``(n_fov_y, n_fov_x)``. ``C`` may be 3 (R/G/B) or N (spectral).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from apollo14.units import nm
from helios.merit import D65_WEIGHTS, d65_weights_at

# ── Primitives (fully generic) ──────────────────────────────────────────────

def fov_heatmap_figure(
    ax_deg: np.ndarray,
    ay_deg: np.ndarray,
    intensity: np.ndarray,
    title: str,
    zmin: float | None = None,
    zmax: float | None = None,
) -> go.Figure:
    """Heatmap on a FOV angular grid."""
    fig = go.Figure(go.Heatmap(
        z=intensity,
        x=ax_deg,
        y=ay_deg,
        colorscale="Viridis",
        zmin=zmin, zmax=zmax,
        colorbar=dict(title="intensity"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="FOV x (deg)",
        yaxis_title="FOV y (deg)",
        yaxis_scaleanchor="x",
        width=460, height=420,
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
) -> go.Figure:
    """Heatmap on a pupil spatial grid. ``values`` shape ``(len(y), len(x))``."""
    fig = go.Figure(go.Heatmap(
        z=values,
        x=x_mm,
        y=y_mm,
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        colorbar=dict(title=colorbar_title),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="pupil x (mm)",
        yaxis_title="pupil y (mm)",
        yaxis_scaleanchor="x",
        width=460, height=420,
        margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig


# ── Composers ────────────────────────────────────────────────────────────────

def _reshape_pupil(response: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """(S, A, C) → (ny, nx, A, C)."""
    S, A, C = response.shape
    if ny * nx != S:
        raise ValueError(f"S={S} does not match pupil grid {ny}x{nx}")
    return response.reshape(ny, nx, A, C)


def _reshape_fov(response: np.ndarray, n_fov_y: int, n_fov_x: int) -> np.ndarray:
    """(S, A, C) → (S, n_fov_y, n_fov_x, C)."""
    S, A, C = response.shape
    if n_fov_y * n_fov_x != A:
        raise ValueError(f"A={A} does not match FOV grid {n_fov_y}x{n_fov_x}")
    return response.reshape(S, n_fov_y, n_fov_x, C)


def _get_d65(wavelengths_nm: np.ndarray | None) -> np.ndarray:
    """Return normalized D65 weights matching the response channels."""
    if wavelengths_nm is not None and len(wavelengths_nm) != 3:
        import jax.numpy as jnp
        wls = jnp.asarray(wavelengths_nm) * nm
        d65 = np.asarray(d65_weights_at(wls))
    else:
        d65 = np.asarray(D65_WEIGHTS)
    return d65 / d65.sum()


def pupil_total_figure(
    response: np.ndarray,
    pupil_x_mm: np.ndarray,
    pupil_y_mm: np.ndarray,
) -> go.Figure:
    """Pupil heatmap — total intensity summed over all channels, FOV-averaged."""
    ny, nx = len(pupil_y_mm), len(pupil_x_mm)
    grid = _reshape_pupil(response, ny, nx)       # (ny, nx, A, C)
    avg = grid.mean(axis=2).sum(axis=-1)          # (ny, nx)

    return pupil_heatmap_figure(
        pupil_x_mm, pupil_y_mm, avg,
        title="Pupil — total intensity (FOV-averaged)",
        colorbar_title="avg intensity",
    )


def white_balance_figure(
    response: np.ndarray,
    pupil_x_mm: np.ndarray,
    pupil_y_mm: np.ndarray,
    wavelengths_nm: np.ndarray | None = None,
) -> go.Figure:
    """Per-cell distance from D65 spectral distribution (FOV-averaged)."""
    d65 = _get_d65(wavelengths_nm)

    ny, nx = len(pupil_y_mm), len(pupil_x_mm)
    grid = _reshape_pupil(response, ny, nx)       # (ny, nx, A, C)
    avg = grid.mean(axis=2)                       # (ny, nx, C)
    total = avg.sum(axis=-1, keepdims=True) + 1e-12
    ratios = avg / total                          # (ny, nx, C)
    err = np.linalg.norm(ratios - d65[None, None, :], axis=-1)  # (ny, nx)

    return pupil_heatmap_figure(
        pupil_x_mm, pupil_y_mm, err,
        title="White-balance error (L2 distance from D65)",
        colorscale="Hot",
        colorbar_title="error",
    )


def fov_global_figures(
    response: np.ndarray,
    scan_angles: np.ndarray,
    wavelengths_nm: np.ndarray | None = None,
) -> list[go.Figure]:
    """Two global FOV maps: pupil-averaged and worst-cell."""
    d65 = _get_d65(wavelengths_nm)

    n_fov_y, n_fov_x = scan_angles.shape[:2]
    ax_deg = np.degrees(scan_angles[0, :, 0])
    ay_deg = np.degrees(scan_angles[:, 0, 1])

    reshaped = _reshape_fov(response, n_fov_y, n_fov_x)   # (S, ny_f, nx_f, C)

    avg = reshaped.sum(axis=-1).mean(axis=0)              # (ny_f, nx_f)

    normed = reshaped / (d65[None, None, None, :] + 1e-12)
    worst_channel = normed.min(axis=-1)                    # (S, ny_f, nx_f)
    worst_cell = worst_channel.min(axis=0)                 # (ny_f, nx_f)

    return [
        fov_heatmap_figure(ax_deg, ay_deg, avg,
                           title="FOV — pupil-averaged (sum of channels)"),
        fov_heatmap_figure(ax_deg, ay_deg, worst_cell,
                           title="FOV — worst pupil cell (D65-normalized min channel)"),
    ]


def per_cell_fov_figure(
    response: np.ndarray,
    scan_angles: np.ndarray,
    pupil_x_mm: np.ndarray,
    pupil_y_mm: np.ndarray,
) -> go.Figure:
    """Per-cell FOV heatmap with a slider stepping through pupil cells."""
    n_fov_y, n_fov_x = scan_angles.shape[:2]
    ax_deg = np.degrees(scan_angles[0, :, 0])
    ay_deg = np.degrees(scan_angles[:, 0, 1])
    ny, nx = len(pupil_y_mm), len(pupil_x_mm)

    pupil_grid = _reshape_pupil(response, ny, nx)         # (ny, nx, A, C)
    fov_grid = pupil_grid.reshape(ny, nx, n_fov_y, n_fov_x, -1)
    per_cell = fov_grid.sum(axis=-1)                      # (ny, nx, ny_f, nx_f)

    zmax = float(per_cell.max())

    fig = go.Figure()
    traces = []
    steps = []
    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            traces.append(go.Heatmap(
                z=per_cell[iy, ix],
                x=ax_deg, y=ay_deg,
                colorscale="Viridis",
                zmin=0.0, zmax=zmax,
                visible=(idx == 0),
                colorbar=dict(title="intensity"),
            ))
            label = f"({pupil_x_mm[ix]:.1f}, {pupil_y_mm[iy]:.1f})"
            steps.append(dict(
                method="update",
                label=label,
                args=[{"visible": [i == idx for i in range(ny * nx)]},
                      {"title": f"Per-cell FOV — pupil {label} mm"}],
            ))
            idx += 1

    for t in traces:
        fig.add_trace(t)

    fig.update_layout(
        title=f"Per-cell FOV — pupil {pupil_x_mm[0]:.1f}, {pupil_y_mm[0]:.1f} mm",
        xaxis_title="FOV x (deg)",
        yaxis_title="FOV y (deg)",
        yaxis_scaleanchor="x",
        width=520, height=520,
        margin=dict(l=50, r=20, t=60, b=90),
        sliders=[dict(active=0, pad={"t": 40}, steps=steps,
                      currentvalue=dict(prefix="cell: "))],
    )
    return fig


# ── Top-level ────────────────────────────────────────────────────────────────

def render_report(run_dir: Path | str) -> Path:
    """Read ``run_dir`` and write ``run_dir/report.html``."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    data = np.load(run_dir / "response.npz")

    response = data["response"]
    pupil_x_mm = data["pupil_x_mm"]
    pupil_y_mm = data["pupil_y_mm"]
    scan_angles = data["scan_angles"]
    wavelengths_nm = data.get("wavelengths_nm", None)

    figs: list[tuple[str, go.Figure]] = []
    figs.append(("pupil_total",
                 pupil_total_figure(response, pupil_x_mm, pupil_y_mm)))
    figs.append(("white_balance",
                 white_balance_figure(response, pupil_x_mm, pupil_y_mm,
                                     wavelengths_nm=wavelengths_nm)))
    figs += list(zip(
        ["fov_avg", "fov_worst"],
        fov_global_figures(response, scan_angles,
                           wavelengths_nm=wavelengths_nm),
        strict=True,
    ))
    figs.append(("per_cell_fov",
                 per_cell_fov_figure(response, scan_angles,
                                     pupil_x_mm, pupil_y_mm)))

    html_parts = [_html_header(manifest)]
    for i, (_, f) in enumerate(figs):
        include_js = "cdn" if i == 0 else False
        html_parts.append(
            f.to_html(full_html=False, include_plotlyjs=include_js))

    out = run_dir / "report.html"
    out.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Apollo14 run report</title>"
        "<style>body{font-family:sans-serif;max-width:1100px;margin:2em auto;}"
        "h1{border-bottom:1px solid #ccc;padding-bottom:.3em;}"
        ".meta{color:#666;font-size:.9em;margin-bottom:2em;}"
        ".row{display:flex;flex-wrap:wrap;gap:1em;}</style>"
        "</head><body>" + "\n".join(html_parts) + "</body></html>"
    )
    return out


def _html_header(manifest: dict) -> str:
    return (
        "<h1>Apollo14 run report</h1>"
        f"<div class='meta'>"
        f"<div><b>git sha:</b> <code>{manifest['git_sha']}</code></div>"
        f"<div><b>timestamp:</b> {manifest['timestamp']}</div>"
        f"<div><b>FOV:</b> "
        f"{np.degrees(manifest['scan']['x_fov']):.1f}° × "
        f"{np.degrees(manifest['scan']['y_fov']):.1f}° "
        f"({manifest['scan']['num_x']}×{manifest['scan']['num_y']} grid)</div>"
        "</div>"
    )
