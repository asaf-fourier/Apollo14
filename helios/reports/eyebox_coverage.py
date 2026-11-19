"""Eyebox beam-coverage report: where every ray lands on the pupil, per FOV angle.

The pupil-brightness report shows *how much* light each eyebox cell gets; this
one shows *where the rays are*, so gaps between the mirrors' footprints are
visible directly. For every FOV scan direction it traces each mirror's
reflected branch onto the pupil plane and records each ray's hit point.

The HTML has three parts:

1. **Ray hits per FOV angle** (slider) — every ray's incident point on the
   pupil plane, coloured by the mirror that reflected it, with the target
   eyebox outlined. Empty regions inside the eyebox are gaps; colours show which
   mirror footprints meet or overlap there.
2. **Coverage per FOV angle** — the fraction of eyebox cells (``cell_size``
   squares) that at least one ray lands in, for every scan direction.
3. **Gap map** — for each eyebox cell, at how many FOV angles it gets no ray.

An empty cell is a real gap only if the rays land closer together than a cell:
any interval at least one ray spacing long contains a ray, so with the *coarser*
of the two lattice spacings below the cell size, sampling alone cannot leave a
cell empty. The report measures both spacings on the pupil (column and row
steps of the beam lattice, fitted per mirror and angle) and warns if the
coarser one is not smaller than a cell.

Pupil-plane coordinates use the pupil's local axes
(:func:`apollo14.geometry.compute_local_axes`), the same frame as the other
pupil reports: +v is world-up, +u is the wearer's right (looking out through
the combiner).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import plotly.graph_objects as go

from apollo14.geometry import compute_local_axes
from apollo14.trace import prepare_route, trace_rays
from apollo14.units import deg, mm

# Validated categorical slots (light surface), one per mirror, fixed order.
MIRROR_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#6250d6", "#e34948")
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"

_trace_rays_jit = jax.jit(trace_rays)


@dataclass
class DirectionHits:
    """Ray hits on the pupil plane for one FOV scan direction."""
    angle_x_deg: float
    angle_y_deg: float
    u_mm: np.ndarray          # (N,) pupil-local horizontal coordinate
    v_mm: np.ndarray          # (N,) pupil-local vertical coordinate
    mirror_index: np.ndarray  # (N,) which mirror's branch the ray came from
    intensity: np.ndarray     # (N,) relative intensity at the pupil
    ray_steps_mm: tuple[float, float]  # pupil spacing between lattice columns, rows


def trace_eyebox_hits(projector, fov_grid, branch_routes, pupil,
                      wavelength) -> list[DirectionHits]:
    """Trace every mirror branch at every FOV direction; collect pupil hits.

    ``branch_routes`` are the unprepared per-mirror routes ending on the
    pupil (e.g. :func:`helios.merit.build_combiner_branch_routes`), in mirror
    order. Only rays that reach the pupil with non-zero intensity are kept.
    """
    prepared = [prepare_route(route, wavelength) for route in branch_routes]
    center = np.asarray(pupil.position, dtype=float)
    local_u, local_v = (np.asarray(axis, dtype=float)
                        for axis in compute_local_axes(pupil.normal))
    angles = np.asarray(fov_grid.angles_grid).reshape(-1, 2)

    all_hits = []
    for direction, (angle_x, angle_y) in zip(fov_grid.flat_directions, angles, strict=True):
        rays = projector.generate_rays(direction=direction, wavelength=wavelength)
        u_parts, v_parts, mirror_parts, intensity_parts, steps = [], [], [], [], []
        for mirror_idx, route in enumerate(prepared):
            result = _trace_rays_jit(route, rays)
            reached = (np.asarray(result.valids[:, -1])
                       & (np.asarray(result.final_intensity) > 0))
            offsets = np.asarray(result.final_pos)[reached] - center
            u_parts.append(offsets @ local_u)
            v_parts.append(offsets @ local_v)
            mirror_parts.append(np.full(int(reached.sum()), mirror_idx))
            intensity_parts.append(np.asarray(result.final_intensity)[reached])
            steps.append(_lattice_steps(np.nonzero(reached)[0], projector.nx,
                                        u_parts[-1], v_parts[-1]))
        valid_steps = [step for step in steps if step is not None]
        all_hits.append(DirectionHits(
            angle_x_deg=float(angle_x) / deg, angle_y_deg=float(angle_y) / deg,
            u_mm=np.concatenate(u_parts) / mm, v_mm=np.concatenate(v_parts) / mm,
            mirror_index=np.concatenate(mirror_parts),
            intensity=np.concatenate(intensity_parts),
            ray_steps_mm=(tuple(np.median(valid_steps, axis=0) / mm) if valid_steps
                          else (float("nan"), float("nan")))))
    return all_hits


def _lattice_steps(ray_indices, lattice_nx, u_values, v_values):
    """Pupil distance between neighbouring beam columns and rows (affine fit).

    Rays launch on an (ny, nx) lattice, row-major; a collimated beam reflected by
    flat mirrors maps that lattice affinely onto the pupil, so fitting
    ``[u, v] ≈ origin + column·S_col + row·S_row`` gives both lattice steps.
    """
    if len(ray_indices) < 4:
        return None
    columns, rows = ray_indices % lattice_nx, ray_indices // lattice_nx
    if np.ptp(columns) == 0 or np.ptp(rows) == 0:
        return None
    design = np.stack([np.ones_like(columns), columns, rows], axis=1).astype(float)
    solution, *_ = np.linalg.lstsq(design, np.stack([u_values, v_values], axis=1), rcond=None)
    return float(np.linalg.norm(solution[1])), float(np.linalg.norm(solution[2]))


def _cell_edges(half_extent_mm: float, cell_size_mm: float) -> np.ndarray:
    num_cells = max(1, round(2 * half_extent_mm / cell_size_mm))
    return np.linspace(-half_extent_mm, half_extent_mm, num_cells + 1)


def coverage_statistics(hits: list[DirectionHits], eyebox_half_u_mm: float,
                        eyebox_half_v_mm: float, cell_size_mm: float) -> dict:
    """Per-direction covered fraction and per-cell uncovered counts."""
    edges_u = _cell_edges(eyebox_half_u_mm, cell_size_mm)
    edges_v = _cell_edges(eyebox_half_v_mm, cell_size_mm)
    covered_fraction, uncovered_counts, empty_cells = [], None, []
    for direction_hits in hits:
        counts, _, _ = np.histogram2d(direction_hits.v_mm, direction_hits.u_mm,
                                      bins=(edges_v, edges_u))
        uncovered = counts == 0
        covered_fraction.append(1.0 - float(uncovered.mean()))
        uncovered_counts = (uncovered.astype(int) if uncovered_counts is None
                            else uncovered_counts + uncovered)
        rows, cols = np.nonzero(uncovered)
        empty_cells.append([(0.5 * (edges_u[col] + edges_u[col + 1]),
                             0.5 * (edges_v[row] + edges_v[row + 1]))
                            for row, col in zip(rows, cols, strict=True)])
    return dict(edges_u=edges_u, edges_v=edges_v,
                covered_fraction=np.asarray(covered_fraction),
                uncovered_counts=uncovered_counts, empty_cells=empty_cells)


def ray_pitch_mm(hits: list[DirectionHits]) -> tuple[float, float]:
    """(column step, row step) of the ray lattice on the pupil, worst over angles."""
    steps = np.array([direction_hits.ray_steps_mm for direction_hits in hits], dtype=float)
    return float(np.nanmax(steps[:, 0])), float(np.nanmax(steps[:, 1]))


def _eyebox_outline(half_u, half_v, name="target eyebox", show_legend=True):
    return go.Scatter(
        x=[-half_u, half_u, half_u, -half_u, -half_u],
        y=[-half_v, -half_v, half_v, half_v, -half_v],
        mode="lines", line=dict(color=TEXT_PRIMARY, width=2, dash="dash"),
        name=name, hoverinfo="name", showlegend=show_legend)


def _hits_figure(hits, stats, num_mirrors, eyebox_half_u, eyebox_half_v,
                 view_margin_mm, title) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(_eyebox_outline(eyebox_half_u, eyebox_half_v))
    num_static = len(figure.data)
    for direction_idx, direction_hits in enumerate(hits):
        for mirror_idx in range(num_mirrors):
            selected = direction_hits.mirror_index == mirror_idx
            figure.add_trace(go.Scattergl(
                x=direction_hits.u_mm[selected], y=direction_hits.v_mm[selected],
                mode="markers", visible=direction_idx == 0,
                marker=dict(size=4, color=MIRROR_COLORS[mirror_idx % len(MIRROR_COLORS)],
                            line=dict(color=TEXT_SECONDARY, width=0.3)),
                name=f"mirror {mirror_idx}", legendgroup=f"mirror {mirror_idx}",
                customdata=direction_hits.intensity[selected],
                hovertemplate=(f"mirror {mirror_idx}<br>u %{{x:.2f}} mm<br>v %{{y:.2f}} mm"
                               "<br>intensity %{customdata:.3e}<extra></extra>")))
    steps = []
    for direction_idx, direction_hits in enumerate(hits):
        visible = [True] * num_static + [False] * (len(hits) * num_mirrors)
        for mirror_idx in range(num_mirrors):
            visible[num_static + direction_idx * num_mirrors + mirror_idx] = True
        covered = 100 * stats["covered_fraction"][direction_idx]
        empty = stats["empty_cells"][direction_idx]
        summary = (f"FOV ({direction_hits.angle_x_deg:+.1f}°, {direction_hits.angle_y_deg:+.1f}°): "
                   f"{len(direction_hits.u_mm)} rays on the pupil, eyebox coverage "
                   f"<b>{covered:.1f}%</b>"
                   + (f" — {len(empty)} empty cell{'s' if len(empty) != 1 else ''}" if empty else
                      " — no gaps"))
        steps.append(dict(method="update",
                          label=f"{direction_hits.angle_x_deg:+.1f}°, {direction_hits.angle_y_deg:+.1f}°",
                          args=[{"visible": visible}, {"annotations": [_note(summary)]}]))
    first = hits[0]
    first_summary = (f"FOV ({first.angle_x_deg:+.1f}°, {first.angle_y_deg:+.1f}°): "
                     f"eyebox coverage <b>{100 * stats['covered_fraction'][0]:.1f}%</b>")
    figure.update_layout(
        title=title, template="plotly_white", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(color=TEXT_PRIMARY), height=760,
        xaxis=dict(title="pupil u (mm, + = wearer's right)",
                   range=[-eyebox_half_u - view_margin_mm, eyebox_half_u + view_margin_mm],
                   zeroline=False, gridcolor="#e8e7e3"),
        yaxis=dict(title="pupil v (mm, + = up)", scaleanchor="x", scaleratio=1,
                   range=[-eyebox_half_v - view_margin_mm, eyebox_half_v + view_margin_mm],
                   zeroline=False, gridcolor="#e8e7e3"),
        legend=dict(title="reflected by", itemsizing="constant"),
        annotations=[_note(first_summary)],
        sliders=[dict(active=0, steps=steps, pad=dict(t=50),
                      currentvalue=dict(prefix="FOV angle (about x, about y): "))])
    return figure


def _note(text):
    return dict(text=text, xref="paper", yref="paper", x=0.0, y=1.06, showarrow=False,
                xanchor="left", font=dict(size=13, color=TEXT_PRIMARY))


def _coverage_by_angle_figure(hits, stats) -> go.Figure:
    angles_x = sorted({round(h.angle_x_deg, 6) for h in hits})
    angles_y = sorted({round(h.angle_y_deg, 6) for h in hits})
    grid = np.full((len(angles_y), len(angles_x)), np.nan)
    for direction_hits, covered in zip(hits, stats["covered_fraction"], strict=True):
        grid[angles_y.index(round(direction_hits.angle_y_deg, 6)),
             angles_x.index(round(direction_hits.angle_x_deg, 6))] = 100 * covered
    figure = go.Figure(go.Heatmap(
        x=[f"{a:+.1f}°" for a in angles_x], y=[f"{a:+.1f}°" for a in angles_y], z=grid,
        zmin=min(90.0, float(np.nanmin(grid))), zmax=100.0,
        colorscale=[[0, "#0d366b"], [0.5, "#3987e5"], [1, "#cde2fb"]],
        colorbar=dict(title="covered %"), xgap=2, ygap=2,
        text=np.round(grid, 1), texttemplate="%{text}", textfont=dict(size=11),
        hovertemplate="FOV (%{x}, %{y})<br>eyebox coverage %{z:.1f}%<extra></extra>"))
    figure.update_layout(
        title="Eyebox coverage per FOV angle (100% = every eyebox cell gets a ray)",
        template="plotly_white", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        xaxis_title="FOV angle about x", yaxis_title="FOV angle about y", height=520)
    return figure


def _gap_map_figure(stats, eyebox_half_u, eyebox_half_v, num_directions) -> go.Figure:
    edges_u, edges_v = stats["edges_u"], stats["edges_v"]
    centers_u = 0.5 * (edges_u[:-1] + edges_u[1:])
    centers_v = 0.5 * (edges_v[:-1] + edges_v[1:])
    figure = go.Figure(go.Heatmap(
        x=centers_u, y=centers_v, z=stats["uncovered_counts"], zmin=0,
        zmax=max(1, int(stats["uncovered_counts"].max())),
        colorscale=[[0, "#fcfcfb"], [0.3, "#86b6ef"], [1, "#0d366b"]],
        colorbar=dict(title="FOV angles<br>with no ray"), xgap=1, ygap=1,
        hovertemplate=("cell (%{x:.2f}, %{y:.2f}) mm<br>no ray at %{z} of "
                       f"{num_directions} FOV angles<extra></extra>")))
    figure.add_trace(_eyebox_outline(eyebox_half_u, eyebox_half_v, show_legend=False))
    figure.update_layout(
        title="Gap map — at how many FOV angles each eyebox cell gets no ray (white = never)",
        template="plotly_white", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=620,
        xaxis=dict(title="pupil u (mm, + = wearer's right)"),
        yaxis=dict(title="pupil v (mm, + = up)", scaleanchor="x", scaleratio=1))
    return figure


def render_eyebox_coverage_report(path: Path, hits: list[DirectionHits], *,
                                  num_mirrors: int, eyebox_half_u_mm: float,
                                  eyebox_half_v_mm: float, cell_size_mm: float = 0.5,
                                  view_margin_mm: float = 2.0,
                                  description: str = "") -> dict:
    """Write the coverage HTML to ``path``; return the coverage statistics."""
    stats = coverage_statistics(hits, eyebox_half_u_mm, eyebox_half_v_mm, cell_size_mm)
    column_step, row_step = ray_pitch_mm(hits)
    pitch = max(column_step, row_step)
    stats["ray_pitch_mm"] = pitch
    stats["ray_steps_mm"] = (column_step, row_step)
    worst = int(np.argmin(stats["covered_fraction"]))
    header = (
        f"<h2>Eyebox beam coverage</h2><p>{description}</p>"
        f"<p>Eyebox {2 * eyebox_half_u_mm:.1f} × {2 * eyebox_half_v_mm:.1f} mm in "
        f"{cell_size_mm:g} mm cells; {len(hits)} FOV angles. Coverage: worst "
        f"<b>{100 * stats['covered_fraction'][worst]:.1f}%</b> at FOV "
        f"({hits[worst].angle_x_deg:+.1f}°, {hits[worst].angle_y_deg:+.1f}°), mean "
        f"{100 * stats['covered_fraction'].mean():.1f}%; "
        f"{int(np.sum(stats['covered_fraction'] < 1.0))} of {len(hits)} angles have gaps.</p>"
        f"<p>Ray spacing on the pupil (worst over angles): {column_step:.3f} mm between "
        f"beam columns, {row_step:.3f} mm between rows"
        + (f" — <b>warning: not finer than the {cell_size_mm:g} mm cells, so empty cells "
           "may be sampling artefacts rather than real gaps</b>" if pitch >= cell_size_mm else
           f" — both finer than the {cell_size_mm:g} mm cells, so every empty cell is a real "
           "gap in the beam")
        + ".</p>")
    figures = [
        _hits_figure(hits, stats, num_mirrors, eyebox_half_u_mm, eyebox_half_v_mm,
                     view_margin_mm, "Ray hits on the pupil plane, coloured by mirror"),
        _coverage_by_angle_figure(hits, stats),
        _gap_map_figure(stats, eyebox_half_u_mm, eyebox_half_v_mm, len(hits)),
    ]
    body = "".join(figure.to_html(full_html=False, include_plotlyjs=(index == 0 and "cdn"))
                   for index, figure in enumerate(figures))
    Path(path).write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Eyebox coverage</title>"
        f"<style>body{{font-family:sans-serif;background:{SURFACE};color:{TEXT_PRIMARY};"
        "margin:1.5em}}</style></head><body>" + header + body + "</body></html>")
    return stats
