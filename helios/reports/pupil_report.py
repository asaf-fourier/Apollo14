"""Top-level pupil report renderer.

Reads ``manifest.json`` + ``response.npz`` from a run directory and writes
``pupil_report.html`` containing three pages:

  1. **Summary** — pupil overview (luminance + ΔD65 with eyebox contour),
     eyebox-quality CDF, and headline numbers.
  2. **Per-cell drill-down** — ΔD65 over FOV per pupil cell with
     brightness contours, plus the visible-color rendering.
  3. **Design diagnostics** — mirror reflectance curves and FOV-global
     views.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from helios.reports.figures.eyebox_cdf import eyebox_quality_cdf_figure
from helios.reports.figures.global_fov import fov_global_figures
from helios.reports.figures.mirrors import mirror_reflectance_figure
from helios.reports.figures.overview import (
    pupil_brightness_figure,
    pupil_d65_distance_figure,
)
from helios.reports.figures.per_cell import per_cell_d65_fov_figure
from helios.reports.figures.projector import (
    mirror_input_spectrum_figure,
    projector_spectrum_figure,
)
from helios.reports.figures.visible_color import per_cell_visible_color_figure
from helios.reports.headline import (
    compute_headline_numbers,
    headline_numbers_html,
)


def render_pupil_report(
    run_dir: Path | str,
    *,
    eyebox_threshold_nits: float | None = None,
    d65_tolerance: float | None = 0.05,
) -> Path:
    """Read ``run_dir`` and write ``run_dir/pupil_report.html``.

    Args:
        run_dir: a directory written by :func:`helios.io.save_run`,
            containing ``manifest.json`` and ``response.npz``.
        eyebox_threshold_nits: optional luminance contour threshold for
            the pupil overview brightness map. ``None`` skips the contour.
        d65_tolerance: ΔD65 contour threshold for the overview color map
            and the second curve in the eyebox-quality CDF.

    Returns:
        Path to the written ``pupil_report.html``.
    """
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    data = np.load(run_dir / "response.npz")

    response = data["response"]
    pupil_x_mm = data["pupil_x_mm"]
    pupil_y_mm = data["pupil_y_mm"]
    scan_angles = data["scan_angles"]
    wavelengths_nm = data.get("wavelengths_nm", None)

    pages = [
        ("Summary", _summary_page(
            response, pupil_x_mm, pupil_y_mm, wavelengths_nm,
            eyebox_threshold_nits, d65_tolerance,
        )),
        ("Per-cell drill-down", _drill_down_page(
            response, scan_angles, pupil_x_mm, pupil_y_mm, wavelengths_nm,
        )),
        ("Design diagnostics", _diagnostics_page(
            manifest, response, scan_angles, wavelengths_nm,
        )),
    ]

    headline = compute_headline_numbers(manifest, response, wavelengths_nm)
    body = [_html_header(manifest), headline_numbers_html(headline)]
    figure_index = 0
    for page_title, blocks in pages:
        body.append(f"<h2>{page_title}</h2>")
        for title, caption, fig in blocks:
            body.append(f"<h3>{title}</h3>")
            if caption:
                body.append(f"<p class='caption'>{caption}</p>")
            include_js = "cdn" if figure_index == 0 else False
            body.append(fig.to_html(full_html=False, include_plotlyjs=include_js))
            figure_index += 1

    # Append the optimization's design variables when an optimization
    # report sits next to the run files. The optimization-report file is
    # only produced by the optimize_pupil* drivers, not by plain
    # ``save_run`` invocations, so this block is skipped when absent.
    opt_report_path = run_dir / "optimization_report.json"
    if opt_report_path.exists():
        opt_report = json.loads(opt_report_path.read_text())
        body.append(_design_variables_html(opt_report))

    out = run_dir / "pupil_report.html"
    out.write_text(_html_wrap("\n".join(body)))
    return out


# ── Design-variables section ────────────────────────────────────────────────


def _design_variables_html(opt_report: dict) -> str:
    """Render the optimization's final params + bounds as HTML tables.

    Reads ``optimization_report.json`` (written by
    :func:`helios.io.save_optimization_report`). All length fields in
    that file are stored in internal units (mm), so wavelengths and
    sigmas are converted back to nm here for display.
    """
    final = opt_report["final_params"]
    initial = opt_report["initial_params"]
    bounds = opt_report.get("param_bounds", {})

    spacings_mm = final["spacings"]
    init_spacings_mm = initial["spacings"]
    curve = final["curve"]
    init_curve = initial["curve"]
    amp = curve["amplitude"]                  # (M, B)
    sigma = curve["sigma"]                    # (M, B), mm
    centers = curve["centers"]                # (M, B), mm
    init_amp = init_curve["amplitude"]
    init_sigma = init_curve["sigma"]

    # Internal units: 1 mm = 1.0, 1 nm = 1e-6. Convert sigmas/centers to nm.
    nm_to_internal = 1e-6
    fwhm_factor = 2.0 * (2.0 * np.log(2.0)) ** 0.5  # 2·sqrt(2·ln 2) ≈ 2.355
    centers_nm = [[c / nm_to_internal for c in row] for row in centers]
    fwhm_nm = [[s * fwhm_factor / nm_to_internal for s in row] for row in sigma]
    init_fwhm_nm = [[s * fwhm_factor / nm_to_internal for s in row]
                    for row in init_sigma]

    num_mirrors = len(amp)
    num_basis = len(amp[0]) if amp else 0

    blocks = ["<h2>Design variables</h2>"]
    blocks.append(
        "<p class='caption'>Final parameter values from the optimization "
        "run. Each row is one mirror; columns are the basis Gaussians at "
        "their fixed center wavelengths. Initial values are shown in "
        "parentheses for reference. Yellow cells are pegged at a "
        "<code>ParamBounds</code> upper limit — when many cells are "
        "yellow the bound is the bottleneck, not the loss landscape.</p>"
    )

    # Bounds box at the top
    if bounds:
        blocks.append(
            "<table class='design-vars'>"
            f"<tr><th>amplitude bounds</th>"
            f"<td>{bounds.get('amplitude_min', 0):.3f} – "
            f"{bounds.get('amplitude_max', 0):.3f}</td></tr>"
            f"<tr><th>FWHM bounds</th>"
            f"<td>{bounds.get('fwhm_min_nm', 0):.1f} – "
            f"{bounds.get('fwhm_max_nm', 0):.1f} nm</td></tr>"
            f"<tr><th>spacing bounds</th>"
            f"<td>{bounds.get('spacing_min_mm', 0):.3f} – "
            f"{bounds.get('spacing_max_mm', 0):.3f} mm</td></tr>"
            "</table>"
        )

    amp_max = float(bounds.get("amplitude_max", float("inf")))
    fwhm_max = float(bounds.get("fwhm_max_nm", float("inf")))

    # Spacings table
    blocks.append("<h3>Mirror inter-spacings</h3>")
    spacing_max_mm = float(bounds.get("spacing_max_mm", float("inf")))
    spacing_min_mm = float(bounds.get("spacing_min_mm", float("-inf")))
    rows = []
    rows.append("<tr><th>gap</th>"
                + "".join(f"<th>m{i}→m{i+1}</th>" for i in range(len(spacings_mm)))
                + "</tr>")
    cells = []
    for i, (s, s_init) in enumerate(zip(spacings_mm, init_spacings_mm)):
        pegged = (s >= spacing_max_mm - 1e-4) or (s <= spacing_min_mm + 1e-4)
        style = " class='pegged'" if pegged else ""
        cells.append(f"<td{style}>{s:.3f}<br>"
                     f"<span class='init'>({s_init:.3f})</span></td>")
    rows.append("<tr><th>mm</th>" + "".join(cells) + "</tr>")
    blocks.append("<table class='design-vars'>" + "".join(rows) + "</table>")

    # Per-mirror amplitude + FWHM tables
    header = ("<tr><th>mirror</th>"
              + "".join(f"<th>{centers_nm[0][b]:.0f} nm</th>"
                        for b in range(num_basis))
              + "</tr>")

    blocks.append("<h3>Reflectance amplitudes (per Gaussian basis)</h3>")
    rows = [header]
    for m in range(num_mirrors):
        cells = [f"<th>m{m}</th>"]
        for b in range(num_basis):
            v = float(amp[m][b])
            v_init = float(init_amp[m][b])
            pegged = v >= amp_max - 1e-4
            style = " class='pegged'" if pegged else ""
            cells.append(f"<td{style}>{v:.4f}<br>"
                         f"<span class='init'>({v_init:.4f})</span></td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    blocks.append("<table class='design-vars'>" + "".join(rows) + "</table>")

    blocks.append("<h3>Reflectance FWHM (nm)</h3>")
    rows = [header]
    for m in range(num_mirrors):
        cells = [f"<th>m{m}</th>"]
        for b in range(num_basis):
            v = float(fwhm_nm[m][b])
            v_init = float(init_fwhm_nm[m][b])
            pegged = v >= fwhm_max - 1e-2
            style = " class='pegged'" if pegged else ""
            cells.append(f"<td{style}>{v:.1f}<br>"
                         f"<span class='init'>({v_init:.1f})</span></td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    blocks.append("<table class='design-vars'>" + "".join(rows) + "</table>")

    return "\n".join(blocks)


# ── Page composers ──────────────────────────────────────────────────────────


def _summary_page(response, pupil_x_mm, pupil_y_mm, wavelengths_nm,
                  eyebox_threshold_nits, d65_tolerance):
    luminance_caption = (
        "Average brightness each pupil cell delivers to the eye, weighted "
        "by the photopic V(λ) curve so values reflect what a viewer "
        "actually perceives. Read it as: where in the eyebox is the image "
        "bright? The dashed contour, when present, marks the eyebox "
        "acceptance threshold — cells inside it clear the brightness target."
    )
    d65_caption = (
        "Per-cell distance from the D65 white point in normalized-channel "
        "space, averaged over the FOV. 0 means the cell looks neutral white; "
        "values near √2 mean it's pathologically tinted toward one color. "
        "The dashed contour marks the ΔD65 tolerance — cells inside it pass "
        "the white-balance check."
    )
    cdf_caption = (
        "How big is the usable eyebox at each brightness threshold? Each "
        "curve gives the fraction of pupil cells clearing a moving "
        "threshold T. The blue curve uses brightness alone; the red curve "
        "additionally requires the cell to be white-balanced (ΔD65 below "
        "tolerance). Read the y-coordinate at your acceptance threshold "
        "to size the usable eyebox."
    )
    return [
        ("Pupil luminance map", luminance_caption,
         pupil_brightness_figure(
             response, pupil_x_mm, pupil_y_mm,
             wavelengths_nm=wavelengths_nm,
             threshold_nits=eyebox_threshold_nits,
         )),
        ("Pupil color drift (ΔD65)", d65_caption,
         pupil_d65_distance_figure(
             response, pupil_x_mm, pupil_y_mm,
             wavelengths_nm=wavelengths_nm,
             tolerance=d65_tolerance,
         )),
        ("Eyebox quality CDF", cdf_caption,
         eyebox_quality_cdf_figure(
             response, pupil_x_mm, pupil_y_mm,
             wavelengths_nm=wavelengths_nm,
             d65_tolerance=d65_tolerance,
         )),
    ]


def _drill_down_page(response, scan_angles, pupil_x_mm, pupil_y_mm,
                     wavelengths_nm):
    d65_fov_caption = (
        "For one pupil cell at a time (use the slider), the heatmap shows "
        "color drift across all FOV angles, and the cyan dashed contours "
        "show where that same cell is brighter or dimmer than its FOV mean "
        "(at 50%, 75%, 100%, 125%, 150%). Surfaces per-cell defects the "
        "FOV-averaged overview hides — a cell that's white on average but "
        "pink at one corner shows up here."
    )
    visible_color_caption = (
        "What a viewer with their eye at this pupil cell would actually "
        "see across the FOV, rendered in sRGB from the cell's spectral "
        "response via CIE 1931 color matching. Each cell is normalized so "
        "its brightest pixel saturates — comparison is qualitative across "
        "cells (same hue ⇒ same chromaticity) but not absolute brightness."
    )
    blocks = [
        ("Per-cell color drift across FOV", d65_fov_caption,
         per_cell_d65_fov_figure(
             response, scan_angles, pupil_x_mm, pupil_y_mm,
             wavelengths_nm=wavelengths_nm,
         )),
    ]
    if wavelengths_nm is not None:
        blocks.append((
            "Per-cell rendered color (sRGB)", visible_color_caption,
            per_cell_visible_color_figure(
                response, scan_angles, pupil_x_mm, pupil_y_mm,
                wavelengths_nm=wavelengths_nm,
            )))
    return blocks


def _diagnostics_page(manifest, response, scan_angles, wavelengths_nm):
    mirror_caption = (
        "One curve per partial mirror, read straight from the run "
        "manifest. Reveals what the optimizer ended up with: per-color "
        "Gaussian peaks, mirrors that specialized on a single channel, "
        "and overall reflectance levels. The legend lists the mirrors in "
        "stack order from projector to pupil."
    )
    projector_caption = (
        "The projector's emitted spectrum (peak-normalized). The R/G/B "
        "primaries are visible as three distinct bumps; the relative "
        "heights determine how much each color the optimizer has to "
        "work with."
    )
    mirror_input_caption = (
        "What spectrum each mirror actually sees. The first curve is the "
        "projector; each subsequent curve is the previous one multiplied "
        "by ``(1 − r_j(λ))`` — the fraction the upstream mirror "
        "transmitted. The dashed curve is what exits the stack after the "
        "last mirror. Reveals where the projector's light is being "
        "absorbed and which mirrors run on a starved spectrum."
    )
    fov_total_caption = (
        "Total light delivered at each FOV direction, summed over all "
        "channels and averaged over pupil cells. Inverts the pupil/FOV "
        "split — instead of asking which pupil cells are bright, it asks "
        "which gaze directions deliver the most light. Dim FOV corners "
        "indicate the optimization's hardest geometry."
    )
    fov_worst_caption = (
        "At each FOV direction, the dimmest D65-normalized channel of "
        "the worst pupil cell. Surfaces FOV angles where the eye sees a "
        "weak or tinted image somewhere on the pupil — even when the "
        "FOV-averaged or pupil-averaged views look healthy."
    )
    fov_total_fig, fov_worst_fig = fov_global_figures(
        response, scan_angles, wavelengths_nm=wavelengths_nm)
    return [
        ("Mirror reflectance r(λ)", mirror_caption,
         mirror_reflectance_figure(manifest)),
        ("Projector spectrum", projector_caption,
         projector_spectrum_figure(manifest)),
        ("Spectrum incident on each mirror", mirror_input_caption,
         mirror_input_spectrum_figure(manifest)),
        ("FOV total intensity (pupil-averaged)", fov_total_caption,
         fov_total_fig),
        ("FOV worst-cell weakness map", fov_worst_caption,
         fov_worst_fig),
    ]


# ── HTML scaffolding ────────────────────────────────────────────────────────


def _html_header(manifest: dict) -> str:
    return (
        "<h1>Apollo14 pupil report</h1>"
        "<div class='meta'>"
        f"<div><b>git sha:</b> <code>{manifest['git_sha']}</code></div>"
        f"<div><b>timestamp:</b> {manifest['timestamp']}</div>"
        f"<div><b>FOV:</b> "
        f"{np.degrees(manifest['scan']['x_fov']):.1f}° × "
        f"{np.degrees(manifest['scan']['y_fov']):.1f}° "
        f"({manifest['scan']['num_x']}×{manifest['scan']['num_y']} grid)</div>"
        "</div>"
    )


def _html_wrap(body: str) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Apollo14 pupil report</title>"
        "<style>body{font-family:sans-serif;max-width:1100px;margin:2em auto;"
        "line-height:1.45;color:#222;}"
        "h1{border-bottom:1px solid #ccc;padding-bottom:.3em;}"
        "h2{margin-top:2em;border-bottom:1px solid #eee;padding-bottom:.2em;}"
        "h3{margin-top:1.6em;margin-bottom:.2em;color:#333;font-size:1.1em;}"
        "p.caption{margin:0 0 .8em 0;color:#555;font-size:.92em;"
        "max-width:780px;}"
        ".meta{color:#666;font-size:.9em;margin-bottom:2em;}"
        ".row{display:flex;flex-wrap:wrap;gap:1em;}"
        ".headline{background:#f7f7f9;border:1px solid #e1e1e8;"
        "border-radius:6px;padding:1em 1.2em;margin-bottom:2em;}"
        ".headline-table{border-collapse:collapse;font-size:.95em;}"
        ".headline-table th{text-align:left;font-weight:600;"
        "padding:.2em .9em .2em 0;color:#444;}"
        ".headline-table td{padding:.2em 0;font-variant-numeric:tabular-nums;}"
        "table.design-vars{border-collapse:collapse;margin:.5em 0 1.5em 0;"
        "font-size:.9em;font-variant-numeric:tabular-nums;}"
        "table.design-vars th,table.design-vars td{padding:.3em .7em;"
        "border:1px solid #ddd;text-align:right;}"
        "table.design-vars th{background:#f3f3f6;font-weight:600;color:#333;}"
        "table.design-vars td.pegged{background:#fff7c4;}"
        "table.design-vars span.init{color:#888;font-size:.85em;}</style>"
        f"</head><body>{body}</body></html>"
    )
