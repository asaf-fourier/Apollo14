"""Portable fixed-eye-position views of Zemax FOV response data.

The normal position-space report fixes one exported FOV direction and displays
all detector positions.  This module writes the transposed view for configured
eye windows: one eye position is fixed and the diagram axes are FOV x and FOV
y.  It consumes the Zemax-derived ``cell_response`` array calculated by
``zemax_pupil_report``; it does not run another ray trace.

Only the Python standard library is used because this file is copied into every
generated bundle and runs on the OpticStudio computer.
"""

import csv
import json
from pathlib import Path

try:
    from .html_plotting import REPORT_LAYOUT_VERSION
except ImportError:  # Standalone copy in a generated Zemax bundle.
    from html_plotting import REPORT_LAYOUT_VERSION  # type: ignore[no-redef]


def write_angular_response_report(report, output_directory):
    """Write one interactive angular-slice viewer and its exact CSV data.

    Args:
        report: Completed ``apollo14.zemax_pupil_report`` mapping. Its response
            array is ordered ``eye_y, eye_x, fov_y, fov_x``.
        output_directory: Root directory containing the parent pupil report.

    Returns:
        A mapping containing parent-report metadata and absolute paths for the
        index writer. The caller adds ``metadata`` and ``files`` to the parent
        JSON before serializing it.
    """
    _validate_report_shape(report)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    report_name = report["configuration"]["report_name"]
    artifact_stem = f"{report_name}_angular_response"
    html_path = output_directory / "angular_response.html"
    csv_path = output_directory / "angular_response.csv"
    slices_directory = output_directory / "angular_response_slices"
    slices_directory.mkdir(exist_ok=True)
    for stale_slice in slices_directory.glob("eye_y_*_x_*.csv"):
        stale_slice.unlink()

    slices = _write_slice_csv_files(report, slices_directory, output_directory)
    _write_long_form_csv(csv_path, report)
    _write_html(html_path, report, slices, csv_path.name)

    title = f"FOV-angle response by eyebox position — {report['detector']['label']} detector"
    files = {
        "angular_response_html": html_path.name,
        "angular_response_csv": csv_path.name,
        "angular_response_slices_directory": slices_directory.name,
    }
    metadata = {
        "name": "fixed_eye_position_fov_response",
        "description": (
            "Each slice fixes one configured eye-window position and plots the "
            "Zemax-derived response over exported FOV x and FOV y samples."
        ),
        "source_array": "arrays.cell_response.values",
        "source_array_axes": ["eye_y", "eye_x", "fov_y", "fov_x"],
        "slice_axes": ["eye_y", "eye_x"],
        "diagram_axes": ["fov_y", "fov_x"],
        "unit": report["quantity"]["unit"],
        "display_scale": "display_scales.cell_response",
        "trace_reused": True,
        "slices": slices,
    }
    return {
        "report_name": artifact_stem,
        "title": title,
        "detail": "Fixed eyebox position; horizontal/vertical axes are FOV x/y",
        "detector_label": report["detector"]["label"],
        "html": str(html_path.resolve()),
        "csv": str(csv_path.resolve()),
        "files": files,
        "metadata": metadata,
    }


def _validate_report_shape(report):
    """Reject incomplete data before creating a partially usable viewer."""
    try:
        eye_x = report["eye_positions"]["x_mm"]
        eye_y = report["eye_positions"]["y_mm"]
        fov_x = report["fov"]["x_deg"]
        fov_y = report["fov"]["y_deg"]
        values = report["arrays"]["cell_response"]["values"]
    except (KeyError, TypeError) as error:
        raise ValueError("Angular response output requires a complete pupil report.") from error

    expected = (len(eye_y), len(eye_x), len(fov_y), len(fov_x))
    if not all(expected):
        raise ValueError("Angular response axes must all contain at least one sample.")
    if len(values) != expected[0]:
        raise ValueError(f"Angular response eye-y dimension does not match {expected!r}.")
    for eye_y_index, eye_row in enumerate(values):
        if len(eye_row) != expected[1]:
            raise ValueError(
                f"Angular response eye-x dimension at y={eye_y_index} does not match {expected!r}."
            )
        for eye_x_index, matrix in enumerate(eye_row):
            if len(matrix) != expected[2] or any(len(row) != expected[3] for row in matrix):
                raise ValueError(
                    "Angular response FOV matrix at eye position "
                    f"({eye_x_index}, {eye_y_index}) does not match {expected!r}."
                )


def _write_slice_csv_files(report, slices_directory, output_directory):
    """Write one FOV matrix per eye position and return its manifest."""
    eye_x = report["eye_positions"]["x_mm"]
    eye_y = report["eye_positions"]["y_mm"]
    fov_x = report["fov"]["x_deg"]
    fov_y = report["fov"]["y_deg"]
    values = report["arrays"]["cell_response"]["values"]
    summaries = {(row["eye_y_index"], row["eye_x_index"]): row for row in report["cell_summary"]}
    slices = []

    for eye_y_index, eye_y_mm in enumerate(eye_y):
        for eye_x_index, eye_x_mm in enumerate(eye_x):
            stem = f"eye_y_{eye_y_index:03d}_x_{eye_x_index:03d}"
            csv_path = slices_directory / f"{stem}.csv"
            _write_matrix_csv(csv_path, fov_x, fov_y, values[eye_y_index][eye_x_index])
            summary = summaries[(eye_y_index, eye_x_index)]
            slices.append(
                {
                    "eye_x_index": eye_x_index,
                    "eye_y_index": eye_y_index,
                    "eye_x_mm": eye_x_mm,
                    "eye_y_mm": eye_y_mm,
                    "mean": summary["mean"],
                    "minimum": summary["minimum"],
                    "maximum": summary["maximum"],
                    "coefficient_of_variation": summary["coefficient_of_variation"],
                    "csv": csv_path.relative_to(output_directory).as_posix(),
                }
            )
    return slices


def _write_matrix_csv(path, fov_x, fov_y, matrix):
    """Write a labelled FOV-y by FOV-x matrix without rounding its values."""
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fov_y_deg\\fov_x_deg", *fov_x])
        # ``strict=`` is intentionally omitted for OpticStudio hosts on Python 3.8.
        for y_value, row in zip(fov_y, matrix):  # noqa: B905
            writer.writerow([y_value, *row])


def _write_long_form_csv(path, report):
    """Write every eye-position/FOV sample as one analysis-friendly row."""
    eye_x = report["eye_positions"]["x_mm"]
    eye_y = report["eye_positions"]["y_mm"]
    fov_x = report["fov"]["x_deg"]
    fov_y = report["fov"]["y_deg"]
    values = report["arrays"]["cell_response"]["values"]
    field_grid = report["fov"]["field_index_by_y_x"]
    field_summary = report["field_summary"]

    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "eye_x_index",
                "eye_y_index",
                "eye_x_mm",
                "eye_y_mm",
                "fov_x_index",
                "fov_y_index",
                "fov_x_deg",
                "fov_y_deg",
                "field_index",
                "source_object_index",
                "source_label",
                "source_sweep_power",
                "eye_window_average_detector_flux_density_source_power_per_mm2",
            ]
        )
        for eye_y_index, eye_y_mm in enumerate(eye_y):
            for eye_x_index, eye_x_mm in enumerate(eye_x):
                matrix = values[eye_y_index][eye_x_index]
                for fov_y_index, fov_y_deg in enumerate(fov_y):
                    for fov_x_index, fov_x_deg in enumerate(fov_x):
                        field_index = field_grid[fov_y_index][fov_x_index]
                        field = field_summary[field_index]
                        writer.writerow(
                            [
                                eye_x_index,
                                eye_y_index,
                                eye_x_mm,
                                eye_y_mm,
                                fov_x_index,
                                fov_y_index,
                                fov_x_deg,
                                fov_y_deg,
                                field_index,
                                field["source_object_index"],
                                field["source_label"],
                                field["source_sweep_power"],
                                matrix[fov_y_index][fov_x_index],
                            ]
                        )


def _write_html(path, report, slices, long_csv_name):
    """Write the offline selector that plots each fixed-position FOV slice."""
    payload = {
        "title": (
            f"FOV-angle response by eyebox position — {report['detector']['label']} detector"
        ),
        "detector_label": report["detector"]["label"],
        "fov": {
            "x_deg": report["fov"]["x_deg"],
            "y_deg": report["fov"]["y_deg"],
        },
        "eye_positions": report["eye_positions"],
        "values": report["arrays"]["cell_response"]["values"],
        "scale": report["display_scales"]["cell_response"],
        "unit": report["quantity"]["unit"],
        "window_area_mm2": report["quantity"]["window_area_mm2"],
        "slices": slices,
        "files": {
            "parent_html": report["files"]["html"],
            "parent_json": report["files"]["json"],
            "long_csv": long_csv_name,
        },
    }
    serialized = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    page = _HTML_TEMPLATE.replace("__REPORT_LAYOUT_VERSION__", REPORT_LAYOUT_VERSION).replace(
        "__ANGULAR_RESPONSE_DATA__", serialized
    )
    Path(path).write_text(page, encoding="utf-8")


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en" data-apollo14-report-layout="__REPORT_LAYOUT_VERSION__">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<title>FOV-angle response by eyebox position</title>
<style>
:root { color-scheme:dark; font-family:Inter,Segoe UI,system-ui,sans-serif;
  background:#0b1020; color:#e8edf8; }
* { box-sizing:border-box; }
body { margin:0; background:linear-gradient(140deg,#111a32,#080b14 60%); }
header, main { max-width:1180px; margin:auto; padding:22px; }
h1 { margin:.1rem 0 .35rem; font-size:clamp(1.55rem,3vw,2.45rem); }
p { color:#b7c2d9; line-height:1.5; }
.tag { display:inline-block; background:#163a59; color:#8cddff; border:1px solid #245f83;
  border-radius:999px; padding:5px 10px; margin:4px 7px 4px 0; font-size:.82rem; }
.links a { color:#79d6ff; margin-right:14px; }
.card { background:#121a2cdd; border:1px solid #27334a; border-radius:13px;
  padding:18px; box-shadow:0 10px 28px #0005; overflow:auto; }
.controls { display:flex; gap:10px; flex-wrap:wrap; align-items:end; margin-bottom:8px; }
label { display:grid; gap:5px; color:#aab8d1; font-size:.84rem; }
select, button { color:#e8edf8; background:#0b1222; border:1px solid #364560;
  border-radius:7px; padding:8px 10px; }
select { min-width:min(27rem,75vw); }
button { cursor:pointer; }
button:hover { border-color:#79d6ff; }
.stats { display:flex; flex-wrap:wrap; gap:9px; margin:12px 0; }
.stat { background:#0a1120; border:1px solid #25324a; border-radius:8px; padding:8px 10px; }
.stat b { display:block; color:#fff; font-variant-numeric:tabular-nums; }
canvas { width:100%; max-width:760px; height:auto; display:block; margin:auto; }
.unit-label { margin:.65rem 0 .35rem; color:#d8e5f7; font-size:.86rem;
  font-weight:600; text-align:center; }
.caption { color:#91a1bc; font-size:.86rem; line-height:1.5; }
code { color:#b7e8ff; }
@media(max-width:560px){ header,main{padding:13px}.card{padding:11px}select{min-width:70vw} }
</style>
</head>
<body>
<header>
  <span class="tag">Calculated on the Zemax computer</span>
  <span class="tag">Fixed eyebox position</span>
  <h1 id="title">FOV-angle response by eyebox position</h1>
  <p>Each slice fixes one physical eye window. The horizontal and vertical axes
  are the exported source FOV angles; every colored cell comes from the
  corresponding Zemax position-space detector trace.</p>
  <p class="links"><a id="parentLink">Parent eyebox report</a>
    <a id="jsonLink">JSON data</a><a id="longCsvLink">All positions CSV</a>
    <a id="sliceCsvLink">Selected position CSV</a></p>
</header>
<main>
  <section class="card">
    <div class="controls">
      <label>Eyebox position<select id="eyePosition"></select></label>
      <button id="previous" type="button">Previous</button>
      <button id="next" type="button">Next</button>
    </div>
    <div class="stats" id="selectedStats"></div>
    <canvas id="angularMap" width="760" height="520"></canvas>
    <div class="unit-label">Color scale units: detector flux density in source-power units/mm².</div>
    <p class="caption" id="caption"></p>
  </section>
</main>
<script type="application/json" id="angularResponseData">__ANGULAR_RESPONSE_DATA__</script>
<script>
const report = JSON.parse(document.getElementById('angularResponseData').textContent);
const COLORS = [[8,20,75],[19,99,164],[31,190,183],[250,220,80],[220,45,34]];
const BACKGROUND = '#0a1120';
const AXIS = '#70809c';
const TEXT = '#b8c5dc';
const MARGIN = {left:82,right:122,top:28,bottom:72};

function formatNumber(value) {
  if (value === null) return 'undefined';
  const number = Number(value);
  const magnitude = Math.abs(number);
  if (magnitude === 0) return '0';
  if (magnitude >= 10000 || magnitude < 0.001) {
    return number.toExponential(2).replace('e+', 'e');
  }
  return Number(number.toPrecision(3)).toString();
}

function formatAxisNumber(value) {
  const number = Math.abs(Number(value)) < 0.005 ? 0 : Number(value);
  return Number(number.toFixed(2)).toString();
}

function colorForFraction(fraction) {
  const bounded = Math.max(0, Math.min(1, fraction));
  const position = bounded * (COLORS.length - 1);
  const lowerIndex = Math.min(COLORS.length - 2, Math.floor(position));
  const amount = position - lowerIndex;
  return `rgb(${COLORS[lowerIndex].map((value, channel) =>
    Math.round(value + (COLORS[lowerIndex + 1][channel] - value) * amount)).join(',')})`;
}

function normalizedValue(value) {
  const span = report.scale.maximum - report.scale.minimum;
  return span === 0 ? 0 : (value - report.scale.minimum) / span;
}

function sampledIndices(count, maximumLabels = 7) {
  if (count <= maximumLabels) return Array.from({length:count}, (_, index) => index);
  const sampled = Array.from({length:maximumLabels}, (_, index) =>
    Math.round(index * (count - 1) / (maximumLabels - 1)));
  return [...new Set(sampled)];
}

function drawRotatedLabel(context, x, y, label) {
  context.save();
  context.translate(x, y);
  context.rotate(-Math.PI / 2);
  context.fillStyle = TEXT;
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(label, 0, 0);
  context.restore();
}

function drawColorScale(context, canvasWidth, plotHeight) {
  const barX = canvasWidth - MARGIN.right + 22;
  const barWidth = 18;
  const samples = 160;
  const sampleHeight = plotHeight / samples;
  for (let index = 0; index < samples; index += 1) {
    const fraction = index / (samples - 1);
    const y = MARGIN.top + plotHeight - (index + 1) * sampleHeight;
    context.fillStyle = colorForFraction(fraction);
    context.fillRect(barX, y, barWidth, sampleHeight + 1);
  }
  context.strokeStyle = AXIS;
  context.strokeRect(barX, MARGIN.top, barWidth, plotHeight);
  context.fillStyle = TEXT;
  context.font = '12px Segoe UI';
  context.textAlign = 'left';
  context.textBaseline = 'middle';
  report.scale.ticks.forEach(value => {
    const y = report.scale.maximum === report.scale.minimum
      ? MARGIN.top + plotHeight / 2
      : MARGIN.top + (1 - normalizedValue(value)) * plotHeight;
    context.beginPath();
    context.moveTo(barX + barWidth, y);
    context.lineTo(barX + barWidth + 5, y);
    context.stroke();
    context.fillText(formatNumber(value), barX + barWidth + 9, y);
  });
}

function drawMap(matrix) {
  const canvas = document.getElementById('angularMap');
  const context = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  const plotWidth = width - MARGIN.left - MARGIN.right;
  const plotHeight = height - MARGIN.top - MARGIN.bottom;
  const rows = report.fov.y_deg.length;
  const columns = report.fov.x_deg.length;
  context.clearRect(0, 0, width, height);
  context.fillStyle = BACKGROUND;
  context.fillRect(0, 0, width, height);

  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const displayRow = rows - 1 - row;
      context.fillStyle = colorForFraction(normalizedValue(matrix[row][column]));
      context.fillRect(
        MARGIN.left + column * plotWidth / columns,
        MARGIN.top + displayRow * plotHeight / rows,
        plotWidth / columns + 0.5,
        plotHeight / rows + 0.5,
      );
    }
  }

  context.strokeStyle = AXIS;
  context.strokeRect(MARGIN.left, MARGIN.top, plotWidth, plotHeight);
  context.fillStyle = TEXT;
  context.font = '12px Segoe UI';
  context.textBaseline = 'middle';
  context.textAlign = 'center';
  sampledIndices(columns).forEach(index => {
    const x = MARGIN.left + (index + 0.5) * plotWidth / columns;
    context.fillText(formatAxisNumber(report.fov.x_deg[index]), x, height - MARGIN.bottom + 20);
  });
  context.fillText('FOV x (deg)', MARGIN.left + plotWidth / 2, height - 13);
  context.textAlign = 'right';
  sampledIndices(rows).forEach(index => {
    const y = MARGIN.top + (rows - index - 0.5) * plotHeight / rows;
    context.fillText(formatAxisNumber(report.fov.y_deg[index]), MARGIN.left - 8, y);
  });
  drawRotatedLabel(context, 16, MARGIN.top + plotHeight / 2, 'FOV y (deg)');
  drawColorScale(context, width, plotHeight);
}

function drawSelected() {
  const selector = document.getElementById('eyePosition');
  const entry = report.slices[Number(selector.value)];
  const matrix = report.values[entry.eye_y_index][entry.eye_x_index];
  drawMap(matrix);
  document.getElementById('sliceCsvLink').href = entry.csv;
  document.getElementById('selectedStats').innerHTML = [
    ['mean', entry.mean],
    ['minimum', entry.minimum],
    ['maximum', entry.maximum],
    ['CV', entry.coefficient_of_variation],
  ].map(([label, value]) =>
    `<span class="stat">${label}<b>${formatNumber(value)}</b></span>`).join('');
  document.getElementById('caption').textContent =
    `Fixed eye center (${formatAxisNumber(entry.eye_x_mm)}, ` +
    `${formatAxisNumber(entry.eye_y_mm)}) mm, indices ` +
    `(x ${entry.eye_x_index}, y ${entry.eye_y_index}). Each diagram uses the same ` +
    `rounded scale from ${formatNumber(report.scale.minimum)} to ` +
    `${formatNumber(report.scale.maximum)}, so colors can be compared directly ` +
    `between eye positions. The ${formatNumber(report.window_area_mm2)} mm² eye ` +
    `window is integrated separately in every Zemax FOV trace.`;
}

function stepSelection(offset) {
  const selector = document.getElementById('eyePosition');
  selector.value = (Number(selector.value) + offset + report.slices.length) % report.slices.length;
  drawSelected();
}

function setup() {
  document.title = report.title;
  document.getElementById('title').textContent = report.title;
  document.getElementById('parentLink').href = report.files.parent_html;
  document.getElementById('jsonLink').href = report.files.parent_json;
  document.getElementById('longCsvLink').href = report.files.long_csv;
  const selector = document.getElementById('eyePosition');
  report.slices.forEach((entry, index) => {
    const option = document.createElement('option');
    option.value = index;
    option.textContent = `x ${formatAxisNumber(entry.eye_x_mm)} mm, ` +
      `y ${formatAxisNumber(entry.eye_y_mm)} mm ` +
      `(indices ${entry.eye_x_index}, ${entry.eye_y_index})`;
    selector.appendChild(option);
  });
  const centerX = Math.floor(report.eye_positions.x_mm.length / 2);
  const centerY = Math.floor(report.eye_positions.y_mm.length / 2);
  selector.value = report.slices.findIndex(entry =>
    entry.eye_x_index === centerX && entry.eye_y_index === centerY);
  selector.addEventListener('change', drawSelected);
  document.getElementById('previous').addEventListener('click', () => stepSelection(-1));
  document.getElementById('next').addEventListener('click', () => stepSelection(1));
  drawSelected();
}

setup();
</script>
</body>
</html>
"""
