"""Dependency-free writers for position-space numeric and graphical files.

This module owns NPZ/CSV/PNG serialization and the per-detector HTML viewer.
It accepts a completed cube through its small read interface and contains no
OpticStudio or Apollo14 runtime dependencies.
"""

import html
import json
import math
import struct
import sys
import zipfile
import zlib
from array import array
from pathlib import Path

try:
    from .html_plotting import (
        REPORT_LAYOUT_VERSION,
        format_compact_number,
    )
    from .position_space_common import (
        CSV_FLOAT_FORMAT,
        _fov_angles_deg,
        _pixel_centers,
        _product,
        _slug,
        _source_label,
    )
except ImportError:  # Standalone copies in a generated Zemax bundle.
    from html_plotting import (  # type: ignore[no-redef]
        REPORT_LAYOUT_VERSION,
        format_compact_number,
    )
    from position_space_common import (  # type: ignore[no-redef]
        CSV_FLOAT_FORMAT,
        _fov_angles_deg,
        _pixel_centers,
        _product,
        _slug,
        _source_label,
    )

_NPY_MAGIC = b"\x93NUMPY"


def _write_npz(npz_path, cube, metadata):
    """Write a NumPy-compatible NPZ without requiring NumPy at runtime."""
    arrays = {
        "flux.npy": (cube.iter_flux(), cube.shape),
        "system_wavelength_um.npy": (
            array("d", cube.spectrum.wavelengths_um),
            (len(cube.spectrum.wavelengths_um),),
        ),
        "system_wavelength_weight.npy": (
            array("d", cube.spectrum.weights),
            (len(cube.spectrum.weights),),
        ),
        "pixel_x_center_mm.npy": (
            array("d", _pixel_centers(cube.half_width_x_mm, cube.pixels_x)),
            (cube.pixels_x,),
        ),
        "pixel_y_center_mm.npy": (
            array("d", _pixel_centers(cube.half_width_y_mm, cube.pixels_y)),
            (cube.pixels_y,),
        ),
    }
    if cube.fov_angles_available:
        arrays["fov_x_deg.npy"] = (
            array("d", [_fov_angles_deg(source)[0] for source in cube.sources]),
            (len(cube.sources),),
        )
        arrays["fov_y_deg.npy"] = (
            array("d", [_fov_angles_deg(source)[1] for source in cube.sources]),
            (len(cube.sources),),
        )
    else:
        arrays["field_index.npy"] = (
            array("d", range(len(cube.sources))),
            (len(cube.sources),),
        )

    with zipfile.ZipFile(npz_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, (values, shape) in arrays.items():
            archive.writestr(name, _npy_bytes(values, shape))
        archive.writestr(
            "metadata.json",
            json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        )


def _npy_bytes(values, shape):
    """Return a little-endian float64 NPY v1.0 member."""
    values = array("d", values)
    shape = tuple(int(value) for value in shape)
    if len(values) != _product(shape):
        raise ValueError(
            f"NPY payload has {len(values)} values but shape {shape} needs {_product(shape)}."
        )

    header_text = "{'descr': '<f8', 'fortran_order': False, 'shape': " + repr(shape) + ", }"
    padding = (16 - (10 + len(header_text) + 1) % 16) % 16
    header = (header_text + " " * padding + "\n").encode("latin1")
    if len(header) > 65535:
        raise ValueError("NPY v1.0 header is too large.")

    if sys.byteorder != "little":
        values.byteswap()
    return _NPY_MAGIC + b"\x01\x00" + struct.pack("<H", len(header)) + header + values.tobytes()


def _write_csv_grid(csv_path, rows):
    with Path(csv_path).open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(",".join(format(value, CSV_FLOAT_FORMAT) for value in row) + "\n")


def _write_heatmap_png(path, rows, maximum):
    """Write a dependency-free RGB PNG using one shared detector scale."""
    height = len(rows)
    width = len(rows[0]) if rows else 0
    if height <= 0 or width <= 0:
        raise ValueError("Cannot create a heatmap for an empty detector grid.")

    scanlines = bytearray()
    for row in rows:
        scanlines.append(0)
        for value in row:
            scanlines.extend(_flux_color(value, maximum))

    raw = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
        + _png_chunk(b"IDAT", zlib.compress(bytes(scanlines), level=9))
        + _png_chunk(b"IEND", b"")
    )
    Path(path).write_bytes(raw)


def _png_chunk(name, data):
    return (
        struct.pack(">I", len(data))
        + name
        + data
        + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
    )


def _flux_color(value, maximum):
    """Map non-negative flux to a compact, monotonically brighter color scale."""
    if maximum <= 0.0 or value <= 0.0:
        return 12, 20, 42
    fraction = min(1.0, max(0.0, float(value) / maximum))
    fraction = math.sqrt(fraction)
    stops = (
        (12, 20, 42),
        (30, 76, 141),
        (36, 148, 154),
        (141, 204, 102),
        (253, 231, 37),
    )
    position = fraction * (len(stops) - 1)
    lower = min(int(position), len(stops) - 2)
    upper = lower + 1
    amount = position - lower
    return tuple(
        round(stops[lower][channel] * (1.0 - amount) + stops[upper][channel] * amount)
        for channel in range(3)
    )


def _color_scale_gradient(maximum, samples=32):
    """Return a CSS gradient sampled from the detector PNG color mapping."""
    if maximum <= 0.0:
        red, green, blue = _flux_color(0.0, 0.0)
        return f"rgb({red}, {green}, {blue})"
    stops = []
    for index in range(samples + 1):
        fraction = index / samples
        red, green, blue = _flux_color(fraction * maximum, maximum)
        stops.append(f"rgb({red}, {green}, {blue}) {100.0 * fraction:.4g}%")
    return "linear-gradient(to top, " + ", ".join(stops) + ")"


def _color_scale_tick_html(color_scale):
    """Render high-to-low numeric labels for a vertical color scale."""
    return "\n".join(
        f"<span>{html.escape(format_compact_number(value))}</span>"
        for value in reversed(color_scale["ticks"])
    )


def _write_html_report(html_path, cube, slice_entries, color_scale):
    """Write an offline field selector for the generated PNG maps."""
    field_options = "\n".join(
        f'<option value="{index}">{html.escape(_field_display(index, source))}</option>'
        for index, source in enumerate(cube.sources)
    )
    slice_json = json.dumps(slice_entries, separators=(",", ":")).replace("</", "<\\/")
    detector_name = html.escape(cube.detector_label)
    shape = " × ".join(str(value) for value in cube.shape)
    source_note = (
        "FOV angle pairs are stored for every field."
        if cube.fov_angles_available
        else "This prescription did not include FOV angle metadata; fields are source order."
    )
    spectrum_text = html.escape(
        ", ".join(
            f"{wavelength:.6g} um (weight {cube.spectrum.weights[spectrum_index]:.6g})"
            for spectrum_index, wavelength in enumerate(cube.spectrum.wavelengths_um)
        )
    )
    scale_maximum = format_compact_number(color_scale["maximum"])
    scale_gradient = html.escape(_color_scale_gradient(color_scale["maximum"]), quote=True)
    scale_ticks = _color_scale_tick_html(color_scale)

    page = f"""<!doctype html>
<html lang="en" data-apollo14-report-layout="{REPORT_LAYOUT_VERSION}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<title>Position-space response — {detector_name}</title>
<style>
  :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
  body {{ max-width: 1100px; margin: 2rem auto; padding: 0 1rem; line-height: 1.45; }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 1rem; align-items: end; }}
  label {{ display: grid; gap: .3rem; font-weight: 600; }}
  select {{ min-width: 18rem; padding: .4rem; }}
  .plot {{
    display: inline-grid;
    grid-template-columns: minmax(0, auto) 8.5rem;
    gap: 1rem;
    align-items: stretch;
    max-width: 100%;
    margin-top: 1rem;
  }}
  .map {{ background: #0c142a; padding: .5rem; min-width: 0; }}
  #map {{
    display: block;
    width: min(72vw, 720px);
    max-width: 100%;
    max-height: 720px;
    image-rendering: auto;
  }}
  .color-scale {{ display: flex; flex-direction: column; min-height: 180px; }}
  .scale-title {{ margin-bottom: .4rem; font-size: .82rem; font-weight: 700; }}
  .scale-title small {{ display: block; font-weight: 400; opacity: .75; }}
  .scale-body {{ display: grid; grid-template-columns: 1.3rem 1fr; gap: .45rem; flex: 1; }}
  .scale-bar {{ border: 1px solid #667085; background: {scale_gradient}; }}
  .scale-labels {{
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    font: 12px ui-monospace, SFMono-Regular, Consolas, monospace;
    font-variant-numeric: tabular-nums;
  }}
  .scale-labels span:only-child {{ margin: auto 0; }}
  dl {{ display: grid; grid-template-columns: max-content 1fr; gap: .35rem .8rem; }}
  dt {{ font-weight: 700; }}
  dd {{ margin: 0; }}
  code {{ overflow-wrap: anywhere; }}
  @media (max-width: 620px) {{
    .plot {{ grid-template-columns: minmax(0, 1fr) 6.5rem; gap: .65rem; }}
    #map {{ width: 100%; }}
  }}
</style>
</head>
<body>
<h1>Position-space detector response: {detector_name}</h1>
<p>The cube shape is <code>{shape}</code>, ordered as
<code>field, pixel_y, pixel_x</code>. {source_note}</p>
<p>Each map is one isolated Clear &amp; Trace operation: detector buffers are
cleared, one field source is enabled, and one polychromatic trace runs with
source wavenumber 0. Its System Wavelengths spectrum is {spectrum_text}.</p>
<p>Each map uses the same square-root color scale from zero to the rounded
display limit <code>{scale_maximum}</code> detector-flux units per pixel. Row zero
is drawn at the top; for a Detector Rectangle this is the local -y row. See
<a href="position_space.json">metadata</a>,
<a href="position_space.npz">NPZ cube</a>, and
<a href="position_space_summary.csv">summary CSV</a> for numeric data.</p>
<div class="controls">
  <label>Field / source<select id="field">{field_options}</select></label>
</div>
<div class="plot">
  <div class="map"><img id="map" alt="Detector flux heatmap"></div>
  <aside class="color-scale" aria-label="Detector flux color scale">
    <div class="scale-title">Flux per pixel<small>square-root color mapping</small></div>
    <div class="scale-body">
      <div class="scale-bar"></div>
      <div class="scale-labels">{scale_ticks}</div>
    </div>
  </aside>
</div>
<h2>Selected slice</h2>
<dl>
  <dt>Source</dt><dd id="source"></dd>
  <dt>Spectrum</dt><dd>Combined System Wavelengths (wavenumber 0)</dd>
  <dt>Total detector flux</dt><dd id="total"></dd>
  <dt>CSV</dt><dd><a id="csv" href="">open raw pixel grid</a></dd>
</dl>
<p>The spectral weights are already applied inside every detector map. Apply
only the intended field/angular weights before combining fields.</p>
<script>
const slices = {slice_json};
const lookup = Object.fromEntries(slices.map(item => [
  String(item.field_index), item
]));
const field = document.getElementById("field");
function formatNumber(value) {{
  const number = Number(value);
  const magnitude = Math.abs(number);
  if (magnitude === 0) return "0";
  if (magnitude >= 10000 || magnitude < 0.001) {{
    return number.toExponential(2).replace("e+", "e");
  }}
  return Number(number.toPrecision(3)).toString();
}}
function update() {{
  const item = lookup[field.value];
  document.getElementById("map").src = item.image;
  document.getElementById("map").alt =
      "Polychromatic detector flux for " + item.source_label;
  document.getElementById("source").textContent = item.source_label;
  document.getElementById("total").textContent = formatNumber(item.total_flux);
  document.getElementById("csv").href = item.csv;
}}
field.addEventListener("change", update);
update();
</script>
</body>
</html>
"""
    Path(html_path).write_text(page, encoding="utf-8")


def _field_display(field_index, source):
    angles = _fov_angles_deg(source)
    label = _source_label(source)
    if angles is None:
        return f"Field {field_index + 1} — {label}"
    return (
        f"Field {field_index + 1} — FOV x {_format_angle_deg(angles[0])}°, "
        f"y {_format_angle_deg(angles[1])}°"
    )


def _format_angle_deg(value, decimal_places=2):
    """Format a signed angle without float noise or redundant trailing zeros."""
    rounded = round(float(value), decimal_places)
    if rounded == 0.0:
        return "0"
    return f"{rounded:+.{decimal_places}f}".rstrip("0").rstrip(".")


def _slice_stem(field_index, source):
    return f"field_{field_index:03d}_{_slug(_source_label(source))}"
