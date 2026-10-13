"""Write a portable coating-stack report inside a Zemax result bundle.

The report is intentionally separate from the detector analyses: it records
the physical layer recipes that were exported to OpticStudio's coating file.
OpticStudio uses those recipes when a ``STACK_M*`` coating is assigned to a
mirror face, while this page makes the exported materials and thicknesses easy
to inspect beside the Zemax-derived position-space reports.

Only the Python standard library is used because this module is copied into
each bundle and executed on the OpticStudio computer after the FOV sweep.
"""

import html
import math
import re
from pathlib import Path

try:
    from .html_plotting import REPORT_LAYOUT_VERSION
except ImportError:  # Standalone copy in a generated Zemax bundle.
    from html_plotting import REPORT_LAYOUT_VERSION  # type: ignore[no-redef]


SCHEMA_VERSION = 1
DEFAULT_REPORT_NAME = "zemax_coating_report"
DEFAULT_REPORT_TITLE = "Zemax coating stacks"
DEFAULT_COATING_FILE = "APOLLO14_COATINGS.DAT"
_SAFE_FILE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_RESERVED_REPORT_NAMES = {"position_space_index"}


def normalize_coating_report(specification, default_coating_file=DEFAULT_COATING_FILE):
    """Validate a coating-report specification and return JSON-safe data.

    ``specification`` describes source data already selected for export.  It
    contains no live OpticStudio objects, which keeps the report reproducible
    and lets the Windows runner create it without Atlas or Apollo14 installed.
    ``None`` deliberately means that no coating report was requested.
    """
    if specification is None:
        return None
    if not isinstance(specification, dict):
        raise TypeError("Coating report specification must be a mapping or None.")

    report_name = _safe_file_name(
        specification.get("report_name", DEFAULT_REPORT_NAME), "Coating report_name"
    )
    if report_name.casefold() in _RESERVED_REPORT_NAMES:
        raise ValueError(
            f"Coating report_name {report_name!r} is reserved by the position-space report index."
        )
    title = _nonempty_text(specification.get("title", DEFAULT_REPORT_TITLE), "Coating report title")
    coating_file = _safe_file_name(
        specification.get("coating_file", default_coating_file), "Coating file name"
    )

    raw_mirrors = specification.get("mirrors", [])
    if not isinstance(raw_mirrors, list):
        raise ValueError("Coating report mirrors must be a list.")
    mirrors = [_normalize_mirror(record) for record in raw_mirrors]
    mirror_indices = [record["index"] for record in mirrors]
    if len(set(mirror_indices)) != len(mirror_indices):
        raise ValueError("Coating report mirror indices must be unique.")
    mirrors.sort(key=lambda record: record["index"])

    return {
        "schema": "apollo14.zemax_coating_report",
        "schema_version": SCHEMA_VERSION,
        "report_name": report_name,
        "title": title,
        "coating_file": coating_file,
        "mirrors": mirrors,
    }


def write_coating_report(specification, output_directory):
    """Write one coating-stack HTML page and return its index entry.

    The returned record is accepted directly by
    :func:`position_space_analysis.write_position_space_index`, so the page
    appears in the fixed navigation pane with the detector and pupil reports.
    """
    report = normalize_coating_report(specification)
    if report is None:
        return None

    output_directory = Path(output_directory)
    legacy_html = output_directory / f"{report['report_name']}.html"
    if legacy_html.is_file():
        legacy_html.unlink()
    report_directory = output_directory / "reports" / report["report_name"]
    report_directory.mkdir(parents=True, exist_ok=True)
    html_path = report_directory / "index.html"
    coating_href = (Path("..") / ".." / ".." / report["coating_file"]).as_posix()
    html_path.write_text(_render_html(report, coating_href), encoding="utf-8")
    return {
        "report_name": report["report_name"],
        "title": report["title"],
        "html": html_path.relative_to(output_directory).as_posix(),
        "detail": f"Physical layer recipes from {report['coating_file']}",
    }


def _normalize_mirror(record):
    if not isinstance(record, dict):
        raise ValueError("Each coating report mirror must be a mapping.")

    mirror_index = _nonnegative_integer(record.get("index"), "Coating report mirror index")
    mirror_name = _nonempty_text(
        record.get("name", f"mirror_{mirror_index}"), "Coating report mirror name"
    )
    stack_coating_name = _optional_safe_file_name(
        record.get("stack_coating_name"), "Physical stack coating name"
    )
    face_coatings = _normalize_face_coatings(record.get("face_coatings", []))
    layers = _normalize_layers(record.get("layers", []))
    if stack_coating_name is None and layers:
        raise ValueError(
            f"Coating report mirror {mirror_index} has layers but no physical stack coating name."
        )
    if stack_coating_name is not None and not layers:
        raise ValueError(
            f"Coating report mirror {mirror_index} names a physical stack but has no layers."
        )

    return {
        "index": mirror_index,
        "name": mirror_name,
        "face_coatings": face_coatings,
        "stack_coating_name": stack_coating_name,
        "layers": layers,
    }


def _normalize_face_coatings(records):
    if not isinstance(records, list):
        raise ValueError("Coating report face_coatings must be a list.")
    normalized = []
    seen_faces = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each coating report face coating must be a mapping.")
        face = _nonempty_text(record.get("face"), "Coating report face name")
        face_key = face.casefold()
        if face_key in seen_faces:
            raise ValueError(f"Coating report lists face {face!r} more than once.")
        seen_faces.add(face_key)
        normalized.append(
            {
                "face": face,
                "coating_name": _safe_file_name(record.get("coating_name"), "Applied coating name"),
            }
        )
    return normalized


def _normalize_layers(records):
    if not isinstance(records, list):
        raise ValueError("Coating report layers must be a list.")
    normalized = []
    for layer_number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Coating report layer {layer_number} must be a mapping.")
        material = _nonempty_text(
            record.get("material"), f"Coating report layer {layer_number} material"
        )
        thickness_nm = _positive_float(
            record.get("thickness_nm"), f"Coating report layer {layer_number} thickness_nm"
        )
        normalized.append({"material": material, "thickness_nm": thickness_nm})
    return normalized


def _render_html(report, coating_href):
    title = html.escape(report["title"])
    cards = "".join(_render_mirror_card(mirror) for mirror in report["mirrors"])
    if not cards:
        cards = '<p class="empty">No mirror coating recipes were included in this export.</p>'

    return f"""<!doctype html>
<html lang="en" data-apollo14-report-layout="{REPORT_LAYOUT_VERSION}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<title>{title}</title>
<style>
:root {{ color-scheme: light; font-family: Inter, "Segoe UI", system-ui, sans-serif;
  background:#fff; color:#202122; --border:#c8ccd1; --muted:#54595d; --link:#3366cc; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:#fff; }}
header, main {{ max-width:1280px; margin:auto; padding:22px; }}
header {{ border-bottom:1px solid var(--border); }}
h1, h2 {{ font-family:Georgia, serif; font-weight:500; }}
h1 {{ margin:.1rem 0 .4rem; font-size:clamp(1.7rem,3vw,2.5rem); }}
h2 {{ margin:0; font-size:1.35rem; }}
h3 {{ margin:1.25rem 0 .55rem; font-size:1rem; }}
p {{ color:var(--muted); line-height:1.5; }}
a {{ color:var(--link); }}
.tag {{ display:inline-block; padding:4px 9px; border:1px solid #a2a9b1;
  border-radius:999px; color:#54595d; font-size:.8rem; }}
.links {{ display:flex; flex-wrap:wrap; gap:14px; margin:1rem 0 0; }}
.mirrors {{ display:grid; gap:18px; }}
.mirror {{ border:1px solid var(--border); border-radius:4px; padding:18px; overflow:auto; }}
.facts {{ display:grid; grid-template-columns:max-content minmax(0,1fr); gap:7px 14px;
  margin:14px 0 0; font-size:.92rem; }}
.facts dt {{ color:var(--muted); }}
.facts dd {{ margin:0; }}
code {{ padding:1px 4px; border-radius:2px; background:#eaecf0; font-family:ui-monospace, monospace; }}
table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
caption {{ padding:0 0 8px; color:var(--muted); text-align:left; font-size:.88rem; }}
th, td {{ padding:8px 10px; border:1px solid var(--border); text-align:left; }}
th {{ background:#eaecf0; font-weight:600; }}
td.number {{ text-align:right; white-space:nowrap; }}
.empty {{ margin:1rem 0 0; color:var(--muted); font-style:italic; }}
.note {{ margin-top:22px; padding-top:14px; border-top:1px solid var(--border); font-size:.86rem; }}
@media(max-width:600px) {{ header,main {{ padding:14px; }} .mirror {{ padding:13px; }}
  .facts {{ grid-template-columns:1fr; gap:3px; }} .facts dd {{ margin-bottom:7px; }} }}
</style>
</head>
<body>
<header>
  <span class="tag">Coating recipe exported for OpticStudio</span>
  <h1>{title}</h1>
  <p>For each Perseus mirror, this page links the coating assigned to its Zemax face
  with the physical film stack exported to the coating file. OpticStudio calculates
  a <code>STACK_M…</code> coating from that stack during its ray trace.</p>
  <p class="links"><a href="{html.escape(coating_href, quote=True)}" target="_blank" rel="noopener">Open exported coating file</a></p>
</header>
<main>
  <section class="mirrors">
    {cards}
  </section>
  <p class="note">Layer thicknesses are displayed to 0.01 nm for readability. The linked
  coating file retains the precision emitted by the export.</p>
</main>
</body>
</html>
"""


def _render_mirror_card(mirror):
    mirror_name = html.escape(mirror["name"])
    face_rows = _render_face_rows(mirror["face_coatings"])
    stack_name = mirror["stack_coating_name"]
    if stack_name is None:
        stack_detail = "No physical film stack was exported for this mirror."
        layer_content = (
            '<p class="empty">The selected face coating has no Atlas physical layer recipe.</p>'
        )
    else:
        escaped_stack_name = html.escape(stack_name)
        stack_detail = f"<code>{escaped_stack_name}</code>"
        layer_content = _render_layer_table(mirror["layers"], escaped_stack_name)

    return f"""
<article class="mirror">
  <h2>Mirror {mirror["index"]} — {mirror_name}</h2>
  <dl class="facts">
    <dt>Applied face coating(s)</dt><dd>{face_rows}</dd>
    <dt>Physical stack in coating file</dt><dd>{stack_detail}</dd>
  </dl>
  {layer_content}
</article>
"""


def _render_face_rows(face_coatings):
    if not face_coatings:
        return "No mirror-face coating was assigned."
    return ", ".join(
        f"{html.escape(record['face'])}: <code>{html.escape(record['coating_name'])}</code>"
        for record in face_coatings
    )


def _render_layer_table(layers, stack_name):
    rows = "".join(
        "<tr>"
        f'<td class="number">{layer_number}</td>'
        f'<td class="number">{_format_thickness_nm(layer["thickness_nm"])}</td>'
        f"<td>{html.escape(layer['material'])}</td>"
        "</tr>"
        for layer_number, layer in enumerate(layers, start=1)
    )
    return f"""
  <h3>Physical layer stack</h3>
  <table>
    <caption>Layer order exported as <code>{stack_name}</code>.</caption>
    <thead><tr><th>Layer</th><th>Thickness (nm)</th><th>Material</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
"""


def _format_thickness_nm(value):
    """Return an evenly rounded thickness label for the HTML table."""
    return f"{value:.2f}"


def _safe_file_name(value, label):
    value = _nonempty_text(value, label)
    if _SAFE_FILE_NAME.fullmatch(value) is None or value in {".", ".."}:
        raise ValueError(f"{label} may contain only letters, digits, dot, underscore, and hyphen.")
    return value


def _optional_safe_file_name(value, label):
    if value is None:
        return None
    return _safe_file_name(value, label)


def _nonempty_text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value


def _nonnegative_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return value


def _positive_float(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a positive finite number.") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be a positive finite number.")
    return result
