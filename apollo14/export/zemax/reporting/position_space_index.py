"""Machine-readable manifest and persistent HTML navigation for Zemax reports.

The report browser is independent of detector-cube serialization. Keeping it
here makes navigation changes local and lets other derived reports register a
page without depending on the cube or native detector archive implementations.
"""

import html
import json
from pathlib import Path

try:
    from .html_plotting import REPORT_LAYOUT_VERSION
    from .position_space_common import SCHEMA_VERSION, _detector_label, _slug
except ImportError:  # Standalone copies in a generated Zemax bundle.
    from html_plotting import REPORT_LAYOUT_VERSION
    from position_space_common import (  # type: ignore[no-redef]
        SCHEMA_VERSION,
        _detector_label,
        _slug,
    )


def write_position_space_index(
    output_directory, reports, *, native_opticstudio=None, derived_reports=None
):
    """Write machine- and human-readable indexes for detector response cubes."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "apollo14.position_space_index",
        "schema_version": SCHEMA_VERSION,
        "description": (
            "One independently traced detector-flux cube per detector. See each "
            "linked position_space.json for axes, units, and normalization."
        ),
        "detectors": list(reports),
    }
    if native_opticstudio is not None:
        payload["native_opticstudio"] = dict(native_opticstudio)
    if derived_reports:
        payload["derived_reports"] = list(derived_reports)
    json_path = output_directory / "position_space_index.json"
    json_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_index_html(
        output_directory / "position_space_index.html",
        payload["detectors"],
        payload.get("derived_reports", []),
    )
    return json_path


def detector_directory_name(detector):
    """Return the safe, stable output-directory name for one detector."""
    return _slug(_detector_label(detector))


def _write_index_html(html_path, reports, derived_reports=()):
    """Write an offline two-pane report browser with persistent navigation."""
    derived_items = [
        {
            "url": str(report.get("html", "")),
            "title": str(report.get("title", report.get("report_name", "Derived report"))),
            "detail": (
                str(report["detail"])
                if report.get("detail")
                else f"Detector {report.get('detector_label', 'unknown')}"
            ),
        }
        for report in derived_reports
        if report.get("html")
    ]
    detector_items = [
        {
            "url": str(report.get("html", "")),
            "title": str(report.get("detector_label", "Detector")),
            "detail": _index_flux_shape(report.get("flux_shape")),
        }
        for report in reports
        if report.get("html")
    ]
    all_items = derived_items + detector_items
    default_item = (
        all_items[0]
        if all_items
        else {
            "url": "about:blank",
            "title": "No reports were generated",
            "detail": "",
        }
    )
    derived_navigation = _index_navigation_section("Analysis reports", derived_items, default_item)
    detector_navigation = _index_navigation_section("Detector maps", detector_items, default_item)
    default_url = html.escape(default_item["url"], quote=True)
    default_title = html.escape(default_item["title"])
    page = f"""<!doctype html>
<html lang="en" data-apollo14-report-layout="{REPORT_LAYOUT_VERSION}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<title>Apollo14 Zemax analysis reports</title>
<style>
  :root {{
    color-scheme: light;
    font-family: Inter, "Segoe UI", system-ui, sans-serif;
    --sidebar-width: 310px;
    --border: #c8ccd1;
    --muted: #54595d;
    --link: #3366cc;
    --active: #eaf3ff;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ width: 100%; height: 100%; margin: 0; overflow: hidden; }}
  body {{ background: #fff; color: #202122; }}
  .sidebar {{
    position: fixed;
    inset: 0 auto 0 0;
    z-index: 10;
    width: var(--sidebar-width);
    overflow-y: auto;
    padding: 20px 18px 28px;
    background: #f8f9fa;
    border-right: 1px solid var(--border);
  }}
  .brand {{ margin: 0 0 5px; font-family: Georgia, serif; font-size: 1.42rem; }}
  .intro {{ margin: 0 0 20px; color: var(--muted); font-size: .86rem; line-height: 1.45; }}
  .nav-section {{ margin: 19px 0 0; }}
  .nav-section h2 {{
    margin: 0 0 7px;
    padding-bottom: 5px;
    border-bottom: 1px solid var(--border);
    font-family: Georgia, serif;
    font-size: 1rem;
    font-weight: 500;
  }}
  .nav-link {{
    display: block;
    margin: 3px 0;
    padding: 8px 9px;
    border-left: 3px solid transparent;
    border-radius: 2px;
    color: var(--link);
    text-decoration: none;
  }}
  .nav-link:hover {{ background: #eaecf0; text-decoration: underline; }}
  .nav-link.active {{
    border-left-color: var(--link);
    background: var(--active);
    color: #202122;
    font-weight: 600;
  }}
  .nav-link small {{
    display: block;
    margin-top: 2px;
    color: var(--muted);
    font-size: .73rem;
    font-weight: 400;
  }}
  .data-link {{ margin-top: 20px; font-size: .82rem; }}
  .data-link a {{ color: var(--link); }}
  .workspace {{
    width: calc(100% - var(--sidebar-width));
    height: 100%;
    margin-left: var(--sidebar-width);
    background: #fff;
  }}
  .toolbar {{
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 0 16px;
    border-bottom: 1px solid var(--border);
    background: #fff;
  }}
  .toolbar strong {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .toolbar a {{ flex: none; color: var(--link); font-size: .86rem; }}
  .report-frame {{
    display: block;
    width: 100%;
    height: calc(100% - 48px);
    border: 0;
    background: #fff;
  }}
  @media (max-width: 760px) {{
    :root {{ --sidebar-width: 220px; }}
    .sidebar {{ padding: 14px 10px 20px; }}
    .brand {{ font-size: 1.1rem; }}
    .intro {{ font-size: .76rem; }}
  }}
</style>
</head>
<body>
<aside class="sidebar" aria-label="Analysis report navigation">
  <h1 class="brand">Apollo14 Zemax results</h1>
  <p class="intro">The menu remains visible while reports open in the right pane.
  Each field map is one combined System Wavelengths trace.</p>
  <nav>
    {derived_navigation}
    {detector_navigation}
  </nav>
  <section class="nav-section data-link">
    <h2>Data</h2>
    <a class="nav-link" href="position_space_index.json" target="report-frame"
      data-report-title="Machine-readable index">Machine-readable index JSON
      <small>Complete report manifest</small></a>
  </section>
</aside>
<main class="workspace">
  <div class="toolbar">
    <strong id="current-title">{default_title}</strong>
    <a id="open-separately" href="{default_url}" target="_blank" rel="noopener">
      Open this report separately
    </a>
  </div>
  <iframe class="report-frame" name="report-frame" id="report-frame"
    src="{default_url}" title="Selected Zemax analysis report"></iframe>
</main>
<script>
const title = document.getElementById("current-title");
const separate = document.getElementById("open-separately");
const links = Array.from(document.querySelectorAll("[data-report-title]"));
function selectReport(link) {{
  links.forEach(item => {{
    const selected = item === link;
    item.classList.toggle("active", selected);
    if (selected) item.setAttribute("aria-current", "page");
    else item.removeAttribute("aria-current");
  }});
  title.textContent = link.dataset.reportTitle;
  separate.href = link.getAttribute("href");
}}
links.forEach(link => link.addEventListener("click", () => selectReport(link)));
</script>
</body>
</html>
"""
    Path(html_path).write_text(page, encoding="utf-8")


def _index_navigation_section(title, items, default_item):
    """Render one group in the persistent index navigation."""
    if not items:
        return ""
    links = "\n".join(_index_navigation_link(item, item is default_item) for item in items)
    return f'<section class="nav-section"><h2>{html.escape(title)}</h2>{links}</section>'


def _index_navigation_link(item, active):
    """Render one report link that targets the persistent right pane."""
    active_attributes = ' active" aria-current="page' if active else ""
    url = html.escape(item["url"], quote=True)
    title = html.escape(item["title"])
    detail = html.escape(item["detail"])
    return (
        f'<a class="nav-link{active_attributes}" href="{url}" '
        f'target="report-frame" data-report-title="{title}">'
        f"{title}<small>{detail}</small></a>"
    )


def _index_flux_shape(shape):
    """Return a concise human-readable detector-cube shape."""
    if not shape:
        return "Detector response cube"
    return "Flux cube: " + " × ".join(str(value) for value in shape)
