"""Assemble a complete, self-contained Zemax bundle on disk.

One call writes everything OpticStudio needs, including generated analysis scripts:

===========================  =================================================
``prescription.json``        every object, placement and setting
``*.POB``                    polygon object (chassis)
``<catalog>.AGF``            glass catalog for the substrate
``<coatings>.DAT``           coating definitions for the mirror stack
``build_and_run.py``         single user-facing build and analysis command
``scripts/*.py``             generated ZOS-API build and sweep scripts
``scripts/reporting/*.py``   internal portable report implementation
``README.md``                install steps and what to verify first
===========================  =================================================
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from apollo14.export.zemax.formats.agf import fit_sellmeier, format_glass_catalog
from apollo14.export.zemax.formats.coating import format_coating_file
from apollo14.export.zemax.prescription import build_prescription
from apollo14.export.zemax.reporting.html_plotting import REPORT_LAYOUT_VERSION
from apollo14.export.zemax.reporting.position_space_analysis import detector_directory_name
from apollo14.export.zemax.reporting.zemax_coating_report import normalize_coating_report
from apollo14.export.zemax.reporting.zemax_pupil_report import (
    normalize_pupil_report_specs,
    validate_pupil_report_geometry,
)
from apollo14.export.zemax.templates.zosapi_script import (
    build_and_run_script_text,
    build_script_text,
    sweep_script_text,
)
from apollo14.system import OpticalSystem

DEFAULT_GLASS_CATALOG = "APOLLO14"
DEFAULT_COATING_FILE = "APOLLO14_COATINGS.DAT"
GENERATED_SCRIPTS_DIRECTORY = Path("scripts")
REPORT_SUPPORT_DIRECTORY = GENERATED_SCRIPTS_DIRECTORY / "reporting"
# These standard-library modules are copied verbatim beside the generated
# reporting runtime. Keep this dependency-first list aligned with their
# standalone-import fallbacks.
PORTABLE_REPORT_MODULES = (
    "html_plotting.py",
    "position_space_common.py",
    "position_space_rendering.py",
    "position_space_core.py",
    "position_space_index.py",
    "position_space_native.py",
    "position_space_analysis.py",
    "zemax_angular_response.py",
    "zemax_pupil_report.py",
    "zemax_coating_report.py",
)


def zemax_glass_name(material_name: str) -> str:
    """A catalog-safe glass name derived from an Apollo14 material name."""
    cleaned = "".join(character for character in material_name if character.isalnum())
    if not cleaned:
        raise ValueError(
            f"Material name {material_name!r} has no usable characters for a Zemax glass name."
        )
    return cleaned.upper()


def export_zemax_bundle(
    output_dir,
    system,
    *,
    chassis_pivot,
    chassis_tilt_deg: float,
    sources,
    trace_wavelengths,
    glass_materials,
    coating_blocks=(),
    coating_names=None,
    face_coatings=None,
    glass_names=None,
    detector_pixels=(140, 180),
    detector_pixels_by_name=None,
    extra_pupils=(),
    pupil_reports=(),
    coating_report=None,
    glass_catalog: str = DEFAULT_GLASS_CATALOG,
    coating_file: str = DEFAULT_COATING_FILE,
    notes: str = "",
    **prescription_kwargs,
) -> Path:
    """Write a Zemax bundle for ``system`` into ``output_dir``.

    ``coating_blocks`` and ``coating_names`` come from
    :mod:`apollo14.export.zemax.formats.coating` — which fidelity rung to export is the
    caller's decision, since it depends on what optimizer and Atlas output is
    available. Everything else is derived from ``system``.

    ``extra_pupils`` appends additional Detector Rectangles without mutating
    the caller's optical system. ``detector_pixels_by_name`` can set their
    individual sampling shapes. ``pupil_reports`` accepts
    :class:`~apollo14.export.zemax.reporting.zemax_pupil_report.PupilReportSpec`
    instances (or
    equivalent mappings) whose detector geometry is validated before any
    bundle files are written.
    ``coating_report`` is optional recipe metadata for the physical mirror
    stacks. The Zemax sweep writes it as a linked HTML analysis page from the
    same serialized input used to create the coating file.

    Returns the bundle directory.
    """
    glass_materials = list(glass_materials)
    coating_blocks = list(coating_blocks)
    resolved_glass_names = _resolve_glass_names(glass_materials, glass_names)
    resolved_coating_names = _resolve_coating_names(system, coating_blocks, coating_names)
    system_for_export = _clone_system_with_extra_pupils(system, extra_pupils)
    prescription = _build_prescription(
        system_for_export,
        chassis_pivot,
        chassis_tilt_deg,
        sources,
        trace_wavelengths,
        resolved_glass_names,
        resolved_coating_names,
        face_coatings,
        detector_pixels,
        coating_file,
        glass_catalog,
        prescription_kwargs,
    )
    _apply_detector_pixel_overrides(prescription, detector_pixels_by_name or {})
    _attach_pupil_reports(prescription, pupil_reports)
    _attach_coating_report(prescription, coating_report, coating_file)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "out").mkdir(parents=True, exist_ok=True)

    _write_prescription(output_dir, prescription, notes)
    fits = _write_polygon_files_and_glass_catalog(
        output_dir, prescription, glass_materials, resolved_glass_names, glass_catalog
    )
    _write_coating_files(output_dir, coating_blocks, coating_file)
    _write_generated_scripts(output_dir)
    _write_readme_file(
        output_dir,
        prescription,
        fits,
        resolved_glass_names,
        glass_catalog,
        coating_file,
        coating_blocks,
        notes,
    )

    return output_dir


def _resolve_glass_names(glass_materials, glass_names):
    """Resolve Apollo14 materials to Zemax-safe glass names."""
    resolved_glass_names = dict(glass_names or {})
    for material in glass_materials:
        resolved_glass_names.setdefault(material.name, zemax_glass_name(material.name))
    return resolved_glass_names


def _resolve_coating_names(system, coating_blocks, coating_names):
    """Resolve mirror names to coating names, defaulting to flat tables."""
    resolved = dict(coating_names or {})
    flat_names = {}
    for block in coating_blocks:
        match = re.fullmatch(r"FLAT_M(\d+)", block.name)
        if match is not None:
            flat_names[int(match.group(1))] = block.name

    for element in system.elements:
        name = getattr(element, "name", None)
        if name is None or not name.startswith("mirror_"):
            continue
        if name in resolved:
            continue
        try:
            mirror_index = int(name.split("_", 1)[1])
        except ValueError:
            continue
        coating_name = flat_names.get(mirror_index)
        if coating_name is not None:
            resolved[name] = coating_name

    return resolved


def _clone_system_with_extra_pupils(system, extra_pupils):
    """Return a shallow copy of ``system`` with extra exported pupils appended."""
    if not extra_pupils:
        return system
    cloned = OpticalSystem(
        elements=list(system.elements),
        env_material=system.env_material,
    )
    for pupil in extra_pupils:
        cloned.add(pupil)
    return cloned


def _apply_detector_pixel_overrides(prescription, detector_pixels_by_name):
    """Apply validated per-detector pixel counts to a prescription."""
    detectors_by_label = _detectors_by_label(prescription)
    unknown = sorted(
        set(detector_pixels_by_name) - set(detectors_by_label),
        key=repr,
    )
    if unknown:
        raise ValueError(
            "Detector pixel overrides reference labels that are not exported: "
            + ", ".join(repr(label) for label in unknown)
            + "."
        )

    for label, pixels in detector_pixels_by_name.items():
        if isinstance(pixels, (str, bytes)) or not hasattr(pixels, "__len__") or len(pixels) != 2:
            raise ValueError(f"Detector {label!r} pixel override must contain (x, y).")
        pixels_x, pixels_y = (_positive_pixel_count(value, label) for value in pixels)
        data = detectors_by_label[label]["data"]
        data["pixels_x"] = pixels_x
        data["pixels_y"] = pixels_y


def _attach_pupil_reports(prescription, pupil_reports):
    """Validate and persist portable derived-report configurations."""
    normalized = normalize_pupil_report_specs(pupil_reports)
    if not normalized:
        return
    detectors_by_label = _detectors_by_label(prescription)
    detector_labels = set(detectors_by_label)
    missing = sorted(
        {
            spec["detector_label"]
            for spec in normalized
            if spec["detector_label"] not in detector_labels
        }
    )
    if missing:
        raise ValueError(
            "Pupil reports reference detector labels that are not exported: "
            + ", ".join(repr(label) for label in missing)
            + "."
        )
    for spec in normalized:
        validate_pupil_report_geometry(detectors_by_label[spec["detector_label"]], spec)
    prescription.document.setdefault("analyses", {})["pupil_reports"] = normalized


def _attach_coating_report(prescription, coating_report, coating_file):
    """Validate and persist one physical coating-stack report configuration."""
    normalized = normalize_coating_report(
        coating_report,
        default_coating_file=coating_file,
    )
    if normalized is None:
        return

    pupil_report_names = {
        str(report.get("report_name", "")).casefold()
        for report in prescription.document.get("analyses", {}).get("pupil_reports", [])
    }
    if normalized["report_name"].casefold() in pupil_report_names:
        raise ValueError(
            f"Coating report name {normalized['report_name']!r} conflicts with a pupil report."
        )
    prescription.document.setdefault("analyses", {})["coating_report"] = normalized


def _detectors_by_label(prescription):
    """Return exported Detector Rectangles keyed by unique non-empty labels."""
    result = {}
    output_directories = {}
    for entry in prescription.objects:
        if entry.get("type") != "detector_rectangle":
            continue
        label = entry.get("comment")
        if not isinstance(label, str) or not label:
            raise ValueError("Every exported detector needs a non-empty label.")
        if label in result:
            raise ValueError(f"Duplicate exported detector label {label!r}.")
        directory = detector_directory_name(entry)
        directory_key = directory.casefold()
        if directory_key in output_directories:
            raise ValueError(
                f"Detector labels {output_directories[directory_key]!r} and "
                f"{label!r} map to the same output directory {directory!r}."
            )
        result[label] = entry
        output_directories[directory_key] = label
    return result


def _positive_pixel_count(value, detector_label):
    """Return one exact positive detector pixel count."""
    if isinstance(value, bool):
        raise ValueError(f"Detector {detector_label!r} pixel counts must be positive integers.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Detector {detector_label!r} pixel counts must be positive integers."
        ) from error
    if result <= 0 or result != value:
        raise ValueError(f"Detector {detector_label!r} pixel counts must be positive integers.")
    return result


def _build_prescription(
    system,
    chassis_pivot,
    chassis_tilt_deg,
    sources,
    trace_wavelengths,
    resolved_glass_names,
    coating_names,
    face_coatings,
    detector_pixels,
    coating_file,
    glass_catalog,
    prescription_kwargs,
):
    """Build the JSON prescription and attach bundle metadata later."""
    return build_prescription(
        system,
        chassis_pivot=chassis_pivot,
        chassis_tilt_deg=chassis_tilt_deg,
        sources=sources,
        trace_wavelengths=trace_wavelengths,
        glass_names=resolved_glass_names,
        coating_names=coating_names,
        face_coatings=face_coatings,
        detector_pixels=detector_pixels,
        coating_file=coating_file,
        glass_catalog=glass_catalog,
        **prescription_kwargs,
    )


def _write_prescription(output_dir, prescription, notes):
    """Write the prescription JSON with bundle metadata."""
    prescription.document["meta"] = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "generator": "apollo14.export",
        "notes": notes,
    }
    (output_dir / "prescription.json").write_text(
        json.dumps(prescription.document, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_polygon_files_and_glass_catalog(
    output_dir, prescription, glass_materials, resolved_glass_names, glass_catalog
):
    """Write POB files and the generated glass catalog, returning fit stats."""
    for file_name, text in prescription.polygon_files.items():
        (output_dir / file_name).write_text(text, encoding="utf-8")
    return _write_glass_catalog(output_dir, glass_materials, resolved_glass_names, glass_catalog)


def _write_glass_catalog(output_dir, glass_materials, resolved_glass_names, glass_catalog):
    """Fit and write the glass catalog, returning the fit records."""
    catalog_lines = []
    fits = {}
    for material in glass_materials:
        fit = fit_sellmeier(material)
        fits[material.name] = fit
        catalog_lines.append(
            format_glass_catalog(material, fit, resolved_glass_names[material.name])
        )
    (output_dir / f"{glass_catalog}.AGF").write_text("".join(catalog_lines), encoding="utf-8")
    return fits


def _write_coating_files(output_dir, coating_blocks, coating_file):
    """Write the Zemax coating file in the bundle root."""
    coating_blocks = list(coating_blocks)
    if not coating_blocks:
        return
    coating_text = format_coating_file(coating_blocks, "Apollo14 combiner mirror stack")
    (output_dir / coating_file).write_text(coating_text, encoding="utf-8")


def _write_generated_scripts(output_dir):
    """Write one root entry point and group implementation files under scripts/."""
    output_dir = Path(output_dir)
    scripts_directory = output_dir / GENERATED_SCRIPTS_DIRECTORY
    reporting_directory = output_dir / REPORT_SUPPORT_DIRECTORY
    scripts_directory.mkdir(parents=True, exist_ok=True)
    reporting_directory.mkdir(parents=True, exist_ok=True)

    # A bundle may be regenerated in place. Remove only files written by the
    # former flat layout so stale modules cannot shadow scripts/reporting/.
    for legacy_name in ("build_zemax_model.py", "run_fov_sweep.py", *PORTABLE_REPORT_MODULES):
        legacy_path = output_dir / legacy_name
        if legacy_path.is_file():
            legacy_path.unlink()

    (output_dir / "build_and_run.py").write_text(build_and_run_script_text(), encoding="utf-8")
    (scripts_directory / "build_zemax_model.py").write_text(build_script_text(), encoding="utf-8")
    (scripts_directory / "run_fov_sweep.py").write_text(sweep_script_text(), encoding="utf-8")

    source_directory = Path(__file__).parent / "reporting"
    for file_name in PORTABLE_REPORT_MODULES:
        source = (source_directory / file_name).read_text(encoding="utf-8")
        (reporting_directory / file_name).write_text(source, encoding="utf-8")


def _write_readme_file(
    output_dir, prescription, fits, glass_names, glass_catalog, coating_file, coating_blocks, notes
):
    """Write the bundle README."""
    (output_dir / "README.md").write_text(
        _readme(
            prescription, fits, glass_names, glass_catalog, coating_file, coating_blocks, notes
        ),
        encoding="utf-8",
    )


def _readme(
    prescription, fits, glass_names, glass_catalog, coating_file, coating_blocks, notes
) -> str:
    return "\n".join(
        [
            _readme_header(notes),
            _readme_install(glass_catalog, coating_file),
            _readme_position_space_analysis(),
            _readme_pupil_reports(prescription),
            _readme_coating_report(prescription),
            _readme_objects(prescription),
            _readme_glass(fits, glass_names),
            _readme_coatings(coating_blocks),
            _readme_checks(),
            _readme_detector_orientation(),
            _readme_expected_bias(),
        ]
    )


def _readme_header(notes):
    return "\n".join(
        [
            "# Apollo14 → Zemax OpticStudio bundle",
            "",
            notes,
            "",
            "Generated by `apollo14.export`. Regenerate rather than editing by hand — the",
            "whole point is that this model cannot drift from the optimizer.",
            "",
            "## Edition requirements",
            "",
            "Non-sequential mode and ZOS-API are both included from the **Professional**",
            "edition upward, and every licence issued from 2023 R1 onward has non-sequential",
            "ray tracing regardless of tier. Only a legacy pre-2023-R1 *Standard* licence —",
            "sequential-only — cannot run this model at all.",
            "",
            "Nothing here needs a Premium-only feature: the chassis is a native polygon",
            "object and the stop is a native Boolean object rather than an imported CAD part;",
            "the sources are analytic rather than measured rayfiles, and the sweep script",
            "disables scattering.",
            "",
            "The build script starts a standalone OpticStudio session. It will not close",
            "a session it did not start.",
        ]
    )


def _readme_install(glass_catalog, coating_file):
    return "\n".join(
        [
            "## Install",
            "",
            "The scripts need Python on the Windows machine with **pythonnet** (`import clr`",
            "is the .NET bridge ZOS-API rides on):",
            "",
            "    pip install pythonnet",
            "",
            "Data files go in OpticStudio's own folders. The exact paths depend on the",
            "install — File → Project Preferences → Folders lists them. Typically under",
            "`{Documents}\\Zemax\\`:",
            "",
            f"1. `{glass_catalog}.AGF` → the **Glass catalog** folder (`Glasscat`).",
            "2. The build script installs the coating file into the configured",
            "   Zemax Coatings folder automatically.",
            "3. The build script installs every freshly generated `.POB` file into the",
            "   active Zemax **Objects** folder (`Polygon Objects` subfolder), replacing",
            "   an older file with the same plain name. Set",
            "   `APOLLO14_ZEMAX_POLYGON_OBJECTS_DIR` only if that folder cannot be",
            "   discovered from OpticStudio.",
            "4. From the bundle directory, build, trace, and generate every report with",
            "   one command:",
            "",
            "       python build_and_run.py",
            "",
            f"   A successful current run ends with `Verified HTML report layout "
            f"{REPORT_LAYOUT_VERSION}`.",
            "   If that line is absent, do not reuse HTML already",
            "   present under `out/`; copy `build_and_run.py` and the complete",
            "   `scripts/` directory from one bundle together and run again.",
            "",
            "   This writes the model and all runtime results under `out/`. To diagnose",
            "   one stage independently, the equivalent two commands are:",
            "",
            "       python scripts/build_zemax_model.py prescription.json apollo14_combiner.zmx",
            "       python scripts/run_fov_sweep.py apollo14_combiner.zmx prescription.json out\\",
            "",
            "`build_zemax_model.py` prints anything it could not set automatically. ZOS-API",
            "property names move between releases, so treat that list as the manual to-do,",
            "not as a failure.",
        ]
    )


def _readme_position_space_analysis():
    return "\n".join(
        [
            "## Position-space detector analysis",
            "",
            "The sweep runs each source direction once. Before tracing, it sets Source",
            "Color to System Wavelengths and source parameter 4, Wavenumber, to zero.",
            "OpticStudio then chooses ray wavelengths from the registered system",
            "wavelengths according to their Wavelength Data Editor weights. Every",
            "detector map therefore contains the combined polychromatic response.",
            "",
            "For every detector the runner records data type 0 from",
            "NCE.GetDetectorData: flux landing in each detector pixel. The primary",
            "artifact is a dense array with axes [field, pixel_y, pixel_x]. The artifact",
            "has no separate wavelength axis or wavelength selector.",
            "",
            "Each field is one exported source. Its source label, object number, intended",
            "direction, FOV x/y angles, and independent source power are recorded in the",
            "metadata. Field does not presume a rectangular angular grid, so custom source",
            "lists remain valid too.",
            "",
            "After the sweep, open out/position_space_index.html. Its fixed left menu",
            "stays visible while analysis reports and detector maps open in the large",
            "right pane, so switching results does not require browser Back or Forward.",
            "The adjacent position_space_index.json is the machine-readable index. For",
            "each detector there are four useful portable files:",
            "",
            "- position_space.npz — NumPy-compatible flux cube plus coordinate arrays.",
            "- position_space.json — schema, axes, detector frame, trace settings, and",
            "  normalization.",
            "- position_space_summary.csv — one total-flux row per field.",
            "- position_space.html — an offline graphical field selector backed by one",
            "  PNG and raw CSV grid per field.",
            "",
            "The PNGs use one detector-wide color scale, so their brightness is comparable",
            "across fields. Every detector-map HTML page shows a numeric color bar beside",
            "the map with rounded tick values. The numeric detector quantity is flux per pixel;",
            "divide by the recorded pixel_area_mm2 only when flux density is wanted.",
            "",
            "Every map uses exactly one source at its data.sweep_power value (1.0 by",
            "default). Spectral weights are already applied inside the polychromatic",
            "trace and are recorded alongside the system wavelengths in NPZ and JSON.",
            "For a physical scene, rescale each non-zero-power map by desired source",
            "power / source_sweep_power, apply any angular sample weight, and then sum",
            "over field. A zero-power map cannot be normalized. Do not apply the spectral",
            "weights again. These detector maps are response functions, not radiance by",
            "themselves.",
            "",
            "### Open a saved field in OpticStudio",
            "",
            "The sweep also saves each just-traced Detector Rectangle state as native",
            "OpticStudio .DDR data under out/zemax_detector_data/. It writes",
            "apollo14_view_position_space.zpl beside apollo14_combiner.zmx and installs",
            "a copy in OpticStudio's configured ZPL macro folder, so it appears in Macro",
            "List. The macro loads every detector's saved data for a selected",
            "field and opens a preconfigured native Detector Viewer in False Color. It",
            "does not alter sources or retrace the model.",
            "",
            "1. Let build_and_run.py finish, then open apollo14_combiner.zmx in",
            "   OpticStudio from this bundle directory.",
            "2. In Programming, use Refresh List if OpticStudio was already open, then",
            "   open Macro List and run apollo14_view_position_space.",
            "3. Enter the one-based field number printed by the macro. It opens Detector",
            "   Viewer with the saved polychromatic data. Choose another detector from",
            "   its selector if needed.",
            "",
            "Run that same macro again to switch fields; no source-power edits, source",
            "wavelength edits, or ray trace are needed. The archive manifest at",
            "out/zemax_detector_data/native_detector_archive.json maps every .DDR file",
            "to its detector and field/source, and records the combined spectrum once.",
            "",
            "The sweep log prints the installed macro path. If automatic installation",
            "warns that the ZPL folder could not be found or written, copy the generated",
            "apollo14_view_position_space.zpl from beside apollo14_combiner.zmx into the",
            "ZPL folder configured in OpticStudio Preferences > Folders, then use",
            "Programming > Refresh List.",
            "",
            "Detector Viewer natively displays Incoherent Irradiance (flux divided by the",
            "constant detector-pixel area), so its spatial pattern is the saved flux map",
            "with a fixed scale factor. The CSV/NPZ response cube remains the exact",
            "flux-per-pixel export. There is one .DDR state per detector and field.",
            "",
            "Detector Viewer shows one detector and one field at a time. Use the generated",
            "HTML report to browse the fields without reopening the native analysis.",
        ]
    )


def _readme_pupil_reports(prescription):
    """Explain reports whose detector input is generated on the Zemax host."""
    reports = prescription.document.get("analyses", {}).get("pupil_reports", [])
    if not reports:
        return ""
    configured = [
        f"- `out/reports/{report['report_name']}/index.html` from detector "
        f"`{report['detector_label']}`: "
        f"{report['eye_positions_x']}×{report['eye_positions_y']} eye positions, "
        f"{report['eye_window_width_mm']:g}×"
        f"{report['eye_window_height_mm']:g} mm integration windows"
        + (
            "; `angular_response.html` in the same report folder provides one "
            "fixed-position FOV-angle slice per eye position."
            if report.get("export_angular_response")
            else "."
        )
        for report in reports
    ]
    return "\n".join(
        [
            "## Automatic Zemax pupil/FOV reports",
            "",
            "`build_and_run.py` creates each configured report after the",
            "FOV sweep:",
            "",
            *configured,
            "",
            "Open `out/position_space_index.html` in a browser on the OpticStudio",
            "machine. Each configured report appears under **Analysis reports** in",
            "the fixed left menu. Each report contains interactive heatmaps for one",
            "eye position over FOV, pupil",
            "mean/worst/CV maps, FOV mean/worst maps, total detector flux, and a CDF.",
            "A configured angular-response page transposes the same Zemax cube: it fixes",
            "one eyebox position and plots response over FOV x and FOV y. It also writes",
            "one exact CSV grid per eye position and one long-form CSV for all samples.",
            "Each heatmap has a numeric color bar with rounded tick values. Its units",
            "are written horizontally below the graph; the CDF has numeric ticks on",
            "both axes and a horizontal unit description.",
            "Each report keeps its HTML, JSON, and CSV files together under",
            "`out/reports/<report-name>/`.",
            "",
            "All response values come from OpticStudio",
            "`NCE.GetDetectorData(..., data=0)` after the sweep. Each report integrates",
            "fractional detector-pixel overlap inside each physical eye window and",
            "divides by its area. Fractional edge pixels assume uniform flux within",
            "one detector pixel. FOV means are unweighted arithmetic means of the",
            "independent source samples; the report does not perform angular integration.",
            "Their unit is OpticStudio source-power units/mm².",
            "They do not copy values from `apollo14_eyebox_report.html` and do not",
            "claim calibrated radiance or luminance.",
        ]
    )


def _readme_coating_report(prescription):
    """Explain the physical coating recipe page when the export includes one."""
    report = prescription.document.get("analyses", {}).get("coating_report")
    if not report:
        return ""
    return "\n".join(
        [
            "## Coating-stack report",
            "",
            "`build_and_run.py` writes "
            f"`out/reports/{report['report_name']}/index.html` after the FOV sweep. "
            "It appears under "
            "**Analysis reports** in `out/position_space_index.html`.",
            "",
            "For every mirror, the page identifies the coating assigned to each Zemax face, "
            "the corresponding physical `STACK_M*` entry, and a table of layer number, "
            "thickness in nm, and material. The table is traceability data from the same "
            f"export input that produced `{report['coating_file']}`; it is not a separate "
            "post-trace calculation. When a `STACK_M*` coating is assigned, OpticStudio "
            "uses its physical stack during the ray trace.",
        ]
    )


def _readme_objects(prescription):
    lines = [
        "## Objects",
        "",
        "| # | type | comment | material | inside of | coating |",
        "|---|------|---------|----------|-----------|---------|",
        _object_rows(prescription),
    ]
    object_comments = {entry["index"]: entry["comment"] for entry in prescription.objects}
    for entry in prescription.objects:
        if entry["type"] != "boolean_native":
            continue
        parent_a = object_comments[entry["data"]["object_a"]]
        parent_b = object_comments[entry["data"]["object_b"]]
        label = entry.get("label", entry["comment"])
        lines.extend(
            [
                "",
                f"The `{label}` aperture is the `{entry['comment']}` Boolean Native row. "
                "Its Object A",
                f"and Object B fields point to the hidden, ray-ignored `{parent_a}` and",
                f"`{parent_b}` rectangular-volume parent rows. The Boolean row carries the",
                "physical position and tilt; its parents are local shape inputs at the origin.",
            ]
        )
    return "\n".join(lines)


def _readme_glass(fits, glass_names):
    return "\n".join(
        [
            "## Glass",
            "",
            "| material | n(d) | V(d) | RMS Δn | max Δn |",
            "|----------|------|------|--------|--------|",
            _fit_rows(fits, glass_names),
            "",
            "The catalog holds a Sellmeier 1 fit of Apollo14's measured index table. It is",
            "valid only across the wavelength range on the record's `LD` line; the residual",
            "above is the fit error against that table.",
        ]
    )


def _readme_coatings(coating_blocks):
    return "\n".join(
        [
            "## Coatings",
            "",
            _coating_rows(coating_blocks),
        ]
    )


def _readme_checks():
    return "\n".join(
        [
            "## Verify these before trusting a number",
            "",
            "The exporter checks everything it can compute on the Apollo14 side — the chassis",
            "vertex round trip, the tilt sign, the width/height axis mapping — but four things",
            "can only be confirmed in OpticStudio:",
            "",
            "1. **Mirrors are inside the chassis.** Each mirror's `Inside Of` must name the",
            "   chassis object. Without it both faces see air and the mirrors refract.",
            "2. **Ray splitting is on**, with the segment and intersection limits raised. A",
            "   cascade of partial mirrors branches hard; stock limits silently cull the",
            "   light this model exists to measure.",
            "3. **Off-axis source tilts.** Sources scanned in two axes need two tilt angles,",
            "   and their composition order is the one convention this exporter assumes",
            "   rather than derives. Each source's comment carries its intended direction",
            "   cosines — check a couple in the 3D layout. The on-axis source uses a single",
            "   tilt and is unambiguous.",
            "4. **Film thickness units.** Physical-stack coatings are written as absolute",
            "   thicknesses in micrometres. If a stack's reflectance comes out wildly wrong,",
            "   this is the first thing to check.",
        ]
    )


def _readme_detector_orientation():
    return "\n".join(
        [
            "## Detector orientation",
            "",
            "The pupil detector's exported frame is **not** Apollo14's pupil frame. Choosing",
            "local +X = world +X keeps the placement to a single unambiguous tilt, but for a",
            "normal of `[0, 0, -1]` that comes out rotated 180° from what",
            "`compute_local_axes` hands the tracer — both in-plane axes flip. A detector CSV",
            "laid over an Apollo14 pupil map without accounting for it is silently upside",
            "down and backwards, and would look like a plausible physics disagreement.",
            "",
            "Both frames are recorded on the detector object in `prescription.json`",
            "(`zemax_local_axes_world` and `apollo14_local_axes_world`); align on those",
            "rather than assuming.",
            "",
            "Position-space arrays follow Detector Rectangle pixel numbering: pixel 1 is",
            "local (-x, -y), x changes first, then rows advance toward +y. The generated",
            "HTML report draws the first (-y) row at the top, so flip it vertically when",
            "comparing visually with a plot whose vertical axis is local +y-up.",
        ]
    )


def _readme_expected_bias():
    return "\n".join(
        [
            "## Expect Zemax to read lower than Apollo14",
            "",
            "The tracer omits physics OpticStudio includes, so a total a few per cent to",
            "low-teens per cent below Apollo14's is expected and explainable:",
            "",
            "- No Fresnel loss at the chassis faces (`snell_refract` carries no amplitude",
            "  coefficients) — around 4% per uncoated surface.",
            "- No bulk absorption: the material's `k` column never reaches the trace.",
            "- TIR rays are terminated rather than followed, so Apollo14 sees no stray paths.",
            "- One reflection per path — no mirror-to-mirror rebounds.",
            "- No polarization anywhere, while the real stack splits s and p hard at the",
            "  ~40° internal incidence angle.",
            "",
            "A *shape* mismatch across the pupil is a different matter and worth chasing.",
        ]
    )


def _object_rows(prescription):
    """Render the exported object table rows."""
    return "\n".join(
        f"| {entry['index']} | {entry['type']} | {entry['comment']} | "
        f"{entry.get('material') or '—'} | "
        f"{entry.get('inside_of') or '—'} | {_display_coating(entry)} |"
        for entry in prescription.objects
    )


def _display_coating(entry):
    """Prefer the visible face coating when the top-level field is empty."""
    coating = entry.get("coating")
    if coating:
        return coating

    face_coatings = entry.get("face_coatings") or {}
    if not face_coatings:
        return "—"

    return ", ".join(
        f"{_face_label(face_number)}: {face_coating}"
        for face_number, face_coating in sorted(
            face_coatings.items(), key=lambda item: int(item[0])
        )
    )


def _face_label(face_number):
    """Render Zemax face numbers in the mirror-side language used upstream."""
    if str(face_number) == "1":
        return "front"
    if str(face_number) == "2":
        return "back"
    return f"face {face_number}"


def _fit_rows(fits, glass_names):
    """Render the fitted glass table rows."""
    return "\n".join(
        f"| {name} → {glass_names[name]} | {fit.index_d:.4f} | "
        f"{fit.abbe_d:.1f} | {fit.rms_index_error:.1e} | "
        f"{fit.max_index_error:.1e} |"
        for name, fit in fits.items()
    )


def _coating_rows(coating_blocks):
    """Render the exported coating list."""
    return "\n".join(f"- `{block.name}`" for block in coating_blocks) or "- (none exported)"
