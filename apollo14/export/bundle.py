"""Assemble a complete, self-contained Zemax bundle on disk.

One call writes everything OpticStudio needs plus the two scripts that drive it:

===========================  =================================================
``prescription.json``        every object, placement and setting
``*.POB``                    polygon objects (chassis, beam stop)
``<catalog>.AGF``            glass catalog for the substrate
``<coatings>.DAT``           coating definitions for the mirror stack
``build_zemax_model.py``     ZOS-API script that constructs the ``.ZMX``
``run_fov_sweep.py``         ZOS-API script that traces and dumps the detector
``README.md``                install steps and what to verify first
===========================  =================================================
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from apollo14.export.agf import fit_sellmeier, format_glass_catalog
from apollo14.export.coating import format_coating_file
from apollo14.export.prescription import build_prescription
from apollo14.export.zosapi_script import (
    build_and_run_script_text,
    build_script_text,
    sweep_script_text,
)

DEFAULT_GLASS_CATALOG = "APOLLO14"
DEFAULT_COATING_FILE = "APOLLO14_COATINGS.DAT"


def zemax_glass_name(material_name: str) -> str:
    """A catalog-safe glass name derived from an Apollo14 material name."""
    cleaned = "".join(character for character in material_name
                      if character.isalnum())
    if not cleaned:
        raise ValueError(f"Material name {material_name!r} has no usable "
                         "characters for a Zemax glass name.")
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
    glass_catalog: str = DEFAULT_GLASS_CATALOG,
    coating_file: str = DEFAULT_COATING_FILE,
    notes: str = "",
    **prescription_kwargs,
) -> Path:
    """Write a Zemax bundle for ``system`` into ``output_dir``.

    ``coating_blocks`` and ``coating_names`` come from
    :mod:`apollo14.export.coating` — which fidelity rung to export is the
    caller's decision, since it depends on what optimizer and Atlas output is
    available. Everything else is derived from ``system``.

    Returns the bundle directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "out").mkdir(parents=True, exist_ok=True)

    glass_materials = list(glass_materials)
    resolved_glass_names = _resolve_glass_names(glass_materials, glass_names)
    resolved_coating_names = _resolve_coating_names(
        system, coating_blocks, coating_names)
    prescription = _build_prescription(
        system, chassis_pivot, chassis_tilt_deg, sources, trace_wavelengths,
        resolved_glass_names, resolved_coating_names, face_coatings, detector_pixels,
        coating_file, glass_catalog, prescription_kwargs)

    _write_prescription(output_dir, prescription, notes)
    fits = _write_polygon_files_and_glass_catalog(
        output_dir, prescription, glass_materials, resolved_glass_names,
        glass_catalog)
    _write_coating_files(output_dir, coating_blocks, coating_file)
    _write_generated_scripts(output_dir)
    _write_readme_file(
        output_dir, prescription, fits, resolved_glass_names, glass_catalog,
        coating_file, coating_blocks, notes)

    return output_dir


def _resolve_glass_names(glass_materials, glass_names):
    """Resolve Apollo14 materials to Zemax-safe glass names."""
    resolved_glass_names = dict(glass_names or {})
    for material in glass_materials:
        resolved_glass_names.setdefault(material.name,
                                        zemax_glass_name(material.name))
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
        except (IndexError, ValueError):
            continue
        coating_name = flat_names.get(mirror_index)
        if coating_name is not None:
            resolved[name] = coating_name

    return resolved


def _build_prescription(system, chassis_pivot, chassis_tilt_deg, sources,
                        trace_wavelengths, resolved_glass_names, coating_names,
                        face_coatings, detector_pixels, coating_file,
                        glass_catalog, prescription_kwargs):
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
        json.dumps(prescription.document, indent=2))


def _write_polygon_files_and_glass_catalog(output_dir, prescription,
                                           glass_materials,
                                           resolved_glass_names,
                                           glass_catalog):
    """Write POB files and the generated glass catalog, returning fit stats."""
    for file_name, text in prescription.polygon_files.items():
        (output_dir / file_name).write_text(text)
    return _write_glass_catalog(output_dir, glass_materials,
                                resolved_glass_names, glass_catalog)


def _write_glass_catalog(output_dir, glass_materials, resolved_glass_names,
                         glass_catalog):
    """Fit and write the glass catalog, returning the fit records."""
    catalog_lines = []
    fits = {}
    for material in glass_materials:
        fit = fit_sellmeier(material)
        fits[material.name] = fit
        catalog_lines.append(format_glass_catalog(
            material, fit, resolved_glass_names[material.name]))
    (output_dir / f"{glass_catalog}.AGF").write_text("".join(catalog_lines))
    return fits


def _write_coating_files(output_dir, coating_blocks, coating_file):
    """Write the Zemax coating file in the bundle root."""
    coating_blocks = list(coating_blocks)
    if not coating_blocks:
        return
    coating_text = format_coating_file(
        coating_blocks, "Apollo14 combiner mirror stack")
    (output_dir / coating_file).write_text(coating_text)


def _write_generated_scripts(output_dir):
    """Write the two generated ZOS-API scripts."""
    (output_dir / "build_zemax_model.py").write_text(build_script_text())
    (output_dir / "run_fov_sweep.py").write_text(sweep_script_text())
    (output_dir / "build_and_run.py").write_text(build_and_run_script_text())


def _write_readme_file(output_dir, prescription, fits, glass_names,
                       glass_catalog, coating_file, coating_blocks, notes):
    """Write the bundle README."""
    (output_dir / "README.md").write_text(_readme(
        prescription, fits, glass_names, glass_catalog, coating_file,
        coating_blocks, notes))


def _readme(prescription, fits, glass_names, glass_catalog, coating_file,
            coating_blocks, notes) -> str:
    return "\n".join([
        _readme_header(notes),
        _readme_install(glass_catalog, coating_file),
        _readme_objects(prescription),
        _readme_glass(fits, glass_names),
        _readme_coatings(coating_blocks),
        _readme_checks(),
        _readme_detector_orientation(),
        _readme_expected_bias(),
    ])


def _readme_header(notes):
    return "\n".join([
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
        "Nothing here needs a Premium-only feature: the chassis and stop are native",
        "polygon objects rather than imported CAD parts, the sources are analytic rather",
        "than measured rayfiles, and the sweep script disables scattering.",
        "",
        "The build script opens a standalone OpticStudio session, which takes a licence",
        "seat. If none is free — usually because OpticStudio is already open — it falls",
        "back to attaching to that running instance, which needs Programming →",
        "Interactive Extension enabled there first. It will not close a session it did",
        "not start.",
    ])


def _readme_install(glass_catalog, coating_file):
    return "\n".join([
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
        "3. every `.POB` file → the **Objects** folder, `Polygon Objects` subfolder.",
        "4. Build the model:",
        "",
        "       python build_zemax_model.py prescription.json apollo14_combiner.zmx",
        "",
        "5. Trace every FOV direction and dump the pupil detector:",
        "",
        "       python run_fov_sweep.py apollo14_combiner.zmx prescription.json out\\",
        "",
        "`build_zemax_model.py` prints anything it could not set automatically. ZOS-API",
        "property names move between releases, so treat that list as the manual to-do,",
        "not as a failure.",
    ])


def _readme_objects(prescription):
    return "\n".join([
        "## Objects",
        "",
        "| # | type | comment | material | inside of | coating |",
        "|---|------|---------|----------|-----------|---------|",
        _object_rows(prescription),
    ])


def _readme_glass(fits, glass_names):
    return "\n".join([
        "## Glass",
        "",
        "| material | n(d) | V(d) | RMS Δn | max Δn |",
        "|----------|------|------|--------|--------|",
        _fit_rows(fits, glass_names),
        "",
        "The catalog holds a Sellmeier 1 fit of Apollo14's measured index table. It is",
        "valid only across the wavelength range on the record's `LD` line; the residual",
        "above is the fit error against that table.",
    ])


def _readme_coatings(coating_blocks):
    return "\n".join([
        "## Coatings",
        "",
        _coating_rows(coating_blocks),
    ])


def _readme_checks():
    return "\n".join([
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
    ])


def _readme_detector_orientation():
    return "\n".join([
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
    ])


def _readme_expected_bias():
    return "\n".join([
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
    ])

    
def _object_rows(prescription):
    """Render the exported object table rows."""
    return "\n".join(
        f"| {entry['index']} | {entry['type']} | {entry['comment']} | "
        f"{entry.get('material') or '—'} | "
        f"{entry.get('inside_of') or '—'} | {_display_coating(entry)} |"
        for entry in prescription.objects)


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
            face_coatings.items(), key=lambda item: int(item[0])))


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
        for name, fit in fits.items())


def _coating_rows(coating_blocks):
    """Render the exported coating list."""
    return "\n".join(f"- `{block.name}`" for block in coating_blocks) \
        or "- (none exported)"
