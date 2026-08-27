"""Export the optimized Perseus combiner to Zemax OpticStudio for validation.

Takes the pupil optimizer's output (per-mirror ``R(λ)`` and mirror spacings) and,
when available, the Atlas coating design built from it, and writes a complete
non-sequential OpticStudio model: polygon objects for the chassis and beam stop,
a glass catalog for the substrate, a coating file, a prescription, and the two
ZOS-API scripts that build and trace it on the Windows machine.

The coating file carries **every** fidelity rung side by side — ideal, flat
``R(λ)``, Atlas ``R(λ, θ)``, and the physical film stack — under distinct names.
``COATING_MODE`` picks which one the exported objects reference; switching rungs
afterwards is then a coating-name edit in OpticStudio rather than a re-export.
Working up the rungs is how the exercise isolates each assumption the tracer
makes: geometry first, then spectrum, then angle, then polarization.

Run::

    python examples/export_perseus_zemax.py

The bundle lands in ``examples/reports/export_perseus_zemax/<timestamp>/``;
copy it to the OpticStudio machine and follow its README.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from apollo14.elements.partial_mirror import PartialMirror
from apollo14.export import export_zemax_bundle
from apollo14.export.coating import (
    flat_table_coating,
    ideal_coating,
    stack_coating,
    table_coating,
)
from apollo14.export.prescription import SourceSpec
from apollo14.materials import agc_m074, air
from apollo14.perseus import (
    PERSEUS_BEAM_HEIGHT,
    PERSEUS_BEAM_WIDTH,
    PERSEUS_COMBINER_CENTER,
    PERSEUS_FOV_AROUND_PROJECTOR_Y,
    PERSEUS_FOV_AROUND_X,
    PERSEUS_LIGHT_POSITION,
    PERSEUS_PANTOSCOPIC_TILT,
    PERSEUS_PROJECTOR_DIRECTION,
    build_perseus_geometry,
)
from apollo14.projector import FovGrid, Projector
from apollo14.system import OpticalSystem
from apollo14.units import nm

# ── What to export ──────────────────────────────────────────────────────────
# REPORT_DIR / COATING_DIR = None → auto-pick the newest run of each.
OPTIMIZE_RUNS_ROOT = Path("examples/reports/optimize_pupil_perseus")
COATING_RUNS_ROOT = Path("examples/reports/design_perseus_mirror_coating")
REPORT_DIR: Path | None = None
COATING_DIR: Path | None = None

# Which rung the exported objects reference: "ideal", "flat", "atlas" or "stack".
COATING_MODE = "ideal"

# FOV sampling for the exported sources. Apollo14 traces one direction at a
# time, so this is also the number of Zemax sources and the number of traces the
# sweep script runs — keep it coarser than the optimizer's own grid.
NUM_FOV_X = 3
NUM_FOV_Y = 3

# Pupil detector resolution. 0.1 mm pixels over the 14 × 18 mm pupil.
DETECTOR_PIXELS = (140, 180)

# Trace wavelengths registered in the Zemax system — the R/G/B lines Apollo14
# optimizes against.
TRACE_WAVELENGTHS_NM = (460.0, 525.0, 630.0)

OUTPUT_ROOT = Path("examples/reports/export_perseus_zemax")


# ── Load the optimizer output ───────────────────────────────────────────────

def latest_run(root: Path, file_name: str) -> Path:
    candidates = sorted(root.glob(f"*/{file_name}"))
    if not candidates:
        raise FileNotFoundError(f"No {file_name} under {root}.")
    return candidates[-1].parent


def load_optimizer_report(report_dir: Path):
    """Return ``(mirrors, spacings)`` from an optimize_pupil_perseus run.

    ``mirrors`` is a list of ``(wavelengths_nm, reflectance)`` in mirror order.
    """
    report = json.loads((report_dir / "optimization_report.json").read_text())

    elements = [element for element in report["system"]["elements"]
                if element["type"] == "PartialMirror"]
    elements.sort(key=lambda element: int(element["name"].split("_")[1]))

    mirrors = []
    for element in elements:
        wavelengths_nm = np.asarray(element["wavelengths"]) / float(nm)
        reflectance = np.asarray(element["reflectance"])
        order = np.argsort(wavelengths_nm)
        mirrors.append((wavelengths_nm[order], reflectance[order]))

    spacings = np.asarray(report["final_params"]["spacings"], dtype=float)
    return mirrors, spacings


def load_coating_design(coating_dir: Path | None, report_dir: Path):
    """Return ``{mirror index: result dict}`` from an Atlas coating design.

    Refuses a design built from a *different* optimizer run than the one being
    exported. Coatings are fitted to a specific set of ``R(λ)`` targets, so
    pairing them with another run's geometry and reflectances would produce a
    model that looks fine and validates nothing.
    """
    if coating_dir is None:
        return {}
    design = json.loads((coating_dir / "coating_design.json").read_text())

    source_report = str(design.get("source_report", "")).rstrip("/")
    if source_report != str(report_dir):
        raise ValueError(
            f"Coating design in {coating_dir} was built from "
            f"{source_report!r}, but this export uses {str(report_dir)!r}. "
            "Re-run examples/design_perseus_mirror_coating.py against the "
            "current optimizer output, or set COATING_DIR explicitly.")

    return {record["index"]: record["result"] for record in design["mirrors"]}


# ── Build the system the optimizer converged on ─────────────────────────────

def build_system(mirrors, spacings) -> OpticalSystem:
    geometry = build_perseus_geometry(spacings=spacings)

    system = OpticalSystem(env_material=air)
    system.add(geometry.chassis)
    system.add(geometry.aperture)
    for mirror_index, (wavelengths_nm, reflectance) in enumerate(mirrors):
        system.add(PartialMirror(
            name=f"mirror_{mirror_index}",
            position=geometry.mirror_positions[mirror_index],
            normal=geometry.mirror_normal.copy(),
            width=geometry.mirror_width,
            height=geometry.mirror_height,
            reflectance=reflectance,
            wavelengths=wavelengths_nm * nm,
        ))
    system.add(geometry.pupil)
    return system


# ── Coatings: every rung, named so they can be swapped in OpticStudio ───────

def build_coatings(mirrors, coating_results, mode: str):
    """Return ``(blocks, names_by_element, notes)`` for all four rungs."""
    blocks = []
    names_by_mode: dict[str, dict[str, str]] = {
        "ideal": {}, "flat": {}, "atlas": {}, "stack": {}}
    notes = []

    reference_wavelength_nm = 550.0

    for mirror_index, (wavelengths_nm, reflectance) in enumerate(mirrors):
        element_name = f"mirror_{mirror_index}"

        reference_reflectance = float(
            np.interp(reference_wavelength_nm, wavelengths_nm, reflectance))
        ideal_name = f"IDEAL_M{mirror_index}"
        blocks.append(ideal_coating(ideal_name, reference_reflectance))
        names_by_mode["ideal"][element_name] = ideal_name

        flat_name = f"FLAT_M{mirror_index}"
        blocks.append(flat_table_coating(flat_name, wavelengths_nm, reflectance))
        names_by_mode["flat"][element_name] = flat_name

        result = coating_results.get(mirror_index)
        if result is None:
            # No Atlas design for this mirror — fall back so the export still
            # builds, and say so rather than quietly shipping a mixed model.
            names_by_mode["atlas"][element_name] = flat_name
            names_by_mode["stack"][element_name] = flat_name
            notes.append(f"mirror_{mirror_index}: no Atlas coating, using "
                         f"{flat_name}")
            continue

        achieved = result["achieved_reflectance"]
        atlas_name = f"ATLAS_M{mirror_index}"
        blocks.append(table_coating(
            atlas_name,
            achieved["wavelengths_nm"],
            achieved["angles_deg"],
            np.asarray(achieved["values"]),
        ))
        names_by_mode["atlas"][element_name] = atlas_name

        stack_name = f"STACK_M{mirror_index}"
        blocks.append(stack_coating(
            stack_name, result["layers"],
            wavelength_range_nm=(float(min(achieved["wavelengths_nm"])),
                                 float(max(achieved["wavelengths_nm"]))),
        ))
        names_by_mode["stack"][element_name] = stack_name

    if mode not in names_by_mode:
        raise ValueError(f"COATING_MODE must be one of {list(names_by_mode)}, "
                         f"got {mode!r}")
    return blocks, names_by_mode[mode], notes


# ── Sources: one per FOV direction ──────────────────────────────────────────

def build_sources(projector: Projector) -> list[SourceSpec]:
    grid = FovGrid(projector.direction, PERSEUS_FOV_AROUND_X,
                   PERSEUS_FOV_AROUND_PROJECTOR_Y, NUM_FOV_X, NUM_FOV_Y)
    angles = np.asarray(grid.flat_angles)

    sources = []
    for direction_index, direction in enumerate(grid):
        direction = np.asarray(direction, dtype=float)
        # The beam's own basis, from the projector — its roll decides how much
        # of the 10 × 2 mm beam clears the 11 × 2 mm stop, so it must come from
        # the same code the tracer uses rather than be re-derived.
        beam_local_x, beam_local_y = projector._compute_basis(direction)
        angle_x_deg, angle_y_deg = np.degrees(angles[direction_index])
        sources.append(SourceSpec(
            label=f"fov_{angle_x_deg:+05.1f}_{angle_y_deg:+05.1f}".replace(
                ".", "p").replace("+", "p").replace("-", "m"),
            position=np.asarray(projector.position, dtype=float),
            direction=direction,
            beam_width=float(projector.beam_width),
            beam_height=float(projector.beam_height),
            beam_local_x=np.asarray(beam_local_x, dtype=float),
            beam_local_y=np.asarray(beam_local_y, dtype=float),
            # Apollo14 traces one direction at a time; the sweep script gives
            # each source the power in turn, so only the first is live here.
            power=1.0 if direction_index == 0 else 0.0,
        ))
    return sources


def main():
    report_dir = REPORT_DIR or latest_run(OPTIMIZE_RUNS_ROOT,
                                          "optimization_report.json")
    try:
        coating_dir = COATING_DIR or latest_run(COATING_RUNS_ROOT,
                                                "coating_design.json")
    except FileNotFoundError:
        coating_dir = None

    mirrors, spacings = load_optimizer_report(report_dir)
    coating_results = load_coating_design(coating_dir, report_dir)
    system = build_system(mirrors, spacings)

    projector = Projector.uniform(
        position=PERSEUS_LIGHT_POSITION,
        direction=PERSEUS_PROJECTOR_DIRECTION,
        beam_width=PERSEUS_BEAM_WIDTH, beam_height=PERSEUS_BEAM_HEIGHT,
        nx=1, ny=1)

    blocks, coating_names, coating_notes = build_coatings(
        mirrors, coating_results, COATING_MODE)
    sources = build_sources(projector)

    print("── Perseus → Zemax OpticStudio ──")
    print(f"optimizer run : {report_dir}")
    print(f"coating run   : {coating_dir or '(none — flat R(λ) only)'}")
    print(f"mirrors       : {len(mirrors)}  spacings "
          f"{np.round(spacings, 4).tolist()}")
    print(f"coating mode  : {COATING_MODE}  "
          f"({len(coating_results)} Atlas designs available)")
    for note in coating_notes:
        print(f"  ! {note}")
    print(f"sources       : {len(sources)} FOV directions "
          f"({NUM_FOV_X}×{NUM_FOV_Y})")

    output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    bundle_dir = export_zemax_bundle(
        output_dir,
        system,
        chassis_pivot=np.asarray(PERSEUS_COMBINER_CENTER, dtype=float),
        # Perseus applies the pantoscopic tilt as a right-handed rotation by
        # the negative of the tilt (see apollo14.perseus conventions).
        chassis_tilt_deg=float(np.degrees(-float(PERSEUS_PANTOSCOPIC_TILT))),
        sources=sources,
        trace_wavelengths=[wavelength * nm
                           for wavelength in TRACE_WAVELENGTHS_NM],
        glass_materials=[agc_m074],
        coating_blocks=blocks,
        coating_names=coating_names,
        detector_pixels=DETECTOR_PIXELS,
        notes=(f"Perseus combiner, {len(mirrors)} mirrors, coating rung "
               f"`{COATING_MODE}`.\n\n"
               f"- optimizer run: `{report_dir}`\n"
               f"- coating run: `{coating_dir or 'none'}`\n"),
    )

    pupil_report = report_dir / "pupil_report.html"
    if not pupil_report.exists():
        raise FileNotFoundError(
            f"The matching optimizer run has no eyebox report: {pupil_report}")
    shutil.copy2(pupil_report, bundle_dir / "apollo14_eyebox_report.html")

    print(f"\nBundle: {output_dir}")
    for path in sorted(output_dir.iterdir()):
        print(f"  {path.name}")
    print("\nCopy to the OpticStudio machine and follow README.md there.")


if __name__ == "__main__":
    main()
