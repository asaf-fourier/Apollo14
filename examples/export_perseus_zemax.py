"""Export the optimized Perseus combiner to Zemax OpticStudio for validation.

Takes the pupil optimizer's output (per-mirror ``R(λ)`` and mirror spacings) and,
when available, the Atlas coating design built from it, and writes a complete
non-sequential OpticStudio model: polygon objects for the chassis and beam stop,
a glass catalog for the substrate, a coating file, a prescription, and the two
ZOS-API scripts that build and trace it on the Windows machine.

The coating file carries **every** fidelity rung side by side — ideal, flat
``R(λ)``, Atlas ``R(λ, θ)``, and the physical film stack — under distinct names.
The face-coating map below chooses which rung each mirror face uses. That keeps
the policy in one small, editable block instead of burying it in the generated
ZOS-API script.

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

from apollo14.elements.pupil import RectangularPupil
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

# Default rung for the mirror front face. Change this value to move the front
# face to a different fidelity rung without touching the exporter internals.
FRONT_FACE_COATING_MODE = "flat"
BACK_FACE_COATING_MODE = None
MIRROR_FACE_COATING_MODES = {
    "front": FRONT_FACE_COATING_MODE,
    "back": BACK_FACE_COATING_MODE,
}

# FOV sampling for the exported sources. Apollo14 traces one direction at a
# time, so this is also the number of Zemax sources and the number of traces the
# sweep script runs — keep it coarser than the optimizer's own grid.
NUM_FOV_X = 3
NUM_FOV_Y = 3

# Source ray budget for the OpticStudio validation sweep. This is much lower
# than the optimizer's internal sampling because the sweep is for regression and
# throughput checks, not for a final Monte Carlo estimate.
SOURCE_ANALYSIS_RAYS = 20_000

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
    """Return ``(mirrors, spacings, eyebox)`` from an optimize_pupil_perseus run.

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
    eyebox = report["eyebox"]
    return mirrors, spacings, eyebox


def load_detector_axes(report_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the saved pupil sampling axes from the optimization run."""
    response_path = report_dir / "response.npz"
    if not response_path.exists():
        raise FileNotFoundError(
            f"Missing response.npz next to {report_dir / 'optimization_report.json'}; "
            "the export needs the saved pupil axes to keep detector sampling "
            "consistent.")
    with np.load(response_path) as data:
        return np.asarray(data["pupil_x_mm"], dtype=float), np.asarray(
            data["pupil_y_mm"], dtype=float)


def detector_pixels_for_axis(axis: np.ndarray) -> int:
    """Return the pixel count implied by one saved detector axis."""
    if axis.size < 2:
        raise ValueError("Detector axis must have at least two samples.")
    return int(axis.size)


def detector_pixels_for_size(width_mm: float, height_mm: float,
                             pitch_x_mm: float, pitch_y_mm: float
                             ) -> tuple[int, int]:
    """Return detector pixels for a rectangle at the saved axis pitch."""
    return (int(round(width_mm / pitch_x_mm)),
            int(round(height_mm / pitch_y_mm)))


def build_eyebox_detector(system: OpticalSystem, eyebox: dict) -> RectangularPupil:
    """Build the exported 8×8 eyebox detector on the physical pupil plane."""
    pupil = next(e for e in system.elements if isinstance(e, RectangularPupil))
    return RectangularPupil(
        name="eyebox",
        position=pupil.position,
        normal=pupil.normal,
        width=2.0 * float(eyebox["half_x"]),
        height=2.0 * float(eyebox["half_y"]),
    )


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

def build_coatings(mirrors, coating_results):
    """Return coating blocks and names for all four rungs."""
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

    return blocks, names_by_mode, notes


def select_mirror_face_coatings(names_by_mode, face_modes):
    """Resolve ``front``/``back`` face policy to per-mirror coating names."""
    face_coatings = {}
    for face_name, mode_name in face_modes.items():
        if mode_name is None:
            continue
        try:
            mode_coatings = names_by_mode[mode_name]
        except KeyError as error:
            raise ValueError(
                f"Unknown face coating mode {mode_name!r} for {face_name!r}; "
                f"choose one of {sorted(names_by_mode)} or None.") from error
        for mirror_name, coating_name in mode_coatings.items():
            face_coatings[(mirror_name, face_name)] = coating_name
    return face_coatings


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
            analysis_rays=SOURCE_ANALYSIS_RAYS,
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

    mirrors, spacings, eyebox = load_optimizer_report(report_dir)
    pupil_x_mm, pupil_y_mm = load_detector_axes(report_dir)
    coating_results = load_coating_design(coating_dir, report_dir)
    system = build_system(mirrors, spacings)
    eyebox_detector = build_eyebox_detector(system, eyebox)
    pupil_pitch_x_mm = abs(float(pupil_x_mm[1] - pupil_x_mm[0]))
    pupil_pitch_y_mm = abs(float(pupil_y_mm[1] - pupil_y_mm[0]))

    pupil = next(e for e in system.elements if isinstance(e, RectangularPupil))
    detector_pixels_by_name = {
        pupil.name: (detector_pixels_for_axis(pupil_x_mm),
                     detector_pixels_for_axis(pupil_y_mm)),
        eyebox_detector.name: detector_pixels_for_size(
            float(eyebox_detector.width), float(eyebox_detector.height),
            pupil_pitch_x_mm, pupil_pitch_y_mm),
    }

    projector = Projector.uniform(
        position=PERSEUS_LIGHT_POSITION,
        direction=PERSEUS_PROJECTOR_DIRECTION,
        beam_width=PERSEUS_BEAM_WIDTH, beam_height=PERSEUS_BEAM_HEIGHT,
        nx=1, ny=1)

    blocks, coating_names_by_mode, coating_notes = build_coatings(
        mirrors, coating_results)
    face_coatings = select_mirror_face_coatings(
        coating_names_by_mode, MIRROR_FACE_COATING_MODES)
    sources = build_sources(projector)

    print("── Perseus → Zemax OpticStudio ──")
    print(f"optimizer run : {report_dir}")
    print(f"coating run   : {coating_dir or '(none — flat R(λ) only)'}")
    print(f"mirrors       : {len(mirrors)}  spacings "
          f"{np.round(spacings, 4).tolist()}")
    print(f"face coatings : {MIRROR_FACE_COATING_MODES}")
    print(f"front mode    : {FRONT_FACE_COATING_MODE}  "
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
        face_coatings=face_coatings,
        detector_pixels_by_name=detector_pixels_by_name,
        extra_pupils=[eyebox_detector],
        notes=(f"Perseus combiner, {len(mirrors)} mirrors, front face mode "
               f"`{FRONT_FACE_COATING_MODE}`.\n\n"
               f"- face coating modes: `{MIRROR_FACE_COATING_MODES}`\n"
               f"- detector pixels: `{detector_pixels_by_name}`\n"
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
