"""Export the optimized Perseus combiner to Zemax OpticStudio for validation.

Takes the pupil optimizer's saved system/projector snapshot and, when available,
the Atlas coating design built from it, and writes a complete non-sequential
OpticStudio model: a polygon object for the chassis, a native Boolean
outer-minus-inner beam stop, a glass catalog for the substrate, a coating file,
a prescription, and the two ZOS-API scripts that build and trace it on the
Windows machine.

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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from apollo14.elements.aperture import RectangularAperture
from apollo14.elements.glass_block import GlassBlock, GlassFace
from apollo14.elements.partial_mirror import PartialMirror
from apollo14.elements.pupil import RectangularPupil
from apollo14.export import export_zemax_bundle
from apollo14.export.coating import (
    flat_table_coating,
    ideal_coating,
    stack_coating,
    table_coating,
)
from apollo14.export.prescription import SourceSpec
from apollo14.materials import agc_m074, air
from apollo14.projector import FovGrid, Projector
from apollo14.spectral import SpectralTable
from apollo14.system import OpticalSystem
from apollo14.units import nm

# ── What to export ──────────────────────────────────────────────────────────
# REPORT_DIR / COATING_DIR = None → auto-pick the newest run of each.
OPTIMIZE_RUNS_ROOT = Path("examples/reports/optimize_pupil_perseus")
COATING_RUNS_ROOT = Path("examples/reports/design_perseus_mirror_coating")
REPORT_DIR: Path | None = None
COATING_DIR: Path | None = None
DETECTOR_SCAN_PIXELS = 200
MIN_RELATIVE_INTENSITY = 1.0e-8
MIN_ABSOLUTE_INTENSITY = 1.0e-8

# Default rung for the mirror faces. Change this value to move the coating
# policy without touching the exporter internals.
FRONT_FACE_COATING_MODE = "atlas"
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
DETECTOR_SCAN_PIXELS = 200


# ── Load the optimizer output ───────────────────────────────────────────────

_MATERIALS_BY_NAME = {air.name: air, agc_m074.name: agc_m074}


@dataclass(frozen=True)
class OptimizerSnapshot:
    """The exact system and source state persisted by the optimizer."""

    system: OpticalSystem
    projector: Projector
    fov_x: float
    fov_y: float
    spacings: np.ndarray
    eyebox: dict
    git_sha: str


def latest_run(root: Path, file_name: str) -> Path:
    candidates = sorted(root.glob(f"*/{file_name}"))
    if not candidates:
        raise FileNotFoundError(f"No {file_name} under {root}.")
    return candidates[-1].parent


def latest_matching_coating_run(
        coating_root: Path, report_dir: Path) -> Path | None:
    """Return the newest coating run built from ``report_dir`` if present."""
    matches: list[Path] = []
    for design_path in coating_root.glob("*/coating_design.json"):
        try:
            design = json.loads(design_path.read_text())
        except json.JSONDecodeError:
            continue
        source_report = str(design.get("source_report", "")).rstrip("/")
        if source_report == str(report_dir):
            matches.append(design_path.parent)
    if not matches:
        return None
    return max(matches, key=lambda path: (path / "coating_design.json").stat().st_mtime)


def coating_source_report(coating_dir: Path) -> str:
    """Read the optimizer report path baked into a coating design."""
    design = json.loads((coating_dir / "coating_design.json").read_text())
    return str(design.get("source_report", "")).rstrip("/")


def _material_named(name: str):
    try:
        return _MATERIALS_BY_NAME[name]
    except KeyError as error:
        raise ValueError(
            f"Optimizer report uses unknown material {name!r}; add it to "
            "_MATERIALS_BY_NAME before exporting.") from error


def _spectral_table(data: dict) -> SpectralTable:
    return SpectralTable.from_samples(data["wavelengths"], data["values"])


def _restore_element(data: dict):
    """Restore one serialized optimizer element without recreating geometry."""
    common = {
        "name": data["name"],
        "position": np.asarray(data["position"], dtype=float),
    }
    if data["type"] == "GlassBlock":
        faces = [
            GlassFace(
                name=face["name"],
                position=np.asarray(face["position"], dtype=float),
                normal=np.asarray(face["normal"], dtype=float),
                vertices=np.asarray(face["vertices"], dtype=float),
                coating_reflectance=_spectral_table(
                    face["coating_reflectance"]),
            )
            for face in data["faces"]
        ]
        return GlassBlock(
            **common,
            material=_material_named(data["material"]),
            faces=faces,
        )
    if data["type"] == "RectangularAperture":
        return RectangularAperture(
            **common,
            normal=np.asarray(data["normal"], dtype=float),
            width=float(data["width"]),
            height=float(data["height"]),
            inner_width=float(data["inner_width"]),
            inner_height=float(data["inner_height"]),
        )
    if data["type"] == "PartialMirror":
        return PartialMirror(
            **common,
            normal=np.asarray(data["normal"], dtype=float),
            width=float(data["width"]),
            height=float(data["height"]),
            wavelengths=np.asarray(data["wavelengths"], dtype=float),
            reflectance=np.asarray(data["reflectance"], dtype=float),
        )
    if data["type"] == "RectangularPupil":
        return RectangularPupil(
            **common,
            normal=np.asarray(data["normal"], dtype=float),
            width=float(data["width"]),
            height=float(data["height"]),
        )
    raise ValueError(f"Unsupported optimizer element type {data['type']!r}.")


def _restore_system(data: dict) -> OpticalSystem:
    system = OpticalSystem(env_material=_material_named(data["env_material"]))
    for element in data["elements"]:
        system.add(_restore_element(element))
    return system


def _restore_projector(data: dict) -> Projector:
    spectrum = data.get("spectrum")
    restored_spectrum = None
    if spectrum is not None:
        restored_spectrum = (
            np.asarray(spectrum["wavelengths"], dtype=float),
            np.asarray(spectrum["radiance"], dtype=float),
        )
    return Projector.uniform(
        position=np.asarray(data["position"], dtype=float),
        direction=np.asarray(data["direction"], dtype=float),
        beam_width=float(data["beam_width"]),
        beam_height=float(data["beam_height"]),
        nx=int(data["nx"]),
        ny=int(data["ny"]),
        falloff_x=float(data["falloff_x"]),
        falloff_y=float(data["falloff_y"]),
        spectrum=restored_spectrum,
    )


def load_optimizer_report(report_dir: Path) -> OptimizerSnapshot:
    """Restore the geometry, source, and FOV saved by the optimizer."""
    report = json.loads((report_dir / "optimization_report.json").read_text())
    projectors = report["projectors"]
    if len(projectors) != 1:
        raise ValueError(
            "Perseus Zemax export requires exactly one saved projector, got "
            f"{len(projectors)}.")
    return OptimizerSnapshot(
        system=_restore_system(report["system"]),
        projector=_restore_projector(projectors[0]),
        fov_x=float(report["fov_grid"]["x_fov"]),
        fov_y=float(report["fov_grid"]["y_fov"]),
        spacings=np.asarray(report["final_params"]["spacings"], dtype=float),
        eyebox=report["eyebox"],
        git_sha=str(report["git_sha"]),
    )


def mirror_samples(system: OpticalSystem) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the saved reflectance samples in numeric mirror order."""
    mirrors = [element for element in system.elements
               if isinstance(element, PartialMirror)]
    mirrors.sort(key=lambda element: int(element.name.split("_")[1]))
    samples = []
    for mirror in mirrors:
        wavelengths_nm = np.asarray(mirror.wavelengths) / float(nm)
        order = np.argsort(wavelengths_nm)
        samples.append((wavelengths_nm[order], np.asarray(mirror.reflectance)[order]))
    return samples


def chassis_pose(system: OpticalSystem) -> tuple[np.ndarray, float]:
    """Get the local POB origin and x-tilt from the saved chassis itself."""
    chassis = next(
        element for element in system.elements
        if isinstance(element, GlassBlock) and element.name == "chassis")
    bottom_normal = np.asarray(chassis.get_face("bottom").normal, dtype=float)
    if not np.isclose(bottom_normal[0], 0.0, atol=1e-6):
        raise ValueError(
            "The Zemax POB exporter supports chassis rotation about world x; "
            f"saved bottom normal is {bottom_normal.tolist()}.")
    tilt_deg = float(np.degrees(np.arctan2(
        bottom_normal[1], -bottom_normal[2])))
    return np.asarray(chassis.position, dtype=float), tilt_deg


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
    """Return the exported detector pixel count for one axis."""
    if axis.size < 2:
        raise ValueError("Detector axis must have at least two samples.")
    return DETECTOR_SCAN_PIXELS


def detector_pixels_for_size(width_mm: float, height_mm: float,
                             pitch_x_mm: float, pitch_y_mm: float
                             ) -> tuple[int, int]:
    """Return detector pixels for a rectangle at the saved axis pitch."""
    return (DETECTOR_SCAN_PIXELS, DETECTOR_SCAN_PIXELS)


def build_eyebox_detector(system: OpticalSystem, eyebox: dict) -> RectangularPupil:
    """Build the exported eyebox detector on the physical pupil plane."""
    pupil = next(e for e in system.elements if isinstance(e, RectangularPupil))
    return RectangularPupil(
        name="eyebox",
        position=pupil.position,
        normal=pupil.normal,
        width=2.0 * float(eyebox["half_x"]),
        height=2.0 * float(eyebox["half_y"]),
    )


def build_ambient_detectors(system: OpticalSystem) -> list[RectangularPupil]:
    """Build four ambient detectors around the chassis+pupil bounding box."""
    offset_mm = 10.0
    chassis = next(
        element for element in system.elements if isinstance(element, GlassBlock)
        and element.name == "chassis")
    pupil = next(e for e in system.elements if isinstance(e, RectangularPupil))

    points = []
    for face in chassis.faces:
        points.extend(np.asarray(face.vertices, dtype=float))

    pupil_position = np.asarray(pupil.position, dtype=float)
    pupil_local_x = np.asarray(pupil._local_x, dtype=float)
    pupil_local_y = np.asarray(pupil._local_y, dtype=float)
    pupil_half_extents = np.asarray(pupil.half_extents, dtype=float)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            points.append(
                pupil_position
                + sx * pupil_half_extents[0] * pupil_local_x
                + sy * pupil_half_extents[1] * pupil_local_y
            )

    bbox = np.asarray(points, dtype=float)
    mins = bbox.min(axis=0)
    maxs = bbox.max(axis=0)
    center = 0.5 * (mins + maxs)
    span_x = float(maxs[0] - mins[0])
    span_y = float(maxs[1] - mins[1])
    span_z = float(maxs[2] - mins[2])
    span_y_with_margin = span_y + 2.0 * offset_mm
    span_z_with_margin = span_z + 2.0 * offset_mm

    ambient_detectors = [
        RectangularPupil(
            name="ambient_y_neg",
            position=np.array(
                [center[0], mins[1] - offset_mm, center[2]], dtype=float),
            normal=np.array([0.0, -1.0, 0.0], dtype=float),
            width=span_x,
            height=span_z_with_margin,
        ),
        RectangularPupil(
            name="ambient_y_pos",
            position=np.array(
                [center[0], maxs[1] + offset_mm, center[2]], dtype=float),
            normal=np.array([0.0, 1.0, 0.0], dtype=float),
            width=span_x,
            height=span_z_with_margin,
        ),
        RectangularPupil(
            name="ambient_z_neg",
            position=np.array(
                [center[0], center[1], mins[2] - offset_mm], dtype=float),
            normal=np.array([0.0, 0.0, -1.0], dtype=float),
            width=span_x,
            height=span_y_with_margin,
        ),
        RectangularPupil(
            name="ambient_z_pos",
            position=np.array(
                [center[0], center[1], maxs[2] + offset_mm], dtype=float),
            normal=np.array([0.0, 0.0, 1.0], dtype=float),
            width=span_x,
            height=span_y_with_margin,
        ),
    ]
    return ambient_detectors


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
        candidates = sorted(COATING_RUNS_ROOT.glob("*/coating_design.json"))
        matching = [
            str(path.parent)
            for path in candidates
            if str(json.loads(path.read_text()).get("source_report", "")).rstrip("/")
            == str(report_dir)
        ]
        suggestion = (
            f"Matching coating run(s) for {str(report_dir)!r}: "
            f"{', '.join(matching)}."
            if matching else
            f"No coating_design.json under {COATING_RUNS_ROOT} was built from "
            f"{str(report_dir)!r}."
        )
        raise ValueError(
            f"Coating design in {coating_dir} was built from "
            f"{source_report!r}, but this export uses {str(report_dir)!r}. "
            f"{suggestion} Re-run examples/design_perseus_mirror_coating.py "
            "against the current optimizer output, or set COATING_DIR "
            "explicitly.")

    return {record["index"]: record["result"] for record in design["mirrors"]}


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

def build_sources(projector: Projector, fov_x: float,
                  fov_y: float) -> list[SourceSpec]:
    grid = FovGrid(projector.direction, fov_x, fov_y, NUM_FOV_X, NUM_FOV_Y)
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
    report_dir_was_auto_selected = REPORT_DIR is None
    try:
        if COATING_DIR is not None:
            coating_dir = COATING_DIR
        else:
            coating_dir = latest_matching_coating_run(COATING_RUNS_ROOT, report_dir)
            if coating_dir is None:
                coating_dir = latest_run(COATING_RUNS_ROOT, "coating_design.json")
    except FileNotFoundError:
        coating_dir = None

    if coating_dir is not None and report_dir_was_auto_selected:
        source_report = coating_source_report(coating_dir)
        if source_report != str(report_dir):
            matching = latest_matching_coating_run(COATING_RUNS_ROOT, report_dir)
            if matching is None:
                print(
                    "warning: no coating design matches the newest optimizer run; "
                    f"falling back to coating source report {source_report}.")
                report_dir = Path(source_report)

    snapshot = load_optimizer_report(report_dir)
    system = snapshot.system
    mirrors = mirror_samples(system)
    pupil_x_mm, pupil_y_mm = load_detector_axes(report_dir)
    coating_results = load_coating_design(coating_dir, report_dir)
    eyebox_detector = build_eyebox_detector(system, snapshot.eyebox)
    ambient_detectors = build_ambient_detectors(system)
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
    detector_pixels_by_name.update({
        detector.name: detector_pixels_for_size(
            float(detector.width), float(detector.height),
            pupil_pitch_x_mm, pupil_pitch_y_mm)
        for detector in ambient_detectors
    })

    blocks, coating_names_by_mode, coating_notes = build_coatings(
        mirrors, coating_results)
    face_coatings = select_mirror_face_coatings(
        coating_names_by_mode, MIRROR_FACE_COATING_MODES)
    sources = build_sources(snapshot.projector, snapshot.fov_x, snapshot.fov_y)
    chassis_origin, chassis_tilt_deg = chassis_pose(system)

    print("── Perseus → Zemax OpticStudio ──")
    print(f"optimizer run : {report_dir}")
    print(f"coating run   : {coating_dir or '(none — flat R(λ) only)'}")
    print(f"mirrors       : {len(mirrors)}  spacings "
          f"{np.round(snapshot.spacings, 4).tolist()}")
    print(f"face coatings : {MIRROR_FACE_COATING_MODES}")
    print(f"mirror mode   : {FRONT_FACE_COATING_MODE}  "
          f"({len(coating_results)} Atlas designs available)")
    print("ambient dets  : "
          f"{', '.join(detector.name for detector in ambient_detectors)}")
    for note in coating_notes:
        print(f"  ! {note}")
    chassis_material_name = next(
        element.material.name
        for element in system.elements
        if isinstance(element, GlassBlock) and element.name == "chassis")
    for mirror_index in range(len(mirrors)):
        mirror_name = f"mirror_{mirror_index}"
        selected_coating = face_coatings.get((mirror_name, "front"))
        if mirror_index in coating_results:
            status = "Atlas coating available"
        else:
            status = f"no Atlas coating, using {selected_coating}"
        print(
            f"  {mirror_name} | {chassis_material_name} | "
            f"{selected_coating} | {status}")
    print(f"sources       : {len(sources)} FOV directions "
          f"({NUM_FOV_X}×{NUM_FOV_Y})")

    output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    bundle_dir = export_zemax_bundle(
        output_dir,
        system,
        chassis_pivot=chassis_origin,
        chassis_tilt_deg=chassis_tilt_deg,
        sources=sources,
        trace_wavelengths=[wavelength * nm
                           for wavelength in TRACE_WAVELENGTHS_NM],
        glass_materials=[next(
            element.material for element in system.elements
            if isinstance(element, GlassBlock) and element.name == "chassis")],
        coating_blocks=blocks,
        face_coatings=face_coatings,
        detector_pixels_by_name=detector_pixels_by_name,
        extra_pupils=[eyebox_detector, *ambient_detectors],
        min_relative_intensity=MIN_RELATIVE_INTENSITY,
        min_absolute_intensity=MIN_ABSOLUTE_INTENSITY,
        notes=(f"Perseus combiner, {len(mirrors)} mirrors, mirror face mode "
               f"`{FRONT_FACE_COATING_MODE}`.\n\n"
               f"- face coating modes: `{MIRROR_FACE_COATING_MODES}`\n"
               f"- detector pixels: `{detector_pixels_by_name}`\n"
               f"- optimizer run: `{report_dir}`\n"
               f"- optimizer git SHA: `{snapshot.git_sha}`\n"
               "- geometry source: serialized optimizer system snapshot\n"
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
