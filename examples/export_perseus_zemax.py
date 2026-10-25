"""Export the optimized Perseus combiner to Zemax OpticStudio for validation.

Takes the pupil optimizer's saved system/projector snapshot and, when available,
the Atlas coating design built from it, and writes a complete non-sequential
OpticStudio model: a polygon object for the chassis, a native Boolean
outer-minus-inner beam stop, a glass catalog for the substrate, a coating file,
a prescription, ZOS-API scripts, and a portable position-space result writer
that build, trace, and report it on the Windows machine.

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
from apollo14.export import (
    ApertureSweepReportSpec,
    EyeboxOverlapReportSpec,
    PupilReportSpec,
    export_zemax_bundle,
)
from apollo14.export.zemax.formats.coating import (
    CoatingBlock,
    flat_table_coating,
    ideal_coating,
    stack_coating,
    table_coating,
    zemax_builtin_ideal_coating,
)
from apollo14.export.zemax.fov_grid import build_export_fov_samples
from apollo14.export.zemax.prescription import SourceSpec
from apollo14.materials import agc_m074, air
from apollo14.projector import Projector
from apollo14.spectral import SpectralTable
from apollo14.system import OpticalSystem
from apollo14.units import nm

# ── What to export ──────────────────────────────────────────────────────────
# REPORT_DIR / COATING_DIR = None → auto-pick the newest run of each.
OPTIMIZE_RUNS_ROOT = Path("examples/reports/optimize_pupil_perseus")
COATING_RUNS_ROOT = Path("examples/reports/design_perseus_mirror_coating")
REPORT_DIR: Path | None = None
COATING_DIR: Path | None = None
DETECTOR_SCAN_SHAPE = (200, 200)
EYE_PUPIL_WINDOW_CELLS = 3
MIN_RELATIVE_INTENSITY = 1.0e-8
MIN_ABSOLUTE_INTENSITY = 1.0e-8

# Default rung for the mirror faces. Change this value to move the coating
# policy without touching the exporter internals.
FRONT_FACE_COATING_MODE = "stack"
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
# The decimal FOV grid is established before source directions are generated,
# so the actual Zemax trace uses clean coordinates such as 0.25° and 0.5°.
FOV_GRID_DECIMALS = 3

# Source ray budget for the OpticStudio validation sweep. This is much lower
# than the optimizer's internal sampling because the sweep is for regression and
# throughput checks, not for a final Monte Carlo estimate.
SOURCE_ANALYSIS_RAYS = 20_000

# Zemax-side aperture tolerance states. These values change only the exported
# Boolean opening's local vertical axis. The full horizontal opening and its
# horizontal center remain identical to the nominal optimized model.
APERTURE_HEIGHT_DELTAS_MM = (0.0, 0.1)#(-0.1, 0.0, 0.1)
APERTURE_VERTICAL_SHIFTS_MM = (0.0)#(-0.1, 0.0, 0.1)

# Trace wavelengths registered in the Zemax system — the R/G/B lines Apollo14
# optimizes against.
TRACE_WAVELENGTHS_NM = (460.0, 525.0, 630.0)

OUTPUT_ROOT = Path("examples/reports/export_perseus_zemax")


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


@dataclass(frozen=True)
class DetectorExportPlan:
    """Detectors, sampling, and derived reports included in one bundle."""

    pupil: RectangularPupil
    eyebox: RectangularPupil
    ambient: tuple[RectangularPupil, ...]
    reports: tuple[PupilReportSpec, ...]
    overlap_report: EyeboxOverlapReportSpec
    pixels_by_name: dict[str, tuple[int, int]]

    @property
    def extra_detectors(self) -> tuple[RectangularPupil, ...]:
        """Detectors appended to the saved optimizer system for export."""
        return (self.eyebox, *self.ambient)


@dataclass(frozen=True)
class CoatingExportPlan:
    """Resolved coating definitions and face assignments for one bundle."""

    blocks: tuple[CoatingBlock, ...]
    face_coatings: dict[tuple[str, str], str]
    chassis_face_coatings: dict[tuple[str, str], str]
    report: dict
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ZemaxExportPlan:
    """Complete, validated inputs needed to write one Perseus Zemax bundle."""

    report_dir: Path
    coating_dir: Path | None
    snapshot: OptimizerSnapshot
    chassis: GlassBlock
    mirrors: tuple[tuple[np.ndarray, np.ndarray], ...]
    coating_results: dict[int, dict]
    detectors: DetectorExportPlan
    aperture_report: ApertureSweepReportSpec
    coatings: CoatingExportPlan
    sources: tuple[SourceSpec, ...]
    chassis_origin: np.ndarray
    chassis_tilt_deg: float


def latest_run(root: Path, file_name: str) -> Path:
    candidates = sorted(root.glob(f"*/{file_name}"))
    if not candidates:
        raise FileNotFoundError(f"No {file_name} under {root}.")
    return candidates[-1].parent


def _read_json(path: Path) -> dict:
    """Read one UTF-8 JSON object used by the export example."""
    return json.loads(path.read_text(encoding="utf-8"))


def latest_matching_coating_run(coating_root: Path, report_dir: Path) -> Path | None:
    """Return the newest coating run built from ``report_dir`` if present."""
    matches: list[Path] = []
    for design_path in coating_root.glob("*/coating_design.json"):
        try:
            design = _read_json(design_path)
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
    design = _read_json(coating_dir / "coating_design.json")
    return str(design.get("source_report", "")).rstrip("/")


def _material_named(name: str):
    try:
        return _MATERIALS_BY_NAME[name]
    except KeyError as error:
        raise ValueError(
            f"Optimizer report uses unknown material {name!r}; add it to "
            "_MATERIALS_BY_NAME before exporting."
        ) from error


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
                coating_reflectance=_spectral_table(face["coating_reflectance"]),
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
    report = _read_json(report_dir / "optimization_report.json")
    projectors = report["projectors"]
    if len(projectors) != 1:
        raise ValueError(
            f"Perseus Zemax export requires exactly one saved projector, got {len(projectors)}."
        )
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
    mirrors = [element for element in system.elements if isinstance(element, PartialMirror)]
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
        element
        for element in system.elements
        if isinstance(element, GlassBlock) and element.name == "chassis"
    )
    bottom_normal = np.asarray(chassis.get_face("bottom").normal, dtype=float)
    if not np.isclose(bottom_normal[0], 0.0, atol=1e-6):
        raise ValueError(
            "The Zemax POB exporter supports chassis rotation about world x; "
            f"saved bottom normal is {bottom_normal.tolist()}."
        )
    tilt_deg = float(np.degrees(np.arctan2(bottom_normal[1], -bottom_normal[2])))
    return np.asarray(chassis.position, dtype=float), tilt_deg


def load_detector_axes(report_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the saved pupil sampling axes from the optimization run."""
    response_path = report_dir / "response.npz"
    if not response_path.exists():
        raise FileNotFoundError(
            f"Missing response.npz next to {report_dir / 'optimization_report.json'}; "
            "the export needs the saved pupil axes to keep detector sampling "
            "consistent."
        )
    with np.load(response_path) as data:
        return np.asarray(data["pupil_x_mm"], dtype=float), np.asarray(
            data["pupil_y_mm"], dtype=float
        )


def detector_axis_pitch_mm(axis: np.ndarray, name: str) -> float:
    """Return the uniform cell pitch of a saved optimizer sampling axis."""
    axis = np.asarray(axis, dtype=float)
    if axis.ndim != 1 or axis.size < 2 or not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must be a finite one-dimensional axis with at least two samples.")
    steps = np.diff(axis)
    if (
        np.any(steps == 0.0)
        or not np.all(np.sign(steps) == np.sign(steps[0]))
        or not np.allclose(np.abs(steps), abs(float(steps[0])), rtol=1.0e-6, atol=1.0e-9)
    ):
        raise ValueError(f"{name} must use one uniform non-zero spacing.")
    return abs(float(steps[0]))


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


def build_zemax_pupil_report_specs(
    pupil: RectangularPupil,
    eyebox_detector: RectangularPupil,
    eyebox: dict,
    pupil_pitch_x_mm: float,
    pupil_pitch_y_mm: float,
) -> list[PupilReportSpec]:
    """Configure complementary Zemax reports for pupil and eyebox detectors.

    The larger pupil detector has enough margin to reproduce the optimizer's
    3x3-cell physical eye window at every saved eyebox position.  The strict
    eyebox detector has no such margin, so its report uses one non-overlapping
    eyebox cell per position.  This preserves all edge cells without silently
    clipping their integration windows at the detector boundary.
    """
    eyebox_width_mm = float(eyebox_detector.width)
    eyebox_height_mm = float(eyebox_detector.height)
    expected_width_mm = 2.0 * float(eyebox["half_x"])
    expected_height_mm = 2.0 * float(eyebox["half_y"])
    if not np.isclose(eyebox_width_mm, expected_width_mm):
        raise ValueError("Eyebox detector width does not match the saved optimizer eyebox.")
    if not np.isclose(eyebox_height_mm, expected_height_mm):
        raise ValueError("Eyebox detector height does not match the saved optimizer eyebox.")

    eye_positions_x = _positive_grid_count(eyebox["nx"], "eyebox nx")
    eye_positions_y = _positive_grid_count(eyebox["ny"], "eyebox ny")
    pupil_pitch_x_mm = _positive_length_mm(pupil_pitch_x_mm, "pupil x pitch")
    pupil_pitch_y_mm = _positive_length_mm(pupil_pitch_y_mm, "pupil y pitch")

    common = {
        "eyebox_width_mm": eyebox_width_mm,
        "eyebox_height_mm": eyebox_height_mm,
        "eye_positions_x": eye_positions_x,
        "eye_positions_y": eye_positions_y,
    }
    return [
        PupilReportSpec(
            detector_label=pupil.name,
            # Match the optimizer's moving physical eye-pupil window while
            # integrating the high-resolution detector by fractional overlap.
            eye_window_width_mm=EYE_PUPIL_WINDOW_CELLS * pupil_pitch_x_mm,
            eye_window_height_mm=EYE_PUPIL_WINDOW_CELLS * pupil_pitch_y_mm,
            report_name="zemax_pupil_report",
            **common,
        ),
        PupilReportSpec(
            detector_label=eyebox_detector.name,
            # The strict detector exactly tiles the configured eyebox.  A
            # single-cell window therefore includes every edge position and
            # measures only flux landing inside the eyebox.
            eye_window_width_mm=eyebox_width_mm / eye_positions_x,
            eye_window_height_mm=eyebox_height_mm / eye_positions_y,
            report_name="zemax_eyebox_report",
            export_angular_response=True,
            **common,
        ),
    ]


def _positive_grid_count(value, name):
    """Return one exact positive grid dimension from saved report metadata."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer.") from error
    if result <= 0 or result != value:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _positive_length_mm(value, name):
    """Return one finite positive length in millimetres."""
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite positive length.") from error
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive length.")
    return result


def build_ambient_detectors(system: OpticalSystem) -> list[RectangularPupil]:
    """Build four ambient detectors around the chassis+pupil bounding box."""
    offset_mm = 10.0
    chassis = next(
        element
        for element in system.elements
        if isinstance(element, GlassBlock) and element.name == "chassis"
    )
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
            position=np.array([center[0], mins[1] - offset_mm, center[2]], dtype=float),
            normal=np.array([0.0, -1.0, 0.0], dtype=float),
            width=span_x,
            height=span_z_with_margin,
        ),
        RectangularPupil(
            name="ambient_y_pos",
            position=np.array([center[0], maxs[1] + offset_mm, center[2]], dtype=float),
            normal=np.array([0.0, 1.0, 0.0], dtype=float),
            width=span_x,
            height=span_z_with_margin,
        ),
        RectangularPupil(
            name="ambient_z_neg",
            position=np.array([center[0], center[1], mins[2] - offset_mm], dtype=float),
            normal=np.array([0.0, 0.0, -1.0], dtype=float),
            width=span_x,
            height=span_y_with_margin,
        ),
        RectangularPupil(
            name="ambient_z_pos",
            position=np.array([center[0], center[1], maxs[2] + offset_mm], dtype=float),
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
    design = _read_json(coating_dir / "coating_design.json")
    source_report = str(design.get("source_report", "")).rstrip("/")
    if source_report != str(report_dir):
        candidates = sorted(COATING_RUNS_ROOT.glob("*/coating_design.json"))
        matching = []
        for path in candidates:
            try:
                candidate = _read_json(path)
            except json.JSONDecodeError:
                continue
            if str(candidate.get("source_report", "")).rstrip("/") == str(report_dir):
                matching.append(str(path.parent))
        suggestion = (
            f"Matching coating run(s) for {str(report_dir)!r}: {', '.join(matching)}."
            if matching
            else f"No coating_design.json under {COATING_RUNS_ROOT} was built from "
            f"{str(report_dir)!r}."
        )
        raise ValueError(
            f"Coating design in {coating_dir} was built from "
            f"{source_report!r}, but this export uses {str(report_dir)!r}. "
            f"{suggestion} Re-run examples/design_perseus_mirror_coating.py "
            "against the current optimizer output, or set COATING_DIR "
            "explicitly."
        )

    return {record["index"]: record["result"] for record in design["mirrors"]}


# ── Coatings: every rung, named so they can be swapped in OpticStudio ───────

IDEAL_COATING_REFERENCE_WAVELENGTH_NM = 550.0


def build_coatings(mirrors, coating_results):
    """Return coating blocks and names for all four rungs."""
    blocks = []
    names_by_mode: dict[str, dict[str, str]] = {"ideal": {}, "flat": {}, "atlas": {}, "stack": {}}
    notes = []

    for mirror_index, (wavelengths_nm, reflectance) in enumerate(mirrors):
        element_name = f"mirror_{mirror_index}"

        reference_reflectance = float(
            np.interp(IDEAL_COATING_REFERENCE_WAVELENGTH_NM, wavelengths_nm, reflectance)
        )
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
            notes.append(f"mirror_{mirror_index}: no Atlas coating, using {flat_name}")
            continue

        achieved = result["achieved_reflectance"]
        atlas_name = f"ATLAS_M{mirror_index}"
        blocks.append(
            table_coating(
                atlas_name,
                achieved["wavelengths_nm"],
                achieved["angles_deg"],
                np.asarray(achieved["values"]),
            )
        )
        names_by_mode["atlas"][element_name] = atlas_name

        stack_name = f"STACK_M{mirror_index}"
        blocks.append(
            stack_coating(
                stack_name,
                result["layers"],
                wavelength_range_nm=(
                    float(min(achieved["wavelengths_nm"])),
                    float(max(achieved["wavelengths_nm"])),
                ),
            )
        )
        names_by_mode["stack"][element_name] = stack_name

    return blocks, names_by_mode, notes


def build_chassis_ideal_coatings(chassis: GlassBlock):
    """Apply OpticStudio's native ideal AR coating to every chassis face.

    Each face uses the optimizer's reflectance at the common Ideal reference
    wavelength.  Faces with the same value share one native ``COAT I.xxx``
    entry in the bundle's active coating file, so it resolves in Zemax without
    custom ``IDEAL_CHASSIS_*`` records.
    """
    blocks_by_name = {}
    face_coatings = {}
    for face in chassis.faces:
        wavelengths_nm = np.asarray(face.coating_reflectance.wavelengths, dtype=float) / float(nm)
        reflectance = np.asarray(face.coating_reflectance.values, dtype=float)
        reference_reflectance = float(
            np.interp(IDEAL_COATING_REFERENCE_WAVELENGTH_NM, wavelengths_nm, reflectance)
        )
        block = zemax_builtin_ideal_coating(1.0 - reference_reflectance)
        blocks_by_name.setdefault(block.name, block)
        face_coatings[(chassis.name, face.name)] = block.name
    return list(blocks_by_name.values()), face_coatings


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
                f"choose one of {sorted(names_by_mode)} or None."
            ) from error
        for mirror_name, coating_name in mode_coatings.items():
            face_coatings[(mirror_name, face_name)] = coating_name
    return face_coatings


def build_zemax_coating_report(mirrors, coating_results, coating_names_by_mode,
                               face_coatings):
    """Describe each exported mirror stack for the Zemax-side HTML report.

    The report data deliberately follows the same resolved face map and stack
    names used to write the coating file. This lets the browser page answer
    both questions a reviewer needs: which coating is applied to a face, and
    which layer recipe belongs to its physical ``STACK_M*`` definition.
    """
    records = []
    for mirror_index in range(len(mirrors)):
        mirror_name = f"mirror_{mirror_index}"
        result = coating_results.get(mirror_index)
        assigned_faces = []
        for face_name in ("front", "back"):
            coating_name = face_coatings.get((mirror_name, face_name))
            if coating_name is not None:
                assigned_faces.append(
                    {"face": face_name, "coating_name": coating_name}
                )

        if result is None:
            stack_coating_name = None
            layers = []
        else:
            stack_coating_name = coating_names_by_mode["stack"][mirror_name]
            layers = [
                {
                    "material": layer["material"],
                    "thickness_nm": layer["thickness_nm"],
                }
                for layer in result["layers"]
            ]

        records.append(
            {
                "index": mirror_index,
                "name": mirror_name,
                "face_coatings": assigned_faces,
                "stack_coating_name": stack_coating_name,
                "layers": layers,
            }
        )

    return {
        "report_name": "zemax_coating_report",
        "title": "Perseus mirror coating stacks",
        "mirrors": records,
    }


# ── Sources: one per FOV direction ──────────────────────────────────────────


def _fov_source_label(angle_x_deg: float, angle_y_deg: float) -> str:
    """Return a filesystem-safe label without discarding configured precision."""

    def token(value):
        rounded = round(float(value), FOV_GRID_DECIMALS)
        if rounded == 0.0:
            rounded = 0.0
        text = f"{rounded:+.{FOV_GRID_DECIMALS}f}".rstrip("0").rstrip(".")
        return text.replace("+", "p").replace("-", "m").replace(".", "p")

    return f"fov_{token(angle_x_deg)}_{token(angle_y_deg)}"


def build_sources(projector: Projector, fov_x: float, fov_y: float) -> list[SourceSpec]:
    samples = build_export_fov_samples(
        projector.direction,
        fov_x,
        fov_y,
        NUM_FOV_X,
        NUM_FOV_Y,
        decimals=FOV_GRID_DECIMALS,
    )

    sources = []
    for direction_index, sample in enumerate(samples):
        # The float64 export grid uses the same basis convention as Projector.
        # Its roll decides how the 10 × 2 mm beam meets the 11 × 2 mm stop.
        direction = sample.direction
        angle_x_deg = sample.angle_x_deg
        angle_y_deg = sample.angle_y_deg
        sources.append(
            SourceSpec(
                label=_fov_source_label(angle_x_deg, angle_y_deg),
                position=np.asarray(projector.position, dtype=float),
                direction=direction,
                beam_width=float(projector.beam_width),
                beam_height=float(projector.beam_height),
                beam_local_x=sample.beam_local_x,
                beam_local_y=sample.beam_local_y,
                # Apollo14 traces one direction at a time. Zero marks every
                # non-first source as initially disabled; the Zemax builder
                # implements that state with zero analysis rays because some
                # ZOS-API releases reject a zero source Power.
                power=1.0 if direction_index == 0 else 0.0,
                analysis_rays=SOURCE_ANALYSIS_RAYS,
                fov_x_deg=float(angle_x_deg),
                fov_y_deg=float(angle_y_deg),
                sweep_power=1.0,
            )
        )
    return sources


# ── Export orchestration ────────────────────────────────────────────────────


def resolve_export_runs(
    configured_report_dir: Path | None,
    configured_coating_dir: Path | None,
) -> tuple[Path, Path | None]:
    """Choose a compatible optimizer run and optional coating-design run."""
    report_dir = configured_report_dir or latest_run(
        OPTIMIZE_RUNS_ROOT, "optimization_report.json"
    )
    report_dir_was_auto_selected = configured_report_dir is None
    try:
        if configured_coating_dir is not None:
            coating_dir = configured_coating_dir
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
                    f"falling back to coating source report {source_report}."
                )
                report_dir = Path(source_report)

    return report_dir, coating_dir


def prepare_detector_export(
    snapshot: OptimizerSnapshot,
    report_dir: Path,
) -> DetectorExportPlan:
    """Build detector objects, sampling sizes, and Zemax-derived reports."""
    system = snapshot.system
    pupil = next(element for element in system.elements if isinstance(element, RectangularPupil))
    eyebox_detector = build_eyebox_detector(system, snapshot.eyebox)
    ambient_detectors = tuple(build_ambient_detectors(system))
    pupil_x_mm, pupil_y_mm = load_detector_axes(report_dir)
    pupil_reports = tuple(
        build_zemax_pupil_report_specs(
            pupil,
            eyebox_detector,
            snapshot.eyebox,
            detector_axis_pitch_mm(pupil_x_mm, "pupil x axis"),
            detector_axis_pitch_mm(pupil_y_mm, "pupil y axis"),
        )
    )
    exported_detectors = (pupil, eyebox_detector, *ambient_detectors)
    return DetectorExportPlan(
        pupil=pupil,
        eyebox=eyebox_detector,
        ambient=ambient_detectors,
        reports=pupil_reports,
        overlap_report=EyeboxOverlapReportSpec(detector_label=eyebox_detector.name),
        pixels_by_name={
            detector.name: DETECTOR_SCAN_SHAPE for detector in exported_detectors
        },
    )


def build_zemax_aperture_report(
    detectors: DetectorExportPlan,
) -> ApertureSweepReportSpec:
    """Configure a compact vertical aperture tolerance sweep in OpticStudio."""
    return ApertureSweepReportSpec(
        aperture_label="aperture",
        detector_labels=(detectors.pupil.name, detectors.eyebox.name),
        opening_height_deltas_mm=APERTURE_HEIGHT_DELTAS_MM,
        vertical_shifts_mm=APERTURE_VERTICAL_SHIFTS_MM,
    )


def prepare_coating_export(
    chassis: GlassBlock,
    mirrors: tuple[tuple[np.ndarray, np.ndarray], ...],
    coating_results: dict[int, dict],
) -> CoatingExportPlan:
    """Build every coating definition, resolved face map, and report input."""
    blocks, names_by_mode, coating_notes = build_coatings(mirrors, coating_results)
    chassis_blocks, chassis_face_coatings = build_chassis_ideal_coatings(chassis)
    blocks.extend(chassis_blocks)
    face_coatings = select_mirror_face_coatings(
        names_by_mode,
        MIRROR_FACE_COATING_MODES,
    )
    face_coatings.update(chassis_face_coatings)
    return CoatingExportPlan(
        blocks=tuple(blocks),
        face_coatings=face_coatings,
        chassis_face_coatings=chassis_face_coatings,
        report=build_zemax_coating_report(
            mirrors,
            coating_results,
            names_by_mode,
            face_coatings,
        ),
        notes=tuple(coating_notes),
    )


def build_export_plan(report_dir: Path, coating_dir: Path | None) -> ZemaxExportPlan:
    """Load and validate all inputs before writing any output files."""
    snapshot = load_optimizer_report(report_dir)
    system = snapshot.system
    mirrors = tuple(mirror_samples(system))
    chassis = next(
        element
        for element in system.elements
        if isinstance(element, GlassBlock) and element.name == "chassis"
    )
    coating_results = load_coating_design(coating_dir, report_dir)
    chassis_origin, chassis_tilt_deg = chassis_pose(system)
    detectors = prepare_detector_export(snapshot, report_dir)
    return ZemaxExportPlan(
        report_dir=report_dir,
        coating_dir=coating_dir,
        snapshot=snapshot,
        chassis=chassis,
        mirrors=mirrors,
        coating_results=coating_results,
        detectors=detectors,
        aperture_report=build_zemax_aperture_report(detectors),
        coatings=prepare_coating_export(chassis, mirrors, coating_results),
        sources=tuple(build_sources(snapshot.projector, snapshot.fov_x, snapshot.fov_y)),
        chassis_origin=chassis_origin,
        chassis_tilt_deg=chassis_tilt_deg,
    )


def print_export_summary(plan: ZemaxExportPlan) -> None:
    """Print the resolved inputs and per-mirror coating choices."""
    snapshot = plan.snapshot
    coatings = plan.coatings

    print("── Perseus → Zemax OpticStudio ──")
    print(f"optimizer run : {plan.report_dir}")
    print(f"coating run   : {plan.coating_dir or '(none — flat R(λ) only)'}")
    print(
        f"mirrors       : {len(plan.mirrors)}  "
        f"spacings {np.round(snapshot.spacings, 4).tolist()}"
    )
    print(f"face coatings : {MIRROR_FACE_COATING_MODES}")
    print(f"chassis coats : {', '.join(coatings.chassis_face_coatings.values())}")
    print(
        f"mirror mode   : {FRONT_FACE_COATING_MODE}  "
        f"({len(plan.coating_results)} Atlas designs available)"
    )
    print(
        "ambient dets  : "
        + ", ".join(detector.name for detector in plan.detectors.ambient)
    )
    print(
        "aperture sweep: height "
        f"{list(plan.aperture_report.opening_height_deltas_mm)} mm × shift "
        f"{list(plan.aperture_report.vertical_shifts_mm)} mm (vertical only)"
    )
    for note in coatings.notes:
        print(f"  ! {note}")
    chassis_material_name = plan.chassis.material.name
    for mirror_index in range(len(plan.mirrors)):
        mirror_name = f"mirror_{mirror_index}"
        selected_coating = coatings.face_coatings.get((mirror_name, "front"))
        if mirror_index in plan.coating_results:
            status = "Atlas coating available"
        else:
            status = f"no Atlas coating, using {selected_coating}"
        print(f"  {mirror_name} | {chassis_material_name} | {selected_coating} | {status}")
    print(f"sources       : {len(plan.sources)} FOV directions ({NUM_FOV_X}×{NUM_FOV_Y})")


def build_bundle_notes(plan: ZemaxExportPlan) -> str:
    """Return provenance notes embedded in the generated bundle README."""
    return (
        f"Perseus combiner, {len(plan.mirrors)} mirrors, mirror face mode "
        f"`{FRONT_FACE_COATING_MODE}`.\n\n"
        f"- face coating modes: `{MIRROR_FACE_COATING_MODES}`\n"
        f"- chassis face coatings: `{plan.coatings.chassis_face_coatings}`\n"
        f"- detector pixels: `{plan.detectors.pixels_by_name}`\n"
        f"- eyebox overlap report: `{plan.detectors.overlap_report.report_name}`\n"
        f"- aperture height deltas (mm): `{plan.aperture_report.opening_height_deltas_mm}`\n"
        f"- aperture vertical shifts (mm): `{plan.aperture_report.vertical_shifts_mm}`\n"
        f"- optimizer run: `{plan.report_dir}`\n"
        f"- optimizer git SHA: `{plan.snapshot.git_sha}`\n"
        "- geometry source: serialized optimizer system snapshot\n"
        f"- coating run: `{plan.coating_dir or 'none'}`\n"
    )


def write_export_bundle(
    plan: ZemaxExportPlan,
    output_dir: Path | None = None,
) -> Path:
    """Serialize one prepared plan to a timestamped Zemax bundle."""
    if output_dir is None:
        output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    return export_zemax_bundle(
        output_dir,
        plan.snapshot.system,
        chassis_pivot=plan.chassis_origin,
        chassis_tilt_deg=plan.chassis_tilt_deg,
        sources=plan.sources,
        trace_wavelengths=[wavelength * nm for wavelength in TRACE_WAVELENGTHS_NM],
        glass_materials=[plan.chassis.material],
        coating_blocks=plan.coatings.blocks,
        face_coatings=plan.coatings.face_coatings,
        detector_pixels_by_name=plan.detectors.pixels_by_name,
        extra_pupils=plan.detectors.extra_detectors,
        pupil_reports=plan.detectors.reports,
        aperture_report=plan.aperture_report,
        coating_report=plan.coatings.report,
        eyebox_overlap_report=plan.detectors.overlap_report,
        min_relative_intensity=MIN_RELATIVE_INTENSITY,
        min_absolute_intensity=MIN_ABSOLUTE_INTENSITY,
        notes=build_bundle_notes(plan),
    )


def copy_optimizer_reference_report(report_dir: Path, bundle_dir: Path) -> Path:
    """Copy the matching Apollo14 pupil report beside the Zemax bundle."""
    apollo14_pupil_report = report_dir / "pupil_report.html"
    if not apollo14_pupil_report.exists():
        raise FileNotFoundError(
            f"The matching optimizer run has no eyebox report: {apollo14_pupil_report}"
        )
    destination = bundle_dir / "apollo14_eyebox_report.html"
    shutil.copy2(apollo14_pupil_report, destination)
    return destination


def print_export_result(bundle_dir: Path) -> None:
    """Print the created bundle and the next user action."""
    print(f"\nBundle: {bundle_dir}")
    for path in sorted(bundle_dir.iterdir()):
        print(f"  {path.name}")
    print("\nCopy to the OpticStudio machine and follow README.md there.")


def main():
    """Resolve inputs, prepare one export plan, and write the bundle."""
    report_dir, coating_dir = resolve_export_runs(REPORT_DIR, COATING_DIR)
    plan = build_export_plan(report_dir, coating_dir)
    print_export_summary(plan)
    bundle_dir = write_export_bundle(plan)
    copy_optimizer_reference_report(plan.report_dir, bundle_dir)
    print_export_result(bundle_dir)


if __name__ == "__main__":
    main()
