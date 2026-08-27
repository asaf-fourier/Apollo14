"""Turn an :class:`~apollo14.system.OpticalSystem` into a Zemax NSC prescription.

The prescription is a plain JSON document describing every non-sequential
object: its type, where it sits, how it is tilted, what it is made of and which
coating it carries. It is the exporter's testable artifact — all the geometry
that could silently be wrong lives here, is computed on this side, and is
checked by ``tests/test_zemax_export.py``. The generated ZOS-API script is then
a thin, dumb reader of this document.

Object order in the document is the order objects are inserted into the
non-sequential component editor, and ``inside_of`` references are 1-based
indices into that order — matching the editor's own numbering.
"""

from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np

from apollo14.elements.aperture import RectangularAperture
from apollo14.elements.glass_block import GlassBlock
from apollo14.elements.partial_mirror import PartialMirror
from apollo14.elements.pupil import RectangularPupil
from apollo14.export.placement import (
    half_extents_in_zemax_frame,
    planar_placement,
    rectangle_frame_to_tilts,
    zemax_plane_axes,
)
from apollo14.export.pob import (
    format_polygon_object,
    polygon_from_aperture,
    polygon_from_glass_block,
)
from apollo14.units import nm

# Ray-splitting defaults. A cascade of M partial mirrors branches into 2^M
# paths, so OpticStudio's stock limits (which assume a handful of splits) cull
# exactly the light this model exists to measure.
DEFAULT_MAX_SEGMENTS = 4000
DEFAULT_MAX_INTERSECTIONS = 400
DEFAULT_MIN_RELATIVE_INTENSITY = 1e-9


class SourceSpec(NamedTuple):
    """One collimated projector beam — Apollo14 traces one per FOV direction.

    ``beam_local_x``/``beam_local_y`` are the axes ``beam_width`` and
    ``beam_height`` are measured along, i.e. what
    ``Projector._compute_basis(direction)`` returns for this direction. They are
    required rather than re-derived: the beam is 10 × 2 mm, so its roll about
    the pointing direction decides how much of it clears the 11 × 2 mm stop, and
    a roll convention invented here instead of taken from the projector would
    quietly change the answer.
    """
    label: str
    position: np.ndarray        # (3,) world mm
    direction: np.ndarray       # (3,) unit
    beam_width: float           # mm, across beam_local_x
    beam_height: float          # mm, across beam_local_y
    beam_local_x: np.ndarray    # (3,) unit
    beam_local_y: np.ndarray    # (3,) unit
    power: float = 1.0
    layout_rays: int = 8
    analysis_rays: int = 200_000


@dataclass
class Prescription:
    """A complete Zemax model: the JSON document plus the files it references."""
    document: dict[str, Any]
    polygon_files: dict[str, str] = field(default_factory=dict)

    @property
    def objects(self) -> list[dict]:
        return self.document["objects"]

    def object_named(self, comment: str) -> dict:
        for entry in self.objects:
            if entry["comment"] == comment:
                return entry
        available = [entry["comment"] for entry in self.objects]
        raise KeyError(f"No exported object {comment!r}. Available: {available}")


def build_prescription(
    system,
    *,
    chassis_pivot,
    chassis_tilt_deg: float,
    sources: list[SourceSpec],
    trace_wavelengths,
    glass_names: dict[str, str],
    coating_names: dict[str, str] | None = None,
    face_coatings: dict[tuple[str, str], str] | None = None,
    detector_pixels: tuple[int, int] = (140, 180),
    coating_file: str | None = None,
    glass_catalog: str | None = None,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
    max_intersections: int = DEFAULT_MAX_INTERSECTIONS,
    min_relative_intensity: float = DEFAULT_MIN_RELATIVE_INTENSITY,
) -> Prescription:
    """Build the prescription for ``system``.

    Args:
        system: the Apollo14 optical system to export.
        chassis_pivot: world point the chassis rotation was applied about
            (the combiner centre for Perseus).
        chassis_tilt_deg: that rotation, in degrees about world x, right-handed
            — for Perseus, minus the pantoscopic tilt. It is undone to recover
            object-local chassis vertices and then re-applied by the NSC row.
        sources: one entry per beam to emit; Apollo14 traces FOV directions one
            at a time, so all but the first are exported with zero power.
        trace_wavelengths: wavelengths to register in the system, in Apollo14's
            internal units.
        glass_names: Apollo14 material name → Zemax catalog glass name.
        coating_names: element name → coating name, for the partial mirrors.
        face_coatings: ``(block name, face name)`` → coating name, for putting
            an AR coating on the chassis faces the beam crosses.
        detector_pixels: ``(x, y)`` pixel counts for the pupil detector.

    Returns:
        A :class:`Prescription`. Polygon object files it references are in
        ``polygon_files``, keyed by file name.
    """
    coating_names = coating_names or {}
    face_coatings = face_coatings or {}

    objects: list[dict] = []
    polygon_files: dict[str, str] = {}
    block_indices: dict[str, int] = {}

    for element in system.elements:
        if isinstance(element, GlassBlock):
            entry, polygon_text, file_name = _export_glass_block(
                element, chassis_pivot, chassis_tilt_deg, glass_names,
                face_coatings)
            polygon_files[file_name] = polygon_text
            block_indices[element.name] = len(objects) + 1
        elif isinstance(element, RectangularAperture):
            entry, polygon_text, file_name = _export_aperture(element)
            polygon_files[file_name] = polygon_text
        elif isinstance(element, PartialMirror):
            entry = _export_partial_mirror(element, coating_names)
        elif isinstance(element, RectangularPupil):
            entry = _export_pupil(element, detector_pixels)
        else:
            raise TypeError(
                f"No Zemax export for element type {type(element).__name__!r} "
                f"({getattr(element, 'name', '?')}).")

        entry["index"] = len(objects) + 1
        objects.append(entry)

    _assign_mirrors_inside_chassis(system, objects, block_indices)

    for source in sources:
        entry = _export_source(source)
        entry["index"] = len(objects) + 1
        objects.append(entry)

    wavelengths_um = [
        float(wavelength) / (1000.0 * nm)
        for wavelength in np.atleast_1d(np.asarray(trace_wavelengths))
    ]

    document = {
        "units": {"lens": "mm", "wavelength": "um"},
        "system": {
            "mode": "non_sequential",
            "wavelengths_um": wavelengths_um,
            "ray_splitting": True,
            "max_segments": int(max_segments),
            "max_intersections": int(max_intersections),
            "min_relative_intensity": float(min_relative_intensity),
            "glass_catalog": glass_catalog,
            "coating_file": coating_file,
        },
        "objects": objects,
    }
    return Prescription(document=document, polygon_files=polygon_files)


def _export_glass_block(block, pivot, tilt_deg, glass_names, face_coatings):
    polygon, placement = polygon_from_glass_block(block, pivot, tilt_deg)
    file_name = f"{block.name}.POB"
    polygon_text = format_polygon_object(
        polygon, f"Apollo14 {block.name} — {block.material.name}")

    material_name = glass_names.get(block.material.name)
    if material_name is None:
        raise KeyError(
            f"No Zemax glass name given for material "
            f"{block.material.name!r}; pass one via glass_names.")

    # Face numbers in the polygon file follow the block's face order.
    coatings_by_face_number = {}
    for face_number, face in enumerate(block.faces, start=1):
        coating = face_coatings.get((block.name, face.name))
        if coating is not None:
            coatings_by_face_number[str(face_number)] = coating

    entry = {
        "type": "polygon",
        "comment": block.name,
        "position": [float(v) for v in placement.position],
        "tilt_deg": [float(v) for v in placement.tilt_deg],
        "material": material_name,
        "inside_of": 0,
        "data": {"polygon_file": file_name},
        "face_coatings": coatings_by_face_number,
    }
    return entry, polygon_text, file_name


def _export_aperture(aperture):
    polygon, placement = polygon_from_aperture(aperture)
    file_name = f"{aperture.name}.POB"
    polygon_text = format_polygon_object(
        polygon, f"Apollo14 {aperture.name} — absorbing beam stop")

    entry = {
        "type": "polygon",
        "comment": aperture.name,
        "position": [float(v) for v in placement.position],
        "tilt_deg": [float(v) for v in placement.tilt_deg],
        "material": "",
        "inside_of": 0,
        "data": {"polygon_file": file_name},
        "face_coatings": {},
    }
    return entry, polygon_text, file_name


def _export_partial_mirror(mirror, coating_names):
    placement = planar_placement(mirror, mirror.name)
    half_x, half_y = half_extents_in_zemax_frame(mirror, mirror.name)

    entry = {
        "type": "rectangle",
        "comment": mirror.name,
        "position": [float(v) for v in placement.position],
        "tilt_deg": [float(v) for v in placement.tilt_deg],
        # Blank material: the rectangle is a coated interface inside the glass,
        # not a medium change. Ray splitting at it is governed by the coating.
        "material": "",
        "inside_of": 0,          # filled in once the chassis index is known
        "data": {"half_width_x": half_x, "half_width_y": half_y},
        "coating": coating_names.get(mirror.name, ""),
        "reference_reflectance": {
            "wavelengths_nm": [float(w) / nm for w in mirror.wavelengths],
            "values": [float(r) for r in mirror.reflectance],
        },
    }
    return entry


def _export_pupil(pupil, detector_pixels):
    placement = planar_placement(pupil, pupil.name)
    half_x, half_y = half_extents_in_zemax_frame(pupil, pupil.name)
    pixels_x, pixels_y = detector_pixels

    entry = {
        "type": "detector_rectangle",
        "comment": pupil.name,
        "position": [float(v) for v in placement.position],
        "tilt_deg": [float(v) for v in placement.tilt_deg],
        "material": "",
        "inside_of": 0,
        "data": {
            "half_width_x": half_x,
            "half_width_y": half_y,
            "pixels_x": int(pixels_x),
            "pixels_y": int(pixels_y),
        },
        **_frame_record(pupil),
    }
    return entry


def _frame_record(element) -> dict:
    """Record both frames a planar element has, so readers can align them.

    The exported frame is chosen for an unambiguous single tilt, which for the
    pupil (normal ``[0, 0, -1]``) comes out rotated 180° from the frame
    ``compute_local_axes`` gives Apollo14 — both in-plane axes flip. A detector
    CSV compared against an Apollo14 pupil map without accounting for that would
    be silently upside down and backwards, so both frames travel with the
    object.
    """
    export_x, export_y = zemax_plane_axes(element.normal, element.name)
    return {
        "zemax_local_axes_world": {
            "x": [float(v) for v in export_x],
            "y": [float(v) for v in export_y],
        },
        "apollo14_local_axes_world": {
            "x": [float(v) for v in np.asarray(element._local_x, dtype=float)],
            "y": [float(v) for v in np.asarray(element._local_y, dtype=float)],
        },
    }


def _export_source(source: SourceSpec):
    direction = np.asarray(source.direction, dtype=float)
    direction = direction / np.linalg.norm(direction)

    # Match the projector's whole frame, not just where it points — see
    # SourceSpec on why the beam's roll matters here.
    tilt_x, tilt_y, tilt_z = rectangle_frame_to_tilts(
        source.beam_local_x, source.beam_local_y, direction, source.label)

    entry = {
        "type": "source_two_angle",
        "comment": (f"{source.label} dir="
                    f"({direction[0]:.6f},{direction[1]:.6f},"
                    f"{direction[2]:.6f})"),
        "label": source.label,
        "position": [float(v) for v in np.asarray(source.position, dtype=float)],
        "tilt_deg": [float(tilt_x), float(tilt_y), float(tilt_z)],
        "material": "",
        "inside_of": 0,
        "data": {
            "half_width_x": float(source.beam_width) / 2.0,
            "half_width_y": float(source.beam_height) / 2.0,
            # Zero half angles make the beam collimated, matching Apollo14's
            # projector, which emits a grid of parallel rays.
            "half_angle_x": 0.0,
            "half_angle_y": 0.0,
            "power": float(source.power),
            "layout_rays": int(source.layout_rays),
            "analysis_rays": int(source.analysis_rays),
        },
        "intended_direction": [float(v) for v in direction],
    }
    return entry


def _assign_mirrors_inside_chassis(system, objects, block_indices) -> None:
    """Point every partial mirror at the glass block that encloses it.

    A zero-thickness rectangle inside a refractive volume has to declare which
    volume it is inside, or OpticStudio treats both of its sides as air and the
    mirror picks up refraction it should not have.
    """
    if not block_indices:
        return
    if len(block_indices) > 1:
        raise NotImplementedError(
            f"Export supports one glass block; found {sorted(block_indices)}. "
            "Enclosing-block selection would need per-mirror containment tests.")

    (chassis_index,) = block_indices.values()
    mirror_names = {element.name for element in system.elements
                    if isinstance(element, PartialMirror)}
    for entry in objects:
        if entry["comment"] in mirror_names:
            entry["inside_of"] = chassis_index
