"""Regression checks for the Perseus Zemax export example."""

import jax.numpy as jnp
import numpy as np

from examples import export_perseus_zemax
from apollo14.elements.pupil import RectangularPupil
from apollo14.elements.glass_block import GlassBlock
from apollo14.materials import agc_m074
from apollo14.export.bundle import zemax_glass_name
from apollo14.export.bundle import (
    _apply_detector_pixel_overrides,
    _clone_system_with_extra_pupils,
)
from apollo14.export.prescription import build_prescription
from apollo14.export.zosapi_script import sweep_script_text
from apollo14.perseus import (
    PERSEUS_COMBINER_CENTER,
    PERSEUS_PANTOSCOPIC_TILT,
)
from apollo14.units import deg, nm
from helios.combiner_params import CombinerParams
from helios.perseus_params import build_parametrized_perseus
from apollo14.export.bundle import _display_coating


def test_perseus_export_defaults_to_flat_coatings():
    assert export_perseus_zemax.FRONT_FACE_COATING_MODE == "flat"


def test_face_coating_modes_resolve_to_mirror_face_names():
    face_coatings = export_perseus_zemax.select_mirror_face_coatings(
        {
            "ideal": {"mirror_0": "IDEAL_M0"},
            "flat": {"mirror_0": "FLAT_M0"},
            "atlas": {"mirror_0": "ATLAS_M0"},
            "stack": {"mirror_0": "STACK_M0"},
        },
        {"front": "flat", "back": "ideal"},
    )

    assert face_coatings == {
        ("mirror_0", "front"): "FLAT_M0",
        ("mirror_0", "back"): "IDEAL_M0",
    }


def test_bundle_summary_shows_face_coating_when_object_coating_is_absent():
    assert _display_coating({
        "coating": None,
        "face_coatings": {"1": "FLAT_M0"},
    }) == "front: FLAT_M0"


def test_export_bundle_helpers_can_add_and_relabel_secondary_detector():
    system = build_parametrized_perseus(
        CombinerParams.initial(),
        probe_wavelengths=jnp.array([550.0]) * nm,
    )
    pupil = next(e for e in system.elements if isinstance(e, RectangularPupil))
    eyebox = RectangularPupil(
        name="eyebox",
        position=pupil.position,
        normal=pupil.normal,
        width=8.0,
        height=8.0,
    )

    cloned = _clone_system_with_extra_pupils(system, [eyebox])
    prescription = build_prescription(
        cloned,
        chassis_pivot=PERSEUS_COMBINER_CENTER,
        chassis_tilt_deg=float(-PERSEUS_PANTOSCOPIC_TILT / deg),
        sources=[],
        trace_wavelengths=jnp.array([550.0]) * nm,
        glass_names={agc_m074.name: zemax_glass_name(agc_m074.name)},
        detector_pixels=(140, 180),
    )
    _apply_detector_pixel_overrides(
        prescription, {"pupil": (140, 180), "eyebox": (80, 80)})

    detectors = [entry for entry in prescription.objects
                 if entry["type"] == "detector_rectangle"]
    assert [entry["comment"] for entry in detectors] == ["pupil", "eyebox"]
    assert detectors[0]["data"]["pixels_x"] == 140
    assert detectors[0]["data"]["pixels_y"] == 180
    assert detectors[1]["data"]["pixels_x"] == 80
    assert detectors[1]["data"]["pixels_y"] == 80


def test_ambient_detectors_bound_the_chassis_and_pupil_box():
    system = build_parametrized_perseus(
        CombinerParams.initial(),
        probe_wavelengths=jnp.array([550.0]) * nm,
    )
    ambient = export_perseus_zemax.build_ambient_detectors(system)

    assert [detector.name for detector in ambient] == [
        "ambient_y_neg",
        "ambient_y_pos",
        "ambient_z_neg",
        "ambient_z_pos",
    ]

    chassis = next(
        element for element in system.elements
        if isinstance(element, GlassBlock) and element.name == "chassis")
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
    span_x = maxs[0] - mins[0]
    span_y = maxs[1] - mins[1]
    span_z = maxs[2] - mins[2]

    assert np.allclose(ambient[0].position[1], mins[1] - 10.0)
    assert np.allclose(ambient[1].position[1], maxs[1] + 10.0)
    assert np.allclose(ambient[2].position[2], mins[2] - 10.0)
    assert np.allclose(ambient[3].position[2], maxs[2] + 10.0)
    assert np.allclose(ambient[0].width, span_x)
    assert np.allclose(ambient[0].height, span_z + 20.0)
    assert np.allclose(ambient[2].width, span_x)
    assert np.allclose(ambient[2].height, span_y + 20.0)


def test_partial_mirrors_inherit_the_chassis_material_name():
    system = build_parametrized_perseus(
        CombinerParams.initial(),
        probe_wavelengths=jnp.array([550.0]) * nm,
    )
    prescription = build_prescription(
        system,
        chassis_pivot=PERSEUS_COMBINER_CENTER,
        chassis_tilt_deg=float(-PERSEUS_PANTOSCOPIC_TILT / deg),
        sources=[],
        trace_wavelengths=jnp.array([550.0]) * nm,
        glass_names={agc_m074.name: zemax_glass_name(agc_m074.name)},
    )

    mirror_materials = [
        entry["material"]
        for entry in prescription.objects
        if entry["type"] == "rectangular_volume" and entry["comment"].startswith("mirror_")
    ]
    assert mirror_materials
    assert set(mirror_materials) == {zemax_glass_name(agc_m074.name)}


def test_generated_sweep_script_handles_multiple_detectors():
    text = sweep_script_text()
    assert "detectors = [entry for entry in prescription[\"objects\"]" in text
    assert "for detector in detectors:" in text
    assert "_run_source_sweep(system, sources, detector, output_directory)" in text
