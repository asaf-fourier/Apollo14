"""Regression checks for the Perseus Zemax export example."""

import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from apollo14.elements.aperture import RectangularAperture
from apollo14.elements.glass_block import GlassBlock
from apollo14.elements.pupil import RectangularPupil
from apollo14.export.bundle import (
    _apply_detector_pixel_overrides,
    _clone_system_with_extra_pupils,
    _display_coating,
    _readme_header,
    _readme_objects,
    zemax_glass_name,
)
from apollo14.export.placement import half_extents_in_zemax_frame, planar_placement
from apollo14.export.prescription import build_prescription
from apollo14.export.zosapi_script import build_script_text, sweep_script_text
from apollo14.materials import agc_m074
from apollo14.perseus import (
    PERSEUS_COMBINER_CENTER,
    PERSEUS_PANTOSCOPIC_TILT,
)
from apollo14.projector import Projector
from apollo14.system import OpticalSystem
from apollo14.units import deg, nm
from examples import export_perseus_zemax
from helios.combiner_params import CombinerParams
from helios.io import _serialize_projector, _serialize_system
from helios.perseus_params import build_parametrized_perseus


def test_perseus_export_defaults_to_atlas_coatings():
    assert export_perseus_zemax.FRONT_FACE_COATING_MODE == "atlas"
    assert export_perseus_zemax.BACK_FACE_COATING_MODE is None


def test_face_coating_modes_resolve_to_mirror_face_names():
    face_coatings = export_perseus_zemax.select_mirror_face_coatings(
        {
            "ideal": {"mirror_0": "IDEAL_M0"},
            "flat": {"mirror_0": "FLAT_M0"},
            "atlas": {"mirror_0": "ATLAS_M0"},
            "stack": {"mirror_0": "STACK_M0"},
        },
        {"front": "atlas", "back": None},
    )

    assert face_coatings == {
        ("mirror_0", "front"): "ATLAS_M0",
    }


def test_bundle_summary_shows_face_coating_when_object_coating_is_absent():
    assert _display_coating({
        "coating": None,
        "face_coatings": {"1": "ATLAS_M0"},
    }) == "front: ATLAS_M0"


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


def test_chassis_all_faces_get_ideal_export_coatings():
    system = build_parametrized_perseus(
        CombinerParams.initial(),
        probe_wavelengths=jnp.array([550.0]) * nm,
    )
    chassis = next(
        element for element in system.elements
        if isinstance(element, GlassBlock) and element.name == "chassis")
    coating_blocks, face_coatings = export_perseus_zemax.build_chassis_ideal_coatings(
        chassis)
    prescription = build_prescription(
        system,
        chassis_pivot=PERSEUS_COMBINER_CENTER,
        chassis_tilt_deg=float(-PERSEUS_PANTOSCOPIC_TILT / deg),
        sources=[],
        trace_wavelengths=jnp.array([550.0]) * nm,
        glass_names={agc_m074.name: zemax_glass_name(agc_m074.name)},
        face_coatings=face_coatings,
    )

    expected_name = "I.995"
    assert [(block.name, block.definition) for block in coating_blocks] == [
        (expected_name, f"COAT {expected_name}")]
    assert face_coatings == {
        ("chassis", face.name): expected_name
        for face in chassis.faces
    }
    assert prescription.object_named("chassis")["face_coatings"] == {
        str(face_number): expected_name
        for face_number, face in enumerate(chassis.faces, start=1)
    }


def test_aperture_exports_as_native_boolean_outer_minus_inner():
    aperture = RectangularAperture(
        name="aperture",
        position=jnp.array([7.0, 35.159, 0.187]),
        normal=jnp.array([0.0, -0.9981, -0.0624]),
        width=14.0,
        height=6.0,
        inner_width=11.0,
        inner_height=2.0,
    )
    prescription = build_prescription(
        OpticalSystem(elements=[aperture]),
        chassis_pivot=jnp.zeros(3),
        chassis_tilt_deg=0.0,
        sources=[],
        trace_wavelengths=jnp.array([550.0]) * nm,
        glass_names={},
    )

    outer = prescription.object_named("aperture outer (A)")
    inner = prescription.object_named("aperture inner (B)")
    boolean = prescription.object_named("aperture")
    expected_outer = half_extents_in_zemax_frame(aperture, "outer")
    opening = SimpleNamespace(
        normal=aperture.normal,
        width=aperture.inner_width,
        height=aperture.inner_height,
        _local_x=aperture._local_x,
    )
    expected_inner = half_extents_in_zemax_frame(opening, "opening")
    expected_placement = planar_placement(aperture, aperture.name)

    assert [outer["index"], inner["index"], boolean["index"]] == [1, 2, 3]
    assert outer["type"] == inner["type"] == "rectangular_volume"
    assert boolean["type"] == "boolean_native"
    assert boolean["comment"] == "A-B"
    assert boolean["material"] == "ABSORB"
    assert boolean["data"] == {"object_a": outer["index"],
                               "object_b": inner["index"]}
    assert outer["position"] == inner["position"] == [0.0, 0.0, 0.0]
    assert outer["tilt_deg"] == inner["tilt_deg"] == [0.0, 0.0, 0.0]
    assert outer["ignore_rays"] is inner["ignore_rays"] is True
    assert outer["do_not_draw"] is inner["do_not_draw"] is True
    assert np.allclose(
        [outer["data"]["x1_half_width"], outer["data"]["y1_half_width"]],
        expected_outer)
    assert np.allclose(
        [inner["data"]["x1_half_width"], inner["data"]["y1_half_width"]],
        expected_inner)
    assert outer["data"]["z_length"] == inner["data"]["z_length"] > 0.0
    assert np.allclose(boolean["position"], expected_placement.position)
    assert np.allclose(boolean["tilt_deg"], expected_placement.tilt_deg)
    assert "aperture.POB" not in prescription.polygon_files
    assert "`aperture` aperture is the `A-B` Boolean Native row" in _readme_objects(
        prescription)


def test_generated_build_script_supports_boolean_native_apertures():
    text = build_script_text()

    assert '"boolean_native": ["BooleanNative", "Boolean Native"]' in text
    assert 'nce_types.RaysIgnoreObjectType, ["Always"]' in text
    assert '["RaysIgnoreObject"]' in text
    assert '["DoNotDrawObject"]' in text
    assert '["ObjectA"]' in text
    assert '["ObjectB"]' in text


def test_generated_sweep_script_handles_multiple_detectors():
    text = sweep_script_text()
    assert "detectors = [entry for entry in prescription[\"objects\"]" in text
    assert "for detector in detectors:" in text
    assert "_run_source_sweep(system, sources, detector, output_directory)" in text


def test_generated_build_script_sets_layout_checkboxes_tolerantly():
    text = build_script_text()
    assert "RAY_TRACE_SPLIT_ATTRIBUTE_CANDIDATES" in text
    assert "NSC3D_LAYOUT_BOOLEAN_ATTRIBUTE_CANDIDATES" in text
    assert '_log_startup("Beginning OpticStudio connection")' in text
    assert "UsePolarization" in text


def test_generated_build_script_logs_opticstudio_startup_attempts():
    text = build_script_text()
    assert "Beginning OpticStudio connection" in text
    assert "Trying CreateNewApplication() for a standalone OpticStudio" in text
    assert "CreateNewApplication() succeeded" in text
    assert "persistent NSC defaults" in text
    assert "FletchRays" in text
    assert "UsePolarization" in text


def test_bundle_readme_mentions_manual_gui_start_for_notauthorized_sessions():
    text = _readme_header("")
    assert "starts a standalone OpticStudio session" in text
    assert "will not close" in text
    assert "a session it did not start" in text


def test_generated_build_script_dumps_layout_inventory_on_missing_settings():
    text = build_script_text()
    assert "HasAnalysisSpecificSettings" in text
    assert "NSC 3D Layout analysis" in text
    assert "NSC 3D Layout settings" in text
    assert "ModifySettings" in text
    assert "USEPOLARIZATION" in text


def test_export_uses_saved_chassis_geometry_for_plain_pob(tmp_path):
    params = CombinerParams.initial()
    system = build_parametrized_perseus(
        params,
        probe_wavelengths=jnp.array([500.0, 600.0]) * nm,
        projector_glass_length=9.0,
    )
    projector = Projector.uniform(
        position=jnp.array([7.0, 40.0, -1.0]),
        direction=jnp.array([0.0, -1.0, 0.2]),
        beam_width=9.0,
        beam_height=1.5,
        nx=3,
        ny=2,
    )
    report = {
        "git_sha": "geometry-snapshot",
        "system": _serialize_system(system),
        "projectors": [_serialize_projector(projector)],
        "fov_grid": {"x_fov": 0.2, "y_fov": 0.3},
        "final_params": {"spacings": np.asarray(params.spacings).tolist()},
        "eyebox": {"half_x": 4.0, "half_y": 4.0, "nx": 8, "ny": 8},
    }
    (tmp_path / "optimization_report.json").write_text(json.dumps(report))

    snapshot = export_perseus_zemax.load_optimizer_report(tmp_path)
    pivot, tilt_deg = export_perseus_zemax.chassis_pose(snapshot.system)
    prescription = build_prescription(
        snapshot.system,
        chassis_pivot=pivot,
        chassis_tilt_deg=tilt_deg,
        sources=[],
        trace_wavelengths=jnp.array([550.0]) * nm,
        glass_names={agc_m074.name: zemax_glass_name(agc_m074.name)},
    )

    assert prescription.object_named("chassis")["data"]["polygon_file"] == (
        "chassis.POB")
    vertices = np.asarray([
        [float(value) for value in line.split()[2:5]]
        for line in prescription.polygon_files["chassis.POB"].splitlines()
        if line.startswith("V ")
    ])
    assert np.isclose(np.ptp(vertices[:4, 1]), 21.0, atol=1e-3)
    assert np.allclose(snapshot.projector.position, projector.position, atol=1e-4)
    assert snapshot.fov_x == 0.2
    assert snapshot.fov_y == 0.3


def test_generated_builder_installs_bundle_pobs_before_creating_model():
    text = build_script_text()
    install_call = (
        "    _install_polygon_files(\n"
        "        prescription_directory, prescription[\"objects\"], application)")

    assert install_call in text
    assert text.index(install_call) < text.index("    system.New(False)")
    assert "filecmp.cmp(source_path, target_path, shallow=False)" in text
    assert "APOLLO14_ZEMAX_POLYGON_OBJECTS_DIR" in text
