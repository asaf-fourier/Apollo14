"""Regression checks for the Perseus Zemax export example."""

import jax.numpy as jnp

from examples import export_perseus_zemax
from apollo14.elements.pupil import RectangularPupil
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


def test_generated_sweep_script_handles_multiple_detectors():
    text = sweep_script_text()
    assert "detectors = [entry for entry in prescription[\"objects\"]" in text
    assert "for detector in detectors:" in text
    assert "_run_source_sweep(system, sources, detector, output_directory)" in text
