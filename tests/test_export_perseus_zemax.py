"""Regression checks for the Perseus Zemax export example."""

from examples import export_perseus_zemax
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
