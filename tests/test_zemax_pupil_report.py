"""Tests for the portable Zemax-derived pupil/FOV report."""

import json

import pytest

from apollo14.export.zemax.reporting.html_plotting import REPORT_LAYOUT_VERSION
from apollo14.export.zemax.reporting.position_space_analysis import (
    PositionSpaceCube,
    SystemSpectrum,
)
from apollo14.export.zemax.reporting.zemax_pupil_report import (
    PupilReportSpec,
    calculate_pupil_report,
    normalize_pupil_report_specs,
    write_pupil_report,
)


def _source(index, x_deg=0.0, y_deg=0.0):
    return {
        "index": index,
        "label": f"field_{index}",
        "fov_angles_deg": [x_deg, y_deg],
        "data": {"sweep_power": 1.0},
    }


def _detector(pixels_x=2, pixels_y=2, half_x=1.0, half_y=1.0):
    return {
        "index": 7,
        "comment": "pupil",
        "data": {
            "pixels_x": pixels_x,
            "pixels_y": pixels_y,
            "half_width_x": half_x,
            "half_width_y": half_y,
        },
        "zemax_local_axes_world": {
            "x": [1.0, 0.0, 0.0],
            "y": [0.0, 1.0, 0.0],
        },
        "apollo14_local_axes_world": {
            "x": [1.0, 0.0, 0.0],
            "y": [0.0, 1.0, 0.0],
        },
    }


def _spec(**overrides):
    values = {
        "detector_label": "pupil",
        "eyebox_width_mm": 1.0,
        "eyebox_height_mm": 1.0,
        "eye_positions_x": 1,
        "eye_positions_y": 1,
        "eye_window_width_mm": 1.0,
        "eye_window_height_mm": 1.0,
    }
    values.update(overrides)
    return PupilReportSpec(**values)


def _cube(detector, sources, fields):
    cube = PositionSpaceCube(detector, sources, SystemSpectrum([0.55], [1.0]))
    for source, rows in zip(sources, fields, strict=True):
        cube.record(source["index"], rows)
    return cube


def test_spec_normalization_validates_names_and_duplicates():
    spec = _spec(report_name="pupil_report-1")
    assert normalize_pupil_report_specs([spec])[0]["detector_label"] == "pupil"

    with pytest.raises(ValueError, match="report_name"):
        _spec(report_name="has a space")
    with pytest.raises(ValueError, match="duplicated"):
        normalize_pupil_report_specs([spec, spec])
    with pytest.raises(ValueError, match="duplicated"):
        normalize_pupil_report_specs(
            [
                spec,
                _spec(report_name="PUPIL_REPORT-1"),
            ]
        )
    with pytest.raises(ValueError, match="reserved"):
        _spec(report_name="position_space_index")
    with pytest.raises(ValueError, match="export_angular_response"):
        _spec(export_angular_response=1)


def test_fractional_pixel_overlap_preserves_window_flux_density():
    sources = [_source(1)]
    cube = _cube(
        _detector(),
        sources,
        [[[1.0, 2.0], [3.0, 4.0]]],
    )

    report = calculate_pupil_report(cube, _spec())

    # The centered 1×1 mm window covers one quarter of each 1×1 mm pixel.
    assert report["arrays"]["cell_response"]["values"] == [[[[2.5]]]]
    assert report["quantity"]["window_area_mm2"] == 1.0
    assert report["provenance"]["apollo14_reference_values_used"] is False
    assert report["statistics"]["angular_integration_performed"] is False
    assert report["field_summary"][0]["source_sweep_power"] == 1.0


def test_report_reshapes_fields_and_writes_self_contained_artifacts(tmp_path):
    sources = [
        _source(1, 1.0, -2.0),
        _source(2, -1.0, 2.0),
        _source(3, -1.0, -2.0),
        _source(4, 1.0, 2.0),
    ]
    detector = _detector(pixels_x=4, pixels_y=4, half_x=2.0, half_y=2.0)
    fields = [[[float(index)] * 4 for _ in range(4)] for index in range(1, 5)]
    cube = _cube(detector, sources, fields)
    (tmp_path / "zemax_pupil_report.html").write_text("stale report")
    legacy_slices = tmp_path / "zemax_pupil_report_angular_response_slices"
    legacy_slices.mkdir()
    (legacy_slices / "stale.csv").write_text("stale slice")

    result = write_pupil_report(
        cube,
        _spec(
            eyebox_width_mm=2.0,
            eyebox_height_mm=2.0,
            eye_positions_x=2,
            eye_positions_y=2,
            eye_window_width_mm=2.0,
            eye_window_height_mm=2.0,
            export_angular_response=True,
        ),
        tmp_path,
    )

    report_directory = tmp_path / "reports" / "zemax_pupil_report"
    assert not (tmp_path / "zemax_pupil_report.html").exists()
    assert not legacy_slices.exists()
    payload = json.loads((report_directory / "report.json").read_text())
    assert payload["fov"]["x_deg"] == [-1.0, 1.0]
    assert payload["fov"]["y_deg"] == [-2.0, 2.0]
    assert payload["arrays"]["cell_response"]["values"][0][0] == [
        [3.0, 1.0],
        [2.0, 4.0],
    ]
    assert payload["display_scales"]["cell_response"] == {
        "minimum": 0.0,
        "maximum": 4.0,
        "step": 1.0,
        "ticks": [0.0, 1.0, 2.0, 3.0, 4.0],
    }
    assert payload["display_scales"]["cdf_cell_mean_flux_density"]["minimum"] == 0.0
    assert payload["angular_response"]["source_array_axes"] == [
        "eye_y",
        "eye_x",
        "fov_y",
        "fov_x",
    ]
    assert payload["angular_response"]["diagram_axes"] == ["fov_y", "fov_x"]
    assert payload["angular_response"]["trace_reused"] is True
    assert len(payload["angular_response"]["slices"]) == 4
    html = (report_directory / "index.html").read_text()
    assert "__REPORT_DATA__" not in html
    assert "apollo14_reference_values_used" in html
    assert "https://" not in html
    assert "<canvas" in html
    assert "function drawColorScale(" in html
    assert "function drawColorScale(context, canvasWidth, scale)" in html
    assert "drawColorScale(context, width, options.scale);" in html
    assert "canvasWidth - 9" not in html
    assert "scale.ticks.forEach" in html
    assert "source-power units/mm²" in html
    assert html.count('class="unit-label"') == 8
    assert "Color scale units: detector flux density in source-power units/mm²." in html
    assert "Common scale with the “Worst FOV” plot" in html
    assert "selection does not rescale the colors" in html
    assert "a curve farther right indicates stronger coverage" in html
    assert "ticks: [0, 0.25, 0.5, 0.75, 1]" in html
    assert payload["title"] == "Zemax pupil and FOV report — pupil detector"
    assert result["title"] == payload["title"]
    assert "document.getElementById('reportTitle').textContent = report.title" in html
    assert "<h2>Total detector flux by FOV</h2>" in html
    assert "configuration.eye_window_width_mm" in html
    assert "angularResponseLink" in html
    assert "FOV-angle slices" in html
    assert f'data-apollo14-report-layout="{REPORT_LAYOUT_VERSION}"' in html
    assert "__REPORT_LAYOUT_VERSION__" not in html
    assert (report_directory / "cell_summary.csv").is_file()
    assert (report_directory / "field_summary.csv").is_file()
    angular_html = report_directory / "angular_response.html"
    angular_csv = report_directory / "angular_response.csv"
    angular_slices = report_directory / "angular_response_slices"
    assert angular_html.is_file()
    assert angular_csv.is_file()
    assert len(list(angular_slices.glob("*.csv"))) == 4
    angular_csv_lines = angular_csv.read_text().splitlines()
    assert len(angular_csv_lines) == 17
    assert "source_object_index,source_label,source_sweep_power" in angular_csv_lines[0]
    assert (angular_slices / "eye_y_000_x_000.csv").read_text().splitlines() == [
        "fov_y_deg\\fov_x_deg,-1.0,1.0",
        "-2.0,3.0,1.0",
        "2.0,2.0,4.0",
    ]
    angular_page = angular_html.read_text()
    assert "__ANGULAR_RESPONSE_DATA__" not in angular_page
    assert "FOV-angle response by eyebox position" in angular_page
    assert "FOV x (deg)" in angular_page
    assert "FOV y (deg)" in angular_page
    assert "Each diagram uses the same" in angular_page
    assert f'data-apollo14-report-layout="{REPORT_LAYOUT_VERSION}"' in angular_page
    assert result["related_reports"][0]["html"] == str(angular_html.resolve())
    assert result["detector_label"] == "pupil"


def test_apollo14_axis_sign_is_applied_to_eye_positions():
    detector = _detector(pixels_x=2, pixels_y=1, half_x=1.0, half_y=0.5)
    detector["apollo14_local_axes_world"]["x"] = [-1.0, 0.0, 0.0]
    cube = _cube(detector, [_source(1)], [[[10.0, 20.0]]])
    report = calculate_pupil_report(
        cube,
        _spec(
            eyebox_width_mm=1.0,
            eye_positions_x=2,
            eye_window_width_mm=0.5,
            eye_window_height_mm=1.0,
        ),
    )

    assert report["coordinate_transform"]["matrix"] == [
        [-1.0, 0.0],
        [0.0, 1.0],
    ]
    assert report["arrays"]["pupil_mean_over_fov"] == [[20.0, 10.0]]


def test_report_rejects_nonrectangular_fov_and_windows_outside_detector():
    sources = [_source(1, 0.0, 0.0), _source(2, 1.0, 1.0)]
    cube = _cube(
        _detector(),
        sources,
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[2.0, 2.0], [2.0, 2.0]],
        ],
    )
    with pytest.raises(ValueError, match="complete rectangular FOV grid"):
        calculate_pupil_report(cube, _spec())

    one_field = _cube(_detector(), [_source(1)], [[[1.0, 1.0], [1.0, 1.0]]])
    with pytest.raises(ValueError, match="outside the detector interval"):
        calculate_pupil_report(
            one_field,
            _spec(eyebox_center_x_mm=0.9, eye_window_width_mm=1.0),
        )
