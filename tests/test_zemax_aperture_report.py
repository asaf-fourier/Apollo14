"""Tests for the Zemax-side aperture tolerance report."""

import csv
import json
import sys
import types

import pytest

import apollo14.export.zemax.reporting.position_space_analysis as position_space_analysis
import apollo14.export.zemax.reporting.zemax_aperture_report as zemax_aperture_report
import apollo14.export.zemax.reporting.zemax_coating_report as zemax_coating_report
import apollo14.export.zemax.reporting.zemax_pupil_report as zemax_pupil_report
from apollo14.export.zemax.reporting.position_space_analysis import (
    PositionSpaceCube,
    SystemSpectrum,
    write_position_space_index,
)
from apollo14.export.zemax.reporting.zemax_aperture_report import (
    ApertureSweepReport,
    ApertureSweepReportSpec,
    resolve_aperture_report,
)
from apollo14.export.zemax.templates.zosapi_script import sweep_script_text
from tests.zosapi_test_helpers import generated_builder_module


def _aperture_objects():
    common = {
        "type": "rectangular_volume",
        "position": [0.0, 0.0, 0.0],
        "tilt_deg": [0.0, 0.0, 0.0],
    }
    return [
        {
            **common,
            "index": 1,
            "comment": "aperture outer (A)",
            "data": {
                "x1_half_width": 7.0,
                "x2_half_width": 7.0,
                "y1_half_width": 3.0,
                "y2_half_width": 3.0,
            },
        },
        {
            **common,
            "index": 2,
            "comment": "aperture inner (B)",
            "data": {
                "x1_half_width": 6.0,
                "x2_half_width": 6.0,
                "y1_half_width": 1.5,
                "y2_half_width": 1.5,
            },
        },
        {
            "index": 3,
            "type": "boolean_native",
            "comment": "A-B",
            "label": "aperture",
            "position": [7.0, 35.0, 0.2],
            "data": {"object_a": 1, "object_b": 2},
            "zemax_local_axes_world": {
                "x": [1.0, 0.0, 0.0],
                "y": [0.0, 0.6, 0.8],
            },
        },
    ]


def _sources():
    return [
        {
            "index": index + 10,
            "label": f"field_{index}",
            "fov_angles_deg": [x_value, y_value],
            "data": {"sweep_power": 1.0},
        }
        for index, (x_value, y_value) in enumerate(
            ((-1.0, -0.5), (1.0, -0.5), (-1.0, 0.5), (1.0, 0.5))
        )
    ]


def _detector(index, label):
    return {
        "index": index,
        "comment": label,
        "data": {
            "pixels_x": 2,
            "pixels_y": 2,
            "half_width_x": 1.0,
            "half_width_y": 1.0,
        },
    }


def _pupil_spec(label, report_name):
    return {
        "detector_label": label,
        "eyebox_width_mm": 2.0,
        "eyebox_height_mm": 2.0,
        "eye_positions_x": 2,
        "eye_positions_y": 2,
        "eye_window_width_mm": 1.0,
        "eye_window_height_mm": 1.0,
        "report_name": report_name,
        "coordinate_frame": "zemax",
    }


def _completed_cube(detector, sources, base_value):
    cube = PositionSpaceCube(detector, sources, SystemSpectrum([0.55]))
    for field_index, source in enumerate(sources):
        value = base_value + field_index
        cube.record(source["index"], [[value, value + 0.1], [value + 0.2, value + 0.3]])
    return cube


def test_aperture_report_resolves_boolean_geometry_and_rejects_clipping():
    resolved = resolve_aperture_report(ApertureSweepReportSpec(), _aperture_objects())

    assert resolved["opening_height_deltas_mm"] == [-0.1, 0.0, 0.1]
    assert resolved["vertical_shifts_mm"] == [-0.1, 0.0, 0.1]
    assert resolved["geometry"]["outer_object_index"] == 1
    assert resolved["geometry"]["inner_object_index"] == 2
    assert resolved["geometry"]["nominal_opening_width_mm"] == 12.0
    assert resolved["geometry"]["nominal_opening_height_mm"] == 3.0
    assert resolved["geometry"]["nominal_aperture_position_mm"] == [7.0, 35.0, 0.2]
    assert resolved["geometry"]["local_vertical_world"] == [0.0, 0.6, 0.8]

    with pytest.raises(ValueError, match="outside the outer frame"):
        resolve_aperture_report(
            ApertureSweepReportSpec(opening_height_deltas_mm=(0.0, 3.1)),
            _aperture_objects(),
        )


def test_aperture_report_normalizes_singleton_variation_axes():
    specification = ApertureSweepReportSpec(
        opening_height_deltas_mm=(0.0),
        vertical_shifts_mm=0.0,
    )

    assert specification.opening_height_deltas_mm == (0.0,)
    assert specification.vertical_shifts_mm == (0.0,)
    resolved = resolve_aperture_report(specification, _aperture_objects())
    assert resolved["opening_height_deltas_mm"] == [0.0]
    assert resolved["vertical_shifts_mm"] == [0.0]


def test_aperture_report_writes_a_single_vertical_shift_axis(tmp_path):
    specification = resolve_aperture_report(
        ApertureSweepReportSpec(
            opening_height_deltas_mm=(0.0, 0.1),
            vertical_shifts_mm=(0.0),
        ),
        _aperture_objects(),
    )
    sources = _sources()
    detectors = [_detector(7, "pupil"), _detector(8, "eyebox")]
    report = ApertureSweepReport(
        specification,
        [
            _pupil_spec("pupil", "zemax_pupil_report"),
            _pupil_spec("eyebox", "zemax_eyebox_report"),
        ],
    )

    for _height_index, _shift_index, height_delta, shift in report.states:
        report.record_state(
            height_delta,
            shift,
            [
                _completed_cube(detectors[0], sources, 100.0 + height_delta),
                _completed_cube(detectors[1], sources, 150.0 + height_delta),
            ],
        )

    report.write(tmp_path)
    payload = json.loads(
        (tmp_path / "reports" / "zemax_aperture_report" / "report.json").read_text()
    )
    assert len(payload["states"]) == 2
    assert payload["configuration"]["vertical_shifts_mm"] == [0.0]
    assert payload["detectors"][0]["state_summary"]["mean_detector_total_flux_over_fov"]
    assert len(payload["detectors"][0]["state_summary"]["mean_detector_total_flux_over_fov"]) == 1


def test_aperture_report_writes_selectable_states_and_exact_tables(tmp_path):
    specification = resolve_aperture_report(ApertureSweepReportSpec(), _aperture_objects())
    sources = _sources()
    detectors = [_detector(7, "pupil"), _detector(8, "eyebox")]
    pupil_specs = [
        _pupil_spec("pupil", "zemax_pupil_report"),
        _pupil_spec("eyebox", "zemax_eyebox_report"),
    ]
    report = ApertureSweepReport(specification, pupil_specs)

    for height_index, shift_index, height_delta, shift in report.states:
        state_value = 100.0 + 10.0 * height_index + shift_index
        report.record_state(
            height_delta,
            shift,
            [
                _completed_cube(detectors[0], sources, state_value),
                _completed_cube(detectors[1], sources, state_value + 50.0),
            ],
        )

    index_record = report.write(tmp_path)
    report_directory = tmp_path / "reports" / "zemax_aperture_report"
    payload = json.loads((report_directory / "report.json").read_text())
    page = (report_directory / "index.html").read_text()

    assert index_record["html"] == "reports/zemax_aperture_report/index.html"
    assert len(payload["states"]) == 9
    assert sum(state["nominal"] for state in payload["states"]) == 1
    assert [detector["label"] for detector in payload["detectors"]] == ["pupil", "eyebox"]
    assert payload["configuration"]["geometry"]["inner_x_half_width_mm"] == 6.0
    assert 'id="detectorSelect"' in page
    assert 'id="heightSelect"' in page
    assert 'id="shiftSelect"' in page
    assert 'id="metricSelect"' in page
    assert "Horizontal opening fixed" in page
    assert "drawColorScale" in page
    assert "common to all configured aperture" in page

    with (report_directory / "state_summary.csv").open(newline="", encoding="utf-8") as handle:
        state_rows = list(csv.DictReader(handle))
    with (report_directory / "field_summary.csv").open(newline="", encoding="utf-8") as handle:
        field_rows = list(csv.DictReader(handle))
    assert len(state_rows) == 9 * 2
    assert len(field_rows) == 9 * 2 * 4

    write_position_space_index(tmp_path, [], derived_reports=[index_record])
    index_page = (tmp_path / "position_space_index.html").read_text()
    assert (
        'href="reports/zemax_aperture_report/index.html" target="report-frame"'
        in index_page
    )


def test_generated_sweep_changes_and_restores_only_vertical_aperture_fields(monkeypatch):
    monkeypatch.setitem(sys.modules, "clr", types.ModuleType("clr"))
    fake_builder = generated_builder_module()
    monkeypatch.setitem(sys.modules, "build_zemax_model", fake_builder)
    monkeypatch.setitem(sys.modules, "position_space_analysis", position_space_analysis)
    monkeypatch.setitem(sys.modules, "zemax_pupil_report", zemax_pupil_report)
    monkeypatch.setitem(sys.modules, "zemax_aperture_report", zemax_aperture_report)
    monkeypatch.setitem(sys.modules, "zemax_coating_report", zemax_coating_report)

    namespace = {"__name__": "generated_sweep_test", "__file__": "/tmp/run_fov_sweep.py"}
    exec(sweep_script_text(), namespace)

    inner = types.SimpleNamespace(
        ObjectData=types.SimpleNamespace(
            X1HalfWidth=6.0,
            X2HalfWidth=6.0,
            Y1HalfWidth=1.5,
            Y2HalfWidth=1.5,
        ),
        XPosition=0.0,
        YPosition=0.0,
    )
    boolean = types.SimpleNamespace(XPosition=7.0, YPosition=35.0, ZPosition=0.2)
    objects_by_index = {2: inner, 3: boolean}
    system = types.SimpleNamespace(
        NCE=types.SimpleNamespace(GetObjectAt=lambda object_index: objects_by_index[object_index])
    )
    specification = resolve_aperture_report(ApertureSweepReportSpec(), _aperture_objects())

    namespace["_set_aperture_vertical_state"](system, specification, 0.1, -0.1)
    assert inner.ObjectData.Y1HalfWidth == pytest.approx(1.55)
    assert inner.ObjectData.Y2HalfWidth == pytest.approx(1.55)
    assert inner.YPosition == 0.0
    assert boolean.YPosition == pytest.approx(34.94)
    assert boolean.ZPosition == pytest.approx(0.12)
    assert boolean.XPosition == 7.0
    assert inner.ObjectData.X1HalfWidth == 6.0
    assert inner.ObjectData.X2HalfWidth == 6.0
    assert inner.XPosition == 0.0

    namespace["_set_aperture_vertical_state"](system, specification, 0.0, 0.0)
    assert inner.ObjectData.Y1HalfWidth == 1.5
    assert inner.ObjectData.Y2HalfWidth == 1.5
    assert boolean.YPosition == 35.0
    assert boolean.ZPosition == 0.2
    assert boolean.XPosition == 7.0
