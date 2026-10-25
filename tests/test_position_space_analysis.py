"""Tests for the field-resolved polychromatic Zemax detector export."""

import ast
import json
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import apollo14.export.zemax.reporting.html_plotting as html_plotting
import apollo14.export.zemax.reporting.position_space_analysis as position_space_analysis
import apollo14.export.zemax.reporting.zemax_aperture_report as zemax_aperture_report
import apollo14.export.zemax.reporting.zemax_coating_report as zemax_coating_report
import apollo14.export.zemax.reporting.zemax_pupil_report as zemax_pupil_report
from apollo14.export.zemax.bundle import (
    GENERATED_SCRIPTS_DIRECTORY,
    PORTABLE_REPORT_MODULES,
    REPORT_SUPPORT_DIRECTORY,
    _readme_position_space_analysis,
    _write_generated_scripts,
)
from apollo14.export.zemax.prescription import SourceSpec, build_prescription
from apollo14.export.zemax.reporting.position_space_analysis import (
    NativeDetectorArchive,
    PositionSpaceCube,
    SystemSpectrum,
    detector_directory_name,
    write_position_space_index,
)
from apollo14.export.zemax.templates.zosapi_script import (
    build_and_run_script_text,
    build_script_text,
    sweep_script_text,
)
from apollo14.system import OpticalSystem
from apollo14.units import nm
from tests.zosapi_test_helpers import generated_builder_module


def _detector():
    return {
        "index": 7,
        "comment": "pupil",
        "data": {
            "pixels_x": 3,
            "pixels_y": 2,
            "half_width_x": 3.0,
            "half_width_y": 1.0,
        },
        "zemax_local_axes_world": {
            "x": [1.0, 0.0, 0.0],
            "y": [0.0, 1.0, 0.0],
        },
        "apollo14_local_axes_world": {
            "x": [-1.0, 0.0, 0.0],
            "y": [0.0, -1.0, 0.0],
        },
    }


def _sources():
    return [
        {
            "index": 11,
            "label": "fov_m1p0_p0p5",
            "fov_angles_deg": [-1.0, 0.5],
            "intended_direction": [-0.01, 0.005, 0.9999],
            "data": {"sweep_power": 1.0},
        },
        {
            "index": 12,
            "label": "fov_p1p0_p0p5",
            "fov_angles_deg": [1.0, 0.5],
            "intended_direction": [0.01, 0.005, 0.9999],
            "data": {"sweep_power": 2.0},
        },
    ]


def _spectrum():
    return SystemSpectrum([0.46, 0.63], [1.0, 2.0])


def _filled_cube():
    cube = PositionSpaceCube(_detector(), _sources(), _spectrum())
    for field_index, source in enumerate(_sources()):
        base = 100.0 * field_index
        cube.record(
            source["index"],
            [
                [base + 0.0, base + 1.0, base + 2.0],
                [base + 3.0, base + 4.0, base + 5.0],
            ],
        )
    return cube


def test_human_readable_numbers_remove_float_noise_and_trailing_zeros():
    source = {
        "index": 1,
        "label": "fov_p05p0_p06p0",
        "fov_angles_deg": [5.000001907, 6.000013351],
    }

    assert position_space_analysis._field_display(8, source) == ("Field 9 — FOV x +5°, y +6°")
    assert position_space_analysis._format_angle_deg(-0.000001) == "0"
    assert position_space_analysis._format_angle_deg(0.499999) == "+0.5"
    assert position_space_analysis._format_angle_deg(0.250001) == "+0.25"
    assert position_space_analysis._format_angle_deg(1.0 / 3.0) == "+0.33"
    assert position_space_analysis.format_compact_number(0.9999) == "1"
    assert detector_directory_name({"index": 7, "comment": "Eye box / scan"}) == ("Eye_box_scan")


def test_position_space_cube_rejects_partial_fov_metadata():
    sources = _sources()
    del sources[1]["fov_angles_deg"]

    with pytest.raises(ValueError, match="every field or for none"):
        PositionSpaceCube(_detector(), sources, _spectrum())


def test_position_space_cube_rejects_malformed_source_metadata():
    malformed_fov = _sources()
    malformed_fov[0]["fov_angles_deg"] = ["not-an-angle", 0.5]
    with pytest.raises(ValueError, match="FOV x angle"):
        PositionSpaceCube(_detector(), malformed_fov, _spectrum())

    negative_power = _sources()
    negative_power[0]["data"]["sweep_power"] = -1.0
    with pytest.raises(ValueError, match="sweep_power must be non-negative"):
        PositionSpaceCube(_detector(), negative_power, _spectrum())


def test_position_space_cube_writes_portable_dense_data_and_graphics(tmp_path):
    cube = _filled_cube()
    result = cube.write(
        tmp_path / "pupil",
        trace_metadata={"use_polarization": True},
    )

    assert cube.shape == (2, 2, 3)
    assert cube.fov_angles_available is True
    assert list(cube.iter_flux())[-1] == pytest.approx(105.0)
    assert cube.field_index(12) == 1
    assert cube.total_flux(1) == pytest.approx(615.0)

    with zipfile.ZipFile(result["npz"]) as archive:
        assert {
            "flux.npy",
            "fov_x_deg.npy",
            "fov_y_deg.npy",
            "system_wavelength_um.npy",
            "system_wavelength_weight.npy",
            "pixel_x_center_mm.npy",
            "pixel_y_center_mm.npy",
            "metadata.json",
        } <= set(archive.namelist())

    with np.load(result["npz"]) as data:
        assert data["flux"].shape == (2, 2, 3)
        assert data["flux"][1, 1, 2] == pytest.approx(105.0)
        assert np.allclose(data["fov_x_deg"], [-1.0, 1.0])
        assert np.allclose(data["fov_y_deg"], [0.5, 0.5])
        assert np.allclose(data["system_wavelength_um"], [0.46, 0.63])
        assert np.allclose(data["system_wavelength_weight"], [1.0, 2.0])
        assert np.allclose(data["pixel_x_center_mm"], [-2.0, 0.0, 2.0])
        assert np.allclose(data["pixel_y_center_mm"], [-0.5, 0.5])

    metadata = json.loads(Path(result["metadata"]).read_text())
    assert metadata["array_order"] == ["field", "pixel_y", "pixel_x"]
    assert metadata["quantity"]["detector_data_code"] == 0
    assert metadata["fields"][1]["source_sweep_power"] == 2.0
    assert metadata["spectrum"]["source_wavenumber"] == 0
    assert metadata["trace"] == {"use_polarization": True}
    assert len(metadata["files"]["slices"]) == 2
    assert "pixel 1 is at local (-x, -y)" in metadata["pixel_row_order"]
    assert metadata["display"]["global_max_flux"] == 105.0
    assert metadata["display"]["color_scale"]["maximum"] == 125.0
    assert metadata["display"]["color_scale"]["ticks"] == [
        0.0,
        25.0,
        50.0,
        75.0,
        100.0,
        125.0,
    ]

    html = Path(result["html"]).read_text()
    assert "function update() {" in html
    assert ":root {" in html
    assert "local -y row" in html
    assert "fov_m1p0_p0p5" in html
    assert 'id="wavelength"' not in html
    assert "wavenumber 0" in html
    assert 'aria-label="Detector flux color scale"' in html
    assert "linear-gradient(to top" in html
    assert "<span>125</span>" in html
    assert "<span>25</span>" in html
    layout_marker = f'data-apollo14-report-layout="{html_plotting.REPORT_LAYOUT_VERSION}"'
    assert layout_marker in html
    assert 'content="no-store, no-cache, must-revalidate"' in html
    assert len(list((tmp_path / "pupil" / "images").glob("*.png"))) == 2
    assert (
        next((tmp_path / "pupil" / "images").glob("*.png"))
        .read_bytes()
        .startswith(b"\x89PNG\r\n\x1a\n")
    )

    summary = (tmp_path / "pupil" / "position_space_summary.csv").read_text().splitlines()
    assert len(summary) == 3
    assert summary[0].startswith("field_index,source_label,source_object_index")

    index_path = write_position_space_index(
        tmp_path,
        [{"detector_label": "pupil", "html": "pupil/position_space.html"}],
    )
    assert json.loads(index_path.read_text())["detectors"][0]["html"] == (
        "pupil/position_space.html"
    )
    index_html = (tmp_path / "position_space_index.html").read_text()
    assert "pupil/position_space.html" in index_html
    assert layout_marker in index_html
    assert "position: fixed;" in index_html
    assert 'name="report-frame"' in index_html
    assert 'target="report-frame"' in index_html
    assert 'src="pupil/position_space.html"' in index_html
    assert "Open this report separately" in index_html


def test_position_space_cube_requires_every_field(tmp_path):
    cube = PositionSpaceCube(_detector(), _sources(), _spectrum())
    cube.record(11, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    with pytest.raises(RuntimeError, match="incomplete"):
        cube.write(tmp_path)


def test_position_space_index_keeps_navigation_while_switching_reports(tmp_path):
    write_position_space_index(
        tmp_path,
        [
            {
                "detector_label": "pupil",
                "flux_shape": [9, 200, 200],
                "html": "pupil/position_space.html",
            }
        ],
        derived_reports=[
            {
                "report_name": "zemax_pupil_report",
                "title": "Zemax pupil and FOV report",
                "detector_label": "pupil",
                "html": "zemax_pupil_report.html",
            }
        ],
    )

    page = (tmp_path / "position_space_index.html").read_text()
    assert '<aside class="sidebar"' in page
    assert '<main class="workspace">' in page
    assert page.index('<aside class="sidebar"') < page.index('<main class="workspace">')
    assert 'src="zemax_pupil_report.html"' in page
    assert 'href="zemax_pupil_report.html" target="report-frame"' in page
    assert 'href="pupil/position_space.html" target="report-frame"' in page
    assert "Flux cube: 9 × 200 × 200" in page
    assert 'class="nav-link active" aria-current="page"' in page
    assert "links.forEach(link => link.addEventListener" in page


def test_native_detector_archive_writes_native_manifest_and_zpl_macro(tmp_path):
    detector = _detector()
    archive = NativeDetectorArchive(
        tmp_path / "out",
        tmp_path,
        _sources(),
        [detector],
        _spectrum(),
    )
    for source in _sources():
        path = archive.data_path(detector["index"], source["index"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"native detector data")
        archive.record(detector["index"], source["index"], path)
    viewer_config = archive.archive_directory / "detector_viewer.cfg"
    viewer_config.write_text("generated settings")

    result = archive.write(viewer_config)

    manifest = json.loads(Path(result["manifest"]).read_text())
    assert manifest["schema_version"] == 3
    assert manifest["format"]["extension"] == ".DDR"
    assert len(manifest["files"]) == 2
    assert manifest["fields"][1]["field_number"] == 2
    assert manifest["fields"][1]["source_sweep_power"] == 2.0
    assert manifest["files"][1]["source_sweep_power"] == 2.0
    assert manifest["spectrum"]["source_wavenumber"] == 0
    assert manifest["spectrum"]["relative_weights"] == [1.0, 2.0]
    assert manifest["viewer"]["settings"] == ("out/zemax_detector_data/detector_viewer.cfg")
    assert "not an OpticStudio Field Data Editor row" in manifest["field_selection"]["meaning"]
    assert "does not sum fields or trace rays" in manifest["field_selection"]["selection_effect"]
    assert manifest["viewer"]["initial_detector_object_index"] == 7
    assert manifest["viewer"]["macro_actions"] == [
        "Prompt for one archived FOV field number.",
        "Load that field's .DDR state into every listed detector object.",
        "Open Detector Viewer in False Color for the first detector.",
    ]
    assert "Run build_and_run.py successfully" in manifest["viewer"]["manual_prerequisites"][0]
    assert "Enter one field number" in manifest["viewer"]["manual_actions"][0]

    macro = Path(result["macro"]).read_text()
    assert 'INPUT "Field number (1-2)", field' in macro
    assert 'INPUT "Wavenumber' not in macro
    assert "A field number selects one independently traced FOV source direction" in macro
    assert "not an OpticStudio Field Data Editor row or a wavelength" in macro
    assert "source object=11" in macro
    assert "sweep power=2" in macro
    assert 'PRINT "  object 7: pupil"' in macro
    assert "Detector Viewer initially shows object 7: pupil" in macro
    assert "use the Detector selector to inspect another object" in macro
    assert "does not trace rays, sum fields, or change sources or wavelengths" in macro
    assert "no wavelength input is required" in macro
    assert "THEN" not in macro
    assert "IF (field == 1)" in macro
    assert "IF (wavenumber" not in macro
    macro_lines = macro.splitlines()
    assert sum(line.lstrip().startswith("IF (") for line in macro_lines) == sum(
        line.strip() == "ENDIF" for line in macro_lines
    )
    assert 'LOADDETECTOR 1, 7, "out\\zemax_detector_data\\detector_7\\field_1.DDR"' in macro
    assert 'OPENANALYSISWINDOW "DVR", "out\\zemax_detector_data\\detector_viewer.cfg"' in macro

    fallback_result = archive.write()
    assert fallback_result["viewer_config"] is None
    fallback_macro = Path(fallback_result["macro"]).read_text()
    assert "COPYFILE default_viewer_config$, viewer_config$" in fallback_macro
    assert (
        'viewer_config$ = lens_path$ + "out\\zemax_detector_data\\'
        'apollo14_detector_viewer_false_color.cfg"'
    ) in fallback_macro
    assert 'MODIFYSETTINGS viewer_config$, DVW_DETECTOR, 7, "DVR"' in fallback_macro
    assert 'MODIFYSETTINGS viewer_config$, DVW_SHOW, 2, "DVR"' in fallback_macro
    assert 'MODIFYSETTINGS viewer_config$, DVW_DATA, 0, "DVR"' in fallback_macro
    assert 'OPENANALYSISWINDOW "DVR", viewer_config$' in fallback_macro


def test_source_prescription_carries_fov_metadata_and_sweep_power():
    source = SourceSpec(
        label="field",
        position=np.array([0.0, 0.0, 0.0]),
        direction=np.array([0.0, 0.0, 1.0]),
        beam_width=4.0,
        beam_height=2.0,
        beam_local_x=np.array([1.0, 0.0, 0.0]),
        beam_local_y=np.array([0.0, 1.0, 0.0]),
        power=0.0,
        fov_x_deg=-2.0,
        fov_y_deg=1.0,
        sweep_power=1.25,
    )
    prescription = build_prescription(
        OpticalSystem(elements=[]),
        chassis_pivot=jnp.zeros(3),
        chassis_tilt_deg=0.0,
        sources=[source],
        trace_wavelengths=jnp.array([550.0]) * nm,
        glass_names={},
    )

    exported = prescription.objects[0]
    assert exported["fov_angles_deg"] == [-2.0, 1.0]
    assert exported["data"]["power"] == 0.0
    assert exported["data"]["sweep_power"] == 1.25
    assert prescription.document["system"]["wavelength_weights"] == [1.0]


def _portable_report_sources():
    source_directory = Path(position_space_analysis.__file__).parent
    return {
        file_name: (source_directory / file_name).read_text()
        for file_name in PORTABLE_REPORT_MODULES
    }


def test_generated_bundle_copies_the_standalone_report_modules(tmp_path):
    (tmp_path / "build_zemax_model.py").write_text("stale builder")
    (tmp_path / "position_space_analysis.py").write_text("stale support module")
    _write_generated_scripts(tmp_path)

    assert (tmp_path / "build_and_run.py").is_file()
    assert (tmp_path / GENERATED_SCRIPTS_DIRECTORY / "build_zemax_model.py").is_file()
    assert (tmp_path / GENERATED_SCRIPTS_DIRECTORY / "run_fov_sweep.py").is_file()
    assert not (tmp_path / "build_zemax_model.py").exists()
    for file_name, source in _portable_report_sources().items():
        generated = tmp_path / REPORT_SUPPORT_DIRECTORY / file_name
        assert generated.read_text() == source
        compile(source, str(generated), "exec")
        assert not (tmp_path / file_name).exists()

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from position_space_analysis import "
                "NativeDetectorArchive, PositionSpaceCube, SystemSpectrum, "
                "write_position_space_index; "
                "from zemax_pupil_report import PupilReportSpec; "
                "from zemax_aperture_report import ApertureSweepReport; "
                "from zemax_angular_response import write_angular_response_report; "
                "from eyebox_overlap_report import EyeboxOverlapReportSpec; "
                "print('portable imports ok')"
            ),
        ],
        cwd=tmp_path / REPORT_SUPPORT_DIRECTORY,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "portable imports ok"

    entrypoint_import = subprocess.run(
        [sys.executable, "-c", "import build_and_run; print('entry point imports ok')"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert entrypoint_import.stdout.strip() == "entry point imports ok"


def test_generated_windows_code_avoids_python310_only_zip_strict():
    """OpticStudio deployments can still host Python versions before 3.10."""
    portable_sources = [
        build_script_text(),
        sweep_script_text(),
        build_and_run_script_text(),
        *_portable_report_sources().values(),
    ]

    assert all("strict=True" not in source for source in portable_sources)


def test_generated_windows_code_parses_with_python38_grammar():
    """Keep copied modules usable by OpticStudio hosts running older Python."""
    portable_sources = {
        "build_zemax_model.py": build_script_text(),
        "run_fov_sweep.py": sweep_script_text(),
        "build_and_run.py": build_and_run_script_text(),
        **_portable_report_sources(),
    }

    for file_name, source in portable_sources.items():
        ast.parse(source, filename=file_name, feature_version=(3, 8))


def test_generated_builder_downcasts_and_disables_zero_power_source(monkeypatch):
    """The builder uses zero analysis rays instead of rejected zero source power."""
    monkeypatch.setitem(sys.modules, "clr", types.ModuleType("clr"))
    namespace = {
        "__name__": "generated_builder_test",
        "__file__": "/tmp/build_zemax_model.py",
    }
    exec(build_script_text(), namespace)

    implementation = types.SimpleNamespace(
        Power=1.0,
        NumberOfLayoutRays=1,
        NumberOfAnalysisRays=1,
        XHalfAngle=0.0,
        YHalfAngle=0.0,
    )
    object_data_interface = types.SimpleNamespace(__implementation__=implementation)

    class ParameterCell:
        def __init__(self, cell_number):
            self.IntegerValue = None
            self._double_value = 1.0 if cell_number == 13 else None

        @property
        def DoubleValue(self):  # noqa: N802 - mirrors ZOS-API.
            return self._double_value

        @DoubleValue.setter
        def DoubleValue(self, value):  # noqa: N802 - mirrors ZOS-API.
            # Reproduce OpticStudio retaining 1.0 when source Power is set to 0.
            if float(value) > 0.0:
                self._double_value = float(value)

    class ObjectRow:
        # Python.NET 3 may expose an unhelpful interface through the property
        # while its explicit getter carries the concrete implementation.
        ObjectData = object()

        def __init__(self):
            self.row_cells = {
                cell_number: ParameterCell(cell_number)
                for cell_number in (11, 12, 13)
            }

        def get_ObjectData(self):  # noqa: N802 - mirrors ZOS-API.
            return object_data_interface

        def GetCellAt(self, cell_number):  # noqa: N802 - mirrors ZOS-API.
            return self.row_cells[cell_number]

    object_row = ObjectRow()
    object_data = namespace["get_object_data_implementation"](object_row)
    assert object_data is implementation

    namespace["_apply_source_angles"](
        [object_data, object_row],
        {
            "half_angle_x": 0.25,
            "half_angle_y": 0.5,
            "power": 0.0,
            "sweep_power": 1.25,
            "layout_rays": 7,
            "analysis_rays": 11,
        },
        "test source",
    )
    zosapi = types.SimpleNamespace(
        Editors=types.SimpleNamespace(
            NCE=types.SimpleNamespace(ObjectColumn=types.SimpleNamespace())
        )
    )
    namespace["_set_source_common_parameters"](
        object_row,
        zosapi,
        {
            "power": 0.0,
            "sweep_power": 1.25,
            "layout_rays": 7,
            "analysis_rays": 11,
        },
        "test source",
    )
    assert implementation.Power == 1.0
    assert implementation.NumberOfLayoutRays == 1
    assert implementation.NumberOfAnalysisRays == 1
    assert implementation.XHalfAngle == 0.25
    assert implementation.YHalfAngle == 0.5
    assert object_row.row_cells[11].IntegerValue == 7
    assert object_row.row_cells[12].IntegerValue == 0
    assert object_row.row_cells[13].DoubleValue == 1.25


def test_build_and_run_rejects_stale_html_report_layouts():
    source = build_and_run_script_text()

    assert "from html_plotting import REPORT_LAYOUT_VERSION" in source
    assert "from position_space_analysis import detector_directory_name" in source
    assert 'BUNDLE_DIRECTORY / "scripts" / "reporting"' in source
    assert 'bundle_dir / "scripts" / "build_zemax_model.py"' in source
    assert 'bundle_dir / "scripts" / "run_fov_sweep.py"' in source
    assert "def _verify_html_report_layout(" in source
    assert "Copy build_and_run.py and the complete scripts directory" in source
    assert "Verified HTML report layout" in source
    assert '"eyebox_overlap_report"' in source
    compile(source, "build_and_run.py", "exec")


def test_build_and_run_expects_the_configured_overlap_page(tmp_path):
    _write_generated_scripts(tmp_path)
    namespace = {
        "__name__": "generated_runner_test",
        "__file__": str(tmp_path / "build_and_run.py"),
    }
    exec(build_and_run_script_text(), namespace)
    prescription_path = tmp_path / "prescription.json"
    prescription_path.write_text(
        json.dumps(
            {
                "objects": [],
                "analyses": {
                    "eyebox_overlap_report": {
                        "report_name": "eyebox_field_overlap",
                    }
                },
            }
        )
    )

    expected = namespace["_expected_html_reports"](tmp_path / "out", prescription_path)

    assert tmp_path / "out" / "reports" / "eyebox_field_overlap" / "index.html" in expected


def test_position_space_readme_explains_output_and_opticstudio_viewer():
    text = _readme_position_space_analysis()

    assert "field, pixel_y, pixel_x" in text
    assert "no separate wavelength axis" in text
    assert "position_space_index.html" in text
    assert "fixed left menu" in text
    assert "right pane" in text
    assert "browser Back or Forward" in text
    assert "position_space.npz" in text
    assert "rounded tick values" in text
    assert "numeric color bar" in text
    assert "System Wavelengths" in text
    assert "Wavenumber" in text
    assert "Detector Viewer" in text
    assert "apollo14_view_position_space.zpl" in text
    assert ".DDR" in text
    assert "not an OpticStudio Field Data Editor row" in text
    assert "Required user action" in text
    assert "original relative locations" in text
    assert "detector object numbers are unchanged" in text
    assert "No manual source-power edit" in text
    assert "zero analysis rays to every other source" in text
    assert "Power remains positive" in text


def test_generated_sweep_uses_sweep_power_and_polychromatic_wavenumber(monkeypatch):
    """Exercise the standalone source-state helpers without OpticStudio."""
    fake_clr = types.ModuleType("clr")
    monkeypatch.setitem(sys.modules, "clr", fake_clr)
    fake_builder = generated_builder_module()
    monkeypatch.setitem(sys.modules, "build_zemax_model", fake_builder)
    monkeypatch.setitem(sys.modules, "position_space_analysis", position_space_analysis)
    monkeypatch.setitem(sys.modules, "zemax_pupil_report", zemax_pupil_report)
    monkeypatch.setitem(sys.modules, "zemax_aperture_report", zemax_aperture_report)
    monkeypatch.setitem(sys.modules, "zemax_coating_report", zemax_coating_report)

    namespace = {
        "__name__": "generated_sweep_test",
        "__file__": "/tmp/run_fov_sweep.py",
    }
    exec(sweep_script_text(), namespace)

    class DotNetDouble:
        """Mimic a Python.NET value that requires explicit float conversion."""

        def __init__(self, value):
            self.value = value

        def __float__(self):
            return float(self.value)

        def __eq__(self, other):
            return False

    class SourceDataImplementation:
        def __init__(self, source):
            self.source = source

        @property
        def Power(self):  # noqa: N802 - mirrors ZOS-API.
            return self.source.power

        @Power.setter
        def Power(self, unused_value):  # noqa: N802 - mirrors ZOS-API.
            # Reproduce the ineffective typed setter in the reported failure.
            return None

    class SourceDataInterface:
        """Mimic a Python.NET 3 interface whose direct setter is ineffective."""

        def __init__(self, source):
            self.source = source
            self.__implementation__ = SourceDataImplementation(source)

        @property
        def Power(self):  # noqa: N802 - mirrors ZOS-API.
            return self.source.power

        @Power.setter
        def Power(self, unused_value):  # noqa: N802 - mirrors ZOS-API.
            return None

    class SourcesDataImplementation:
        IsSourcesAvailable = True

        def __init__(self):
            self.SourceColor = "Monochromatic"

    class SourcesDataInterface:
        """Expose source settings only through the concrete implementation."""

        def __init__(self):
            self.__implementation__ = SourcesDataImplementation()

        @property
        def IsSourcesAvailable(self):  # noqa: N802 - mirrors ZOS-API.
            return self.__implementation__.IsSourcesAvailable

        @property
        def SourceColor(self):  # noqa: N802 - mirrors ZOS-API.
            return self.__implementation__.SourceColor

        @SourceColor.setter
        def SourceColor(self, unused_value):  # noqa: N802 - mirrors ZOS-API.
            return None

    class Cell:
        def __init__(self, source, column, writable):
            self.source = source
            self.column = column
            self.writable = writable
            self.value = None

        @property
        def IntegerValue(self):  # noqa: N802 - mirrors ZOS-API.
            if self.column not in ("Par2", "Par4"):
                raise TypeError("Expected Integer, got 'Double'")
            return self.value

        @IntegerValue.setter
        def IntegerValue(self, value):  # noqa: N802 - mirrors ZOS-API.
            if self.column not in ("Par2", "Par4"):
                raise TypeError("Expected Integer, got 'Double'")
            if self.writable:
                self.value = int(value)

        @property
        def DoubleValue(self):  # noqa: N802 - mirrors ZOS-API.
            if self.column != "Par3":
                raise TypeError("Expected Double, got 'Integer'")
            return DotNetDouble(self.source.power)

        @DoubleValue.setter
        def DoubleValue(self, value):  # noqa: N802 - mirrors ZOS-API.
            if self.column != "Par3":
                raise TypeError("Expected Double, got 'Integer'")
            self.source.power_write_attempts.append(float(value))
            if self.writable and float(value) > 0.0:
                self.source.power = float(value)

    class SourceObject:
        def __init__(self):
            self.power = 1.0
            self.power_write_attempts = []
            self.ObjectData = SourceDataInterface(self)
            self.SourcesData = SourcesDataInterface()
            self.cells = {}
            self.row_cells = {}

        def GetObjectCell(self, column):  # noqa: N802 - mirrors ZOS-API.
            # This route silently ignores writes in the reported environment.
            return self.cells.setdefault(column, Cell(self, column, False))

        def GetCellAt(self, cell_number):  # noqa: N802 - mirrors ZOS-API.
            column = f"Par{cell_number - 10}"
            return self.row_cells.setdefault(
                cell_number, Cell(self, column, True))

        def get_ObjectData(self):  # noqa: N802 - mirrors ZOS-API.
            return self.ObjectData

    source_objects = {1: SourceObject(), 2: SourceObject()}
    editor = types.SimpleNamespace(GetObjectAt=lambda object_index: source_objects[object_index])
    zosapi = types.SimpleNamespace(
        Editors=types.SimpleNamespace(
            NCE=types.SimpleNamespace(
                ObjectColumn=types.SimpleNamespace(
                    Par2="Par2", Par3="Par3", Par4="Par4"),
                SourceColorMode=types.SimpleNamespace(SystemWavelengths="SystemWavelengths"),
            )
        )
    )
    sources = [
        {
            "index": 1,
            "label": "first",
            "data": {
                "power": 1.0,
                "sweep_power": 1.0,
                "layout_rays": 8,
                "analysis_rays": 17,
            },
        },
        {
            "index": 2,
            "label": "second",
            "data": {
                "power": 0.0,
                "sweep_power": 1.5,
                "layout_rays": 8,
                "analysis_rays": 23,
            },
        },
    ]

    namespace["configure_sources_for_polychromatic_trace"](editor, sources, zosapi)
    namespace["set_only_active_source"](editor, sources, 2, zosapi)

    assert source_objects[1].SourcesData.SourceColor == "SystemWavelengths"
    assert source_objects[2].SourcesData.SourceColor == "SystemWavelengths"
    assert source_objects[1].row_cells[14].IntegerValue == 0
    assert source_objects[2].row_cells[14].IntegerValue == 0
    assert source_objects[1].row_cells[12].IntegerValue == 0
    assert source_objects[2].row_cells[12].IntegerValue == 23
    assert source_objects[1].power == 1.0
    assert float(source_objects[2].row_cells[13].DoubleValue) == 1.5
    assert source_objects[1].power_write_attempts == []
    assert source_objects[2].power_write_attempts == [1.5]
    with pytest.raises(KeyError, match="not unique"):
        namespace["set_only_active_source"](editor, sources, 3, zosapi)

    invalid_sources = [
        {**source, "data": dict(source["data"])} for source in sources
    ]
    invalid_sources[1]["data"]["sweep_power"] = "invalid"
    state_before_error = [
        (source_objects[index].power, source_objects[index].row_cells[12].IntegerValue)
        for index in (1, 2)
    ]
    with pytest.raises(ValueError, match="invalid sweep_power"):
        namespace["set_only_active_source"](editor, invalid_sources, 2, zosapi)
    assert [
        (source_objects[index].power, source_objects[index].row_cells[12].IntegerValue)
        for index in (1, 2)
    ] == state_before_error

    failed_system = types.SimpleNamespace(
        NCE=types.SimpleNamespace(GetDetectorData=lambda detector_index, pixel, data: (False, 0.0))
    )
    with pytest.raises(RuntimeError, match="rejected detector"):
        namespace["_read_detector_with_nce"](failed_system, 7, 1, 1)

    class FailingTrace:
        closed = False

        def __init__(self):
            self.events = []

        def ClearDetectors(self, detector_number):  # noqa: N802 - mirrors ZOS-API.
            self.events.append(("clear_all", detector_number))
            return None

        def ClearDetectorObject(self, detector_index):  # noqa: N802 - mirrors ZOS-API.
            self.events.append(("clear_object", detector_index))
            return None

        def RunAndWaitForCompletion(self):  # noqa: N802 - mirrors ZOS-API.
            self.events.append(("trace", None))
            raise RuntimeError("trace failed")

        def Close(self):  # noqa: N802 - mirrors ZOS-API.
            self.closed = True

    ray_trace = FailingTrace()
    trace_system = types.SimpleNamespace(
        NCE=types.SimpleNamespace(
            GetDetectorData=lambda detector_index, pixel, data: (True, 0.0)
        ),
        Tools=types.SimpleNamespace(OpenNSCRayTrace=lambda: ray_trace),
    )
    with pytest.raises(RuntimeError, match="trace failed"):
        namespace["clear_and_trace"](trace_system, [{"index": 7}])
    assert ray_trace.events == [
        ("clear_all", 0),
        ("clear_object", 7),
        ("trace", None),
    ]
    assert ray_trace.closed is True

    uncleared_trace = FailingTrace()
    uncleared_system = types.SimpleNamespace(
        NCE=types.SimpleNamespace(
            GetDetectorData=lambda detector_index, pixel, data: (True, 2.5)
        ),
        Tools=types.SimpleNamespace(OpenNSCRayTrace=lambda: uncleared_trace),
    )
    with pytest.raises(RuntimeError, match="accumulated detector map"):
        namespace["clear_and_trace"](uncleared_system, [{"index": 7}])
    assert uncleared_trace.events == [
        ("clear_all", 0),
        ("clear_object", 7),
    ]
    assert uncleared_trace.closed is True


def test_generated_sweep_traces_once_per_field_for_all_detectors(tmp_path, monkeypatch):
    """The generated runner should not retrace identical states per detector."""
    fake_clr = types.ModuleType("clr")
    monkeypatch.setitem(sys.modules, "clr", fake_clr)
    fake_builder = generated_builder_module()
    monkeypatch.setitem(sys.modules, "build_zemax_model", fake_builder)
    monkeypatch.setitem(sys.modules, "position_space_analysis", position_space_analysis)
    monkeypatch.setitem(sys.modules, "zemax_pupil_report", zemax_pupil_report)
    monkeypatch.setitem(sys.modules, "zemax_aperture_report", zemax_aperture_report)
    monkeypatch.setitem(sys.modules, "zemax_coating_report", zemax_coating_report)

    namespace = {
        "__name__": "generated_sweep_test",
        "__file__": "/tmp/run_fov_sweep.py",
    }
    exec(sweep_script_text(), namespace)

    class Cell:
        IntegerValue = None
        DoubleValue = None

    class SourceObject:
        def __init__(self):
            self.ObjectData = types.SimpleNamespace(Power=None)
            self.SourcesData = types.SimpleNamespace(
                IsSourcesAvailable=True,
                SourceColor=None,
            )
            self.cells = {}

        def GetObjectCell(self, column):  # noqa: N802 - mirrors ZOS-API.
            return self.cells.setdefault(column, Cell())

    source_objects = {1: SourceObject(), 2: SourceObject()}
    native_saves = []

    def save_detector(detector_index, path):
        Path(path).write_bytes(f"detector {detector_index}".encode())
        native_saves.append((detector_index, Path(path)))
        return True

    editor = types.SimpleNamespace(
        GetObjectAt=lambda object_index: source_objects[object_index],
        SaveDetector=save_detector,
    )

    viewer_settings = []
    untyped_viewer_settings = []
    concrete_viewer_settings = []

    class DetectorViewerSettings:
        def __init__(self):
            self.Detector = types.SimpleNamespace(
                SetDetectorNumber=lambda detector_index: setattr(
                    self, "detector_index", detector_index
                )
            )
            self.detector_index = None
            self.ShowAs = None
            self.DataType = None
            self.Smoothing = None
            self.RayDatabaseFilename = None
            self.Filter = None

        def SaveTo(self, path):  # noqa: N802 - mirrors ZOS-API.
            Path(path).write_text("detector viewer settings")
            viewer_settings.append(self)

    class DetectorViewer:
        def __init__(self):
            self.settings = DetectorViewerSettings()
            self.untyped_settings = object()
            concrete_viewer_settings.append(self.settings)
            untyped_viewer_settings.append(self.untyped_settings)

        def GetSettings(self):  # noqa: N802 - mirrors ZOS-API.
            return self.untyped_settings

        def Close(self):  # noqa: N802 - mirrors ZOS-API.
            return None

    system = types.SimpleNamespace(
        NCE=editor,
        Analyses=types.SimpleNamespace(New_DetectorViewer=DetectorViewer),
    )
    cast_inputs = []

    def cast_detector_viewer_settings(settings):
        cast_inputs.append(settings)
        assert settings is untyped_viewer_settings[-1]
        return concrete_viewer_settings[-1]

    zosapi = types.SimpleNamespace(
        Editors=types.SimpleNamespace(
            NCE=types.SimpleNamespace(
                ObjectColumn=types.SimpleNamespace(
                    Par2="Par2", Par3="Par3", Par4="Par4"),
                SourceColorMode=types.SimpleNamespace(SystemWavelengths="SystemWavelengths"),
            )
        ),
        Analysis=types.SimpleNamespace(
            DetectorViewerShowAsTypes=types.SimpleNamespace(FalseColor="FalseColor"),
            DetectorViewerShowDataTypes=types.SimpleNamespace(
                IncoherentIrradiance="IncoherentIrradiance"
            ),
            Settings=types.SimpleNamespace(
                RayTracing=types.SimpleNamespace(IAS_DetectorViewer=cast_detector_viewer_settings)
            ),
        ),
    )
    sources = [
        {
            "index": 1,
            "label": "left",
            "fov_angles_deg": [-1.0, 0.0],
            "data": {
                "power": 1.0,
                "sweep_power": 1.0,
                "layout_rays": 8,
                "analysis_rays": 10,
            },
        },
        {
            "index": 2,
            "label": "right",
            "fov_angles_deg": [1.0, 0.0],
            "data": {
                "power": 0.0,
                "sweep_power": 1.0,
                "layout_rays": 8,
                "analysis_rays": 10,
            },
        },
    ]
    detectors = [
        {
            "index": 7,
            "comment": "pupil",
            "data": {
                "pixels_x": 2,
                "pixels_y": 2,
                "half_width_x": 1.0,
                "half_width_y": 1.0,
            },
        },
        {
            "index": 8,
            "comment": "eyebox",
            "data": {
                "pixels_x": 2,
                "pixels_y": 2,
                "half_width_x": 1.0,
                "half_width_y": 1.0,
            },
        },
    ]
    traced = []
    namespace["clear_and_trace"] = (
        lambda unused_system, active_detectors: traced.append(
            [detector["index"] for detector in active_detectors])
    )

    def fake_read_detector(unused_system, detector_index, pixels_x, pixels_y):
        active_source = next(
            index
            for index, source in source_objects.items()
            if source.cells["Par2"].IntegerValue > 0
        )
        wavenumber = source_objects[active_source].cells["Par4"].IntegerValue
        assert wavenumber == 0
        value = 100.0 * active_source + detector_index
        return [
            [value + column for column in range(pixels_x)] for _ in range(pixels_y)
        ], "fake flux"

    namespace["read_detector"] = fake_read_detector
    refreshed_file_lists = []
    application = types.SimpleNamespace(
        ZPLDir=str(tmp_path / "zemax_macros"),
        UpdateFileLists=lambda: refreshed_file_lists.append(True),
    )
    index_path = namespace["_run_position_space_sweep"](
        system,
        zosapi,
        sources,
        detectors,
        _spectrum(),
        tmp_path / "out",
        application=application,
        pupil_report_specs=[
            {
                "detector_label": "pupil",
                "eyebox_width_mm": 1.0,
                "eyebox_height_mm": 1.0,
                "eye_positions_x": 1,
                "eye_positions_y": 1,
                "eye_window_width_mm": 1.0,
                "eye_window_height_mm": 1.0,
                "coordinate_frame": "zemax",
            },
            {
                "detector_label": "eyebox",
                "eyebox_width_mm": 1.0,
                "eyebox_height_mm": 1.0,
                "eye_positions_x": 1,
                "eye_positions_y": 1,
                "eye_window_width_mm": 1.0,
                "eye_window_height_mm": 1.0,
                "coordinate_frame": "zemax",
                "report_name": "zemax_eyebox_report",
                "export_angular_response": True,
            },
        ],
        eyebox_overlap_report={"detector_label": "eyebox"},
    )

    assert traced == [[7, 8], [7, 8]]
    assert len(native_saves) == 4
    assert all(path.is_file() for _, path in native_saves)
    assert cast_inputs == [untyped_viewer_settings[0]]
    assert viewer_settings[0].detector_index == 7
    assert viewer_settings[0].ShowAs == "FalseColor"
    assert viewer_settings[0].DataType == "IncoherentIrradiance"
    assert viewer_settings[0].Smoothing == 0
    assert viewer_settings[0].RayDatabaseFilename == ""
    assert viewer_settings[0].Filter == ""
    index = json.loads(Path(index_path).read_text())
    assert [entry["detector_label"] for entry in index["detectors"]] == ["pupil", "eyebox"]
    assert [report["html"] for report in index["derived_reports"]] == [
        "reports/zemax_pupil_report/index.html",
        "reports/zemax_eyebox_report/index.html",
        "reports/zemax_eyebox_report/angular_response.html",
        "reports/eyebox_field_overlap/index.html",
    ]
    assert [report["detector_label"] for report in index["derived_reports"]] == [
        "pupil",
        "eyebox",
        "eyebox",
        "eyebox",
    ]
    pupil_directory = tmp_path / "out" / "reports" / "zemax_pupil_report"
    eyebox_directory = tmp_path / "out" / "reports" / "zemax_eyebox_report"
    assert (pupil_directory / "index.html").is_file()
    assert (eyebox_directory / "index.html").is_file()
    assert (eyebox_directory / "angular_response.html").is_file()
    assert (eyebox_directory / "angular_response.csv").is_file()
    overlap_directory = tmp_path / "out" / "reports" / "eyebox_field_overlap"
    assert (overlap_directory / "index.html").is_file()
    overlap_report = json.loads((overlap_directory / "report.json").read_text())
    assert overlap_report["detector"]["label"] == "eyebox"
    assert overlap_report["overlap_polygon"]["area_mm2"] == pytest.approx(4.0)
    pupil_report = json.loads((pupil_directory / "report.json").read_text())
    assert pupil_report["provenance"]["calculation_engine"].startswith("Zemax")
    eyebox_report = json.loads((eyebox_directory / "report.json").read_text())
    assert eyebox_report["detector"]["label"] == "eyebox"
    assert index["native_opticstudio"]["records"] == 4
    installed_macro = tmp_path / "zemax_macros" / "apollo14_view_position_space.zpl"
    assert Path(index["native_opticstudio"]["macro_list_entry"]) == installed_macro
    assert (
        installed_macro.read_text() == (tmp_path / "apollo14_view_position_space.zpl").read_text()
    )
    assert refreshed_file_lists == [True]
    assert (tmp_path / "out" / "zemax_detector_data" / "detector_7" / "field_1.DDR").is_file()
    native_manifest = json.loads(
        (tmp_path / "out" / "zemax_detector_data" / "native_detector_archive.json").read_text()
    )
    assert len(native_manifest["files"]) == 4
    assert native_manifest["spectrum"]["source_wavenumber"] == 0
    macro = (tmp_path / "apollo14_view_position_space.zpl").read_text()
    assert 'LOADDETECTOR 1, 7, "out\\zemax_detector_data\\detector_7\\field_1.DDR"' in macro
    assert 'INPUT "Wavenumber' not in macro
    assert 'OPENANALYSISWINDOW "DVR", ' in macro
    with np.load(tmp_path / "out" / "pupil" / "position_space.npz") as data:
        assert data["flux"].shape == (2, 2, 2)
        assert data["flux"][0, 0].tolist() == [107.0, 108.0]
        assert data["flux"][1, 0].tolist() == [207.0, 208.0]
        assert data["system_wavelength_weight"].tolist() == [1.0, 2.0]
