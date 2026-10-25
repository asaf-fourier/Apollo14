"""Zemax-side vertical-aperture tolerance sweep and portable HTML report.

The aperture exported by Apollo14 is a Boolean Native ``outer - inner``
object. This module describes two controlled vertical tolerances: the inner
rectangular volume's local-Y half widths set the clear-opening height, and the
Boolean object moves along its own local-Y axis. Local-X dimensions and the
global X position are recorded as invariants and never written by the sweep.

The module is copied into every Zemax bundle, so it intentionally uses only
the Python standard library and the portable pupil-report interface.
"""

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

try:
    from .html_plotting import REPORT_LAYOUT_VERSION, nice_scale
    from .zemax_pupil_report import (
        PupilReportSpec,
        calculate_pupil_report,
    )
except ImportError:  # Standalone copies in a generated Zemax bundle.
    from html_plotting import REPORT_LAYOUT_VERSION, nice_scale  # type: ignore[no-redef]
    from zemax_pupil_report import (  # type: ignore[no-redef]
        PupilReportSpec,
        calculate_pupil_report,
    )


SCHEMA_VERSION = 1
DEFAULT_REPORT_NAME = "zemax_aperture_report"
DEFAULT_REPORT_TITLE = "Zemax vertical aperture tolerance sweep"
_SAFE_REPORT_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_RESERVED_REPORT_NAMES = {"position_space_index"}
_FLOAT_TOLERANCE = 1.0e-9


@dataclass(frozen=True)
class ApertureSweepReportSpec:
    """User-facing configuration for one vertical aperture sweep.

    ``opening_height_deltas_mm`` changes the *full* clear-opening height.  A
    +0.1 mm state therefore adds +0.05 mm to each local-Y half width.
    ``vertical_shifts_mm`` moves the complete aperture along the Boolean
    object's local-Y axis. The two axes are crossed to form all states. Either
    axis may be a scalar for a one-state sweep; values are stored as tuples.
    """

    aperture_label: str = "aperture"
    detector_labels: tuple = ("pupil", "eyebox")
    opening_height_deltas_mm: tuple = (-0.1, 0.0, 0.1)
    vertical_shifts_mm: tuple = (-0.1, 0.0, 0.1)
    report_name: str = DEFAULT_REPORT_NAME
    title: str = DEFAULT_REPORT_TITLE

    def __post_init__(self):
        _nonempty_text(self.aperture_label, "Aperture label")
        _safe_report_name(self.report_name)
        _nonempty_text(self.title, "Aperture report title")
        detector_labels = tuple(_unique_nonempty_texts(
            self.detector_labels, "Aperture report detector_labels"))
        opening_height_deltas_mm = _variation_axis(
            self.opening_height_deltas_mm,
            "Aperture opening_height_deltas_mm",
        )
        vertical_shifts_mm = _variation_axis(
            self.vertical_shifts_mm,
            "Aperture vertical_shifts_mm",
        )
        object.__setattr__(self, "detector_labels", detector_labels)
        object.__setattr__(self, "opening_height_deltas_mm", opening_height_deltas_mm)
        object.__setattr__(self, "vertical_shifts_mm", vertical_shifts_mm)

    @classmethod
    def from_mapping(cls, value):
        """Validate and construct a spec from public exporter input."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError("Aperture report specification must be a mapping.")
        try:
            return cls(**value)
        except TypeError as error:
            raise ValueError(f"Invalid aperture report specification: {error}") from error

    def as_dict(self):
        """Return the unresolved JSON-safe request."""
        return {
            "schema": "apollo14.zemax_aperture_report",
            "schema_version": SCHEMA_VERSION,
            "report_name": self.report_name,
            "title": self.title,
            "aperture_label": self.aperture_label,
            "detector_labels": list(self.detector_labels),
            "opening_height_deltas_mm": [
                float(value) for value in self.opening_height_deltas_mm
            ],
            "vertical_shifts_mm": [float(value) for value in self.vertical_shifts_mm],
        }


def resolve_aperture_report(specification, objects):
    """Resolve one public sweep request against exported prescription objects.

    The returned mapping contains exact object indices and nominal dimensions
    used by the Windows runner.  Resolving this during export catches a missing
    or incompatible aperture before an expensive OpticStudio trace starts.
    """
    if specification is None:
        return None
    spec = ApertureSweepReportSpec.from_mapping(specification)
    resolved = spec.as_dict()
    resolved["geometry"] = _resolve_aperture_geometry(spec, objects)
    return normalize_resolved_aperture_report(resolved)


def normalize_resolved_aperture_report(specification):
    """Validate a fully resolved prescription mapping used on the Zemax host."""
    if specification is None:
        return None
    if not isinstance(specification, dict):
        raise TypeError("Resolved aperture report specification must be a mapping.")
    spec = ApertureSweepReportSpec.from_mapping(
        {
            key: specification[key]
            for key in (
                "aperture_label",
                "detector_labels",
                "opening_height_deltas_mm",
                "vertical_shifts_mm",
                "report_name",
                "title",
            )
            if key in specification
        }
    )
    geometry = _normalize_geometry(specification.get("geometry"))
    normalized = spec.as_dict()
    normalized["geometry"] = geometry
    _validate_aperture_states(normalized)
    return normalized


def _resolve_aperture_geometry(spec, objects):
    objects = list(objects)
    matching = [
        entry
        for entry in objects
        if entry.get("type") == "boolean_native"
        and entry.get("label") == spec.aperture_label
    ]
    if len(matching) != 1:
        raise ValueError(
            f"Aperture report expected one Boolean Native object labelled "
            f"{spec.aperture_label!r}; found {len(matching)}."
        )
    boolean = matching[0]
    objects_by_index = {}
    for entry in objects:
        index = _positive_int(entry.get("index"), "Exported object index")
        if index in objects_by_index:
            raise ValueError(f"Duplicate exported object index {index}.")
        objects_by_index[index] = entry
    try:
        outer = objects_by_index[int(boolean["data"]["object_a"])]
        inner = objects_by_index[int(boolean["data"]["object_b"])]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Aperture {spec.aperture_label!r} has invalid Boolean Native operands."
        ) from error
    if outer.get("type") != "rectangular_volume" or inner.get("type") != "rectangular_volume":
        raise ValueError(
            f"Aperture {spec.aperture_label!r} must subtract one rectangular "
            "volume from another."
        )

    outer_dimensions = _rectangular_volume_dimensions(outer, "outer aperture volume")
    inner_dimensions = _rectangular_volume_dimensions(inner, "inner aperture volume")
    outer_position = _pose_vector(outer.get("position"), "outer aperture position")
    inner_position = _pose_vector(inner.get("position"), "inner aperture position")
    boolean_position = _pose_vector(boolean.get("position"), "Boolean aperture position")
    outer_tilt = _pose_vector(outer.get("tilt_deg"), "outer aperture tilt")
    inner_tilt = _pose_vector(inner.get("tilt_deg"), "inner aperture tilt")
    if not _vectors_close(outer_tilt, inner_tilt):
        raise ValueError("Aperture Boolean parents must share one local frame for a vertical sweep.")

    try:
        local_vertical = _unit_vector(
            boolean["zemax_local_axes_world"]["y"],
            "Boolean aperture local-Y world axis",
        )
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Aperture report requires the exported Boolean local-Y world axis."
        ) from error
    if abs(local_vertical[0]) > _FLOAT_TOLERANCE:
        raise ValueError(
            "A vertical-only aperture sweep requires local Y to have no global X component."
        )

    geometry = {
        "boolean_object_index": int(boolean["index"]),
        "outer_object_index": int(outer["index"]),
        "inner_object_index": int(inner["index"]),
        "horizontal_axis": "Boolean-parent local X",
        "vertical_axis": "Boolean-parent local Y",
        "outer_x_half_width_mm": outer_dimensions["x_half_width_mm"],
        "outer_y_half_width_mm": outer_dimensions["y_half_width_mm"],
        "inner_x_half_width_mm": inner_dimensions["x_half_width_mm"],
        "nominal_inner_y_half_width_mm": inner_dimensions["y_half_width_mm"],
        "outer_x_position_mm": outer_position[0],
        "outer_y_position_mm": outer_position[1],
        "nominal_inner_x_position_mm": inner_position[0],
        "nominal_inner_y_position_mm": inner_position[1],
        "nominal_aperture_position_mm": boolean_position,
        "local_vertical_world": local_vertical,
        "nominal_opening_width_mm": 2.0 * inner_dimensions["x_half_width_mm"],
        "nominal_opening_height_mm": 2.0 * inner_dimensions["y_half_width_mm"],
    }
    normalized = _normalize_geometry(geometry)
    _validate_horizontal_opening(normalized)
    return normalized


def _rectangular_volume_dimensions(entry, description):
    data = entry.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"{description} has no geometry data.")
    x1 = _positive_float(data.get("x1_half_width"), f"{description} X1 half width")
    x2 = _positive_float(data.get("x2_half_width"), f"{description} X2 half width")
    y1 = _positive_float(data.get("y1_half_width"), f"{description} Y1 half width")
    y2 = _positive_float(data.get("y2_half_width"), f"{description} Y2 half width")
    if not _close(x1, x2) or not _close(y1, y2):
        raise ValueError(
            f"{description} is tapered; the aperture sweep requires equal end half widths."
        )
    return {"x_half_width_mm": x1, "y_half_width_mm": y1}


def _normalize_geometry(value):
    if not isinstance(value, dict):
        raise ValueError("Resolved aperture report requires a geometry mapping.")
    result = {
        "boolean_object_index": _positive_int(
            value.get("boolean_object_index"), "Aperture Boolean object index"
        ),
        "outer_object_index": _positive_int(
            value.get("outer_object_index"), "Aperture outer object index"
        ),
        "inner_object_index": _positive_int(
            value.get("inner_object_index"), "Aperture inner object index"
        ),
        "horizontal_axis": _nonempty_text(
            value.get("horizontal_axis"), "Aperture horizontal axis"
        ),
        "vertical_axis": _nonempty_text(value.get("vertical_axis"), "Aperture vertical axis"),
    }
    for name in (
        "outer_x_half_width_mm",
        "outer_y_half_width_mm",
        "inner_x_half_width_mm",
        "nominal_inner_y_half_width_mm",
        "nominal_opening_width_mm",
        "nominal_opening_height_mm",
    ):
        result[name] = _positive_float(value.get(name), f"Aperture geometry {name}")
    for name in (
        "outer_x_position_mm",
        "outer_y_position_mm",
        "nominal_inner_x_position_mm",
        "nominal_inner_y_position_mm",
    ):
        result[name] = _finite_float(value.get(name), f"Aperture geometry {name}")
    result["nominal_aperture_position_mm"] = _pose_vector(
        value.get("nominal_aperture_position_mm"),
        "Aperture geometry nominal_aperture_position_mm",
    )
    result["local_vertical_world"] = _unit_vector(
        value.get("local_vertical_world"),
        "Aperture geometry local_vertical_world",
    )
    if abs(result["local_vertical_world"][0]) > _FLOAT_TOLERANCE:
        raise ValueError(
            "A vertical-only aperture sweep requires local Y to have no global X component."
        )
    if not _close(result["nominal_opening_width_mm"], 2.0 * result["inner_x_half_width_mm"]):
        raise ValueError("Aperture nominal opening width disagrees with its X half width.")
    if not _close(
        result["nominal_opening_height_mm"],
        2.0 * result["nominal_inner_y_half_width_mm"],
    ):
        raise ValueError("Aperture nominal opening height disagrees with its Y half width.")
    _validate_horizontal_opening(result)
    return result


def _validate_horizontal_opening(geometry):
    offset = geometry["nominal_inner_x_position_mm"] - geometry["outer_x_position_mm"]
    if abs(offset) + geometry["inner_x_half_width_mm"] > (
        geometry["outer_x_half_width_mm"] + _FLOAT_TOLERANCE
    ):
        raise ValueError("The nominal aperture opening extends outside its outer X bounds.")


def _validate_aperture_states(specification):
    geometry = specification["geometry"]
    outer_half_height = geometry["outer_y_half_width_mm"]
    outer_center = geometry["outer_y_position_mm"]
    nominal_inner_center = geometry["nominal_inner_y_position_mm"]
    nominal_inner_half_height = geometry["nominal_inner_y_half_width_mm"]
    for height_delta in specification["opening_height_deltas_mm"]:
        inner_half_height = nominal_inner_half_height + height_delta / 2.0
        if inner_half_height <= 0.0:
            raise ValueError(
                f"Aperture opening-height delta {height_delta:g} mm makes the opening empty."
            )
        center_offset = nominal_inner_center - outer_center
        if abs(center_offset) + inner_half_height > outer_half_height + _FLOAT_TOLERANCE:
            raise ValueError(
                f"Aperture opening-height delta {height_delta:g} mm extends outside "
                "the outer frame."
            )


class ApertureSweepReport:
    """Collect completed detector cubes for every configured aperture state."""

    def __init__(self, specification, pupil_report_specs):
        self.specification = normalize_resolved_aperture_report(specification)
        self._pupil_specs = {
            spec.detector_label: spec
            for spec in (PupilReportSpec.from_mapping(value) for value in pupil_report_specs)
        }
        missing = [
            label
            for label in self.specification["detector_labels"]
            if label not in self._pupil_specs
        ]
        if missing:
            raise ValueError(
                "Aperture report detectors need matching pupil-report configurations: "
                + ", ".join(repr(label) for label in missing)
                + "."
            )
        self._records = {}

    @property
    def detector_labels(self):
        return tuple(self.specification["detector_labels"])

    @property
    def states(self):
        """Iterate configured states in stable height-major order."""
        for height_index, height_delta in enumerate(
            self.specification["opening_height_deltas_mm"]
        ):
            for shift_index, shift in enumerate(self.specification["vertical_shifts_mm"]):
                yield height_index, shift_index, height_delta, shift

    def record_state(self, opening_height_delta_mm, vertical_shift_mm, cubes):
        """Reduce completed cubes into the compact arrays used by the report."""
        height_index = _axis_value_index(
            self.specification["opening_height_deltas_mm"],
            opening_height_delta_mm,
            "opening-height delta",
        )
        shift_index = _axis_value_index(
            self.specification["vertical_shifts_mm"],
            vertical_shift_mm,
            "vertical shift",
        )
        state_key = (height_index, shift_index)
        if state_key in self._records:
            raise ValueError(f"Aperture state {state_key} was recorded more than once.")

        cubes_by_label = {cube.detector_label: cube for cube in cubes}
        missing = [label for label in self.detector_labels if label not in cubes_by_label]
        if missing:
            raise ValueError(
                "Aperture state is missing detector cubes: "
                + ", ".join(repr(label) for label in missing)
                + "."
            )
        detector_records = []
        for label in self.detector_labels:
            cube = cubes_by_label[label]
            pupil_report = calculate_pupil_report(cube, self._pupil_specs[label])
            detector_records.append(_reduce_pupil_report(pupil_report))
        self._records[state_key] = {
            "height_index": height_index,
            "shift_index": shift_index,
            "opening_height_delta_mm": float(opening_height_delta_mm),
            "vertical_shift_mm": float(vertical_shift_mm),
            "opening_height_mm": (
                self.specification["geometry"]["nominal_opening_height_mm"]
                + float(opening_height_delta_mm)
            ),
            "aperture_position_mm": _shifted_aperture_position(
                self.specification["geometry"], vertical_shift_mm
            ),
            "nominal": _close(opening_height_delta_mm, 0.0)
            and _close(vertical_shift_mm, 0.0),
            "detectors": detector_records,
        }

    def write(self, output_directory):
        """Write HTML, JSON and CSV artifacts and return an index record."""
        report = self._build_report()
        output_directory = Path(output_directory)
        report_directory = (
            output_directory / "reports" / self.specification["report_name"]
        )
        report_directory.mkdir(parents=True, exist_ok=True)
        html_path = report_directory / "index.html"
        json_path = report_directory / "report.json"
        state_csv_path = report_directory / "state_summary.csv"
        field_csv_path = report_directory / "field_summary.csv"
        report["files"] = {
            "html": html_path.name,
            "json": json_path.name,
            "state_summary_csv": state_csv_path.name,
            "field_summary_csv": field_csv_path.name,
        }
        json_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _write_state_summary_csv(state_csv_path, report)
        _write_field_summary_csv(field_csv_path, report)
        _write_html(html_path, report)
        return {
            "report_name": self.specification["report_name"],
            "title": self.specification["title"],
            "html": html_path.relative_to(output_directory).as_posix(),
            "json": json_path.relative_to(output_directory).as_posix(),
            "state_summary_csv": state_csv_path.relative_to(output_directory).as_posix(),
            "field_summary_csv": field_csv_path.relative_to(output_directory).as_posix(),
            "detail": (
                f"{len(self.specification['opening_height_deltas_mm'])} opening heights × "
                f"{len(self.specification['vertical_shifts_mm'])} vertical shifts"
            ),
        }

    def _build_report(self):
        expected_keys = {
            (height_index, shift_index)
            for height_index in range(len(self.specification["opening_height_deltas_mm"]))
            for shift_index in range(len(self.specification["vertical_shifts_mm"]))
        }
        missing = sorted(expected_keys - set(self._records))
        if missing:
            raise RuntimeError(
                "Aperture response is incomplete; missing state indices "
                + ", ".join(str(value) for value in missing)
                + "."
            )
        ordered_states = [
            self._records[(height_index, shift_index)]
            for height_index in range(len(self.specification["opening_height_deltas_mm"]))
            for shift_index in range(len(self.specification["vertical_shifts_mm"]))
        ]
        detectors = [
            _assemble_detector_report(label, ordered_states)
            for label in self.detector_labels
        ]
        fov = detectors[0]["fov"]
        return {
            "schema": "apollo14.zemax_aperture_report",
            "schema_version": SCHEMA_VERSION,
            "title": self.specification["title"],
            "configuration": self.specification,
            "provenance": {
                "calculation_engine": "Zemax OpticStudio non-sequential ray trace",
                "detector_readout": "NCE.GetDetectorData(detector, pixel, data=0)",
                "spectral_mode": "polychromatic_system_wavelengths",
                "source_wavenumber": 0,
                "apollo14_reference_values_used": False,
                "nominal_state_reused_from_primary_sweep": True,
            },
            "quantity": {
                "detector_total_flux_unit": "OpticStudio system source-power units",
                "eye_window_flux_density_unit": (
                    "OpticStudio system source-power units/mm^2"
                ),
                "fov_mean": (
                    "Unweighted arithmetic mean across independently traced FOV samples."
                ),
                "fov_worst": "Minimum across independently traced FOV samples.",
            },
            "fov": fov,
            "detectors": detectors,
            "states": ordered_states,
        }


def _reduce_pupil_report(report):
    arrays = report["arrays"]
    field_rows = report["field_summary"]
    detector_totals = [row["detector_total_flux"] for row in field_rows]
    eye_means = [row["eye_position_mean"] for row in field_rows]
    eye_worsts = [row["eye_position_minimum"] for row in field_rows]
    return {
        "label": report["detector"]["label"],
        "object_index": report["detector"]["object_index"],
        "fov": report["fov"],
        "eye_positions": report["eye_positions"],
        "eye_window": {
            "width_mm": report["configuration"]["eye_window_width_mm"],
            "height_mm": report["configuration"]["eye_window_height_mm"],
            "coordinate_frame": report["configuration"]["coordinate_frame"],
        },
        "arrays": {
            "fov_detector_total_flux": arrays["fov_detector_total_flux"],
            "fov_mean_eye_window_flux_density": arrays["fov_mean_over_eye_positions"],
            "fov_worst_eye_window_flux_density": arrays["fov_worst_over_eye_positions"],
            "eye_mean_over_fov_flux_density": arrays["pupil_mean_over_fov"],
            "eye_worst_over_fov_flux_density": arrays["pupil_worst_over_fov"],
        },
        "summary": {
            "detector_total_flux_mean_over_fov": _mean(detector_totals),
            "detector_total_flux_worst_fov": min(detector_totals),
            "detector_total_flux_best_fov": max(detector_totals),
            "eye_window_flux_density_mean": _mean(eye_means),
            "eye_window_flux_density_worst": min(eye_worsts),
        },
        "field_summary": [
            {
                key: row[key]
                for key in (
                    "field_index",
                    "source_label",
                    "source_object_index",
                    "fov_x_deg",
                    "fov_y_deg",
                    "detector_total_flux",
                    "eye_position_mean",
                    "eye_position_minimum",
                    "eye_position_maximum",
                )
            }
            for row in field_rows
        ],
    }


def _assemble_detector_report(label, states):
    detector_states = []
    for state in states:
        matches = [record for record in state["detectors"] if record["label"] == label]
        if len(matches) != 1:
            raise RuntimeError(
                f"Aperture state ({state['height_index']}, {state['shift_index']}) "
                f"does not contain exactly one detector {label!r}."
            )
        detector_states.append(matches[0])
    first = detector_states[0]
    metric_names = tuple(first["arrays"])
    scales = {}
    for metric_name in metric_names:
        values = []
        for record in detector_states:
            values.extend(_flatten_matrix(record["arrays"][metric_name]))
        scales[metric_name] = nice_scale(min(values), max(values), include_zero=True)

    height_count = 1 + max(state["height_index"] for state in states)
    shift_count = 1 + max(state["shift_index"] for state in states)
    state_mean = [[None for _ in range(height_count)] for _ in range(shift_count)]
    state_worst = [[None for _ in range(height_count)] for _ in range(shift_count)]
    for state_index, state in enumerate(states):
        detector = detector_states[state_index]
        state_mean[state["shift_index"]][state["height_index"]] = detector["summary"][
            "detector_total_flux_mean_over_fov"
        ]
        state_worst[state["shift_index"]][state["height_index"]] = detector["summary"][
            "detector_total_flux_worst_fov"
        ]
    state_values = _flatten_matrix(state_mean) + _flatten_matrix(state_worst)
    scales["state_detector_total_flux"] = nice_scale(
        min(state_values), max(state_values), include_zero=True
    )
    return {
        "label": label,
        "object_index": first["object_index"],
        "fov": first["fov"],
        "eye_positions": first["eye_positions"],
        "eye_window": first["eye_window"],
        "states": detector_states,
        "state_summary": {
            "mean_detector_total_flux_over_fov": state_mean,
            "worst_fov_detector_total_flux": state_worst,
        },
        "display_scales": scales,
    }


def _write_state_summary_csv(path, report):
    fieldnames = [
        "detector_label",
        "height_index",
        "shift_index",
        "opening_height_delta_mm",
        "opening_height_mm",
        "vertical_shift_mm",
        "aperture_position_x_mm",
        "aperture_position_y_mm",
        "aperture_position_z_mm",
        "nominal",
        "detector_total_flux_mean_over_fov",
        "detector_total_flux_worst_fov",
        "detector_total_flux_best_fov",
        "eye_window_flux_density_mean",
        "eye_window_flux_density_worst",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for state in report["states"]:
            for detector in state["detectors"]:
                writer.writerow(
                    {
                        "detector_label": detector["label"],
                        **{
                            key: state[key]
                            for key in (
                                "height_index",
                                "shift_index",
                                "opening_height_delta_mm",
                                "opening_height_mm",
                                "vertical_shift_mm",
                                "nominal",
                            )
                        },
                        "aperture_position_x_mm": state["aperture_position_mm"][0],
                        "aperture_position_y_mm": state["aperture_position_mm"][1],
                        "aperture_position_z_mm": state["aperture_position_mm"][2],
                        **detector["summary"],
                    }
                )


def _write_field_summary_csv(path, report):
    fieldnames = [
        "detector_label",
        "height_index",
        "shift_index",
        "opening_height_delta_mm",
        "opening_height_mm",
        "vertical_shift_mm",
        "field_index",
        "source_label",
        "source_object_index",
        "fov_x_deg",
        "fov_y_deg",
        "detector_total_flux",
        "eye_position_mean",
        "eye_position_minimum",
        "eye_position_maximum",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for state in report["states"]:
            state_columns = {
                key: state[key]
                for key in (
                    "height_index",
                    "shift_index",
                    "opening_height_delta_mm",
                    "opening_height_mm",
                    "vertical_shift_mm",
                )
            }
            for detector in state["detectors"]:
                for field in detector["field_summary"]:
                    writer.writerow(
                        {
                            "detector_label": detector["label"],
                            **state_columns,
                            **field,
                        }
                    )


def _write_html(path, report):
    payload = json.dumps(report, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    page = _HTML_TEMPLATE.replace("__REPORT_LAYOUT_VERSION__", REPORT_LAYOUT_VERSION).replace(
        "__REPORT_DATA__", payload
    )
    Path(path).write_text(page, encoding="utf-8")


def _variation_axis(values, description):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{description} must be a numeric sequence.")
    try:
        values = list(values)
    except TypeError:
        values = [values]
    if not values:
        raise ValueError(f"{description} must not be empty.")
    normalized = [_finite_float(value, description) for value in values]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{description} values must be unique.")
    if normalized != sorted(normalized):
        raise ValueError(f"{description} must be sorted from low to high.")
    if not any(_close(value, 0.0) for value in normalized):
        raise ValueError(f"{description} must include the nominal 0 mm state.")
    return tuple(normalized)


def _unique_nonempty_texts(values, description):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{description} must be a sequence of labels.")
    labels = [_nonempty_text(value, description) for value in values]
    if not labels:
        raise ValueError(f"{description} must not be empty.")
    if len({label.casefold() for label in labels}) != len(labels):
        raise ValueError(f"{description} must contain unique labels.")
    return labels


def _safe_report_name(value):
    value = _nonempty_text(value, "Aperture report_name")
    if _SAFE_REPORT_NAME.fullmatch(value) is None:
        raise ValueError(
            "Aperture report_name may contain only letters, digits, dot, underscore, and hyphen."
        )
    if value.casefold() in _RESERVED_REPORT_NAMES:
        raise ValueError(f"Aperture report_name {value!r} is reserved.")
    return value


def _axis_value_index(axis, value, description):
    value = _finite_float(value, description)
    matches = [index for index, candidate in enumerate(axis) if _close(candidate, value)]
    if len(matches) != 1:
        raise ValueError(f"{description} {value:g} mm is not a configured aperture state.")
    return matches[0]


def _shifted_aperture_position(geometry, vertical_shift_mm):
    shift = _finite_float(vertical_shift_mm, "Aperture vertical shift")
    nominal = geometry["nominal_aperture_position_mm"]
    direction = geometry["local_vertical_world"]
    return [nominal[index] + shift * direction[index] for index in range(3)]


def _pose_vector(value, description):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{description} must contain exactly three values.")
    return [_finite_float(component, description) for component in value]


def _unit_vector(value, description):
    vector = _pose_vector(value, description)
    norm = math.sqrt(math.fsum(component * component for component in vector))
    if norm <= 0.0:
        raise ValueError(f"{description} must be non-zero.")
    normalized = [component / norm for component in vector]
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError(f"{description} must be a unit vector.")
    return normalized


def _vectors_close(left, right):
    return len(left) == len(right) and all(
        _close(left[index], right[index]) for index in range(len(left))
    )


def _flatten_matrix(matrix):
    return [float(value) for row in matrix for value in row if value is not None]


def _mean(values):
    return math.fsum(values) / len(values)


def _close(left, right):
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=_FLOAT_TOLERANCE)


def _finite_float(value, description):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must be a finite number.") from error
    if not math.isfinite(result):
        raise ValueError(f"{description} must be a finite number.")
    return result


def _positive_float(value, description):
    result = _finite_float(value, description)
    if result <= 0.0:
        raise ValueError(f"{description} must be positive.")
    return result


def _positive_int(value, description):
    if isinstance(value, bool):
        raise ValueError(f"{description} must be a positive integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{description} must be a positive integer.") from error
    if result <= 0 or result != value:
        raise ValueError(f"{description} must be a positive integer.")
    return result


def _nonempty_text(value, description):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be non-empty text.")
    return value.strip()


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en" data-apollo14-report-layout="__REPORT_LAYOUT_VERSION__">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<title>Zemax aperture sweep</title>
<style>
:root { color-scheme:dark; font-family:Inter,Segoe UI,system-ui,sans-serif;
  background:#0b1020; color:#e8edf8; }
* { box-sizing:border-box; }
body { margin:0; background:linear-gradient(140deg,#111a32,#080b14 60%); }
header,main { max-width:1500px; margin:auto; padding:22px; }
h1 { margin:.1rem 0 .35rem; font-size:clamp(1.6rem,3vw,2.6rem); }
h2 { margin:.1rem 0 .8rem; font-size:1.15rem; }
p { color:#b7c2d9; line-height:1.5; }
.tag { display:inline-block; background:#163a59; color:#8cddff; border:1px solid #245f83;
  border-radius:999px; padding:5px 10px; margin:4px 7px 4px 0; font-size:.82rem; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:16px; }
.card { background:#121a2cdd; border:1px solid #27334a; border-radius:13px;
  padding:16px; box-shadow:0 10px 28px #0005; overflow:auto; }
.wide { grid-column:1/-1; }
.controls { display:flex; gap:14px; flex-wrap:wrap; align-items:end; margin-bottom:10px; }
label { display:grid; gap:5px; color:#aab8d1; font-size:.84rem; }
select { color:#e8edf8; background:#0b1222; border:1px solid #364560;
  border-radius:7px; padding:7px 10px; min-width:180px; }
canvas { width:100%; max-width:650px; height:auto; display:block; margin:auto; }
.stats { display:flex; flex-wrap:wrap; gap:9px; margin:10px 0; }
.stat { background:#0a1120; border:1px solid #25324a; border-radius:8px; padding:8px 10px; }
.stat b { display:block; color:#fff; font-variant-numeric:tabular-nums; }
.unit-label { margin:.55rem 0 .35rem; color:#d8e5f7; font-size:.84rem;
  font-weight:600; text-align:center; }
.caption { min-height:3.5em; color:#91a1bc; font-size:.84rem; line-height:1.45; }
.links a { color:#79d6ff; margin-right:14px; }
code { color:#b7e8ff; }
@media(max-width:500px){ header,main{padding:13px}.grid{grid-template-columns:1fr}.card{padding:11px} }
</style>
</head>
<body>
<header>
  <span class="tag">Calculated on the Zemax computer</span>
  <span class="tag" id="stateCountTag">Vertical aperture states</span>
  <span class="tag">Horizontal opening fixed</span>
  <h1 id="reportTitle">Zemax aperture sweep</h1>
  <p id="subtitle"></p>
  <p class="links"><a id="jsonLink">JSON data</a><a id="stateCsvLink">State CSV</a>
    <a id="fieldCsvLink">Field CSV</a></p>
</header>
<main class="grid">
  <section class="card wide">
    <h2>Selected aperture state</h2>
    <div class="controls">
      <label>Detector<select id="detectorSelect"></select></label>
      <label>Vertical opening-height change<select id="heightSelect"></select></label>
      <label>Vertical aperture shift<select id="shiftSelect"></select></label>
      <label>Displayed result<select id="metricSelect"></select></label>
    </div>
    <div class="stats" id="selectedStats"></div>
    <canvas id="selectedPlot" width="680" height="470"></canvas>
    <div class="unit-label" id="selectedUnit"></div>
    <div class="caption" id="selectedCaption"></div>
  </section>
  <section class="card">
    <h2>Mean total detector flux over FOV</h2>
    <canvas id="stateMean" width="620" height="450"></canvas>
    <div class="unit-label">Color scale units: total detector flux in source-power units.</div>
    <div class="caption">Each cell is one selectable aperture state. Arithmetic mean over all exported FOV samples. Common scale with the worst-FOV plot.</div>
  </section>
  <section class="card">
    <h2>Worst-FOV total detector flux</h2>
    <canvas id="stateWorst" width="620" height="450"></canvas>
    <div class="unit-label">Color scale units: total detector flux in source-power units.</div>
    <div class="caption">Minimum total detector flux across exported FOV samples. It shares the mean plot's color scale; equal colors are equal Zemax values.</div>
  </section>
  <section class="card wide"><h2>Geometry and calculation</h2><p id="method"></p></section>
</main>
<script type="application/json" id="reportData">__REPORT_DATA__</script>
<script>
const report = JSON.parse(document.getElementById('reportData').textContent);
const configuration = report.configuration;
const geometry = configuration.geometry;
const COLORS = [[8,20,75],[19,99,164],[31,190,183],[250,220,80],[220,45,34]];
const CANVAS_BACKGROUND = '#0a1120';
const AXIS_COLOR = '#70809c';
const TEXT_COLOR = '#b8c5dc';
const NULL_COLOR = '#303747';
const HEATMAP_MARGIN = {left:78,right:112,top:25,bottom:70};
const METRICS = [
  {key:'fov_detector_total_flux', label:'Total detector flux by FOV', axes:'fov',
    unit:'Color scale units: total detector flux in source-power units.'},
  {key:'fov_mean_eye_window_flux_density', label:'Mean eye-window response by FOV', axes:'fov',
    unit:'Color scale units: mean eye-window detector flux density in source-power units/mm².'},
  {key:'fov_worst_eye_window_flux_density', label:'Worst eye-window response by FOV', axes:'fov',
    unit:'Color scale units: worst eye-window detector flux density in source-power units/mm².'},
  {key:'eye_mean_over_fov_flux_density', label:'Mean over FOV by eye position', axes:'eye',
    unit:'Color scale units: mean detector flux density in source-power units/mm².'},
  {key:'eye_worst_over_fov_flux_density', label:'Worst FOV by eye position', axes:'eye',
    unit:'Color scale units: worst-FOV detector flux density in source-power units/mm².'},
];

function formatNumber(value) {
  const number = Number(value);
  const magnitude = Math.abs(number);
  if (magnitude === 0) return '0';
  if (magnitude >= 10000 || magnitude < 0.001) return number.toExponential(2).replace('e+','e');
  return Number(number.toPrecision(3)).toString();
}
function formatAxisNumber(value) {
  const number = Math.abs(Number(value)) < 0.0005 ? 0 : Number(value);
  return Number(number.toFixed(3)).toString();
}
function formatDelta(value) {
  const number = Number(value);
  if (Math.abs(number) < 1e-12) return '0 mm (nominal)';
  return `${number > 0 ? '+' : ''}${formatAxisNumber(number)} mm`;
}
function option(value,text) {
  const element = document.createElement('option');
  element.value = value;
  element.textContent = text;
  return element;
}
function selectedDetector() {
  return report.detectors[Number(document.getElementById('detectorSelect').value)];
}
function selectedStateIndex() {
  const heightIndex = Number(document.getElementById('heightSelect').value);
  const shiftIndex = Number(document.getElementById('shiftSelect').value);
  return {
    heightIndex,
    shiftIndex,
    flatIndex: heightIndex * configuration.vertical_shifts_mm.length + shiftIndex,
  };
}
function selectedState() {
  const selection = selectedStateIndex();
  return {
    geometry: report.states[selection.flatIndex],
    detector: selectedDetector().states[selection.flatIndex],
    ...selection,
  };
}
function setup() {
  document.title = report.title;
  document.getElementById('reportTitle').textContent = report.title;
  document.getElementById('stateCountTag').textContent =
    `${configuration.opening_height_deltas_mm.length * configuration.vertical_shifts_mm.length} ` +
    `vertical aperture states`;
  document.getElementById('subtitle').textContent =
    `${configuration.opening_height_deltas_mm.length} opening heights × ` +
    `${configuration.vertical_shifts_mm.length} vertical shifts × ` +
    `${report.fov.x_deg.length}×${report.fov.y_deg.length} FOV samples. ` +
    `Nominal opening ${formatAxisNumber(geometry.nominal_opening_width_mm)}×` +
    `${formatAxisNumber(geometry.nominal_opening_height_mm)} mm.`;
  document.getElementById('jsonLink').href = report.files.json;
  document.getElementById('stateCsvLink').href = report.files.state_summary_csv;
  document.getElementById('fieldCsvLink').href = report.files.field_summary_csv;

  const detectorSelect = document.getElementById('detectorSelect');
  report.detectors.forEach((detector,index) => detectorSelect.appendChild(option(index,detector.label)));
  const heightSelect = document.getElementById('heightSelect');
  configuration.opening_height_deltas_mm.forEach((value,index) => {
    const actual = geometry.nominal_opening_height_mm + value;
    heightSelect.appendChild(option(index,`${formatDelta(value)} → ${formatAxisNumber(actual)} mm`));
  });
  const shiftSelect = document.getElementById('shiftSelect');
  configuration.vertical_shifts_mm.forEach((value,index) => shiftSelect.appendChild(option(index,formatDelta(value))));
  const metricSelect = document.getElementById('metricSelect');
  METRICS.forEach((metric,index) => metricSelect.appendChild(option(index,metric.label)));
  heightSelect.value = configuration.opening_height_deltas_mm.findIndex(value => Math.abs(value) < 1e-12);
  shiftSelect.value = configuration.vertical_shifts_mm.findIndex(value => Math.abs(value) < 1e-12);
  [detectorSelect,heightSelect,shiftSelect,metricSelect].forEach(element => element.addEventListener('change',redraw));
  writeMethod();
  redraw();
}
function redraw() {
  const detector = selectedDetector();
  const state = selectedState();
  const metric = METRICS[Number(document.getElementById('metricSelect').value)];
  const axes = metric.axes === 'fov'
    ? {x:detector.fov.x_deg,y:detector.fov.y_deg,xLabel:'FOV x (deg)',yLabel:'FOV y (deg)'}
    : {x:detector.eye_positions.x_mm,y:detector.eye_positions.y_mm,xLabel:'eye x (mm)',yLabel:'eye y (mm)'};
  drawHeatmap({id:'selectedPlot',matrix:state.detector.arrays[metric.key],
    xValues:axes.x,yValues:axes.y,xLabel:axes.xLabel,yLabel:axes.yLabel,
    scale:detector.display_scales[metric.key]});
  document.getElementById('selectedUnit').textContent = metric.unit;
  const summary = state.detector.summary;
  document.getElementById('selectedStats').innerHTML = [
    ['opening height',`${formatAxisNumber(state.geometry.opening_height_mm)} mm`],
    ['vertical shift',formatDelta(state.geometry.vertical_shift_mm)],
    ['mean total flux',formatNumber(summary.detector_total_flux_mean_over_fov)],
    ['worst FOV total',formatNumber(summary.detector_total_flux_worst_fov)],
    ['mean eye response',formatNumber(summary.eye_window_flux_density_mean)],
    ['worst eye response',formatNumber(summary.eye_window_flux_density_worst)],
  ].map(([label,value]) => `<span class="stat">${label}<b>${value}</b></span>`).join('');
  document.getElementById('selectedCaption').textContent =
    `${metric.label} for detector ${detector.label}. The scale is common to all configured aperture ` +
    `states for this detector and result, so changing a state does not rescale its colors.`;
  const highlight = {column:state.heightIndex,row:state.shiftIndex};
  drawHeatmap({id:'stateMean',matrix:detector.state_summary.mean_detector_total_flux_over_fov,
    xValues:configuration.opening_height_deltas_mm,yValues:configuration.vertical_shifts_mm,
    xLabel:'opening-height change (mm)',yLabel:'vertical shift (mm)',
    scale:detector.display_scales.state_detector_total_flux,highlight});
  drawHeatmap({id:'stateWorst',matrix:detector.state_summary.worst_fov_detector_total_flux,
    xValues:configuration.opening_height_deltas_mm,yValues:configuration.vertical_shifts_mm,
    xLabel:'opening-height change (mm)',yLabel:'vertical shift (mm)',
    scale:detector.display_scales.state_detector_total_flux,highlight});
}
function writeMethod() {
  document.getElementById('method').innerHTML =
    `The clear opening is Boolean object B (row ${geometry.inner_object_index}) subtracted ` +
    `from object A (row ${geometry.outer_object_index}). Opening height changes write only ` +
    `B's two local-Y half widths; a +0.1 mm full-height change adds +0.05 mm to each one. ` +
    `Vertical shift moves the complete Boolean aperture (row ${geometry.boolean_object_index}) ` +
    `along its local-Y world direction [${geometry.local_vertical_world.map(formatAxisNumber).join(', ')}]. ` +
    `The local-X half width remains ` +
    `<code>${formatAxisNumber(geometry.inner_x_half_width_mm)} mm</code> and its X position ` +
    `remains <code>${formatAxisNumber(geometry.nominal_inner_x_position_mm)} mm</code>; the ` +
    `Boolean object's global X position also remains ` +
    `<code>${formatAxisNumber(geometry.nominal_aperture_position_mm[0])} mm</code> for every ` +
    `state. Each non-nominal state is traced in OpticStudio with the same polychromatic System ` +
    `Wavelengths setup and source powers as the main sweep; the nominal state reuses that main ` +
    `sweep. Detector values come from <code>${report.provenance.detector_readout}</code>. The ` +
    `runner restores the nominal opening height and Boolean aperture position in memory when ` +
    `the sweep ends.`;
}
function colorForFraction(fraction) {
  const bounded = Math.max(0,Math.min(1,fraction));
  const position = bounded*(COLORS.length-1);
  const lowerIndex = Math.min(COLORS.length-2,Math.floor(position));
  const amount = position-lowerIndex;
  const channels = COLORS[lowerIndex].map((value,channel) => Math.round(value+(COLORS[lowerIndex+1][channel]-value)*amount));
  return `rgb(${channels.join(',')})`;
}
function normalizedValue(value,scale) {
  const span = scale.maximum-scale.minimum;
  return span === 0 ? 0 : (value-scale.minimum)/span;
}
function drawHeatmap(options) {
  const canvas = document.getElementById(options.id);
  const context = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  const plotWidth = width-HEATMAP_MARGIN.left-HEATMAP_MARGIN.right;
  const plotHeight = height-HEATMAP_MARGIN.top-HEATMAP_MARGIN.bottom;
  const rows = options.matrix.length;
  const columns = options.matrix[0].length;
  context.clearRect(0,0,width,height);
  context.fillStyle = CANVAS_BACKGROUND;
  context.fillRect(0,0,width,height);
  for (let row=0; row<rows; row+=1) {
    for (let column=0; column<columns; column+=1) {
      const value = options.matrix[row][column];
      const displayRow = rows-1-row;
      context.fillStyle = value === null ? NULL_COLOR : colorForFraction(normalizedValue(value,options.scale));
      context.fillRect(HEATMAP_MARGIN.left+column*plotWidth/columns,
        HEATMAP_MARGIN.top+displayRow*plotHeight/rows,
        plotWidth/columns+.5,plotHeight/rows+.5);
    }
  }
  if (options.highlight) {
    const displayRow = rows-1-options.highlight.row;
    context.strokeStyle = '#fff';
    context.lineWidth = 3;
    context.strokeRect(HEATMAP_MARGIN.left+options.highlight.column*plotWidth/columns+1.5,
      HEATMAP_MARGIN.top+displayRow*plotHeight/rows+1.5,
      plotWidth/columns-3,plotHeight/rows-3);
    context.lineWidth = 1;
  }
  drawHeatmapAxes(context,width,height,options);
  drawColorScale(context,width,options.scale);
}
function drawHeatmapAxes(context,width,height,options) {
  const plotWidth = width-HEATMAP_MARGIN.left-HEATMAP_MARGIN.right;
  const plotHeight = height-HEATMAP_MARGIN.top-HEATMAP_MARGIN.bottom;
  context.strokeStyle = AXIS_COLOR;
  context.strokeRect(HEATMAP_MARGIN.left,HEATMAP_MARGIN.top,plotWidth,plotHeight);
  context.fillStyle = TEXT_COLOR;
  context.font = '12px Segoe UI';
  context.textBaseline = 'middle';
  context.textAlign = 'center';
  sampledIndices(options.xValues.length).forEach(index => {
    const x = HEATMAP_MARGIN.left+(index+.5)*plotWidth/options.xValues.length;
    context.fillText(formatAxisNumber(options.xValues[index]),x,height-HEATMAP_MARGIN.bottom+19);
  });
  context.fillText(options.xLabel,HEATMAP_MARGIN.left+plotWidth/2,height-12);
  context.textAlign = 'right';
  sampledIndices(options.yValues.length).forEach(index => {
    const y = HEATMAP_MARGIN.top+(options.yValues.length-index-.5)*plotHeight/options.yValues.length;
    context.fillText(formatAxisNumber(options.yValues[index]),HEATMAP_MARGIN.left-8,y);
  });
  drawRotatedLabel(context,15,HEATMAP_MARGIN.top+plotHeight/2,options.yLabel);
}
function sampledIndices(count,maximumLabels=6) {
  if (count <= maximumLabels) return Array.from({length:count},(_,index)=>index);
  return [...new Set(Array.from({length:maximumLabels},(_,index)=>Math.round(index*(count-1)/(maximumLabels-1))))];
}
function drawColorScale(context,canvasWidth,scale) {
  const plotHeight = context.canvas.height-HEATMAP_MARGIN.top-HEATMAP_MARGIN.bottom;
  const barX = canvasWidth-HEATMAP_MARGIN.right+22;
  const barWidth = 17;
  const sampleCount = 160;
  const sampleHeight = plotHeight/sampleCount;
  for (let index=0; index<sampleCount; index+=1) {
    const fraction = index/(sampleCount-1);
    const y = HEATMAP_MARGIN.top+plotHeight-(index+1)*sampleHeight;
    context.fillStyle = colorForFraction(fraction);
    context.fillRect(barX,y,barWidth,sampleHeight+1);
  }
  context.strokeStyle = AXIS_COLOR;
  context.strokeRect(barX,HEATMAP_MARGIN.top,barWidth,plotHeight);
  context.fillStyle = TEXT_COLOR;
  context.font = '12px Segoe UI';
  context.textAlign = 'left';
  context.textBaseline = 'middle';
  scale.ticks.forEach(value => {
    const fraction = normalizedValue(value,scale);
    const y = scale.maximum === scale.minimum ? HEATMAP_MARGIN.top+plotHeight/2 : HEATMAP_MARGIN.top+(1-fraction)*plotHeight;
    context.beginPath();
    context.moveTo(barX+barWidth,y);
    context.lineTo(barX+barWidth+5,y);
    context.stroke();
    context.fillText(formatNumber(value),barX+barWidth+9,y);
  });
}
function drawRotatedLabel(context,x,y,label) {
  context.save();
  context.translate(x,y);
  context.rotate(-Math.PI/2);
  context.fillStyle = TEXT_COLOR;
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(label,0,0);
  context.restore();
}
setup();
</script>
</body>
</html>
"""
