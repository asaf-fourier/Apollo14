"""Portable Zemax-derived pupil and FOV report generation.

This module is copied into every Zemax bundle.  It intentionally depends only
on the Python standard library and the small public interface exposed by
``PositionSpaceCube``.  OpticStudio supplies flux per detector pixel; this
module aggregates that flux into physical eye windows and writes a
self-contained HTML report plus JSON and CSV data.

The reported eye-window quantity is average detector flux density::

    sum(pixel_flux * fractional_pixel_overlap) / eye_window_area_mm2

It is expressed in OpticStudio source-power units/mm².  It is not luminance or
radiance because the polychromatic detector result has no calibrated source
radiometry or angular measure.
"""

import csv
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

try:
    from .html_plotting import REPORT_LAYOUT_VERSION, nice_scale
    from .zemax_angular_response import write_angular_response_report
except ImportError:  # Standalone copy in a generated Zemax bundle.
    from html_plotting import (  # type: ignore[no-redef]
        REPORT_LAYOUT_VERSION,
        nice_scale,
    )
    from zemax_angular_response import (  # type: ignore[no-redef]
        write_angular_response_report,
    )

SCHEMA_VERSION = 3
_SAFE_REPORT_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_RESERVED_REPORT_NAMES = {"position_space_index"}
_AXIS_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class PupilReportSpec:
    """Configuration for one Zemax-derived pupil report.

    Eye-position centers are distributed uniformly through the configured
    eyebox.  ``eye_window_*`` gives the physical integration rectangle around
    each center.  Coordinates use the detector's Apollo14 local frame by
    default; the exported axis metadata supplies the signed permutation into
    the Zemax Detector Rectangle frame. ``export_angular_response`` adds a
    dedicated viewer and exact CSV slices with one eye position fixed while
    FOV x and FOV y vary.
    """

    detector_label: str
    eyebox_width_mm: float
    eyebox_height_mm: float
    eye_positions_x: int
    eye_positions_y: int
    eye_window_width_mm: float
    eye_window_height_mm: float
    report_name: str = "zemax_pupil_report"
    eyebox_center_x_mm: float = 0.0
    eyebox_center_y_mm: float = 0.0
    coordinate_frame: str = "apollo14"
    export_angular_response: bool = False

    def __post_init__(self):
        if not isinstance(self.detector_label, str) or not self.detector_label:
            raise ValueError("Pupil report detector_label must be a non-empty string.")
        if (
            not isinstance(self.report_name, str)
            or _SAFE_REPORT_NAME.fullmatch(self.report_name) is None
        ):
            raise ValueError(
                "Pupil report_name may contain only letters, digits, dot, underscore, and hyphen."
            )
        if self.report_name.casefold() in _RESERVED_REPORT_NAMES:
            raise ValueError(
                f"Pupil report_name {self.report_name!r} is reserved by the "
                "position-space report index."
            )
        if self.coordinate_frame not in {"apollo14", "zemax"}:
            raise ValueError("Pupil report coordinate_frame must be 'apollo14' or 'zemax'.")
        if not isinstance(self.export_angular_response, bool):
            raise ValueError("Pupil report export_angular_response must be a boolean.")
        for name in (
            "eyebox_width_mm",
            "eyebox_height_mm",
            "eye_window_width_mm",
            "eye_window_height_mm",
        ):
            _positive_float(getattr(self, name), name)
        for name in ("eyebox_center_x_mm", "eyebox_center_y_mm"):
            _finite_float(getattr(self, name), name)
        for name in ("eye_positions_x", "eye_positions_y"):
            _positive_int(getattr(self, name), name)

    @classmethod
    def from_mapping(cls, value):
        """Validate and construct a spec from prescription JSON data."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError("A pupil report specification must be a mapping.")
        try:
            return cls(**value)
        except TypeError as error:
            raise ValueError(f"Invalid pupil report specification: {error}") from error

    def as_dict(self):
        """Return a JSON-serializable prescription representation."""
        return {
            "detector_label": self.detector_label,
            "eyebox_width_mm": float(self.eyebox_width_mm),
            "eyebox_height_mm": float(self.eyebox_height_mm),
            "eye_positions_x": int(self.eye_positions_x),
            "eye_positions_y": int(self.eye_positions_y),
            "eye_window_width_mm": float(self.eye_window_width_mm),
            "eye_window_height_mm": float(self.eye_window_height_mm),
            "report_name": self.report_name,
            "eyebox_center_x_mm": float(self.eyebox_center_x_mm),
            "eyebox_center_y_mm": float(self.eyebox_center_y_mm),
            "coordinate_frame": self.coordinate_frame,
            "export_angular_response": self.export_angular_response,
        }


@dataclass(frozen=True)
class _EyeWindowAggregation:
    """Intermediate response values shared by report summaries and plots."""

    cell_response: list
    cell_summary_rows: list
    field_values_by_field: list
    all_response_values: list


def normalize_pupil_report_specs(specs):
    """Return validated, JSON-serializable report specifications."""
    return [spec.as_dict() for spec in _validated_pupil_report_specs(specs)]


def _validated_pupil_report_specs(specs):
    """Return validated specs after checking output-name uniqueness once."""
    normalized = [PupilReportSpec.from_mapping(spec) for spec in specs]
    seen = set()
    duplicates = set()
    for spec in normalized:
        name_key = spec.report_name.casefold()
        if name_key in seen:
            duplicates.add(spec.report_name)
        seen.add(name_key)
    if duplicates:
        raise ValueError(
            "Pupil report names must be unique; duplicated: " + ", ".join(sorted(duplicates)) + "."
        )
    return normalized


def write_configured_pupil_reports(cubes, specs, output_directory):
    """Write every configured report from completed position-space cubes.

    Args:
        cubes: Iterable of completed ``PositionSpaceCube`` objects.
        specs: ``PupilReportSpec`` objects or their prescription mappings.
        output_directory: Root sweep output directory.

    Returns:
        A list of index records containing absolute artifact paths.
    """
    cubes_by_label = {}
    for cube in cubes:
        if cube.detector_label in cubes_by_label:
            raise ValueError(
                f"Duplicate detector label {cube.detector_label!r} in pupil report input."
            )
        cubes_by_label[cube.detector_label] = cube

    normalized = _validated_pupil_report_specs(specs)
    results = []
    for spec in normalized:
        try:
            cube = cubes_by_label[spec.detector_label]
        except KeyError as error:
            raise ValueError(
                f"Pupil report {spec.report_name!r} requests detector "
                f"{spec.detector_label!r}, which is not in the sweep."
            ) from error
        results.append(write_pupil_report(cube, spec, output_directory))
    return results


def validate_pupil_report_geometry(detector, spec):
    """Validate every configured eye window against its detector bounds.

    This check can run while the bundle is created, before OpticStudio spends
    time tracing the configured fields.
    """
    spec = PupilReportSpec.from_mapping(spec)
    transform, eye_x, eye_y, window_width, window_height = _sampling_geometry(detector, spec)
    data = detector.get("data", {})
    half_width_x = _positive_float(data.get("half_width_x"), "detector half_width_x")
    half_width_y = _positive_float(data.get("half_width_y"), "detector half_width_y")
    _positive_int(data.get("pixels_x"), "detector pixels_x")
    _positive_int(data.get("pixels_y"), "detector pixels_y")

    for y_index, center_y in enumerate(eye_y):
        for x_index, center_x in enumerate(eye_x):
            zemax_center_x, zemax_center_y = _transform_point(transform, center_x, center_y)
            _bounded_interval(
                zemax_center_x - window_width / 2.0,
                zemax_center_x + window_width / 2.0,
                half_width_x,
                f"eye cell ({x_index}, {y_index}) x window",
            )
            _bounded_interval(
                zemax_center_y - window_height / 2.0,
                zemax_center_y + window_height / 2.0,
                half_width_y,
                f"eye cell ({x_index}, {y_index}) y window",
            )


def write_pupil_report(cube, spec, output_directory):
    """Calculate and write one complete Zemax-derived pupil report."""
    spec = PupilReportSpec.from_mapping(spec)
    report = calculate_pupil_report(cube, spec)

    output_directory = Path(output_directory)
    _remove_legacy_flat_report(output_directory, spec.report_name)
    report_directory = output_directory / "reports" / spec.report_name
    report_directory.mkdir(parents=True, exist_ok=True)
    json_path = report_directory / "report.json"
    cell_csv_path = report_directory / "cell_summary.csv"
    field_csv_path = report_directory / "field_summary.csv"
    html_path = report_directory / "index.html"
    report["files"] = {
        "html": html_path.name,
        "json": json_path.name,
        "cell_summary_csv": cell_csv_path.name,
        "field_summary_csv": field_csv_path.name,
    }

    related_reports = []
    if spec.export_angular_response:
        angular_result = write_angular_response_report(report, report_directory)
        report["files"].update(angular_result["files"])
        report["angular_response"] = angular_result["metadata"]
        related_reports.append(
            {
                key: value
                for key, value in angular_result.items()
                if key not in {"files", "metadata"}
            }
        )

    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_cell_summary(cell_csv_path, report)
    _write_field_summary(field_csv_path, report)
    _write_html(html_path, report)
    return {
        "report_name": spec.report_name,
        "title": report["title"],
        "detector_label": spec.detector_label,
        "html": str(html_path.resolve()),
        "json": str(json_path.resolve()),
        "cell_summary_csv": str(cell_csv_path.resolve()),
        "field_summary_csv": str(field_csv_path.resolve()),
        "related_reports": related_reports,
    }


def _remove_legacy_flat_report(output_directory, report_name):
    """Remove artifacts owned by the former flat ``out/`` report layout."""
    legacy_files = (
        f"{report_name}.html",
        f"{report_name}.json",
        f"{report_name}_cell_summary.csv",
        f"{report_name}_field_summary.csv",
        f"{report_name}_angular_response.html",
        f"{report_name}_angular_response.csv",
    )
    for file_name in legacy_files:
        path = output_directory / file_name
        if path.is_file():
            path.unlink()
    legacy_slices = output_directory / f"{report_name}_angular_response_slices"
    if legacy_slices.is_dir():
        shutil.rmtree(legacy_slices)


def calculate_pupil_report(cube, spec):
    """Return the complete report data without performing file I/O."""
    spec = PupilReportSpec.from_mapping(spec)
    cube.assert_complete()
    fov = _rectangular_fov_grid(cube.sources)
    transform, eye_x, eye_y, zemax_window_width, zemax_window_height = _sampling_geometry(
        cube.detector, spec
    )
    window_area_mm2 = spec.eye_window_width_mm * spec.eye_window_height_mm

    aggregation = _integrate_eye_windows(
        cube,
        spec,
        fov,
        transform,
        eye_x,
        eye_y,
        zemax_window_width,
        zemax_window_height,
    )
    cell_summary_rows = aggregation.cell_summary_rows

    pupil_mean = _summary_map(cell_summary_rows, spec.eye_positions_x, spec.eye_positions_y, "mean")
    pupil_worst = _summary_map(
        cell_summary_rows, spec.eye_positions_x, spec.eye_positions_y, "minimum"
    )
    pupil_cv = _summary_map(
        cell_summary_rows, spec.eye_positions_x, spec.eye_positions_y, "coefficient_of_variation"
    )

    field_summary_rows = _summarize_fields(cube, fov, aggregation.field_values_by_field)

    fov_mean = _reshape_by_fov([row["eye_position_mean"] for row in field_summary_rows], fov)
    fov_worst = _reshape_by_fov([row["eye_position_minimum"] for row in field_summary_rows], fov)
    fov_total = _reshape_by_fov([row["detector_total_flux"] for row in field_summary_rows], fov)
    sorted_means = sorted(row["mean"] for row in cell_summary_rows)

    display_values = {
        "cell_response": aggregation.all_response_values,
        "pupil_flux_density": _flatten_matrices(pupil_mean, pupil_worst),
        "pupil_cv": [value for row in pupil_cv for value in row if value is not None],
        "fov_flux_density": _flatten_matrices(fov_mean, fov_worst),
        "fov_detector_total_flux": [value for row in fov_total for value in row],
        "cdf_cell_mean_flux_density": sorted_means,
    }
    return {
        "schema": "apollo14.zemax_pupil_report",
        "schema_version": SCHEMA_VERSION,
        "title": (f"Zemax pupil and FOV report — {cube.detector_label} detector"),
        "provenance": {
            "calculation_engine": "Zemax OpticStudio detector trace",
            "detector_readout": "NCE.GetDetectorData(detector, pixel, data=0)",
            "postprocessing": (
                "Fractional detector-pixel overlap integration and statistics "
                "performed by the portable bundle report module."
            ),
            "apollo14_reference_values_used": False,
            "spectral_mode": "polychromatic_system_wavelengths",
            "source_wavenumber": 0,
        },
        "configuration": spec.as_dict(),
        "detector": {
            "label": cube.detector_label,
            "object_index": cube.detector.get("index"),
            "pixels_x": cube.pixels_x,
            "pixels_y": cube.pixels_y,
            "half_width_x_mm": cube.half_width_x_mm,
            "half_width_y_mm": cube.half_width_y_mm,
            "zemax_local_axes_world": cube.detector.get("zemax_local_axes_world"),
            "apollo14_local_axes_world": cube.detector.get("apollo14_local_axes_world"),
        },
        "coordinate_transform": {
            "input_frame": spec.coordinate_frame,
            "output_frame": "zemax_detector",
            "matrix": transform,
            "description": (
                "[zemax_x, zemax_y]^T = matrix * "
                f"[{spec.coordinate_frame}_x, {spec.coordinate_frame}_y]^T"
            ),
        },
        "quantity": {
            "name": "eye_window_average_detector_flux_density",
            "unit": "OpticStudio system source-power units/mm^2",
            "calculation": (
                "sum(detector pixel flux * fractional pixel overlap) / eye-window area"
            ),
            "edge_pixel_assumption": (
                "Flux is uniform within a detector pixel when only part of that "
                "pixel overlaps the eye window."
            ),
            "window_area_mm2": window_area_mm2,
            "classification": "detector response; not calibrated radiance or luminance",
        },
        "statistics": {
            "mean_over_fov": (
                "Unweighted arithmetic mean across the independently traced FOV "
                "samples at their recorded source_sweep_power values."
            ),
            "worst_over_fov": "Minimum across the independently traced FOV samples.",
            "mean_over_eye_positions": (
                "Unweighted arithmetic mean across the configured eye positions."
            ),
            "worst_over_eye_positions": ("Minimum across the configured eye positions."),
            "coefficient_of_variation": "Population standard deviation / absolute mean.",
            "angular_integration_performed": False,
        },
        "spectrum": cube.spectrum.metadata(),
        "fov": fov,
        "eye_positions": {
            "x_mm": eye_x,
            "y_mm": eye_y,
            "axis_order": ["eye_y", "eye_x"],
        },
        "arrays": {
            "cell_response": {
                "axes": ["eye_y", "eye_x", "fov_y", "fov_x"],
                "values": aggregation.cell_response,
            },
            "pupil_mean_over_fov": pupil_mean,
            "pupil_worst_over_fov": pupil_worst,
            "pupil_cv_over_fov": pupil_cv,
            "fov_mean_over_eye_positions": fov_mean,
            "fov_worst_over_eye_positions": fov_worst,
            "fov_detector_total_flux": fov_total,
        },
        "cell_summary": cell_summary_rows,
        "field_summary": field_summary_rows,
        "cdf": {
            "cell_mean_flux_density": sorted_means,
            "cumulative_fraction": [
                (index + 1) / len(sorted_means) for index in range(len(sorted_means))
            ],
        },
        "display_ranges": {name: _finite_range(values) for name, values in display_values.items()},
        "display_scales": {
            name: _rounded_display_scale(values) for name, values in display_values.items()
        },
    }


def _integrate_eye_windows(
    cube,
    spec,
    fov,
    transform,
    eye_x,
    eye_y,
    zemax_window_width,
    zemax_window_height,
):
    """Integrate every physical eye window for every independently traced field."""
    window_area_mm2 = spec.eye_window_width_mm * spec.eye_window_height_mm
    field_rows = [cube.flat_field(index) for index in range(len(cube.sources))]
    field_values_by_field = [[] for _source in cube.sources]
    cell_response = []
    cell_summary_rows = []
    all_response_values = []

    for y_index, center_y in enumerate(eye_y):
        response_row = []
        for x_index, center_x in enumerate(eye_x):
            zemax_center_x, zemax_center_y = _transform_point(transform, center_x, center_y)
            description = f"eye cell ({x_index}, {y_index})"
            x_weights = _overlap_fractions(
                zemax_center_x - zemax_window_width / 2.0,
                zemax_center_x + zemax_window_width / 2.0,
                cube.half_width_x_mm,
                cube.pixels_x,
                description + " x window",
            )
            y_weights = _overlap_fractions(
                zemax_center_y - zemax_window_height / 2.0,
                zemax_center_y + zemax_window_height / 2.0,
                cube.half_width_y_mm,
                cube.pixels_y,
                description + " y window",
            )
            field_values = [
                _weighted_window_flux(flat, cube.pixels_x, x_weights, y_weights) / window_area_mm2
                for flat in field_rows
            ]
            for field_index, value in enumerate(field_values):
                field_values_by_field[field_index].append(value)
            all_response_values.extend(field_values)
            cell_summary_rows.append(
                {
                    "eye_x_index": x_index,
                    "eye_y_index": y_index,
                    "eye_x_mm": center_x,
                    "eye_y_mm": center_y,
                    **_statistics(field_values),
                }
            )
            response_row.append(_reshape_by_fov(field_values, fov))
        cell_response.append(response_row)

    return _EyeWindowAggregation(
        cell_response=cell_response,
        cell_summary_rows=cell_summary_rows,
        field_values_by_field=field_values_by_field,
        all_response_values=all_response_values,
    )


def _summarize_fields(cube, fov, field_values_by_field):
    """Summarize each FOV trace across all configured eye positions."""
    rows = []
    for field_index, source in enumerate(cube.sources):
        values = field_values_by_field[field_index]
        rows.append(
            {
                "field_index": field_index,
                "source_label": _source_label(source),
                "source_object_index": source.get("index"),
                "source_sweep_power": _source_sweep_power(source),
                "fov_x_deg": fov["field_x_deg"][field_index],
                "fov_y_deg": fov["field_y_deg"][field_index],
                "detector_total_flux": cube.total_flux(field_index),
                "eye_position_mean": _mean(values),
                "eye_position_minimum": min(values),
                "eye_position_maximum": max(values),
                "eye_position_coefficient_of_variation": _coefficient_of_variation(values),
            }
        )
    return rows


def _sampling_geometry(detector, spec):
    """Return transformed axes, centers, and window dimensions for a report."""
    transform = _coordinate_transform(detector, spec.coordinate_frame)
    eye_x = _cell_centers(spec.eyebox_center_x_mm, spec.eyebox_width_mm, spec.eye_positions_x)
    eye_y = _cell_centers(spec.eyebox_center_y_mm, spec.eyebox_height_mm, spec.eye_positions_y)
    window_width, window_height = _transformed_window_size(
        transform, spec.eye_window_width_mm, spec.eye_window_height_mm
    )
    return transform, eye_x, eye_y, window_width, window_height


def _transformed_window_size(transform, width, height):
    """Return an axis-aligned window size after a signed-axis permutation."""
    return (
        abs(transform[0][0]) * width + abs(transform[0][1]) * height,
        abs(transform[1][0]) * width + abs(transform[1][1]) * height,
    )


def _transform_point(transform, x_value, y_value):
    """Apply a two-dimensional signed-axis permutation to one point."""
    return (
        transform[0][0] * x_value + transform[0][1] * y_value,
        transform[1][0] * x_value + transform[1][1] * y_value,
    )


def _rectangular_fov_grid(sources):
    """Validate source angles and return a deterministic rectangular index."""
    records = []
    for field_index, source in enumerate(sources):
        angles = source.get("fov_angles_deg")
        if not isinstance(angles, (list, tuple)) or len(angles) != 2:
            raise ValueError("A Zemax pupil report requires fov_angles_deg on every source.")
        x_value = _finite_float(angles[0], "source fov_x_deg")
        y_value = _finite_float(angles[1], "source fov_y_deg")
        records.append((x_value, y_value, field_index))
    pairs = [(x, y) for x, y, _ in records]
    if len(set(pairs)) != len(pairs):
        raise ValueError("A Zemax pupil report cannot use duplicate FOV angle pairs.")
    x_axis = sorted({x for x, _, _ in records})
    y_axis = sorted({y for _, y, _ in records})
    if len(x_axis) * len(y_axis) != len(records):
        raise ValueError(
            "A Zemax pupil report requires a complete rectangular FOV grid; "
            f"got {len(records)} sources for {len(x_axis)} x {len(y_axis)} axes."
        )
    x_lookup = {value: index for index, value in enumerate(x_axis)}
    y_lookup = {value: index for index, value in enumerate(y_axis)}
    field_grid = [[None for _ in x_axis] for _ in y_axis]
    field_x_index = [None] * len(records)
    field_y_index = [None] * len(records)
    field_x_deg = [None] * len(records)
    field_y_deg = [None] * len(records)
    labels = [None] * len(records)
    for x_value, y_value, field_index in records:
        x_index = x_lookup[x_value]
        y_index = y_lookup[y_value]
        field_grid[y_index][x_index] = field_index
        field_x_index[field_index] = x_index
        field_y_index[field_index] = y_index
        field_x_deg[field_index] = x_value
        field_y_deg[field_index] = y_value
        labels[field_index] = _source_label(sources[field_index])
    if any(value is None for row in field_grid for value in row):
        raise ValueError("The FOV source grid contains a missing angle pair.")
    return {
        "x_deg": x_axis,
        "y_deg": y_axis,
        "field_index_by_y_x": field_grid,
        "field_x_index": field_x_index,
        "field_y_index": field_y_index,
        "field_x_deg": field_x_deg,
        "field_y_deg": field_y_deg,
        "source_labels": labels,
        "axis_order": ["fov_y", "fov_x"],
    }


def _coordinate_transform(detector, coordinate_frame):
    if coordinate_frame == "zemax":
        return [[1.0, 0.0], [0.0, 1.0]]
    try:
        zemax = detector["zemax_local_axes_world"]
        apollo = detector["apollo14_local_axes_world"]
        zemax_axes = [zemax["x"], zemax["y"]]
        apollo_axes = [apollo["x"], apollo["y"]]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Apollo14-frame pupil reporting requires both exported local-axis "
            "descriptions on the detector."
        ) from error
    matrix = [
        [_dot(zemax_axis, apollo_axis) for apollo_axis in apollo_axes] for zemax_axis in zemax_axes
    ]
    for row in matrix:
        magnitudes = sorted(abs(value) for value in row)
        if not (magnitudes[0] <= _AXIS_TOLERANCE and abs(magnitudes[1] - 1.0) <= _AXIS_TOLERANCE):
            raise ValueError(
                "Pupil report frames must differ by an axis-aligned signed "
                f"permutation; got {matrix!r}."
            )
    for column in range(2):
        magnitudes = sorted(abs(matrix[row][column]) for row in range(2))
        if not (magnitudes[0] <= _AXIS_TOLERANCE and abs(magnitudes[1] - 1.0) <= _AXIS_TOLERANCE):
            raise ValueError(
                f"Pupil report frames must map each input axis exactly once; got {matrix!r}."
            )
    return [[_snap_axis_value(value) for value in row] for row in matrix]


def _overlap_fractions(low, high, half_width, pixels, description):
    """Return ``(pixel index, covered fraction)`` for one physical interval."""
    pixels = _positive_int(pixels, "detector pixels")
    low, high, half_width = _bounded_interval(low, high, half_width, description)
    pixel_width = 2.0 * half_width / pixels
    result = []
    for pixel_index in range(pixels):
        pixel_low = -half_width + pixel_index * pixel_width
        pixel_high = pixel_low + pixel_width
        overlap = max(0.0, min(high, pixel_high) - max(low, pixel_low))
        if overlap > 0.0:
            result.append((pixel_index, overlap / pixel_width))
    return result


def _bounded_interval(low, high, half_width, description):
    """Validate an interval against symmetric detector bounds and clamp noise."""
    half_width = _positive_float(half_width, "detector half width")
    low = _finite_float(low, description + " low edge")
    high = _finite_float(high, description + " high edge")
    tolerance = 1.0e-9 * max(1.0, half_width)
    if low < -half_width - tolerance or high > half_width + tolerance:
        raise ValueError(
            f"{description} [{low:.9g}, {high:.9g}] mm lies outside the "
            f"detector interval [{-half_width:.9g}, {half_width:.9g}] mm."
        )
    low = max(low, -half_width)
    high = min(high, half_width)
    if high <= low:
        raise ValueError(f"{description} has no positive width.")
    return low, high, half_width


def _weighted_window_flux(flat, pixels_x, x_weights, y_weights):
    return math.fsum(
        flat[y_index * pixels_x + x_index] * x_fraction * y_fraction
        for y_index, y_fraction in y_weights
        for x_index, x_fraction in x_weights
    )


def _reshape_by_fov(field_values, fov):
    return [[field_values[field_index] for field_index in row] for row in fov["field_index_by_y_x"]]


def _summary_map(rows, nx, ny, key):
    result = [[None for _ in range(nx)] for _ in range(ny)]
    for row in rows:
        result[row["eye_y_index"]][row["eye_x_index"]] = row[key]
    return result


def _statistics(values):
    return {
        "mean": _mean(values),
        "minimum": min(values),
        "maximum": max(values),
        "coefficient_of_variation": _coefficient_of_variation(values),
    }


def _mean(values):
    return math.fsum(values) / len(values)


def _coefficient_of_variation(values):
    mean = _mean(values)
    if mean == 0.0:
        return None
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / abs(mean)


def _finite_range(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return [0.0, 0.0]
    return [min(values), max(values)]


def _rounded_display_scale(values):
    """Return rounded ticks and include zero as a meaningful response origin."""
    minimum, maximum = _finite_range(values)
    return nice_scale(minimum, maximum, include_zero=True)


def _flatten_matrices(*matrices):
    """Flatten row-major matrices into one display-scale sample list."""
    return [value for matrix in matrices for row in matrix for value in row]


def _cell_centers(center, width, count):
    pitch = width / count
    return [center - width / 2.0 + (index + 0.5) * pitch for index in range(count)]


def _dot(left, right):
    if len(left) != 3 or len(right) != 3:
        raise ValueError("Exported local axes must contain three components.")
    return math.fsum(float(left[index]) * float(right[index]) for index in range(3))


def _snap_axis_value(value):
    if abs(value) <= _AXIS_TOLERANCE:
        return 0.0
    return 1.0 if value > 0.0 else -1.0


def _source_label(source):
    label = source.get("label")
    if isinstance(label, str) and label:
        return label
    return f"source_{source.get('index', 'unknown')}"


def _source_sweep_power(source):
    data = source.get("data", {})
    if not isinstance(data, dict):
        raise ValueError(f"Source {_source_label(source)!r} data must be a mapping.")
    value = _finite_float(
        data.get("sweep_power", 1.0),
        f"Source {_source_label(source)!r} sweep_power",
    )
    if value < 0.0:
        raise ValueError(f"Source {_source_label(source)!r} sweep_power must be non-negative.")
    return value


def _finite_float(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number.") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


def _positive_float(value, name):
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _positive_int(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer.") from error
    if result <= 0 or result != value:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _write_cell_summary(path, report):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "eye_x_index",
            "eye_y_index",
            "eye_x_mm",
            "eye_y_mm",
            "mean",
            "minimum",
            "maximum",
            "coefficient_of_variation",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["cell_summary"])


def _write_field_summary(path, report):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "field_index",
            "source_label",
            "source_object_index",
            "source_sweep_power",
            "fov_x_deg",
            "fov_y_deg",
            "detector_total_flux",
            "eye_position_mean",
            "eye_position_minimum",
            "eye_position_maximum",
            "eye_position_coefficient_of_variation",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["field_summary"])


def _write_html(path, report):
    payload = json.dumps(report, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    page = _HTML_TEMPLATE.replace("__REPORT_LAYOUT_VERSION__", REPORT_LAYOUT_VERSION).replace(
        "__REPORT_DATA__", payload
    )
    Path(path).write_text(page, encoding="utf-8")


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en" data-apollo14-report-layout="__REPORT_LAYOUT_VERSION__">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<title>Zemax pupil and FOV report</title>
<style>
:root { color-scheme: dark; font-family: Inter, Segoe UI, system-ui, sans-serif;
  background:#0b1020; color:#e8edf8; }
* { box-sizing:border-box; }
body { margin:0; background:linear-gradient(140deg,#111a32,#080b14 60%); }
header, main { max-width:1500px; margin:auto; padding:22px; }
h1 { margin:.1rem 0 .35rem; font-size:clamp(1.6rem,3vw,2.6rem); }
h2 { margin:.1rem 0 .8rem; font-size:1.15rem; }
p { color:#b7c2d9; line-height:1.5; }
.tag { display:inline-block; background:#163a59; color:#8cddff; border:1px solid #245f83;
  border-radius:999px; padding:5px 10px; margin:4px 7px 4px 0; font-size:.82rem; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:16px; }
.card { background:#121a2cdd; border:1px solid #27334a; border-radius:13px;
  padding:16px; box-shadow:0 10px 28px #0005; overflow:auto; }
.wide { grid-column:1/-1; }
.controls { display:flex; gap:14px; flex-wrap:wrap; align-items:end; margin-bottom:8px; }
label { display:grid; gap:5px; color:#aab8d1; font-size:.84rem; }
select { color:#e8edf8; background:#0b1222; border:1px solid #364560;
  border-radius:7px; padding:7px 10px; min-width:145px; }
canvas { width:100%; max-width:580px; height:auto; display:block; margin:auto; }
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
  <span class="tag">Polychromatic System Wavelengths</span>
  <h1 id="reportTitle">Zemax pupil and FOV report</h1>
  <p id="subtitle"></p>
  <p class="links"><a id="jsonLink">JSON data</a><a id="cellCsvLink">Cell CSV</a>
    <a id="fieldCsvLink">Field CSV</a><a id="angularResponseLink" hidden>FOV-angle slices</a></p>
</header>
<main class="grid">
  <section class="card wide">
    <h2>One eye position across field of view</h2>
    <div class="controls">
      <label>Eye x position<select id="eyeX"></select></label>
      <label>Eye y position<select id="eyeY"></select></label>
    </div>
    <div class="stats" id="selectedStats"></div>
    <canvas id="selected" width="620" height="440"></canvas>
    <div class="unit-label">Color scale units: detector flux density in source-power units/mm².</div>
    <div class="caption" id="selectedCaption"></div>
  </section>
  <section class="card"><h2>Mean over FOV at each eye position</h2>
    <canvas id="pupilMean" width="560" height="430"></canvas>
    <div class="unit-label">Color scale units: detector flux density in source-power units/mm².</div>
    <div class="caption">Arithmetic mean across all exported FOV samples at each eye position. Common scale with the “Worst FOV” plot, so equal colors represent equal response values in both graphs.</div></section>
  <section class="card"><h2>Worst FOV at each eye position</h2>
    <canvas id="pupilWorst" width="560" height="430"></canvas>
    <div class="unit-label">Color scale units: detector flux density in source-power units/mm².</div>
    <div class="caption">Minimum response across all exported FOV samples at each eye position. It uses the same scale as the mean plot; dim colors identify eye positions whose weakest FOV response is low.</div></section>
  <section class="card"><h2>FOV coefficient of variation</h2>
    <canvas id="pupilCv" width="560" height="430"></canvas>
    <div class="unit-label">Color scale: coefficient of variation (dimensionless).</div>
    <div class="caption">Population standard deviation across FOV divided by the absolute mean at each eye position. Lower values indicate more uniform response across viewing directions; zero means no sampled FOV variation.</div></section>
  <section class="card"><h2>Mean over eye positions by FOV</h2>
    <canvas id="fovMean" width="560" height="430"></canvas>
    <div class="unit-label">Color scale units: detector flux density in source-power units/mm².</div>
    <div class="caption">Arithmetic mean across all configured eye positions at each FOV direction. Common scale with the “Worst eye position” plot, allowing direct color-by-color comparison.</div></section>
  <section class="card"><h2>Worst eye position by FOV</h2>
    <canvas id="fovWorst" width="560" height="430"></canvas>
    <div class="unit-label">Color scale units: detector flux density in source-power units/mm².</div>
    <div class="caption">Minimum eye-window flux density across all configured eye positions at each FOV direction. It uses the same scale as the mean plot and reveals directions with localized dim regions.</div></section>
  <section class="card"><h2>Total detector flux by FOV</h2>
    <canvas id="fovTotal" width="560" height="430"></canvas>
    <div class="unit-label">Color scale units: total detector flux in source-power units.</div>
    <div class="caption">Raw sum of every pixel on the configured Zemax detector for each FOV direction. This includes the whole detector and is not divided by eye-window area.</div></section>
  <section class="card"><h2>CDF of eye-position means</h2>
    <canvas id="cdf" width="560" height="430"></canvas>
    <div class="unit-label">Horizontal-axis units: mean detector flux density in source-power units/mm²; vertical axis: cumulative fraction.</div>
    <div class="caption">For every eye position, the report first calculates the unweighted mean across exported FOV samples. The curve shows the fraction of eye positions at or below each response value; a curve farther right indicates stronger coverage.</div></section>
  <section class="card wide"><h2>Calculation and provenance</h2><p id="method"></p></section>
</main>
<script type="application/json" id="reportData">__REPORT_DATA__</script>
<script>
const report = JSON.parse(document.getElementById('reportData').textContent);
const arrays = report.arrays;
const fov = report.fov;
const eye = report.eye_positions;
const scales = report.display_scales;

const COLORS = [
  [8, 20, 75],
  [19, 99, 164],
  [31, 190, 183],
  [250, 220, 80],
  [220, 45, 34],
];
const CANVAS_BACKGROUND = '#0a1120';
const GRID_COLOR = '#27344b';
const AXIS_COLOR = '#70809c';
const TEXT_COLOR = '#b8c5dc';
const NULL_COLOR = '#303747';
const HEATMAP_MARGIN = {left: 75, right: 110, top: 24, bottom: 66};

const STATIC_HEATMAPS = [
  {
    id: 'pupilMean',
    matrix: arrays.pupil_mean_over_fov,
    xValues: eye.x_mm,
    yValues: eye.y_mm,
    scale: scales.pupil_flux_density,
    xLabel: 'eye x (mm)',
    yLabel: 'eye y (mm)',
  },
  {
    id: 'pupilWorst',
    matrix: arrays.pupil_worst_over_fov,
    xValues: eye.x_mm,
    yValues: eye.y_mm,
    scale: scales.pupil_flux_density,
    xLabel: 'eye x (mm)',
    yLabel: 'eye y (mm)',
  },
  {
    id: 'pupilCv',
    matrix: arrays.pupil_cv_over_fov,
    xValues: eye.x_mm,
    yValues: eye.y_mm,
    scale: scales.pupil_cv,
    xLabel: 'eye x (mm)',
    yLabel: 'eye y (mm)',
  },
  {
    id: 'fovMean',
    matrix: arrays.fov_mean_over_eye_positions,
    xValues: fov.x_deg,
    yValues: fov.y_deg,
    scale: scales.fov_flux_density,
    xLabel: 'FOV x (deg)',
    yLabel: 'FOV y (deg)',
  },
  {
    id: 'fovWorst',
    matrix: arrays.fov_worst_over_eye_positions,
    xValues: fov.x_deg,
    yValues: fov.y_deg,
    scale: scales.fov_flux_density,
    xLabel: 'FOV x (deg)',
    yLabel: 'FOV y (deg)',
  },
  {
    id: 'fovTotal',
    matrix: arrays.fov_detector_total_flux,
    xValues: fov.x_deg,
    yValues: fov.y_deg,
    scale: scales.fov_detector_total_flux,
    xLabel: 'FOV x (deg)',
    yLabel: 'FOV y (deg)',
  },
];

function formatNumber(value) {
  if (value === null) return 'undefined';
  const number = Number(value);
  const magnitude = Math.abs(number);
  if (magnitude === 0) return '0';
  if (magnitude >= 10000 || magnitude < 0.001) {
    return number.toExponential(2).replace('e+', 'e');
  }
  return Number(number.toPrecision(3)).toString();
}

function formatAxisNumber(value) {
  const number = Math.abs(Number(value)) < 0.005 ? 0 : Number(value);
  return Number(number.toFixed(2)).toString();
}

function createOption(value, text) {
  const element = document.createElement('option');
  element.value = value;
  element.textContent = text;
  return element;
}

function setup() {
  document.title = report.title;
  document.getElementById('reportTitle').textContent = report.title;
  const configuration = report.configuration;
  document.getElementById('subtitle').textContent =
    `Detector ${report.detector.label} · ` +
    `${eye.x_mm.length}×${eye.y_mm.length} eye positions · ` +
    `${fov.x_deg.length}×${fov.y_deg.length} FOV samples · ` +
    `${formatAxisNumber(configuration.eye_window_width_mm)}×` +
    `${formatAxisNumber(configuration.eye_window_height_mm)} mm windows`;
  document.getElementById('jsonLink').href = report.files.json;
  document.getElementById('cellCsvLink').href = report.files.cell_summary_csv;
  document.getElementById('fieldCsvLink').href = report.files.field_summary_csv;
  if (report.files.angular_response_html) {
    const angularLink = document.getElementById('angularResponseLink');
    angularLink.href = report.files.angular_response_html;
    angularLink.hidden = false;
  }

  const eyeXSelect = document.getElementById('eyeX');
  const eyeYSelect = document.getElementById('eyeY');
  eye.x_mm.forEach((value, index) => {
    eyeXSelect.appendChild(
      createOption(index, `${formatAxisNumber(value)} mm (x ${index})`));
  });
  eye.y_mm.forEach((value, index) => {
    eyeYSelect.appendChild(
      createOption(index, `${formatAxisNumber(value)} mm (y ${index})`));
  });
  eyeXSelect.value = Math.floor(eye.x_mm.length / 2);
  eyeYSelect.value = Math.floor(eye.y_mm.length / 2);
  eyeXSelect.addEventListener('change', drawSelectedEyePosition);
  eyeYSelect.addEventListener('change', drawSelectedEyePosition);

  drawSelectedEyePosition();
  STATIC_HEATMAPS.forEach(drawHeatmap);
  drawCdf();
  writeMethodDescription();
}

function drawSelectedEyePosition() {
  const eyeXIndex = Number(document.getElementById('eyeX').value);
  const eyeYIndex = Number(document.getElementById('eyeY').value);
  const summary = report.cell_summary.find(row =>
    row.eye_x_index === eyeXIndex && row.eye_y_index === eyeYIndex);

  drawHeatmap({
    id: 'selected',
    matrix: arrays.cell_response.values[eyeYIndex][eyeXIndex],
    xValues: fov.x_deg,
    yValues: fov.y_deg,
    scale: scales.cell_response,
    xLabel: 'FOV x (deg)',
    yLabel: 'FOV y (deg)',
  });

  const statistics = [
    ['mean', summary.mean],
    ['minimum', summary.minimum],
    ['maximum', summary.maximum],
    ['CV', summary.coefficient_of_variation],
  ];
  document.getElementById('selectedStats').innerHTML = statistics
    .map(([label, value]) =>
      `<span class="stat">${label}<b>${formatNumber(value)}</b></span>`)
    .join('');
  document.getElementById('selectedCaption').textContent =
    `Eye center (${formatAxisNumber(summary.eye_x_mm)}, ` +
    `${formatAxisNumber(summary.eye_y_mm)}) mm. ` +
    'Each colored cell is the detector flux density for one FOV direction. ' +
    'The color scale is shared across every selectable eye position, so changing ' +
    'the selection does not rescale the colors. FOV x is horizontal; FOV y ' +
    'increases upward.';
}

function writeMethodDescription() {
  const quantity = report.quantity;
  const provenance = report.provenance;
  const configuration = report.configuration;
  document.getElementById('method').innerHTML =
    `Every plotted response starts with <code>${provenance.detector_readout}</code>. ` +
    `${quantity.calculation}. Fractional edge pixels assume flux is uniform ` +
    `inside one detector pixel. The eye window is ` +
    `${configuration.eye_window_width_mm} mm × ` +
    `${configuration.eye_window_height_mm} mm in the ` +
    `${configuration.coordinate_frame} detector frame. Means over FOV are ` +
    `unweighted arithmetic means of the independently traced FOV samples; ` +
    `no angular integration is performed. Units are ${quantity.unit}. No values ` +
    `from the pre-export Apollo14 pupil report are used. These values are ` +
    `detector response, not calibrated radiance or luminance.`;
}

function colorForFraction(fraction) {
  const bounded = Math.max(0, Math.min(1, fraction));
  const position = bounded * (COLORS.length - 1);
  const lowerIndex = Math.min(COLORS.length - 2, Math.floor(position));
  const amount = position - lowerIndex;
  const lower = COLORS[lowerIndex];
  const upper = COLORS[lowerIndex + 1];
  const channels = lower.map((value, channel) =>
    Math.round(value + (upper[channel] - value) * amount));
  return `rgb(${channels.join(',')})`;
}

function normalizedValue(value, scale) {
  const span = scale.maximum - scale.minimum;
  if (span === 0) return 0;
  return (value - scale.minimum) / span;
}

function drawHeatmap(options) {
  const canvas = document.getElementById(options.id);
  const context = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  const plotWidth = width - HEATMAP_MARGIN.left - HEATMAP_MARGIN.right;
  const plotHeight = height - HEATMAP_MARGIN.top - HEATMAP_MARGIN.bottom;
  const rowCount = options.matrix.length;
  const columnCount = options.matrix[0].length;

  context.clearRect(0, 0, width, height);
  context.fillStyle = CANVAS_BACKGROUND;
  context.fillRect(0, 0, width, height);

  for (let row = 0; row < rowCount; row += 1) {
    for (let column = 0; column < columnCount; column += 1) {
      const value = options.matrix[row][column];
      const displayRow = rowCount - 1 - row;
      context.fillStyle = value === null
        ? NULL_COLOR
        : colorForFraction(normalizedValue(value, options.scale));
      context.fillRect(
        HEATMAP_MARGIN.left + column * plotWidth / columnCount,
        HEATMAP_MARGIN.top + displayRow * plotHeight / rowCount,
        plotWidth / columnCount + 0.5,
        plotHeight / rowCount + 0.5,
      );
    }
  }

  drawHeatmapAxes(context, width, height, options);
  drawColorScale(context, width, options.scale);
}

function drawHeatmapAxes(context, width, height, options) {
  const plotWidth = width - HEATMAP_MARGIN.left - HEATMAP_MARGIN.right;
  const plotHeight = height - HEATMAP_MARGIN.top - HEATMAP_MARGIN.bottom;
  const rowCount = options.yValues.length;
  const columnCount = options.xValues.length;

  context.strokeStyle = AXIS_COLOR;
  context.strokeRect(
    HEATMAP_MARGIN.left,
    HEATMAP_MARGIN.top,
    plotWidth,
    plotHeight,
  );
  context.fillStyle = TEXT_COLOR;
  context.font = '12px Segoe UI';
  context.textBaseline = 'middle';
  context.textAlign = 'center';
  sampledIndices(columnCount).forEach(index => {
    const x = HEATMAP_MARGIN.left + (index + 0.5) * plotWidth / columnCount;
    context.fillText(
      formatAxisNumber(options.xValues[index]),
      x,
      height - HEATMAP_MARGIN.bottom + 19,
    );
  });
  context.fillText(
    options.xLabel,
    HEATMAP_MARGIN.left + plotWidth / 2,
    height - 11,
  );

  context.textAlign = 'right';
  sampledIndices(rowCount).forEach(index => {
    const y = HEATMAP_MARGIN.top + (rowCount - index - 0.5) * plotHeight / rowCount;
    context.fillText(
      formatAxisNumber(options.yValues[index]),
      HEATMAP_MARGIN.left - 7,
      y,
    );
  });
  drawRotatedLabel(
    context,
    14,
    HEATMAP_MARGIN.top + plotHeight / 2,
    options.yLabel,
  );
}

function sampledIndices(count, maximumLabels = 6) {
  if (count <= maximumLabels) {
    return Array.from({length: count}, (_, index) => index);
  }
  const sampled = Array.from({length: maximumLabels}, (_, index) =>
    Math.round(index * (count - 1) / (maximumLabels - 1)));
  return [...new Set(sampled)];
}

function drawColorScale(context, canvasWidth, scale) {
  const plotHeight = context.canvas.height - HEATMAP_MARGIN.top - HEATMAP_MARGIN.bottom;
  const barX = canvasWidth - HEATMAP_MARGIN.right + 22;
  const barWidth = 17;
  const sampleCount = 160;
  const sampleHeight = plotHeight / sampleCount;

  for (let index = 0; index < sampleCount; index += 1) {
    const fraction = index / (sampleCount - 1);
    const y = HEATMAP_MARGIN.top + plotHeight - (index + 1) * sampleHeight;
    context.fillStyle = colorForFraction(fraction);
    context.fillRect(barX, y, barWidth, sampleHeight + 1);
  }
  context.strokeStyle = AXIS_COLOR;
  context.strokeRect(barX, HEATMAP_MARGIN.top, barWidth, plotHeight);

  context.fillStyle = TEXT_COLOR;
  context.font = '12px Segoe UI';
  context.textAlign = 'left';
  context.textBaseline = 'middle';
  scale.ticks.forEach(value => {
    const fraction = normalizedValue(value, scale);
    const y = scale.maximum === scale.minimum
      ? HEATMAP_MARGIN.top + plotHeight / 2
      : HEATMAP_MARGIN.top + (1 - fraction) * plotHeight;
    context.beginPath();
    context.moveTo(barX + barWidth, y);
    context.lineTo(barX + barWidth + 5, y);
    context.stroke();
    context.fillText(formatNumber(value), barX + barWidth + 9, y);
  });
}

function drawRotatedLabel(context, x, y, label) {
  context.save();
  context.translate(x, y);
  context.rotate(-Math.PI / 2);
  context.fillStyle = TEXT_COLOR;
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(label, 0, 0);
  context.restore();
}

function drawCdf() {
  const canvas = document.getElementById('cdf');
  const context = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  const margin = {left: 82, right: 24, top: 24, bottom: 70};
  const xScale = scales.cdf_cell_mean_flux_density;
  const yScale = {
    minimum: 0,
    maximum: 1,
    ticks: [0, 0.25, 0.5, 0.75, 1],
  };
  const xValues = report.cdf.cell_mean_flux_density;
  const yValues = report.cdf.cumulative_fraction;

  context.clearRect(0, 0, width, height);
  context.fillStyle = CANVAS_BACKGROUND;
  context.fillRect(0, 0, width, height);
  drawCartesianAxes(
    context,
    width,
    height,
    margin,
    xScale,
    yScale,
    'mean eye-window flux density',
    'cumulative fraction',
  );

  context.beginPath();
  xValues.forEach((value, index) => {
    const x = coordinateForScale(
      value,
      xScale,
      margin.left,
      width - margin.right,
    );
    const y = coordinateForScale(
      yValues[index],
      yScale,
      height - margin.bottom,
      margin.top,
    );
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.strokeStyle = '#55d6be';
  context.lineWidth = 3;
  context.stroke();
  xValues.forEach((value, index) => {
    const x = coordinateForScale(
      value,
      xScale,
      margin.left,
      width - margin.right,
    );
    const y = coordinateForScale(
      yValues[index],
      yScale,
      height - margin.bottom,
      margin.top,
    );
    context.beginPath();
    context.arc(x, y, 2.5, 0, 2 * Math.PI);
    context.fillStyle = '#55d6be';
    context.fill();
  });
}

function drawCartesianAxes(
  context,
  width,
  height,
  margin,
  xScale,
  yScale,
  xLabel,
  yLabel,
) {
  const plotLeft = margin.left;
  const plotRight = width - margin.right;
  const plotTop = margin.top;
  const plotBottom = height - margin.bottom;

  context.font = '12px Segoe UI';
  context.lineWidth = 1;
  context.textBaseline = 'middle';
  xScale.ticks.forEach(value => {
    const x = coordinateForScale(value, xScale, plotLeft, plotRight);
    context.strokeStyle = GRID_COLOR;
    context.beginPath();
    context.moveTo(x, plotTop);
    context.lineTo(x, plotBottom);
    context.stroke();
    context.fillStyle = TEXT_COLOR;
    context.textAlign = 'center';
    context.fillText(formatNumber(value), x, plotBottom + 19);
  });
  yScale.ticks.forEach(value => {
    const y = coordinateForScale(value, yScale, plotBottom, plotTop);
    context.strokeStyle = GRID_COLOR;
    context.beginPath();
    context.moveTo(plotLeft, y);
    context.lineTo(plotRight, y);
    context.stroke();
    context.fillStyle = TEXT_COLOR;
    context.textAlign = 'right';
    context.fillText(formatNumber(value), plotLeft - 8, y);
  });

  context.strokeStyle = AXIS_COLOR;
  context.strokeRect(plotLeft, plotTop, plotRight - plotLeft, plotBottom - plotTop);
  context.fillStyle = TEXT_COLOR;
  context.textAlign = 'center';
  context.fillText(xLabel, (plotLeft + plotRight) / 2, height - 13);
  drawRotatedLabel(context, 15, (plotTop + plotBottom) / 2, yLabel);
}

function coordinateForScale(value, scale, start, end) {
  const span = scale.maximum - scale.minimum;
  const fraction = span === 0 ? 0.5 : (value - scale.minimum) / span;
  return start + fraction * (end - start);
}

setup();
</script>
</body>
</html>
"""
