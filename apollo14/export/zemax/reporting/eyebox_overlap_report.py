"""Portable configuration and artifacts for the eyebox-overlap report.

The generated Zemax sweep calls this module directly from completed
``PositionSpaceCube`` objects.  It uses only the Python standard library so a
bundle does not require Apollo14, NumPy, or image-processing packages on the
OpticStudio computer.
"""

import csv
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

try:
    from .eyebox_overlap_core import (
        DEFAULT_DENSE_ZONE_SETTINGS,
        DenseZoneSettings,
        calculate_eyebox_overlap,
    )
    from .eyebox_overlap_report_template import HTML_TEMPLATE
    from .html_plotting import REPORT_LAYOUT_VERSION
    from .position_space_common import _pixel_centers, _slug
except ImportError:  # Standalone copies in a generated Zemax bundle.
    from eyebox_overlap_core import (  # type: ignore[no-redef]
        DEFAULT_DENSE_ZONE_SETTINGS,
        DenseZoneSettings,
        calculate_eyebox_overlap,
    )
    from eyebox_overlap_report_template import HTML_TEMPLATE  # type: ignore[no-redef]
    from html_plotting import REPORT_LAYOUT_VERSION  # type: ignore[no-redef]
    from position_space_common import _pixel_centers, _slug  # type: ignore[no-redef]


DEFAULT_REPORT_NAME = "eyebox_field_overlap"
DEFAULT_REPORT_TITLE = "Eyebox field overlap"
SPEC_SCHEMA = "apollo14.eyebox_overlap_report_spec"
SPEC_SCHEMA_VERSION = 1
_SAFE_REPORT_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_RESERVED_REPORT_NAMES = {"position_space_index"}
_STALE_REPORT_FILENAMES = (
    "field_rectangles.csv",
    "field_quads.csv",
    "quad_overlap.csv",
)


# Prescription configuration ------------------------------------------------


@dataclass(frozen=True)
class EyeboxOverlapReportSpec:
    """Configuration for one overlap report derived from a detector cube."""

    detector_label: str = "eyebox"
    report_name: str = DEFAULT_REPORT_NAME
    title: str = DEFAULT_REPORT_TITLE
    dense_zone_settings: DenseZoneSettings = field(
        default_factory=lambda: DEFAULT_DENSE_ZONE_SETTINGS
    )

    def __post_init__(self):
        if not isinstance(self.detector_label, str) or not self.detector_label.strip():
            raise ValueError("Eyebox overlap detector_label must be a non-empty string.")
        if (
            not isinstance(self.report_name, str)
            or _SAFE_REPORT_NAME.fullmatch(self.report_name) is None
        ):
            raise ValueError(
                "Eyebox overlap report_name may contain only letters, digits, dot, "
                "underscore, and hyphen."
            )
        if self.report_name.casefold() in _RESERVED_REPORT_NAMES:
            raise ValueError(
                f"Eyebox overlap report_name {self.report_name!r} is reserved by the "
                "position-space report index."
            )
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("Eyebox overlap report title must be a non-empty string.")
        settings = DenseZoneSettings.from_mapping(self.dense_zone_settings)
        object.__setattr__(self, "detector_label", self.detector_label.strip())
        object.__setattr__(self, "title", self.title.strip())
        object.__setattr__(self, "dense_zone_settings", settings)

    @classmethod
    def from_mapping(cls, value):
        """Validate public exporter input or prescription JSON data."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("An eyebox overlap report specification must be a mapping.")
        allowed_keys = {
            "schema",
            "schema_version",
            "detector_label",
            "report_name",
            "title",
            "dense_zone_settings",
        }
        unknown_keys = sorted(set(value) - allowed_keys)
        if unknown_keys:
            raise ValueError("Invalid eyebox overlap report keys: " + ", ".join(unknown_keys) + ".")
        if "schema" in value and value["schema"] != SPEC_SCHEMA:
            raise ValueError(f"Eyebox overlap report schema must be {SPEC_SCHEMA!r}.")
        if "schema_version" in value and (
            isinstance(value["schema_version"], bool)
            or value["schema_version"] != SPEC_SCHEMA_VERSION
        ):
            raise ValueError(f"Eyebox overlap report schema_version must be {SPEC_SCHEMA_VERSION}.")
        accepted = {
            key: value[key]
            for key in (
                "detector_label",
                "report_name",
                "title",
                "dense_zone_settings",
            )
            if key in value
        }
        try:
            return cls(**accepted)
        except TypeError as error:
            raise ValueError(f"Invalid eyebox overlap report specification: {error}") from error

    def as_dict(self):
        """Return a JSON-safe prescription representation."""
        return {
            "schema": SPEC_SCHEMA,
            "schema_version": SPEC_SCHEMA_VERSION,
            "detector_label": self.detector_label,
            "report_name": self.report_name,
            "title": self.title,
            "dense_zone_settings": self.dense_zone_settings.as_spec_dict(),
        }


def normalize_eyebox_overlap_report(specification):
    """Return one validated JSON-safe report request, or ``None``."""
    if specification is None:
        return None
    return EyeboxOverlapReportSpec.from_mapping(specification).as_dict()


# Completed-cube adapter ----------------------------------------------------


def write_configured_eyebox_overlap_report(cubes, specification, output_directory):
    """Calculate and write the configured report from completed detector cubes."""
    if specification is None:
        return None
    spec = EyeboxOverlapReportSpec.from_mapping(specification)
    cubes_by_label = {}
    for cube in cubes:
        if cube.detector_label in cubes_by_label:
            raise ValueError(
                f"Duplicate detector label {cube.detector_label!r} in overlap report input."
            )
        cubes_by_label[cube.detector_label] = cube
    try:
        cube = cubes_by_label[spec.detector_label]
    except KeyError as error:
        raise ValueError(
            f"Eyebox overlap report {spec.report_name!r} requests detector "
            f"{spec.detector_label!r}, which is not in the sweep."
        ) from error

    flux = [cube.rows(field_index) for field_index in range(len(cube.sources))]
    x_centers = _pixel_centers(cube.half_width_x_mm, cube.pixels_x)
    y_centers = _pixel_centers(cube.half_width_y_mm, cube.pixels_y)
    result = calculate_eyebox_overlap(
        flux,
        x_centers,
        y_centers,
        detector_label=cube.detector_label,
        field_metadata=cube.sources,
        dense_zone_settings=spec.dense_zone_settings,
    )

    output_directory = Path(output_directory)
    detector_directory = output_directory / _slug(cube.detector_label)
    report_directory = output_directory / "reports" / spec.report_name
    paths = write_eyebox_overlap_report(
        result,
        report_directory,
        source_npz_path=detector_directory / "position_space.npz",
        source_metadata_path=detector_directory / "position_space.json",
        title=spec.title,
    )
    return {
        "report_name": spec.report_name,
        "title": spec.title,
        "detail": f"All-field dense overlap · Detector {cube.detector_label}",
        "detector_label": cube.detector_label,
        **paths,
    }


# Report artifact writer ----------------------------------------------------


def write_eyebox_overlap_report(
    result,
    output_directory,
    *,
    source_npz_path=None,
    source_metadata_path=None,
    title=None,
):
    """Write the offline polygon-overlap report and numeric artifacts."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    html_path = output_directory / "index.html"
    json_path = output_directory / "report.json"
    csv_path = output_directory / "contained_overlap_quad.csv"
    for filename in _STALE_REPORT_FILENAMES:
        path = output_directory / filename
        if path.is_file():
            path.unlink()

    payload = result.as_dict()
    if title is not None:
        payload["title"] = str(title)
    files = {
        "html": html_path.name,
        "json": json_path.name,
        "contained_overlap_quad_csv": csv_path.name,
        "source_npz": _relative_file_link(source_npz_path, output_directory),
        "source_metadata": _relative_file_link(source_metadata_path, output_directory),
        "field_images": _field_image_links(
            result,
            source_npz_path,
            source_metadata_path,
            output_directory,
        ),
    }
    if source_npz_path is not None:
        source_html = Path(source_npz_path).with_name("position_space.html")
        files["source_position_space_html"] = _relative_file_link(
            source_html,
            output_directory,
        )
    payload["files"] = files

    json_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_contained_overlap_quad_csv(csv_path, result.contained_overlap_quad)
    html_path.write_text(_render_html(payload), encoding="utf-8")
    return {
        "html": html_path.resolve(),
        "json": json_path.resolve(),
        "contained_overlap_quad_csv": csv_path.resolve(),
    }


# Links to existing detector artifacts -------------------------------------


def _relative_file_link(path, output_directory):
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    return Path(os.path.relpath(path.resolve(), output_directory.resolve())).as_posix()


def _metadata_image_paths(source_metadata_path):
    if source_metadata_path is None:
        return {}
    metadata_path = Path(source_metadata_path)
    if not metadata_path.is_file():
        return {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    files = metadata.get("files") if isinstance(metadata, dict) else None
    slices = files.get("slices") if isinstance(files, dict) else None
    if not isinstance(slices, list):
        return {}

    image_paths = {}
    for item in slices:
        if not isinstance(item, dict):
            continue
        field_index = item.get("field_index")
        image = item.get("image")
        if isinstance(field_index, int) and isinstance(image, str):
            candidate = metadata_path.parent / image
            if candidate.is_file():
                image_paths[field_index] = candidate
    return image_paths


def _field_image_links(
    result,
    source_npz_path,
    source_metadata_path,
    output_directory,
):
    image_paths = _metadata_image_paths(source_metadata_path)
    if source_npz_path is not None:
        image_directory = Path(source_npz_path).parent / "images"
        for field_bound in result.fields:
            if field_bound.field_index in image_paths:
                continue
            matches = sorted(image_directory.glob(f"field_{field_bound.field_index:03d}_*.png"))
            if len(matches) == 1:
                image_paths[field_bound.field_index] = matches[0]

    links = []
    for field_bound in result.fields:
        image = _relative_file_link(image_paths.get(field_bound.field_index), output_directory)
        if image is not None:
            links.append(
                {
                    "field_index": field_bound.field_index,
                    "source_label": field_bound.source_label,
                    "image": image,
                }
            )
    return links


# Small output serializers --------------------------------------------------


def _write_contained_overlap_quad_csv(path, quad):
    headers = ["vertex", "x_mm", "y_mm", "outgoing_side", "side_length_mm"]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        if quad is None:
            return
        for index, values in enumerate(
            zip(  # noqa: B905 - copied bundles retain Python 3.8 compatibility.
                quad.vertices_mm, quad.side_names, quad.side_lengths_mm
            ),
            start=1,
        ):
            (x, y), side_name, side_length = values
            writer.writerow(
                {
                    "vertex": f"V{index}",
                    "x_mm": x,
                    "y_mm": y,
                    "outgoing_side": side_name,
                    "side_length_mm": side_length,
                }
            )


def _render_html(payload):
    serialized = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__REPORT_LAYOUT_VERSION__", REPORT_LAYOUT_VERSION).replace(
        "__REPORT_DATA__", serialized
    )


__all__ = [
    "DEFAULT_REPORT_NAME",
    "DEFAULT_REPORT_TITLE",
    "EyeboxOverlapReportSpec",
    "normalize_eyebox_overlap_report",
    "write_configured_eyebox_overlap_report",
    "write_eyebox_overlap_report",
]
