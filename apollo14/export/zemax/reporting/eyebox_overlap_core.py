"""Portable dense-zone analysis for field-resolved eyebox detector maps.

This module is copied into every Zemax export bundle.  It therefore uses only
the Python standard library and accepts ordinary nested sequences instead of
NumPy arrays.  It owns flux validation, local-density filtering, field
metadata, and assembly of the result.  Generic convex geometry lives in
``eyebox_overlap_geometry`` and artifact output lives in
``eyebox_overlap_report``.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

try:
    from .eyebox_overlap_geometry import (
        BoundingQuad,
        Polygon,
        Rectangle,
        convex_bounding_polygon,
        intersect_convex_polygons,
        largest_inscribed_quad,
        polygon_dict,
        rectangle_from_spans,
    )
except ImportError:  # Standalone copies in a generated Zemax bundle.
    from eyebox_overlap_geometry import (  # type: ignore[no-redef]
        BoundingQuad,
        Polygon,
        Rectangle,
        convex_bounding_polygon,
        intersect_convex_polygons,
        largest_inscribed_quad,
        polygon_dict,
        rectangle_from_spans,
    )

SCHEMA = "apollo14.eyebox_field_overlap"
SCHEMA_VERSION = 13


# Analysis configuration ----------------------------------------------------


@dataclass(frozen=True)
class DenseZoneSettings:
    """Thresholds used to isolate each field's dense numeric support."""

    neighborhood_fraction_of_short_detector_axis: float = 0.05
    minimum_local_positive_pixel_fraction: float = 0.5
    minimum_local_flux_density_fraction_of_field_max: float = 0.1

    def __post_init__(self):
        values = {
            "neighborhood_fraction_of_short_detector_axis": (
                self.neighborhood_fraction_of_short_detector_axis
            ),
            "minimum_local_positive_pixel_fraction": (self.minimum_local_positive_pixel_fraction),
            "minimum_local_flux_density_fraction_of_field_max": (
                self.minimum_local_flux_density_fraction_of_field_max
            ),
        }
        for name, value in values.items():
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must be finite and in the interval (0, 1].") from error
            if not math.isfinite(number) or not 0.0 < number <= 1.0:
                raise ValueError(f"{name} must be finite and in the interval (0, 1].")
            object.__setattr__(self, name, number)

    @classmethod
    def from_mapping(cls, value):
        """Return validated settings from a settings object or mapping."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("Dense-zone settings must be a mapping.")
        try:
            return cls(**dict(value))
        except TypeError as error:
            raise ValueError(f"Invalid dense-zone settings: {error}") from error

    def as_spec_dict(self):
        """Return the unresolved JSON-safe settings."""
        return {
            "neighborhood_fraction_of_short_detector_axis": (
                self.neighborhood_fraction_of_short_detector_axis
            ),
            "minimum_local_positive_pixel_fraction": (self.minimum_local_positive_pixel_fraction),
            "minimum_local_flux_density_fraction_of_field_max": (
                self.minimum_local_flux_density_fraction_of_field_max
            ),
        }

    def as_dict(self, neighborhood_pixels):
        """Return settings plus the resolution-dependent neighborhood size."""
        return {
            "dense_neighborhood_fraction_of_short_detector_axis": (
                self.neighborhood_fraction_of_short_detector_axis
            ),
            "dense_neighborhood_pixels": int(neighborhood_pixels),
            "minimum_local_positive_pixel_fraction": (self.minimum_local_positive_pixel_fraction),
            "minimum_local_flux_density_fraction_of_field_max": (
                self.minimum_local_flux_density_fraction_of_field_max
            ),
        }


DEFAULT_DENSE_ZONE_SETTINGS = DenseZoneSettings()


# Report result models -------------------------------------------------------


@dataclass(frozen=True)
class FieldBounds:
    """Positive-flux support and dense-zone bounds for one field."""

    field_index: int
    source_label: str
    fov_x_deg: object
    fov_y_deg: object
    peak_flux: float
    support_pixel_count: int
    support_area_mm2: float
    dense_zone_pixel_count: int
    dense_zone_area_mm2: float
    polygon: Polygon

    def as_dict(self):
        return {
            "field_index": self.field_index,
            "source_label": self.source_label,
            "fov_x_deg": self.fov_x_deg,
            "fov_y_deg": self.fov_y_deg,
            "peak_flux": self.peak_flux,
            "support_pixel_count": self.support_pixel_count,
            "support_area_mm2": self.support_area_mm2,
            "dense_zone_pixel_count": self.dense_zone_pixel_count,
            "dense_zone_area_mm2": self.dense_zone_area_mm2,
            "bounding_polygon": self.polygon.as_dict(),
        }


@dataclass(frozen=True)
class _FieldDescriptor:
    source_label: str
    fov_x_deg: object
    fov_y_deg: object


@dataclass(frozen=True)
class EyeboxOverlapResult:
    """Complete numeric result for field-resolved eyebox detector maps."""

    detector_label: str
    pixel_x_edges_mm: tuple
    pixel_y_edges_mm: tuple
    dense_zone_settings: DenseZoneSettings
    dense_window_pixels: int
    fields: tuple
    overlap_polygon: object
    contained_overlap_quad: object

    @property
    def bounding_polygon_intersection(self):
        """Compatibility alias for the all-field polygon intersection."""
        return self.overlap_polygon

    @property
    def bounding_polygon_inscribed_quad(self):
        """Compatibility alias for the contained overlap quad."""
        return self.contained_overlap_quad

    @property
    def detector_rectangle(self):
        return rectangle_from_spans(
            0,
            len(self.pixel_x_edges_mm) - 1,
            0,
            len(self.pixel_y_edges_mm) - 1,
            self.pixel_x_edges_mm,
            self.pixel_y_edges_mm,
        )

    def as_dict(self):
        detector = self.detector_rectangle
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "title": f"Eyebox field-polygon overlap report — {self.detector_label} detector",
            "provenance": {
                "calculation": "Numeric positive-flux support from position_space.npz.",
                "image_processing_used": False,
                "support_rule": "A pixel belongs to a field when its numeric flux is positive.",
                "dense_zone_definition": (
                    "The dense zone contains positive pixels whose local occupancy and local "
                    "mean flux meet the configured fractions. Its convex hull bridges internal "
                    "scan gaps."
                ),
                "coordinate_frame": (
                    "Vertices are detector-local (x, y) coordinates in millimetres."
                ),
                "field_polygon_definition": (
                    "Each field bound is the convex hull of that field's numeric dense zone."
                ),
                "overlap_polygon_definition": (
                    "The overlap polygon is the geometric intersection of every field's dense-"
                    "zone polygon."
                ),
                "contained_quad_definition": (
                    "The reported quad is the maximum-area four-vertex approximation selected "
                    "from the overlap polygon's vertices, so the entire quad remains inside the "
                    "overlap polygon."
                ),
                "quad_vertex_and_side_order": (
                    "V1 through V4 follow the overlap polygon counter-clockwise. Top, Bottom, "
                    "Left, and Right are assigned from each edge's outward direction."
                ),
            },
            "configuration": self.dense_zone_settings.as_dict(self.dense_window_pixels),
            "detector": {
                "label": self.detector_label,
                "pixels_x": len(self.pixel_x_edges_mm) - 1,
                "pixels_y": len(self.pixel_y_edges_mm) - 1,
                "bounding_rectangle": detector.as_dict(),
            },
            "fields": [field.as_dict() for field in self.fields],
            "overlap_polygon": polygon_dict(self.overlap_polygon),
            "contained_overlap_quad": polygon_dict(self.contained_overlap_quad),
        }


# Public analysis entry point -----------------------------------------------


def calculate_eyebox_overlap(
    flux,
    pixel_x_center_mm,
    pixel_y_center_mm,
    *,
    detector_label="eyebox",
    field_metadata=None,
    dense_zone_settings=DEFAULT_DENSE_ZONE_SETTINGS,
):
    """Calculate field polygons, their overlap, and a contained quad.

    ``flux`` must use ``(field, pixel_y, pixel_x)`` order.  Every field is
    processed independently from its numeric flux map.
    """
    flux_cube, field_count, pixels_y, pixels_x = _flux_cube(flux)
    if not isinstance(detector_label, str) or not detector_label.strip():
        raise ValueError("detector_label must be a non-empty string.")
    if not isinstance(dense_zone_settings, DenseZoneSettings):
        raise TypeError("dense_zone_settings must be a DenseZoneSettings instance.")

    x_edges = _pixel_edges(pixel_x_center_mm, pixels_x, "pixel_x_center_mm")
    y_edges = _pixel_edges(pixel_y_center_mm, pixels_y, "pixel_y_center_mm")
    descriptors = _field_descriptors(field_metadata, field_count)
    dense_window_pixels = _dense_window_size(
        pixels_y,
        pixels_x,
        dense_zone_settings.neighborhood_fraction_of_short_detector_axis,
    )

    fields = []
    polygons = []
    invalid_fields = []
    for field_index, field_flux in enumerate(flux_cube):
        peak_flux = max(max(row) for row in field_flux)
        if peak_flux <= 0.0:
            invalid_fields.append(field_index)
            continue
        support_mask = tuple(tuple(value > 0.0 for value in row) for row in field_flux)
        dense_mask = _dense_support_mask(
            field_flux,
            support_mask,
            dense_window_pixels,
            dense_zone_settings,
        )
        polygon = convex_bounding_polygon(dense_mask, x_edges, y_edges)
        polygons.append(polygon)
        descriptor = descriptors[field_index]
        support_count, support_area = _mask_metrics(support_mask, x_edges, y_edges)
        dense_count, dense_area = _mask_metrics(dense_mask, x_edges, y_edges)
        fields.append(
            FieldBounds(
                field_index=field_index,
                source_label=descriptor.source_label,
                fov_x_deg=descriptor.fov_x_deg,
                fov_y_deg=descriptor.fov_y_deg,
                peak_flux=peak_flux,
                support_pixel_count=support_count,
                support_area_mm2=support_area,
                dense_zone_pixel_count=dense_count,
                dense_zone_area_mm2=dense_area,
                polygon=polygon,
            )
        )

    if invalid_fields:
        indices = ", ".join(str(index) for index in invalid_fields)
        raise ValueError(
            f"Every field must contain positive flux; invalid field indices: {indices}."
        )

    overlap_polygon = intersect_convex_polygons(polygons)
    return EyeboxOverlapResult(
        detector_label=detector_label.strip(),
        pixel_x_edges_mm=x_edges,
        pixel_y_edges_mm=y_edges,
        dense_zone_settings=dense_zone_settings,
        dense_window_pixels=dense_window_pixels,
        fields=tuple(fields),
        overlap_polygon=overlap_polygon,
        contained_overlap_quad=largest_inscribed_quad(overlap_polygon),
    )


# Numeric input and dense-zone extraction ----------------------------------


def _flux_cube(value):
    """Validate and copy a non-empty rectangular field/y/x flux cube."""
    if isinstance(value, (str, bytes)):
        raise ValueError("flux must be a non-empty array in (field, pixel_y, pixel_x) order.")
    try:
        raw_fields = list(value)
    except TypeError as error:
        raise ValueError(
            "flux must be a non-empty array in (field, pixel_y, pixel_x) order."
        ) from error
    if not raw_fields:
        raise ValueError("flux must be a non-empty array in (field, pixel_y, pixel_x) order.")

    fields = []
    pixels_y = None
    pixels_x = None
    for raw_field in raw_fields:
        try:
            raw_rows = list(raw_field)
        except TypeError as error:
            raise ValueError(
                "flux must be a non-empty array in (field, pixel_y, pixel_x) order."
            ) from error
        if not raw_rows:
            raise ValueError("flux must be a non-empty array in (field, pixel_y, pixel_x) order.")
        if pixels_y is None:
            pixels_y = len(raw_rows)
        elif len(raw_rows) != pixels_y:
            raise ValueError("flux must use one rectangular detector shape for every field.")

        rows = []
        for raw_row in raw_rows:
            try:
                raw_values = list(raw_row)
            except TypeError as error:
                raise ValueError("Every flux detector row must be a sequence.") from error
            if not raw_values:
                raise ValueError(
                    "flux must be a non-empty array in (field, pixel_y, pixel_x) order."
                )
            if pixels_x is None:
                pixels_x = len(raw_values)
            elif len(raw_values) != pixels_x:
                raise ValueError("flux must use one rectangular detector shape for every field.")
            values = []
            for raw_value in raw_values:
                try:
                    number = float(raw_value)
                except (TypeError, ValueError) as error:
                    raise ValueError("flux must contain only finite values.") from error
                if not math.isfinite(number):
                    raise ValueError("flux must contain only finite values.")
                if number < 0.0:
                    raise ValueError("flux must not contain negative values.")
                values.append(number)
            rows.append(tuple(values))
        fields.append(tuple(rows))
    return tuple(fields), len(fields), pixels_y, pixels_x


def _pixel_edges(centers, expected_count, name):
    """Derive uniform detector-pixel outer edges from pixel centers."""
    if isinstance(centers, (str, bytes)):
        raise ValueError(f"{name} must contain at least two values matching the flux axis.")
    try:
        values = tuple(float(value) for value in centers)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only finite values.") from error
    if len(values) != expected_count or expected_count < 2:
        raise ValueError(f"{name} must contain at least two values matching the flux axis.")
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain only finite values.")
    steps = tuple(
        end - start
        for start, end in zip(values[:-1], values[1:])  # noqa: B905, RUF007
    )
    if any(step == 0.0 for step in steps) or any(
        (step > 0.0) != (steps[0] > 0.0) for step in steps
    ):
        raise ValueError(f"{name} must be strictly monotonic.")
    if any(not math.isclose(step, steps[0], rel_tol=1.0e-6, abs_tol=1.0e-9) for step in steps):
        raise ValueError(f"{name} must use uniform detector-pixel spacing.")
    half_step = steps[0] / 2.0
    return (values[0] - half_step, *(value + half_step for value in values))


def _mask_metrics(mask, x_edges, y_edges):
    """Return true-pixel count and physical area for one detector mask."""
    count = 0
    areas = []
    for y_index, row in enumerate(mask):
        pixel_height = abs(y_edges[y_index + 1] - y_edges[y_index])
        for x_index, selected in enumerate(row):
            if selected:
                count += 1
                areas.append(pixel_height * abs(x_edges[x_index + 1] - x_edges[x_index]))
    return count, math.fsum(areas)


def _dense_window_size(pixels_y, pixels_x, detector_fraction):
    """Return an odd local-density window scaled to detector resolution."""
    size = max(1, round(min(pixels_y, pixels_x) * detector_fraction))
    if size % 2 == 0:
        size += 1
    maximum = min(pixels_y, pixels_x)
    if size > maximum:
        size = maximum if maximum % 2 == 1 else maximum - 1
    return max(1, size)


def _box_sum(values, radius):
    """Return clipped square-neighborhood sums from an integral image."""
    rows = len(values)
    columns = len(values[0])
    integral = [[0.0] * (columns + 1) for _unused in range(rows + 1)]
    for y_index, row in enumerate(values):
        running = 0.0
        previous_integral = integral[y_index]
        current_integral = integral[y_index + 1]
        for x_index, value in enumerate(row):
            running += float(value)
            current_integral[x_index + 1] = previous_integral[x_index + 1] + running

    result = []
    for y_index in range(rows):
        y_start = max(0, y_index - radius)
        y_stop = min(rows, y_index + radius + 1)
        row = []
        for x_index in range(columns):
            x_start = max(0, x_index - radius)
            x_stop = min(columns, x_index + radius + 1)
            row.append(
                integral[y_stop][x_stop]
                - integral[y_start][x_stop]
                - integral[y_stop][x_start]
                + integral[y_start][x_start]
            )
        result.append(tuple(row))
    return tuple(result)


def _dense_support_mask(flux, support_mask, window_pixels, settings):
    """Keep supported pixels with sufficient local occupancy and flux density."""
    radius = window_pixels // 2
    positive_counts = _box_sum(support_mask, radius)
    available_counts = _box_sum(
        tuple(tuple(True for _value in row) for row in support_mask),
        radius,
    )
    local_flux_sums = _box_sum(flux, radius)
    local_means = tuple(
        tuple(
            total / available
            for total, available in zip(total_row, available_row)  # noqa: B905
        )
        for total_row, available_row in zip(  # noqa: B905
            local_flux_sums, available_counts
        )
    )
    maximum_local_mean = max(max(row) for row in local_means)
    dense = tuple(
        tuple(
            supported
            and positive >= settings.minimum_local_positive_pixel_fraction * available
            and local_mean
            >= settings.minimum_local_flux_density_fraction_of_field_max * maximum_local_mean
            for supported, positive, available, local_mean in zip(  # noqa: B905
                support_row,
                positive_row,
                available_row,
                mean_row,
            )
        )
        for support_row, positive_row, available_row, mean_row in zip(  # noqa: B905
            support_mask,
            positive_counts,
            available_counts,
            local_means,
        )
    )
    if any(any(row) for row in dense):
        return dense
    return tuple(tuple(row) for row in support_mask)


# Field metadata normalization ----------------------------------------------


def _field_descriptors(field_metadata, field_count):
    if field_metadata is None:
        field_metadata = [{} for _unused in range(field_count)]
    if isinstance(field_metadata, (str, bytes)) or len(field_metadata) != field_count:
        raise ValueError("field_metadata must contain one mapping for every field.")
    descriptors = []
    for index, metadata in enumerate(field_metadata):
        if not isinstance(metadata, Mapping):
            raise ValueError("Every field_metadata entry must be a mapping.")
        source_label = metadata.get("source_label", metadata.get("label", f"field_{index}"))
        if not isinstance(source_label, str) or not source_label:
            source_label = f"field_{index}"
        fov_x_deg, fov_y_deg = _field_fov(metadata)
        descriptors.append(_FieldDescriptor(source_label, fov_x_deg, fov_y_deg))
    return descriptors


def _field_fov(metadata):
    fov = metadata.get("fov_angles_deg")
    if isinstance(fov, Sequence) and not isinstance(fov, (str, bytes)) and len(fov) == 2:
        return _optional_finite_float(fov[0]), _optional_finite_float(fov[1])
    return (
        _optional_finite_float(metadata.get("fov_x_deg")),
        _optional_finite_float(metadata.get("fov_y_deg")),
    )


def _optional_finite_float(value):
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("FOV metadata values must be finite numbers.") from error
    if not math.isfinite(result):
        raise ValueError("FOV metadata values must be finite numbers.")
    return result


__all__ = [
    "DEFAULT_DENSE_ZONE_SETTINGS",
    "BoundingQuad",
    "DenseZoneSettings",
    "EyeboxOverlapResult",
    "FieldBounds",
    "Polygon",
    "Rectangle",
    "calculate_eyebox_overlap",
]
