"""Portable convex geometry used by the eyebox-overlap analysis.

This module owns shapes and geometry algorithms only.  Dense-zone extraction
from detector flux belongs in :mod:`eyebox_overlap_core`, while file and HTML
output belongs in :mod:`eyebox_overlap_report`.  Keeping the geometry here
makes those boundaries visible and lets the local and Zemax workflows execute
the same implementation.

The file is copied verbatim into generated Zemax bundles, so it deliberately
uses only the Python standard library and syntax supported by Python 3.8.
"""

import math
from dataclasses import dataclass
from itertools import permutations

_GEOMETRY_TOLERANCE = 1.0e-10


# Public value objects -------------------------------------------------------


@dataclass(frozen=True)
class Rectangle:
    """One detector-aligned rectangle, with exclusive pixel-stop indices."""

    x_start_index: int
    x_stop_index: int
    y_start_index: int
    y_stop_index: int
    x_min_mm: float
    x_max_mm: float
    y_min_mm: float
    y_max_mm: float

    @property
    def width_mm(self):
        return self.x_max_mm - self.x_min_mm

    @property
    def height_mm(self):
        return self.y_max_mm - self.y_min_mm

    @property
    def area_mm2(self):
        return self.width_mm * self.height_mm

    @property
    def pixel_count(self):
        return (self.x_stop_index - self.x_start_index) * (
            self.y_stop_index - self.y_start_index
        )

    def as_dict(self):
        return {
            "pixel_span": {
                "x_start_index": self.x_start_index,
                "x_stop_index_exclusive": self.x_stop_index,
                "y_start_index": self.y_start_index,
                "y_stop_index_exclusive": self.y_stop_index,
            },
            "bounds_mm": {
                "x_min": self.x_min_mm,
                "x_max": self.x_max_mm,
                "y_min": self.y_min_mm,
                "y_max": self.y_max_mm,
            },
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "area_mm2": self.area_mm2,
            "pixel_count": self.pixel_count,
        }


@dataclass(frozen=True)
class Polygon:
    """One counter-clockwise polygon in detector-local millimetres."""

    vertices_mm: tuple

    @property
    def area_mm2(self):
        return _polygon_area(self.vertices_mm)

    @property
    def side_lengths_mm(self):
        return tuple(
            math.hypot(end[0] - start[0], end[1] - start[1])
            for start, end in _cyclic_pairs(self.vertices_mm)
        )

    def as_dict(self):
        return {
            "vertices_mm": [list(vertex) for vertex in self.vertices_mm],
            "vertex_axis_order": ["x", "y"],
            "vertex_count": len(self.vertices_mm),
            "side_lengths_mm": list(self.side_lengths_mm),
            "area_mm2": self.area_mm2,
        }


@dataclass(frozen=True)
class BoundingQuad(Polygon):
    """Four-sided convex approximation inscribed in an overlap polygon."""

    def __post_init__(self):
        if len(self.vertices_mm) != 4:
            raise ValueError("A bounding quad must contain exactly four vertices.")
        if _signed_polygon_area(self.vertices_mm) <= _GEOMETRY_TOLERANCE:
            raise ValueError("A bounding quad must have counter-clockwise positive area.")
        if min(self.side_lengths_mm) <= _GEOMETRY_TOLERANCE:
            raise ValueError("A bounding quad must not contain zero-length sides.")

    @property
    def side_names(self):
        """Name each edge from its outward direction, in vertex order."""
        outward_normals = []
        for start, end in _cyclic_pairs(self.vertices_mm):
            edge_x = end[0] - start[0]
            edge_y = end[1] - start[1]
            length = math.hypot(edge_x, edge_y)
            outward_normals.append((edge_y / length, -edge_x / length))
        named_directions = (
            ("Top", (0.0, 1.0)),
            ("Bottom", (0.0, -1.0)),
            ("Left", (-1.0, 0.0)),
            ("Right", (1.0, 0.0)),
        )
        assignment = max(
            permutations(named_directions),
            key=lambda candidate: sum(
                normal[0] * direction[0] + normal[1] * direction[1]
                for normal, (_name, direction) in zip(  # noqa: B905
                    outward_normals, candidate
                )
            ),
        )
        return tuple(name for name, _direction in assignment)

    def as_dict(self):
        payload = super().as_dict()
        payload["side_names"] = list(self.side_names)
        return payload


# Detector-mask boundaries --------------------------------------------------


def rectangle_from_spans(x_start, x_stop, y_start, y_stop, x_edges, y_edges):
    """Build a physical rectangle from exclusive detector-pixel spans."""
    if x_start >= x_stop or y_start >= y_stop:
        raise ValueError("A rectangle needs a non-empty pixel span.")
    x_first, x_last = float(x_edges[x_start]), float(x_edges[x_stop])
    y_first, y_last = float(y_edges[y_start]), float(y_edges[y_stop])
    return Rectangle(
        x_start_index=x_start,
        x_stop_index=x_stop,
        y_start_index=y_start,
        y_stop_index=y_stop,
        x_min_mm=min(x_first, x_last),
        x_max_mm=max(x_first, x_last),
        y_min_mm=min(y_first, y_last),
        y_max_mm=max(y_first, y_last),
    )


def convex_bounding_polygon(mask, x_edges, y_edges):
    """Return the convex hull of all selected pixels' physical corners."""
    corners = set()
    for y_index, row in enumerate(mask):
        for x_index, selected in enumerate(row):
            if not selected:
                continue
            corners.update(
                (
                    (x_edges[x_index], y_edges[y_index]),
                    (x_edges[x_index + 1], y_edges[y_index]),
                    (x_edges[x_index + 1], y_edges[y_index + 1]),
                    (x_edges[x_index], y_edges[y_index + 1]),
                )
            )
    if not corners:
        raise ValueError("A field bounding polygon needs at least one supported pixel.")
    hull = _convex_hull(corners)
    return Polygon(vertices_mm=tuple(_clean_point(vertex) for vertex in hull))


# Convex overlap and contained quad -----------------------------------------


def points_in_convex_polygon(points, polygon):
    """Return one containment flag per point, including the boundary."""
    orientation = 1.0 if _signed_polygon_area(polygon.vertices_mm) >= 0.0 else -1.0
    result = []
    for point in points:
        x, y = float(point[0]), float(point[1])
        inside = True
        for start, end in _cyclic_pairs(polygon.vertices_mm):
            cross = (end[0] - start[0]) * (y - start[1]) - (end[1] - start[1]) * (
                x - start[0]
            )
            if orientation * cross < -_GEOMETRY_TOLERANCE:
                inside = False
                break
        result.append(inside)
    return result


def largest_inscribed_quad(polygon):
    """Return the maximum-area four-vertex approximation of a convex polygon."""
    if polygon is None:
        return None
    vertices = polygon.vertices_mm
    vertex_count = len(vertices)
    if vertex_count < 4:
        return None

    best_vertices = None
    best_key = None
    for first_index in range(vertex_count - 3):
        first = vertices[first_index]
        for third_index in range(first_index + 2, vertex_count - 1):
            third = vertices[third_index]
            second_index = max(
                range(first_index + 1, third_index),
                key=lambda index: (_turn(first, vertices[index], third), -index),
            )
            fourth_index = max(
                range(third_index + 1, vertex_count),
                key=lambda index: (_turn(first, third, vertices[index]), -index),
            )
            indices = (first_index, second_index, third_index, fourth_index)
            candidate = tuple(vertices[index] for index in indices)
            side_lengths = tuple(
                math.hypot(end[0] - start[0], end[1] - start[1])
                for start, end in _cyclic_pairs(candidate)
            )
            key = (
                _polygon_area(candidate),
                min(side_lengths),
                -max(side_lengths),
                -float(first_index),
                -float(second_index),
                -float(third_index),
                -float(fourth_index),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_vertices = candidate

    if best_vertices is None:
        return None
    quad = BoundingQuad(vertices_mm=best_vertices)
    if not all(points_in_convex_polygon(quad.vertices_mm, polygon)):
        raise RuntimeError("Internal error: an inscribed quad escaped its field polygon.")
    return quad


def intersect_convex_polygons(polygons):
    """Return the exact convex intersection of all supplied polygons."""
    if not polygons:
        return None
    subject = list(polygons[0].vertices_mm)
    for clip_polygon in polygons[1:]:
        subject = _clip_convex_polygon(subject, clip_polygon.vertices_mm)
        if len(subject) < 3:
            return None
    subject = _clean_polygon(subject)
    if len(subject) < 3 or _polygon_area(subject) <= _GEOMETRY_TOLERANCE:
        return None
    if _signed_polygon_area(subject) < 0.0:
        subject.reverse()
    return Polygon(vertices_mm=tuple(_clean_point(point) for point in subject))


def polygon_dict(polygon):
    """Serialize an optional polygon for a report payload."""
    return None if polygon is None else polygon.as_dict()


# Internal convex-polygon primitives ---------------------------------------


def _convex_hull(points):
    ordered = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(ordered) < 3:
        raise ValueError("A supported detector pixel must provide a two-dimensional hull.")
    lower = []
    for point in ordered:
        while len(lower) >= 2 and _turn(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and _turn(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _clip_convex_polygon(subject, clip):
    output = list(subject)
    clip_vertices = list(clip)
    if _signed_polygon_area(clip_vertices) < 0.0:
        clip_vertices.reverse()
    for clip_start, clip_end in _cyclic_pairs(clip_vertices):
        input_vertices = output
        output = []
        if not input_vertices:
            break
        segment_start = input_vertices[-1]
        start_inside = _inside_half_plane(segment_start, clip_start, clip_end)
        for segment_end in input_vertices:
            end_inside = _inside_half_plane(segment_end, clip_start, clip_end)
            if end_inside:
                if not start_inside:
                    output.append(
                        _line_intersection(segment_start, segment_end, clip_start, clip_end)
                    )
                output.append(segment_end)
            elif start_inside:
                output.append(
                    _line_intersection(segment_start, segment_end, clip_start, clip_end)
                )
            segment_start = segment_end
            start_inside = end_inside
    return _clean_polygon(output)


def _inside_half_plane(point, edge_start, edge_end):
    return (
        _cross_2d(
            (edge_end[0] - edge_start[0], edge_end[1] - edge_start[1]),
            (point[0] - edge_start[0], point[1] - edge_start[1]),
        )
        >= -_GEOMETRY_TOLERANCE
    )


def _line_intersection(segment_start, segment_end, line_start, line_end):
    segment_direction = (
        segment_end[0] - segment_start[0],
        segment_end[1] - segment_start[1],
    )
    line_direction = (line_end[0] - line_start[0], line_end[1] - line_start[1])
    denominator = _cross_2d(segment_direction, line_direction)
    if abs(denominator) <= _GEOMETRY_TOLERANCE:
        return _clean_point(segment_end)
    amount = (
        _cross_2d(
            (line_start[0] - segment_start[0], line_start[1] - segment_start[1]),
            line_direction,
        )
        / denominator
    )
    return _clean_point(
        (
            segment_start[0] + amount * segment_direction[0],
            segment_start[1] + amount * segment_direction[1],
        )
    )


def _clean_polygon(points):
    cleaned = []
    for point in points:
        point = _clean_point(point)
        if not cleaned or not _points_close(point, cleaned[-1]):
            cleaned.append(point)
    if len(cleaned) > 1 and _points_close(cleaned[0], cleaned[-1]):
        cleaned.pop()

    changed = True
    while changed and len(cleaned) > 3:
        changed = False
        for index, point in enumerate(cleaned):
            previous = cleaned[index - 1]
            following = cleaned[(index + 1) % len(cleaned)]
            if abs(_turn(previous, point, following)) <= _GEOMETRY_TOLERANCE:
                cleaned.pop(index)
                changed = True
                break
    return cleaned


def _polygon_area(vertices):
    return abs(_signed_polygon_area(vertices))


def _signed_polygon_area(vertices):
    if len(vertices) < 3:
        return 0.0
    return 0.5 * math.fsum(
        first[0] * second[1] - second[0] * first[1]
        for first, second in _cyclic_pairs(vertices)
    )


def _cyclic_pairs(values):
    values = tuple(values)
    return zip(values, values[1:] + values[:1])  # noqa: B905


def _turn(origin, first, second):
    return (first[0] - origin[0]) * (second[1] - origin[1]) - (
        first[1] - origin[1]
    ) * (second[0] - origin[0])


def _cross_2d(first, second):
    return float(first[0] * second[1] - first[1] * second[0])


def _clean_point(point):
    values = []
    for coordinate in point:
        value = float(coordinate)
        values.append(0.0 if abs(value) <= _GEOMETRY_TOLERANCE else value)
    return values[0], values[1]


def _points_close(first, second):
    return all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=_GEOMETRY_TOLERANCE)
        for a, b in zip(first, second)  # noqa: B905
    )


__all__ = [
    "BoundingQuad",
    "Polygon",
    "Rectangle",
    "convex_bounding_polygon",
    "intersect_convex_polygons",
    "largest_inscribed_quad",
    "points_in_convex_polygon",
    "polygon_dict",
    "rectangle_from_spans",
]
