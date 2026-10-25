"""Create a local numeric overlap report from one exported eyebox detector cube.

Example:

    python examples/analyze_eyebox_overlap.py \
        examples/reports/export_perseus_zemax/2026-10-19_12-54-54/out/eyebox/position_space.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

from apollo14.eyebox_overlap import (
    calculate_eyebox_overlap,
    load_position_space,
    write_eyebox_overlap_report,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Intersect the numeric field dense-zone polygons and approximate their overlap "
            "with a contained quad."
        )
    )
    parser.add_argument(
        "position_space_npz",
        type=Path,
        help="Path to the eyebox position_space.npz file.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional path to position_space.json; defaults to the sibling file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Report output directory. Defaults to "
            "<position-space-npz>/../../reports/eyebox_field_overlap."
        ),
    )
    return parser.parse_args()


def _default_output_directory(npz_path: Path) -> Path:
    """Place a derived report beside the existing Zemax output reports."""
    return npz_path.resolve().parent.parent / "reports" / "eyebox_field_overlap"


def main() -> None:
    """Load one detector cube, calculate overlap, and write local artifacts."""
    arguments = _arguments()
    position_space = load_position_space(arguments.position_space_npz, arguments.metadata)
    detector = position_space.metadata.get("detector", {})
    detector_label = detector.get("label", "eyebox") if isinstance(detector, dict) else "eyebox"
    fields = position_space.metadata.get("fields")
    if not isinstance(fields, list):
        fields = None

    result = calculate_eyebox_overlap(
        position_space.flux,
        position_space.pixel_x_center_mm,
        position_space.pixel_y_center_mm,
        detector_label=detector_label,
        field_metadata=fields,
    )
    output_directory = arguments.output_dir or _default_output_directory(position_space.npz_path)
    paths = write_eyebox_overlap_report(
        result,
        output_directory,
        source_npz_path=position_space.npz_path,
        source_metadata_path=position_space.metadata_path,
    )

    polygon_intersection = result.overlap_polygon
    contained_quad = result.contained_overlap_quad
    print(f"Wrote local report: {paths['html']}")
    print(f"Wrote contained overlap quad vertices and sides: {paths['contained_overlap_quad_csv']}")
    if polygon_intersection is None:
        print("Field bounding polygons do not have a common area intersection.")
    else:
        print(
            "Field-polygon intersection: "
            f"{polygon_intersection.area_mm2:.6g} mm² "
            f"({len(polygon_intersection.vertices_mm)} vertices)"
        )
    if contained_quad is None:
        print("The field-polygon intersection cannot contain a four-vertex quad.")
    else:
        print(f"Contained overlap quad: {contained_quad.area_mm2:.6g} mm²")


if __name__ == "__main__":
    main()
