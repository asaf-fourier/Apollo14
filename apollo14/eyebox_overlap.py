"""Local NPZ facade for the portable eyebox-overlap implementation.

Geometry and report generation live in the standard-library Zemax reporting
runtime. This module adds NumPy archive loading for local development, then
re-exports the shared public API.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from apollo14.export.zemax.reporting.eyebox_overlap_core import (
    DEFAULT_DENSE_ZONE_SETTINGS,
    BoundingQuad,
    DenseZoneSettings,
    EyeboxOverlapResult,
    FieldBounds,
    Polygon,
    Rectangle,
    calculate_eyebox_overlap,
)
from apollo14.export.zemax.reporting.eyebox_overlap_geometry import (
    intersect_convex_polygons as _intersect_convex_polygons,
)
from apollo14.export.zemax.reporting.eyebox_overlap_geometry import (
    largest_inscribed_quad as _largest_inscribed_quad,
)
from apollo14.export.zemax.reporting.eyebox_overlap_geometry import (
    points_in_convex_polygon as _points_in_convex_polygon,
)
from apollo14.export.zemax.reporting.eyebox_overlap_report import (
    write_eyebox_overlap_report,
)

__all__ = [
    "DEFAULT_DENSE_ZONE_SETTINGS",
    "BoundingQuad",
    "DenseZoneSettings",
    "EyeboxOverlapResult",
    "FieldBounds",
    "Polygon",
    "PositionSpaceData",
    "Rectangle",
    "_intersect_convex_polygons",
    "_largest_inscribed_quad",
    "_points_in_convex_polygon",
    "calculate_eyebox_overlap",
    "load_position_space",
    "write_eyebox_overlap_report",
]


@dataclass(frozen=True)
class PositionSpaceData:
    """Numeric position-space data and its optional JSON metadata."""

    flux: np.ndarray
    pixel_x_center_mm: np.ndarray
    pixel_y_center_mm: np.ndarray
    metadata: Mapping[str, Any]
    npz_path: Path
    metadata_path: Path | None


def load_position_space(
    npz_path: str | Path,
    metadata_path: str | Path | None = None,
) -> PositionSpaceData:
    """Load numeric arrays and metadata from an exported detector directory.

    ``metadata_path`` defaults to the sibling ``position_space.json`` file.
    When that file is absent, the JSON embedded in the NPZ archive is used.
    """
    npz_path = Path(npz_path).resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"Position-space NPZ file does not exist: {npz_path}")

    requested_metadata_path = None if metadata_path is None else Path(metadata_path).resolve()
    default_metadata_path = npz_path.with_name("position_space.json")
    resolved_metadata_path = requested_metadata_path
    if resolved_metadata_path is None and default_metadata_path.is_file():
        resolved_metadata_path = default_metadata_path

    with np.load(npz_path, allow_pickle=False) as archive:
        required = {"flux", "pixel_x_center_mm", "pixel_y_center_mm"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(
                "Position-space NPZ file is missing required arrays: " + ", ".join(missing) + "."
            )
        flux = np.array(archive["flux"], dtype=float, copy=True)
        x_centers = np.array(archive["pixel_x_center_mm"], dtype=float, copy=True)
        y_centers = np.array(archive["pixel_y_center_mm"], dtype=float, copy=True)
        archive_metadata = _metadata_from_archive(archive)

    if resolved_metadata_path is not None:
        if not resolved_metadata_path.is_file():
            raise FileNotFoundError(
                f"Position-space metadata file does not exist: {resolved_metadata_path}"
            )
        metadata = json.loads(resolved_metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = archive_metadata

    if not isinstance(metadata, dict):
        raise ValueError("Position-space metadata must be a JSON object.")
    return PositionSpaceData(
        flux=flux,
        pixel_x_center_mm=x_centers,
        pixel_y_center_mm=y_centers,
        metadata=metadata,
        npz_path=npz_path,
        metadata_path=resolved_metadata_path,
    )


def _metadata_from_archive(archive: Any) -> dict[str, Any]:
    """Read the optional metadata JSON stored inside one NPZ archive."""
    if "metadata.json" not in archive.files:
        return {}
    raw = archive["metadata.json"]
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise ValueError("Embedded position-space metadata must be text JSON.")
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise ValueError("Embedded position-space metadata must be a JSON object.")
    return decoded
