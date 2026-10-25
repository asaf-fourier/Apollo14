"""Shared schema constants and validation helpers for position-space output.

Every position-space module is copied into a generated Zemax bundle, so these
helpers deliberately use only the Python standard library. Centralizing field,
detector, path, and numeric validation keeps the focused modules consistent.
"""

import math
import os
import re
from pathlib import Path

SCHEMA_VERSION = 2
NATIVE_DETECTOR_ARCHIVE_SCHEMA_VERSION = 3
CSV_FLOAT_FORMAT = ".9e"
SPECTRAL_MODE = "polychromatic_system_wavelengths"
SOURCE_WAVENUMBER = 0
DETECTOR_FLUX_UNIT = "OpticStudio system source-power units per detector pixel"


def _source_label(source):
    label = source.get("label")
    if isinstance(label, str) and label:
        return label
    return f"source_{source.get('index', 'unknown')}"


def _detector_label(detector):
    label = detector.get("comment")
    if isinstance(label, str) and label:
        return label
    return f"detector_{detector.get('index', 'unknown')}"


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


def _fov_angles_deg(source):
    angles = source.get("fov_angles_deg")
    if angles is None:
        return None
    label = _source_label(source)
    if not isinstance(angles, (list, tuple)) or len(angles) != 2:
        raise ValueError(f"Source {label!r} fov_angles_deg must contain exactly two values.")
    return (
        _finite_float(angles[0], f"Source {label!r} FOV x angle"),
        _finite_float(angles[1], f"Source {label!r} FOV y angle"),
    )


def _pixel_centers(half_width_mm, pixels):
    width = 2.0 * half_width_mm
    return [-half_width_mm + (index + 0.5) * width / pixels for index in range(pixels)]


def _relative_path(path, base):
    return Path(path).relative_to(base).as_posix()


def _relative_path_from(path, base):
    """Return a slash-normalized path suitable for a manifest or ZPL macro."""
    return os.path.relpath(Path(path), Path(base)).replace(os.sep, "/")


def _unique_indices(entries, kind):
    result = {}
    for position, entry in enumerate(entries):
        index = _positive_int(entry.get("index"), f"{kind} index")
        if index in result:
            raise ValueError(f"Duplicate {kind} index {index}.")
        result[index] = position
    return result


def _zpl_string(value):
    """Quote a generated path for a ZPL string literal."""
    value = str(value).replace("/", "\\")
    if '"' in value or "\r" in value or "\n" in value:
        raise ValueError(f"Cannot represent path {value!r} as a ZPL string.")
    return f'"{value}"'


def _zpl_comment_text(value):
    """Make generated human-facing metadata safe for a ZPL PRINT string."""
    return str(value).replace("\r", " ").replace("\n", " ").replace('"', "'")


def _positive_int(value, label):
    original = value
    if isinstance(original, bool):
        raise ValueError(f"{label} must be a positive integer; got {original!r}.")
    try:
        result = int(original)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a positive integer; got {original!r}.") from error
    if result <= 0 or result != original:
        raise ValueError(f"{label} must be a positive integer; got {original!r}.")
    return result


def _finite_float(value, label):
    """Return a finite floating-point value with a contextual error."""
    original = value
    try:
        result = float(original)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number; got {original!r}.") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number; got {original!r}.")
    return result


def _positive_float(value, label):
    result = _finite_float(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be a positive number; got {value!r}.")
    return result


def _product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def _slug(value):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_") or "unnamed"
