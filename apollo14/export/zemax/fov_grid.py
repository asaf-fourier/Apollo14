"""Numerically clean FOV sampling for exported optical models.

Apollo14's differentiable tracer uses JAX arrays, normally at float32.  That is
appropriate for optimization but can turn an intended 6 degree export sample
into 6.000013 degrees.  An external prescription should make its sampling grid
explicit and stable, so this module builds the export grid in NumPy float64
from decimal-rounded angles and derives the actual direction and beam frame
from those same angles.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExportFovSample:
    """One clean FOV coordinate and the beam frame derived from it."""

    angle_x_deg: float
    angle_y_deg: float
    direction: np.ndarray
    beam_local_x: np.ndarray
    beam_local_y: np.ndarray


def build_export_fov_samples(base_direction, x_fov_rad, y_fov_rad, num_x, num_y, *, decimals=3):
    """Return a row-major float64 FOV grid built from rounded degree values.

    The rotation order and reference-axis convention match
    :class:`apollo14.projector.FovGrid`.  Unlike that optimization grid, the
    angle axes are rounded *before* rotation, so metadata and traced source
    directions represent the same clean coordinates.
    """
    num_x = _positive_int(num_x, "num_x")
    num_y = _positive_int(num_y, "num_y")
    decimals = _nonnegative_int(decimals, "decimals")
    base_direction = _unit(base_direction, "base_direction")

    x_axis_deg = rounded_fov_axis_deg(x_fov_rad, num_x, decimals=decimals)
    y_axis_deg = rounded_fov_axis_deg(y_fov_rad, num_y, decimals=decimals)
    reference_axis = _reference_axis(base_direction)
    rotation_axis_x = _unit(np.cross(base_direction, reference_axis), "FOV rotation axis x")
    rotation_axis_y = _unit(np.cross(rotation_axis_x, base_direction), "FOV rotation axis y")

    samples = []
    for angle_y_deg in y_axis_deg:
        for angle_x_deg in x_axis_deg:
            direction = _rotate_about_axis(
                base_direction,
                rotation_axis_y,
                math.radians(angle_y_deg),
            )
            direction = _rotate_about_axis(
                direction,
                rotation_axis_x,
                math.radians(angle_x_deg),
            )
            direction = _unit(direction, "exported FOV direction")
            beam_local_x, beam_local_y = _beam_basis(direction)
            samples.append(
                ExportFovSample(
                    angle_x_deg=angle_x_deg,
                    angle_y_deg=angle_y_deg,
                    direction=direction,
                    beam_local_x=beam_local_x,
                    beam_local_y=beam_local_y,
                )
            )
    return samples


def rounded_fov_axis_deg(full_fov_rad, count, *, decimals=3):
    """Return a symmetric FOV axis whose decimal values drive the trace."""
    count = _positive_int(count, "FOV axis count")
    decimals = _nonnegative_int(decimals, "decimals")
    try:
        full_fov_rad = float(full_fov_rad)
    except (TypeError, ValueError) as error:
        raise ValueError("Full FOV must be a finite non-negative angle.") from error
    if not math.isfinite(full_fov_rad) or full_fov_rad < 0.0:
        raise ValueError("Full FOV must be a finite non-negative angle.")
    if count == 1:
        return [0.0]

    full_fov_deg = _clean_decimal(math.degrees(full_fov_rad), decimals)
    raw_axis = np.linspace(-full_fov_deg / 2.0, full_fov_deg / 2.0, count)
    axis = [_clean_decimal(value, decimals) for value in raw_axis]
    if len(set(axis)) != count:
        raise ValueError(
            f"FOV axis samples collapse together at {decimals} decimal places; "
            "increase decimals or reduce the sample count."
        )
    return axis


def _beam_basis(direction):
    """Match ``Projector._compute_basis`` using float64 arithmetic."""
    direction = _unit(direction, "beam direction")
    reference_axis = _reference_axis(direction)
    local_x = _unit(np.cross(direction, reference_axis), "beam local x")
    local_y = _unit(np.cross(local_x, direction), "beam local y")
    return local_x, local_y


def _reference_axis(direction):
    """Choose the same non-parallel reference axis as ``Projector``."""
    if abs(direction[2]) < 0.9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return np.array([1.0, 0.0, 0.0], dtype=np.float64)


def _rotate_about_axis(vector, axis, angle_rad):
    """Rodrigues rotation in float64."""
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return (
        vector * cosine
        + np.cross(axis, vector) * sine
        + axis * np.dot(axis, vector) * (1.0 - cosine)
    )


def _unit(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite three-component vector.")
    norm = float(np.linalg.norm(result))
    if norm == 0.0:
        raise ValueError(f"{name} must be non-zero.")
    return result / norm


def _clean_decimal(value, decimals):
    rounded = round(float(value), decimals)
    return 0.0 if rounded == 0.0 else rounded


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


def _nonnegative_int(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a non-negative integer.") from error
    if result < 0 or result != value:
        raise ValueError(f"{name} must be a non-negative integer.")
    return result
