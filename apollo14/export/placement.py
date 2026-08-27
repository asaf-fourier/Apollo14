"""Map Apollo14 world geometry onto Zemax non-sequential object placements.

Apollo14 places a planar element with a ``position`` and a ``normal`` plus an
implicit local frame from :func:`apollo14.geometry.compute_local_axes`. A Zemax
NSC object instead carries a position and three tilt angles, and its geometry is
defined in a local frame whose ``+Z`` is the surface normal.

Both conventions are honoured here by choosing the Zemax local frame so that
**local +X is always world +X**. Every Perseus element has its normal in the
world y–z plane (the pantoscopic tilt, the projector tilt and the mirror tilt
all rotate about world x), so that choice reduces the placement to a *single*
``Tilt About X`` — one rotation, one unambiguous sign, no Euler-order
assumptions. Elements whose normal leaves the y–z plane are rejected rather
than silently mis-tilted.

Coordinates map through unchanged: Apollo14's internal length unit is the
millimetre (``apollo14.units.mm == 1.0``) and the exported Zemax system is set
to millimetre lens units, so world ``(x, y, z)`` is Zemax global ``(X, Y, Z)``.

Rotation convention
-------------------
``Tilt About X`` is taken to be a right-handed rotation of the object about its
local x axis, i.e. it carries ``+Z`` to ``(0, -sin, cos)``. This matches
:func:`apollo14.geometry.rotate_vectors`, whose positive angle is likewise
right-handed. Source directions additionally need a second tilt; see
:func:`direction_to_tilts` for the ordering assumption that entails.
"""

import math
from typing import NamedTuple

import numpy as np

# A normal is treated as lying in the world y–z plane when its x component is
# below this. Perseus normals are exactly zero there up to float32 round-off.
IN_PLANE_TOLERANCE = 1e-5

# Axis-identification tolerance when matching Apollo14's local frame against the
# exported one — these are unit vectors either parallel or perpendicular, so the
# comparison is never marginal.
AXIS_MATCH_TOLERANCE = 1e-3


class Placement(NamedTuple):
    """Where a Zemax NSC object sits: position plus the three tilt angles."""
    position: tuple[float, float, float]     # world mm
    tilt_deg: tuple[float, float, float]     # (about X, about Y, about Z)


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def require_in_yz_plane(normal, what: str) -> np.ndarray:
    """Return ``normal`` as a unit vector, rejecting any x component.

    The single-tilt placement below is only valid for normals in the world
    y–z plane. Raising here keeps a mis-oriented element from being exported
    with a plausible-looking but wrong tilt.
    """
    unit_normal = _unit(normal)
    if abs(unit_normal[0]) > IN_PLANE_TOLERANCE:
        raise ValueError(
            f"{what}: normal {tuple(np.round(unit_normal, 6))} has an x "
            f"component of {unit_normal[0]:.3e}, so it cannot be placed with a "
            "single Tilt About X. Extend placement.py with a two-axis tilt "
            "before exporting this element.")
    return unit_normal


def tilt_about_x_for_normal(normal, what: str = "element") -> float:
    """Degrees of ``Tilt About X`` that carry local +Z onto ``normal``."""
    unit_normal = require_in_yz_plane(normal, what)
    return math.degrees(math.atan2(-unit_normal[1], unit_normal[2]))


def zemax_plane_axes(normal, what: str = "element"):
    """The exported local frame for a planar element: ``(local_x, local_y)``.

    ``local_x`` is world +X and ``local_y`` completes a right-handed frame with
    the normal as local +Z, so the frame is reached from world axes by the
    single rotation :func:`tilt_about_x_for_normal` returns.
    """
    unit_normal = require_in_yz_plane(normal, what)
    local_x = np.array([1.0, 0.0, 0.0])
    local_y = np.array([0.0, unit_normal[2], -unit_normal[1]])
    return local_x, local_y


def half_extents_in_zemax_frame(element, what: str = "element"):
    """``(half_x, half_y)`` of a planar element in the exported local frame.

    Apollo14 measures ``width`` along its own ``local_x`` and ``height`` along
    its ``local_y``; :func:`apollo14.geometry.compute_local_axes` may pick those
    axes swapped or negated relative to the exported frame. The two frames share
    a plane, so each Apollo14 axis is parallel to one exported axis — this
    figures out which and returns the half extents in exported order.
    """
    element_local_x = _unit(element._local_x)
    export_local_x, export_local_y = zemax_plane_axes(element.normal, what)

    along_export_x = abs(float(np.dot(element_local_x, export_local_x)))
    along_export_y = abs(float(np.dot(element_local_x, export_local_y)))

    if abs(along_export_x - 1.0) < AXIS_MATCH_TOLERANCE:
        return float(element.width) / 2.0, float(element.height) / 2.0
    if abs(along_export_y - 1.0) < AXIS_MATCH_TOLERANCE:
        # Apollo14's width axis is the exported height axis — swap the extents
        # so the rectangle keeps its physical shape.
        return float(element.height) / 2.0, float(element.width) / 2.0
    raise ValueError(
        f"{what}: local x axis {tuple(np.round(element_local_x, 6))} is "
        "parallel to neither exported axis "
        f"({tuple(np.round(export_local_x, 6))}, "
        f"{tuple(np.round(export_local_y, 6))}); cannot map width/height.")


def planar_placement(element, what: str = "element") -> Placement:
    """Placement for a planar Apollo14 element (mirror, aperture, pupil)."""
    position = np.asarray(element.position, dtype=float)
    return Placement(
        position=tuple(float(v) for v in position),
        tilt_deg=(tilt_about_x_for_normal(element.normal, what), 0.0, 0.0),
    )


def tilts_to_rotation(tilt_x_deg: float, tilt_y_deg: float,
                      tilt_z_deg: float) -> np.ndarray:
    """The object rotation matrix Zemax builds from three tilt angles.

    OpticStudio rotates an object about x first, then y, then z, taking each
    rotation about the object's own (already rotated) axes — equivalently, an
    extrinsic z-y-x sequence. Both descriptions compose to::

        R = Rx(tilt_x) · Ry(tilt_y) · Rz(tilt_z)

    The matrix maps object-local axes onto world axes, so its columns are the
    world directions of local +X, +Y and +Z.
    """
    angle_x = math.radians(tilt_x_deg)
    angle_y = math.radians(tilt_y_deg)
    angle_z = math.radians(tilt_z_deg)

    rotation_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(angle_x), -math.sin(angle_x)],
        [0.0, math.sin(angle_x), math.cos(angle_x)],
    ])
    rotation_y = np.array([
        [math.cos(angle_y), 0.0, math.sin(angle_y)],
        [0.0, 1.0, 0.0],
        [-math.sin(angle_y), 0.0, math.cos(angle_y)],
    ])
    rotation_z = np.array([
        [math.cos(angle_z), -math.sin(angle_z), 0.0],
        [math.sin(angle_z), math.cos(angle_z), 0.0],
        [0.0, 0.0, 1.0],
    ])
    return rotation_x @ rotation_y @ rotation_z


def rotation_to_tilts(rotation, what: str = "object"):
    """Decompose a rotation matrix into ``(tilt_x, tilt_y, tilt_z)`` degrees.

    Inverts :func:`tilts_to_rotation`. Raises when the rotation is within a
    degree of the gimbal-lock pose (local +Z along world ±X), where the x and z
    tilts stop being separable — no Perseus beam comes near it, and silently
    returning one of the infinitely many solutions would be worse than failing.
    """
    rotation = np.asarray(rotation, dtype=float)
    sin_tilt_y = float(np.clip(rotation[0, 2], -1.0, 1.0))
    cos_tilt_y = math.sqrt(max(0.0, 1.0 - sin_tilt_y ** 2))

    if cos_tilt_y < math.sin(math.radians(1.0)):
        raise ValueError(
            f"{what}: orientation is at gimbal lock (local +Z within 1° of "
            "world ±X); tilt about x and z are degenerate there.")

    tilt_y = math.degrees(math.asin(sin_tilt_y))
    tilt_x = math.degrees(math.atan2(-rotation[1, 2], rotation[2, 2]))
    tilt_z = math.degrees(math.atan2(-rotation[0, 1], rotation[0, 0]))
    return tilt_x, tilt_y, tilt_z


def frame_to_tilts(local_x, local_y, local_z, what: str = "object"):
    """Tilts that carry the object's local axes onto the given world axes.

    Used for the projector, whose *roll* matters: the beam cross-section is
    10 × 2 mm, so exporting the right pointing direction with the wrong roll
    would present a differently-shaped beam to the 11 × 2 mm stop. Aiming the
    direction alone is not enough — the full frame has to be matched.
    """
    rotation = np.stack([_unit(local_x), _unit(local_y), _unit(local_z)],
                        axis=1)

    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > 1e-6:
        raise ValueError(
            f"{what}: local axes are not a right-handed orthonormal frame "
            f"(determinant {determinant:.6f}).")

    tilts = rotation_to_tilts(rotation, what)
    reconstructed = tilts_to_rotation(*tilts)
    error = float(np.max(np.abs(reconstructed - rotation)))
    # Frames arrive from float32 JAX arrays, so the floor is ~1e-7; this catches
    # a wrong decomposition, not round-off.
    if error > 1e-6:
        raise ValueError(
            f"{what}: tilt decomposition does not reproduce the frame "
            f"(max element error {error:.3e}).")
    return tilts


def rectangle_frame_to_tilts(width_axis, height_axis, normal,
                             what: str = "object"):
    """Tilts for a centred rectangle spanning ``width_axis`` × ``height_axis``.

    Apollo14's projector builds its beam basis as ``local_y = local_x ×
    direction``, which makes ``(local_x, local_y, direction)`` *left*-handed.
    A Zemax object frame must be right-handed, so the width axis is flipped when
    needed. That is a genuine no-op for the emitted beam: the rectangle is
    centred on the object, so reversing the axis it is measured along maps it
    onto itself.
    """
    width_axis = _unit(width_axis)
    height_axis = _unit(height_axis)
    normal = _unit(normal)

    if float(np.dot(np.cross(width_axis, height_axis), normal)) < 0.0:
        width_axis = -width_axis

    return frame_to_tilts(width_axis, height_axis, normal, what)


def direction_to_tilts(direction, what: str = "object"):
    """``(tilt_x, tilt_y)`` degrees aiming local +Z along ``direction``.

    Leaves roll unspecified — only for objects where roll is irrelevant. Under
    ``R = Rx·Ry·Rz`` a local +Z of ``(sin ty, −sin tx·cos ty, cos tx·cos ty)``
    gives the closed form below.
    """
    unit_direction = _unit(direction)
    tilt_y = math.degrees(math.asin(float(np.clip(unit_direction[0], -1.0, 1.0))))
    tilt_x = math.degrees(math.atan2(-float(unit_direction[1]),
                                     float(unit_direction[2])))
    return tilt_x, tilt_y


def to_local_frame(world_points, pivot, tilt_deg_about_x):
    """Undo a world rotation about x, returning points in an object-local frame.

    Perseus builds its chassis axis-aligned and then applies the pantoscopic
    tilt about the combiner centre. Exporting the *tilted* vertices with a
    zero-tilt object would work but hides the tilt from the NSC editor; undoing
    it here lets the object carry its real position and tilt, so the tilt stays
    adjustable in OpticStudio.
    """
    world_points = np.asarray(world_points, dtype=float)
    pivot = np.asarray(pivot, dtype=float)
    angle = math.radians(tilt_deg_about_x)
    cos_angle, sin_angle = math.cos(angle), math.sin(angle)
    # Inverse of a right-handed rotation about +X.
    inverse_rotation = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_angle, sin_angle],
        [0.0, -sin_angle, cos_angle],
    ])
    return (world_points - pivot) @ inverse_rotation.T


def to_local_direction(world_directions, tilt_deg_about_x):
    """Undo the same rotation for direction vectors — no pivot translation."""
    world_directions = np.asarray(world_directions, dtype=float)
    angle = math.radians(tilt_deg_about_x)
    cos_angle, sin_angle = math.cos(angle), math.sin(angle)
    inverse_rotation = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_angle, sin_angle],
        [0.0, -sin_angle, cos_angle],
    ])
    return world_directions @ inverse_rotation.T


def from_local_frame(local_points, pivot, tilt_deg_about_x):
    """Inverse of :func:`to_local_frame` — used to verify the round trip."""
    local_points = np.asarray(local_points, dtype=float)
    pivot = np.asarray(pivot, dtype=float)
    angle = math.radians(tilt_deg_about_x)
    cos_angle, sin_angle = math.cos(angle), math.sin(angle)
    rotation = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_angle, -sin_angle],
        [0.0, sin_angle, cos_angle],
    ])
    return local_points @ rotation.T + pivot
