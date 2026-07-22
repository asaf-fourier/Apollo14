"""Perseus combiner system definition.

Perseus is a variant of the Talos / flat combiner with two rigid-body
tilts plus a reworked mirror stack and chassis:

- **Pantoscopic tilt** — the combiner (chassis + mirror stack) is rotated
  forward about a horizontal (world-x) axis **through the exit-pupil
  center**. The projector, aperture, and pupil stay put, so the eye box
  is fixed in space and the tilted combiner rotates beneath it (the
  "combiner only, about the pupil" convention).
- **Projector tilt** — the emission axis is rotated about the same
  world-x axis, away from straight-down ``[0, -1, 0]``. The beam-defining
  aperture follows the beam (its normal tracks the projector direction);
  the aperture is *not* affected by the pantoscopic tilt.

Sign conventions (right-handed about world-x ``[1, 0, 0]``):

- ``pantoscopic_tilt > 0`` rotates the **top** of the combiner toward the
  eye (+z) and the bottom away. Flip the sign for the opposite lean.
- ``projector_tilt > 0`` swings the downward beam toward −z (away from
  the eye). ``0`` is straight down, matching Talos / the flat system.

The geometry derivation lives in :func:`build_perseus_geometry` so both
the baked-reflectance reference :func:`build_perseus_system` here and the
differentiable :func:`helios.perseus_params.build_parametrized_perseus`
share one source of truth. ``apollo14`` never imports ``helios``.
"""

import math
from typing import NamedTuple

import jax.numpy as jnp

from apollo14.combiner import DEFAULT_LIGHT_POSITION, compensated_reflectances
from apollo14.elements.aperture import RectangularAperture
from apollo14.elements.glass_block import GlassBlock
from apollo14.elements.partial_mirror import (
    DEFAULT_MIRROR_WAVELENGTHS,
    PartialMirror,
)
from apollo14.elements.pupil import RectangularPupil
from apollo14.geometry import normalize, rotate_points, rotate_vectors
from apollo14.materials import agc_m074, air
from apollo14.system import OpticalSystem
from apollo14.units import deg, mm

# ── Rotation axis / nominal beam ────────────────────────────────────────────

X_AXIS = jnp.array([1.0, 0.0, 0.0])          # pantoscopic / projector tilt axis
STRAIGHT_DOWN = jnp.array([0.0, -1.0, 0.0])  # untilted projector direction

# ── Perseus defaults (differ from Talos: 8 mirrors, thin chassis, tilted) ───

PERSEUS_LIGHT_POSITION = DEFAULT_LIGHT_POSITION      # [7, 31, 1] mm
PERSEUS_NUM_MIRRORS = 8
PERSEUS_DISTANCE_BETWEEN_MIRRORS = 0.93 * mm
PERSEUS_MIRROR_ANGLE = 48.0 * deg
PERSEUS_CHASSIS_X = 14.0 * mm
PERSEUS_CHASSIS_Y = 20.0 * mm
PERSEUS_CHASSIS_Z = 1.7 * mm                         # thin (Talos is 2.0 mm)
PERSEUS_SKEW_ANGLE = 6.0 * deg                       # chassis z-wedge (prism)
PERSEUS_FIRST_MIRROR_OFFSET_Y = 5.0 * mm
PERSEUS_EYE_RELIEF = 15.0 * mm
PERSEUS_PUPIL_OFFSET_X = 0.0 * mm
PERSEUS_PUPIL_OFFSET_Y = -2.38 * mm
PERSEUS_PUPIL_WIDTH = 14.0 * mm
PERSEUS_PUPIL_HEIGHT = 18.0 * mm
PERSEUS_APERTURE_OUTER = (14.0 * mm, 6.0 * mm)       # (width, height)
PERSEUS_APERTURE_INNER = (10.0 * mm, 2.0 * mm)       # clear opening
PERSEUS_BASE_RATIO = 0.05 * 6 / PERSEUS_NUM_MIRRORS  # per-mirror pick-off

# The two defining tilts. Pantoscopic defaults nonzero (it *is* Perseus);
# projector tilt defaults to 0 (straight down) until a convention/value is set.
PERSEUS_PANTOSCOPIC_TILT = 10.0 * deg
PERSEUS_PROJECTOR_TILT = 0.0 * deg


def uniform_spacings(num_mirrors: int,
                     distance: float = PERSEUS_DISTANCE_BETWEEN_MIRRORS
                     ) -> jnp.ndarray:
    """``(num_mirrors - 1,)`` equal perpendicular gaps between mirrors."""
    return jnp.full((num_mirrors - 1,), distance)


class PerseusGeometry(NamedTuple):
    """Fully-placed Perseus geometry — everything except mirror reflectance.

    ``mirror_positions``/``mirror_normal`` already carry the pantoscopic
    tilt; a caller adds :class:`PartialMirror` elements using its own
    reflectance source (baked curve or design-variable curve).
    """
    chassis: GlassBlock
    aperture: RectangularAperture
    pupil: RectangularPupil
    mirror_positions: jnp.ndarray   # (M, 3), tilted
    mirror_normal: jnp.ndarray      # (3,), tilted
    mirror_width: float
    mirror_height: float
    light_position: jnp.ndarray     # (3,) projector location
    light_direction: jnp.ndarray    # (3,) tilted beam axis
    pupil_center: jnp.ndarray       # (3,) pantoscopic pivot


def build_perseus_geometry(
    *,
    spacings: jnp.ndarray,
    mirror_angle: float = PERSEUS_MIRROR_ANGLE,
    chassis_x: float = PERSEUS_CHASSIS_X,
    chassis_y: float = PERSEUS_CHASSIS_Y,
    chassis_z: float = PERSEUS_CHASSIS_Z,
    skew_angle: float = PERSEUS_SKEW_ANGLE,
    first_mirror_offset_y: float = PERSEUS_FIRST_MIRROR_OFFSET_Y,
    eye_relief: float = PERSEUS_EYE_RELIEF,
    pupil_offset_x: float = PERSEUS_PUPIL_OFFSET_X,
    pupil_offset_y: float = PERSEUS_PUPIL_OFFSET_Y,
    pupil_width: float = PERSEUS_PUPIL_WIDTH,
    pupil_height: float = PERSEUS_PUPIL_HEIGHT,
    aperture_outer: tuple[float, float] = PERSEUS_APERTURE_OUTER,
    aperture_inner: tuple[float, float] = PERSEUS_APERTURE_INNER,
    light_position: jnp.ndarray = PERSEUS_LIGHT_POSITION,
    pantoscopic_tilt: float = PERSEUS_PANTOSCOPIC_TILT,
    projector_tilt: float = PERSEUS_PROJECTOR_TILT,
) -> PerseusGeometry:
    """Place every Perseus element, applying both tilts.

    ``spacings`` is a ``(M-1,)`` array of perpendicular mirror-plane gaps;
    the mirror count ``M`` is inferred from it. All other geometry knobs
    default to the Perseus constants. Returns a :class:`PerseusGeometry`;
    reflectance is left to the caller.
    """
    chassis_center = jnp.array([chassis_x / 2.0, chassis_y, chassis_z / 2.0])

    # Exit pupil — fixed in space; it is also the pantoscopic pivot.
    pupil_center = jnp.array([
        chassis_x / 2.0 + pupil_offset_x,
        chassis_y + pupil_offset_y,
        eye_relief + chassis_z,
    ])
    pupil = RectangularPupil(
        name="pupil", position=pupil_center,
        normal=jnp.array([0.0, 0.0, -1.0]),
        width=pupil_width, height=pupil_height,
    )

    # Projector beam axis, tilted about world-x from straight down.
    light_direction = normalize(
        rotate_vectors(STRAIGHT_DOWN, X_AXIS, projector_tilt))

    # Beam-defining stop, half a mm downstream of the emitter along the beam.
    # Follows the projector tilt; unaffected by the pantoscopic tilt.
    aperture = RectangularAperture(
        name="aperture",
        position=light_position + 0.5 * mm * light_direction,
        normal=light_direction,
        width=aperture_outer[0], height=aperture_outer[1],
        inner_width=aperture_inner[0], inner_height=aperture_inner[1],
    )

    # Chassis: axis-aligned wedge, translated to place, then pantoscopically
    # rotated about the pupil.
    z_skew = float(chassis_z * math.tan(skew_angle))
    chassis = (GlassBlock.create_chassis(
        name="chassis", x=float(chassis_x), y=float(chassis_y),
        z=float(chassis_z), material=agc_m074, z_skew=z_skew)
        .translate(chassis_center)
        .rotate(X_AXIS, pantoscopic_tilt, pupil_center))

    # Mirror stack in the untilted frame, then rotated about the pupil.
    normal_angle = math.pi / 2.0 - mirror_angle
    mirror_normal = jnp.array([0.0, math.sin(normal_angle), math.cos(normal_angle)])
    mirror_width = float(chassis_x)
    mirror_height = float(chassis_z / math.cos(mirror_angle))

    first_mirror_center = chassis_center + jnp.array(
        [0.0, first_mirror_offset_y, 0.0])
    mirror_edge_to_center_y = 0.5 * math.sqrt(
        mirror_height ** 2 - float(chassis_z) ** 2)
    first_pos = first_mirror_center - jnp.array(
        [0.0, mirror_edge_to_center_y, 0.0])

    # Each perpendicular gap projects onto y by 1/sin(normal_angle).
    unit_offset = jnp.array([0.0, 1.0 / math.sin(normal_angle), 0.0])
    cumulative_offset = jnp.concatenate(
        [jnp.zeros(1), jnp.cumsum(jnp.asarray(spacings))])  # (M,)
    mirror_positions = (
        first_pos[None, :] - cumulative_offset[:, None] * unit_offset[None, :])

    mirror_positions = rotate_points(
        mirror_positions, X_AXIS, pantoscopic_tilt, pupil_center)
    mirror_normal = normalize(
        rotate_vectors(mirror_normal, X_AXIS, pantoscopic_tilt))

    return PerseusGeometry(
        chassis=chassis, aperture=aperture, pupil=pupil,
        mirror_positions=mirror_positions, mirror_normal=mirror_normal,
        mirror_width=mirror_width, mirror_height=mirror_height,
        light_position=jnp.asarray(light_position),
        light_direction=light_direction, pupil_center=pupil_center,
    )


def build_perseus_system(
    *,
    num_mirrors: int = PERSEUS_NUM_MIRRORS,
    distance_between_mirrors: float = PERSEUS_DISTANCE_BETWEEN_MIRRORS,
    base_ratio: float = PERSEUS_BASE_RATIO,
    pantoscopic_tilt: float = PERSEUS_PANTOSCOPIC_TILT,
    projector_tilt: float = PERSEUS_PROJECTOR_TILT,
    chassis_z: float = PERSEUS_CHASSIS_Z,
    **geometry_kwargs,
) -> OpticalSystem:
    """Build the Perseus reference system with baked flat reflectances.

    Each mirror gets a wavelength-flat, Talos-compensated reflectance
    (``base_ratio / (1 - i·base_ratio)``) on the default mirror wavelength
    grid — the fixed-design analogue of the flat driver's warm start.
    Extra ``geometry_kwargs`` pass straight through to
    :func:`build_perseus_geometry` (chassis dims, eye relief, pupil, …).
    """
    geometry = build_perseus_geometry(
        spacings=uniform_spacings(num_mirrors, distance_between_mirrors),
        pantoscopic_tilt=pantoscopic_tilt, projector_tilt=projector_tilt,
        chassis_z=chassis_z, **geometry_kwargs)

    reflectance_table = compensated_reflectances(
        jnp.full_like(DEFAULT_MIRROR_WAVELENGTHS, base_ratio), num_mirrors)

    system = OpticalSystem(env_material=air)
    system.add(geometry.chassis)
    system.add(geometry.aperture)
    for mirror_idx in range(num_mirrors):
        system.add(PartialMirror(
            name=f"mirror_{mirror_idx}",
            position=geometry.mirror_positions[mirror_idx],
            normal=geometry.mirror_normal.copy(),
            width=geometry.mirror_width,
            height=geometry.mirror_height,
            reflectance=reflectance_table[mirror_idx],
            wavelengths=DEFAULT_MIRROR_WAVELENGTHS,
        ))
    system.add(geometry.pupil)
    return system
