"""Perseus combiner system definition — the tuned reference geometry.

Perseus is a tilted, cascaded partial-mirror combiner. This module is the
**single source of truth** for its geometry: the constants below and
:func:`build_perseus_geometry` reproduce exactly the layout dialed in by
``examples/visualize_perseus.py`` (projector tilt + placement, pantoscopic
tilt about the combiner center, mirror core, tuned beam-defining aperture,
and the eye-relief-centered pupil). Both the baked-reflectance reference
:func:`build_perseus_system` and the differentiable
:func:`helios.perseus_params.build_parametrized_perseus` build on it, so the
optimizer and the visualization can never drift. ``apollo14`` never imports
``helios``.

Conventions (rotations are **counter-clockwise about world-x** ``[1,0,0]``;
``+`` swings toward +z, the eye side; ``0`` is straight down / untilted):

- ``projector_tilt`` leans the emitted beam that far CCW from straight down.
- ``pantoscopic_tilt`` leans the combiner (chassis + mirror stack) that far
  CCW, rotating the whole assembly about the **combiner center**.
- The mirror plane normals sit ``mirror_normal_angle`` off the (untilted)
  combiner normal ``+z``; a top-side ``prism_angle`` wedge is baked into the
  chassis as a z-skew.

Because these are CCW rotations, they apply the *negative* angle to
``geometry.rotate_vectors`` (whose positive angle is right-handed, swinging
toward -z) — matching the visualization exactly.
"""

import math
from typing import NamedTuple

import jax.numpy as jnp

from apollo14.combiner import compensated_reflectances
from apollo14.elements.aperture import RectangularAperture
from apollo14.elements.glass_block import GlassBlock, validate_reflectance_table
from apollo14.elements.partial_mirror import (
    DEFAULT_MIRROR_WAVELENGTHS,
    PartialMirror,
)
from apollo14.elements.pupil import RectangularPupil
from apollo14.geometry import (
    compute_local_axes,
    normalize,
    rotate_points,
    rotate_vectors,
)
from apollo14.materials import agc_m074, air
from apollo14.spectral import SpectralTable
from apollo14.system import OpticalSystem
from apollo14.units import deg, mm, nm

# ── Rotation axis / nominal beam ────────────────────────────────────────────

X_AXIS = jnp.array([1.0, 0.0, 0.0])          # pantoscopic / projector tilt axis
STRAIGHT_DOWN = jnp.array([0.0, -1.0, 0.0])  # untilted projector direction

# ── Projector ───────────────────────────────────────────────────────────────

PERSEUS_LIGHT_POSITION = jnp.array([7.0 * mm, 31.0 * mm, 0.6 * mm])
PERSEUS_PROJECTOR_TILT = 13.422 * deg        # CCW about world-x from straight down
PERSEUS_BEAM_WIDTH = 10.0 * mm
PERSEUS_BEAM_HEIGHT = 2.0 * mm

# Emitted beam axis (CCW ⇒ negative angle into the right-handed rotation).
PERSEUS_PROJECTOR_DIRECTION = normalize(
    rotate_vectors(STRAIGHT_DOWN, X_AXIS, -PERSEUS_PROJECTOR_TILT))

# ── Combiner (glass chassis + cascade of tilted partial mirrors) ────────────

PERSEUS_PANTOSCOPIC_TILT = 7.0 * deg         # CCW about world-x, about combiner center
PERSEUS_NUM_MIRRORS = 8
PERSEUS_PRISM_ANGLE = 10.578 * deg           # top-side wedge (chassis z-skew)
# Mirror plane normal sits this far off the (untilted) combiner normal +z; the
# equivalent mirror tilt from horizontal is 90° − this = 50.289°.
PERSEUS_MIRROR_NORMAL_ANGLE = (90.0 - 50.289) * deg
# The mirror stack fills a fixed y-extent; the per-mirror gap is derived from it
# and the mirror count, so changing the count repacks the same region (more
# mirrors ⇒ smaller gaps). See spacings_for_count().
PERSEUS_MIRROR_STACK_SPAN = 9.8 * mm         # y-extent the mirror-center stack fills
PERSEUS_MIRROR_Y_SPACING = (                 # per-mirror gap at the default count
    PERSEUS_MIRROR_STACK_SPAN / (PERSEUS_NUM_MIRRORS - 1))
PERSEUS_CHASSIS_Z = 3.0 * mm                 # full chassis depth (z) — "COMBINER_DEPTH"
PERSEUS_MIRROR_Z_EXTENT = 1.2 * mm           # mirror core, centered in the depth
PERSEUS_COMBINER_CENTER = jnp.array([7.0 * mm, 24.0 * mm, 1.6 * mm])
PERSEUS_COMBINER_WIDTH = 14.0 * mm           # chassis x
PERSEUS_COMBINER_HEIGHT = 12.0 * mm          # chassis y (holds the stack)
PERSEUS_AR_REFLECTANCE = validate_reflectance_table(
    SpectralTable.constant(
        0.005, jnp.array([380.0, 780.0]) * nm))  # 0.5% residual per face
PERSEUS_BASE_RATIO = 0.05 * 6 / PERSEUS_NUM_MIRRORS   # per-mirror pick-off

# ── Beam-defining aperture (opaque frame with a clear opening) ──────────────
# Oriented parallel to the combiner prism (entry) face; its opening is sized to
# pass every mirror-hitting ray and recentred along its height axis so a tight
# 2 mm opening sits over the densest band (see examples/visualize_perseus.py).

PERSEUS_APERTURE_OFFSET = 0.2 * mm           # downstream of the projector along the beam
PERSEUS_APERTURE_DROP_Y = 0.6 * mm           # extra drop, straight down the y axis
PERSEUS_APERTURE_HEIGHT_SHIFT = 0.15 * mm    # recentre along the opening's height axis
PERSEUS_APERTURE_OUTER = (14.0 * mm, 6.0 * mm)   # (width, height) opaque frame
PERSEUS_APERTURE_INNER = (12.0 * mm, 3.0 * mm)   # (width, height) clear opening

# ── Target pupil (eyebox detector) ──────────────────────────────────────────
# Fixed in space (not rotated with the combiner); centred on where the reflected
# light lands at the eye-relief plane over the full FOV scan.

PERSEUS_EYE_RELIEF = 16.0 * mm
PERSEUS_PUPIL_OFFSET_X = 0.0 * mm
PERSEUS_PUPIL_CENTER_Y = 21.13 * mm
PERSEUS_PUPIL_WIDTH = 14.0 * mm
PERSEUS_PUPIL_HEIGHT = 18.0 * mm
PERSEUS_EYEBOX_SIZE = 8.0 * mm               # target eyebox (square) at the pupil center

# ── FOV scan (used by examples/optimizer) ───────────────────────────────────
# 10° about world-x, 12° about the in-plane perpendicular axis.
PERSEUS_FOV_AROUND_X = 10.0 * deg
PERSEUS_FOV_AROUND_PROJECTOR_Y = 12.0 * deg


def spacings_for_count(num_mirrors: int,
                       stack_span: float = PERSEUS_MIRROR_STACK_SPAN
                       ) -> jnp.ndarray:
    """``(num_mirrors - 1,)`` equal mirror-center y-gaps summing to ``stack_span``.

    Changing ``num_mirrors`` keeps the stack's total y-extent fixed at
    ``stack_span`` and just repacks the mirrors — more mirrors ⇒ smaller gaps —
    so a single count knob controls the whole layout.
    """
    return jnp.full((num_mirrors - 1,), stack_span / (num_mirrors - 1))


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
    pupil_center: jnp.ndarray       # (3,) pupil center


def build_perseus_geometry(
    *,
    spacings: jnp.ndarray,
    mirror_normal_angle: float = PERSEUS_MIRROR_NORMAL_ANGLE,
    mirror_z_extent: float = PERSEUS_MIRROR_Z_EXTENT,
    combiner_center: jnp.ndarray = PERSEUS_COMBINER_CENTER,
    combiner_width: float = PERSEUS_COMBINER_WIDTH,
    combiner_height: float = PERSEUS_COMBINER_HEIGHT,
    chassis_z: float = PERSEUS_CHASSIS_Z,
    prism_angle: float = PERSEUS_PRISM_ANGLE,
    eye_relief: float = PERSEUS_EYE_RELIEF,
    pupil_offset_x: float = PERSEUS_PUPIL_OFFSET_X,
    pupil_center_y: float = PERSEUS_PUPIL_CENTER_Y,
    pupil_width: float = PERSEUS_PUPIL_WIDTH,
    pupil_height: float = PERSEUS_PUPIL_HEIGHT,
    aperture_offset: float = PERSEUS_APERTURE_OFFSET,
    aperture_drop_y: float = PERSEUS_APERTURE_DROP_Y,
    aperture_height_shift: float = PERSEUS_APERTURE_HEIGHT_SHIFT,
    aperture_outer: tuple[float, float] = PERSEUS_APERTURE_OUTER,
    aperture_inner: tuple[float, float] = PERSEUS_APERTURE_INNER,
    light_position: jnp.ndarray = PERSEUS_LIGHT_POSITION,
    pantoscopic_tilt: float = PERSEUS_PANTOSCOPIC_TILT,
    projector_tilt: float = PERSEUS_PROJECTOR_TILT,
    ar_reflectance: SpectralTable = PERSEUS_AR_REFLECTANCE,
) -> PerseusGeometry:
    """Place every Perseus element, reproducing the visualization exactly.

    ``spacings`` is a ``(M-1,)`` array of mirror-center gaps along y; the
    mirror count ``M`` is inferred from it and the stack is centered on the
    combiner center. All other knobs default to the Perseus constants above.
    Returns a :class:`PerseusGeometry`; reflectance is left to the caller.
    """
    combiner_center = jnp.asarray(combiner_center)
    light_position = jnp.asarray(light_position)

    # Projector beam axis — CCW about world-x from straight down.
    light_direction = normalize(
        rotate_vectors(STRAIGHT_DOWN, X_AXIS, -projector_tilt))

    # Aperture: parallel to the combiner's prism (entry) face — the inward
    # prism-face normal [0, -cos(prism), -sin(prism)] leaned CCW by the
    # pantoscopic tilt. Positioned downstream of the projector along the beam,
    # dropped in y, then recentred along its height axis so the tight opening
    # sits over the densest band of mirror-hitting rays.
    aperture_normal = normalize(rotate_vectors(
        jnp.array([0.0, -math.cos(float(prism_angle)),
                   -math.sin(float(prism_angle))]),
        X_AXIS, -pantoscopic_tilt))
    aperture_height_axis = compute_local_axes(aperture_normal)[1]
    aperture_position = (light_position
                         + aperture_offset * light_direction
                         - jnp.array([0.0, aperture_drop_y, 0.0])
                         + aperture_height_shift * aperture_height_axis)
    aperture = RectangularAperture(
        name="aperture", position=aperture_position, normal=aperture_normal,
        width=aperture_outer[0], height=aperture_outer[1],
        inner_width=aperture_inner[0], inner_height=aperture_inner[1],
    )

    # Mirror plane orientation (untilted): normal sits mirror_normal_angle off
    # the combiner normal (+z). Mirrors span the chassis width in x and only the
    # central mirror-core in z (height = core / sin(angle)).
    angle = float(mirror_normal_angle)
    mirror_normal = jnp.array([0.0, math.sin(angle), math.cos(angle)])
    mirror_width = float(combiner_width)
    mirror_height = float(mirror_z_extent / math.sin(angle))

    # Chassis with the top-side prism baked in as a z-skew wedge, placed at the
    # combiner center in the untilted frame.
    z_skew = float(chassis_z * math.tan(float(prism_angle)))
    chassis = GlassBlock.create_chassis(
        name="chassis", x=float(combiner_width), y=float(combiner_height),
        z=float(chassis_z), material=agc_m074, z_skew=z_skew,
        coating_reflectance=ar_reflectance,
    ).translate(combiner_center)

    # Mirror centers: consecutive y-gaps from ``spacings``, the stack centered
    # on the combiner center; all share the combiner x/z center (step only in y).
    cumulative_offset = jnp.concatenate(
        [jnp.zeros(1), jnp.cumsum(jnp.asarray(spacings))])  # (M,)
    num_mirrors = cumulative_offset.shape[0]
    stack_span = cumulative_offset[-1]
    top_y = combiner_center[1] + stack_span / 2.0
    mirror_y = top_y - cumulative_offset
    mirror_positions = jnp.stack([
        jnp.full((num_mirrors,), combiner_center[0]),
        mirror_y,
        jnp.full((num_mirrors,), combiner_center[2]),
    ], axis=1)  # (M, 3)

    # Apply the pantoscopic tilt (CCW about world-x) about the combiner center.
    tilt = -pantoscopic_tilt
    chassis = chassis.rotate(X_AXIS, tilt, combiner_center)
    mirror_positions = rotate_points(mirror_positions, X_AXIS, tilt, combiner_center)
    mirror_normal = normalize(rotate_vectors(mirror_normal, X_AXIS, tilt))

    # Pupil — fixed in space, on the eye-relief plane past the front face,
    # centered on where the reflected light lands (not rotated with the stack).
    pupil_center = jnp.array([
        combiner_center[0] + pupil_offset_x,
        pupil_center_y,
        combiner_center[2] + chassis_z / 2.0 + eye_relief,
    ])
    pupil = RectangularPupil(
        name="pupil", position=pupil_center,
        normal=jnp.array([0.0, 0.0, -1.0]),
        width=pupil_width, height=pupil_height,
    )

    return PerseusGeometry(
        chassis=chassis, aperture=aperture, pupil=pupil,
        mirror_positions=mirror_positions, mirror_normal=mirror_normal,
        mirror_width=mirror_width, mirror_height=mirror_height,
        light_position=light_position, light_direction=light_direction,
        pupil_center=pupil_center,
    )


def build_perseus_system(
    *,
    num_mirrors: int = PERSEUS_NUM_MIRRORS,
    stack_span: float = PERSEUS_MIRROR_STACK_SPAN,
    base_ratio: float = PERSEUS_BASE_RATIO,
    pantoscopic_tilt: float = PERSEUS_PANTOSCOPIC_TILT,
    projector_tilt: float = PERSEUS_PROJECTOR_TILT,
    chassis_z: float = PERSEUS_CHASSIS_Z,
    **geometry_kwargs,
) -> OpticalSystem:
    """Build the Perseus reference system with baked flat reflectances.

    ``num_mirrors`` is the single layout knob: the mirror-center gaps are
    derived so the stack always fills ``stack_span`` (see
    :func:`spacings_for_count`). Each mirror gets a wavelength-flat,
    Talos-compensated reflectance (``base_ratio / (1 - i·base_ratio)``). Extra
    ``geometry_kwargs`` pass straight through to :func:`build_perseus_geometry`.
    """
    geometry = build_perseus_geometry(
        spacings=spacings_for_count(num_mirrors, stack_span),
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
