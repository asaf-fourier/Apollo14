import jax.numpy as jnp

from apollo14.units import mm, nm, deg
from apollo14.materials import air, agc_m074
from apollo14.elements.surface import PartialMirror
from apollo14.elements.glass_block import GlassBlock
from apollo14.elements.aperture import RectangularAperture
from apollo14.elements.pupil import RectangularPupil
from apollo14.system import OpticalSystem


# ── Talos projector / scan defaults (not part of the optical system) ────────

DEFAULT_LIGHT_POSITION = jnp.array([7.0 * mm, 31.0 * mm, 1.0 * mm])
DEFAULT_LIGHT_DIRECTION = jnp.array([0.0, -1.0, 0.0])
DEFAULT_WAVELENGTH = 550.0 * nm
DEFAULT_BEAM_WIDTH = 4.0 * mm
DEFAULT_BEAM_HEIGHT = 2.0 * mm
DEFAULT_X_FOV = 7.0 * deg
DEFAULT_Y_FOV = 7.0 * deg
DEFAULT_NUM_X_STEPS = 5
DEFAULT_NUM_Y_STEPS = 5


def build_default_system(stage_margin: float = 10.0) -> OpticalSystem:
    """Build the Talos reference combiner system.

    Creates a 6-mirror AR combiner: glass chassis with cascaded partial
    mirrors, aperture, and rectangular pupil. All parameters match the
    Apollo13 Talos reference design.

    Args:
        stage_margin: Margin around system elements for the Stage boundary.
            Set to 0 to skip adding a stage.

    Returns:
        OpticalSystem with all elements added.
    """
    system = OpticalSystem(env_material=air)

    # ── Chassis ──────────────────────────────────────────────────────────
    chassis_dims = jnp.array([14.0 * mm, 20.0 * mm, 2.0 * mm])
    chassis_center = jnp.array([chassis_dims[0] / 2, 20.0 * mm, chassis_dims[2] / 2])
    cx, cy, cz = chassis_dims
    skew_angle = 6.0 * deg
    z_skew = float(cz * jnp.tan(skew_angle))

    chassis = GlassBlock.create_chassis(
        name="chassis", x=float(cx), y=float(cy), z=float(cz),
        material=agc_m074, z_skew=z_skew,
    ).translate(chassis_center)
    system.add(chassis)

    # ── Aperture ─────────────────────────────────────────────────────────
    system.add(RectangularAperture(
        name="aperture",
        position=DEFAULT_LIGHT_POSITION - jnp.array([0.0, 0.5 * mm, 0.0]),
        normal=DEFAULT_LIGHT_DIRECTION,
        width=4.0 * mm,
        height=1.0 * mm,
    ))

    # ── Mirrors ──────────────────────────────────────────────────────────
    num_mirrors = 6
    mirror_angle = 48.0 * deg
    normal_angle = jnp.pi / 2 - mirror_angle
    mirror_normal = jnp.array([0.0, float(jnp.sin(normal_angle)),
                                float(jnp.cos(normal_angle))])
    mirror_x_width = float(cx)
    mirror_y_width = float(cz / jnp.cos(mirror_angle))
    reflection_ratio = jnp.array([0.05, 0.05, 0.05])

    first_mirror_center = chassis_center + jnp.array([0.0, 5.0 * mm, 0.0])
    distance_between_mirrors = 1.47 * mm

    mirror_edge_to_center_y = 0.5 * jnp.sqrt(mirror_y_width ** 2 - float(cz) ** 2)
    first_pos = first_mirror_center - jnp.array([0.0, float(mirror_edge_to_center_y), 0.0])
    mirror_offset_y = distance_between_mirrors / mirror_normal[1]
    mirror_offset = jnp.array([0.0, float(mirror_offset_y), 0.0])

    # Non-sequential tracer uses scalar reflectance (mean across colors)
    global_refl = float(jnp.mean(reflection_ratio))
    transmitted_light = 1.0

    for i in range(num_mirrors):
        refl_ratio = global_refl / transmitted_light
        transmitted_light -= global_refl
        trans_ratio = 1.0 - refl_ratio

        system.add(PartialMirror(
            name=f"mirror_{i}",
            position=first_pos - i * mirror_offset,
            normal=mirror_normal.copy(),
            width=mirror_x_width,
            height=mirror_y_width,
            transmission_ratio=trans_ratio,
            reflection_ratio=refl_ratio,
        ))

    # ── Pupil ────────────────────────────────────────────────────────────
    eye_relief = 15.0 * mm
    system.add(RectangularPupil(
        name="pupil",
        position=jnp.array([float(chassis_center[0]),
                             float(chassis_center[1]) - 2 * mm,
                             eye_relief + float(cz)]),
        normal=jnp.array([0.0, 0.0, -1.0]),
        width=10.0 * mm,
        height=14.0 * mm,
    ))

    # ── Stage boundary ───────────────────────────────────────────────────
    if stage_margin > 0:
        from apollo14.stage import Stage
        stage = Stage.from_system(system, margin=stage_margin)
        stage.add_to_system(system)

    return system
