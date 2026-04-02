from dataclasses import dataclass

import jax.numpy as jnp

from apollo14.units import mm, nm, deg
from apollo14.materials import air, agc_m074, Material
from apollo14.elements.surface import PartialMirror
from apollo14.elements.glass_block import GlassBlock
from apollo14.elements.aperture import RectangularAperture
from apollo14.elements.pupil import Pupil
from apollo14.system import OpticalSystem


@dataclass
class MirrorConfig:
    normal: jnp.ndarray
    angle_with_horizon: float
    x_width: float
    y_width: float
    reflection_ratio: float


@dataclass
class ChassisConfig:
    dimensions: jnp.ndarray   # (3,) x, y, z
    center: jnp.ndarray       # (3,)
    first_mirror_center: jnp.ndarray  # (3,)
    distance_between_mirrors: float
    skew_angle: float
    material: Material
    pupil_spacing: float


@dataclass
class LightConfig:
    position: jnp.ndarray    # (3,)
    direction: jnp.ndarray   # (3,)
    beam_width: float
    beam_height: float
    wavelength: float
    x_fov: float
    y_fov: float
    num_x_steps: int
    num_y_steps: int


@dataclass
class PupilConfig:
    center: jnp.ndarray
    normal: jnp.ndarray
    eye_relief: float
    radius: float


@dataclass
class ApertureConfig:
    center: jnp.ndarray
    normal: jnp.ndarray
    width: float
    height: float


@dataclass
class CombinerConfig:
    mirror: MirrorConfig
    chassis: ChassisConfig
    light: LightConfig
    pupil: PupilConfig
    aperture: ApertureConfig
    num_mirrors: int = 6

    @classmethod
    def default(cls):
        """Development configuration matching Apollo13's Talos defaults."""
        chassis_dims = jnp.array([14.0 * mm, 20.0 * mm, 2.0 * mm])
        chassis_center = jnp.array([chassis_dims[0] / 2, 20.0 * mm, chassis_dims[2] / 2])

        light_y = chassis_center[1] + chassis_dims[1] / 2 + 1.0 * mm
        light_pos = jnp.array([chassis_dims[0] / 2, light_y, chassis_dims[2] / 2])
        light_dir = jnp.array([0.0, -1.0, 0.0])

        aperture_config = ApertureConfig(
            center=light_pos - jnp.array([0.0, 0.5 * mm, 0.0]),
            normal=light_dir,
            width=4.0 * mm,
            height=1.0 * mm,
        )

        eye_relief = 15.0 * mm
        pupil_config = PupilConfig(
            normal=jnp.array([0.0, 0.0, -1.0]),
            center=jnp.array([chassis_center[0], chassis_center[1],
                              eye_relief + chassis_dims[2]]),
            eye_relief=eye_relief,
            radius=4.0 * mm,
        )

        chassis_config = ChassisConfig(
            dimensions=chassis_dims,
            center=chassis_center,
            first_mirror_center=chassis_center + jnp.array([0.0, 5.0 * mm, 0.0]),
            distance_between_mirrors=1.47 * mm,
            skew_angle=6.0 * deg,
            material=agc_m074,
            pupil_spacing=2 * pupil_config.radius,
        )

        mirror_angle = 48.0 * deg
        normal_angle = jnp.pi / 2 - mirror_angle
        mirror_config = MirrorConfig(
            normal=jnp.array([0.0, float(jnp.sin(normal_angle)), float(jnp.cos(normal_angle))]),
            angle_with_horizon=mirror_angle,
            x_width=float(chassis_dims[0]),
            y_width=float(chassis_dims[2] / jnp.cos(mirror_angle)),
            reflection_ratio=0.05,
        )

        light_config = LightConfig(
            position=light_pos,
            direction=light_dir,
            beam_width=4.0 * mm,
            beam_height=2.0 * mm,
            wavelength=550.0 * nm,
            x_fov=7.0 * deg,
            y_fov=7.0 * deg,
            num_x_steps=5,
            num_y_steps=5,
        )

        return cls(
            mirror=mirror_config,
            chassis=chassis_config,
            light=light_config,
            pupil=pupil_config,
            aperture=aperture_config,
        )


def build_system(config: CombinerConfig, stage_margin: float = 10.0) -> OpticalSystem:
    """Build an OpticalSystem from a CombinerConfig.

    Args:
        stage_margin: Margin around the system elements for the Stage
            boundary. Set to 0 to skip adding a stage.
    """
    system = OpticalSystem(env_material=air)

    # Chassis (glass block)
    cx, cy, cz = config.chassis.dimensions
    z_skew = float(cz * jnp.tan(config.chassis.skew_angle))
    chassis = GlassBlock.create_chassis(
        name="chassis", x=float(cx), y=float(cy), z=float(cz),
        material=config.chassis.material, z_skew=z_skew,
    ).translate(config.chassis.center)
    system.add(chassis)

    # Aperture
    system.add(RectangularAperture(
        name="aperture",
        position=config.aperture.center,
        normal=config.aperture.normal,
        width=config.aperture.width,
        height=config.aperture.height,
    ))

    # Mirrors
    mirror_y_width = config.mirror.y_width
    chassis_z = float(config.chassis.dimensions[2])
    mirror_edge_to_center_y = 0.5 * jnp.sqrt(mirror_y_width ** 2 - chassis_z ** 2)

    first_pos = config.chassis.first_mirror_center - jnp.array([0.0, float(mirror_edge_to_center_y), 0.0])
    mirror_offset_y = config.chassis.distance_between_mirrors / config.mirror.normal[1]
    mirror_offset = jnp.array([0.0, float(mirror_offset_y), 0.0])

    global_refl = config.mirror.reflection_ratio
    transmitted_light = 1.0

    for i in range(config.num_mirrors):
        refl_ratio = global_refl / transmitted_light
        transmitted_light -= global_refl
        trans_ratio = 1.0 - refl_ratio

        system.add(PartialMirror(
            name=f"mirror_{i}",
            position=first_pos - i * mirror_offset,
            normal=config.mirror.normal.copy(),
            width=config.mirror.x_width,
            height=config.mirror.y_width,
            transmission_ratio=trans_ratio,
            reflection_ratio=refl_ratio,
        ))

    # Pupil
    system.add(Pupil(
        name="pupil",
        position=config.pupil.center,
        normal=config.pupil.normal,
        radius=config.pupil.radius,
    ))

    # Stage boundary
    if stage_margin > 0:
        from apollo14.stage import Stage
        stage = Stage.from_system(system, margin=stage_margin)
        stage.add_to_system(system)

    return system
