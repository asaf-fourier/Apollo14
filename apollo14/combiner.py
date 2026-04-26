"""Talos reference combiner system definition."""

import jax.numpy as jnp

from apollo14.elements.aperture import RectangularAperture
from apollo14.elements.glass_block import GlassBlock
from apollo14.elements.partial_mirror import (
    DEFAULT_MIRROR_WAVELENGTHS,
    PartialMirror,
)
from apollo14.elements.pupil import RectangularPupil
from apollo14.materials import agc_m074, air
from apollo14.system import OpticalSystem
from apollo14.units import deg, mm, nm

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


def compensated_reflectances(ratio, num_mirrors, num_samples: int | None = None):
    """Compute per-mirror reflectances compensated for upstream losses.

    Each mirror reflects ``ratio`` of the *original* beam intensity. Because
    earlier mirrors absorb light, later mirrors need a higher local
    reflectance to achieve the same absolute reflected amount::

        r[i] = ratio / (1 - i * ratio)

    Args:
        ratio: Target fraction of original intensity reflected by each mirror.
            Either a scalar (broadcast across all spectral samples) or a
            ``(K,)`` curve sampled on the mirror's wavelength grid.
        num_mirrors: Number of mirrors (M).
        num_samples: Only used when ``ratio`` is a scalar — the length of
            the spectral grid to broadcast to. Defaults to the length of
            ``DEFAULT_MIRROR_WAVELENGTHS``.

    Returns:
        ``(M, K)`` array of per-mirror spectral reflectance curves.
    """
    ratio = jnp.asarray(ratio)
    if ratio.ndim == 0:
        k = num_samples if num_samples is not None \
            else int(DEFAULT_MIRROR_WAVELENGTHS.shape[0])
        ratio = jnp.broadcast_to(ratio, (k,))
    i = jnp.arange(num_mirrors)[:, None]  # (M, 1)
    return ratio[None, :] / (1.0 - i * ratio[None, :])  # (M, K)


def build_default_system() -> OpticalSystem:
    """Build the Talos reference combiner system.

    Creates a 6-mirror AR combiner: glass chassis with cascaded partial
    mirrors, aperture, and rectangular pupil. All parameters match the
    Apollo13 Talos reference design.

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
        width=6.0 * mm,
        height=3.0 * mm,
        inner_width=10.0 * mm,
        inner_height=2.0 * mm,
    ))

    # ── Mirrors ──────────────────────────────────────────────────────────
    num_mirrors = 6
    mirror_angle = 48.0 * deg
    normal_angle = jnp.pi / 2 - mirror_angle
    mirror_normal = jnp.array([0.0, float(jnp.sin(normal_angle)),
                                float(jnp.cos(normal_angle))])
    mirror_x_width = float(cx)
    mirror_y_width = float(cz / jnp.cos(mirror_angle))

    # Per-mirror compensated reflectances — flat 5% curve across the default
    # spectral grid (equal absolute intensity per mirror).
    mirror_wavelengths = DEFAULT_MIRROR_WAVELENGTHS
    base_reflectance = jnp.full_like(mirror_wavelengths, 0.05)
    refl_table = compensated_reflectances(base_reflectance, num_mirrors)

    first_mirror_center = chassis_center + jnp.array([0.0, 5.0 * mm, 0.0])
    distance_between_mirrors = 1.47 * mm

    mirror_edge_to_center_y = 0.5 * jnp.sqrt(mirror_y_width ** 2 - float(cz) ** 2)
    first_pos = first_mirror_center - jnp.array([0.0, float(mirror_edge_to_center_y), 0.0])
    mirror_offset_y = distance_between_mirrors / mirror_normal[1]
    mirror_offset = jnp.array([0.0, float(mirror_offset_y), 0.0])

    for i in range(num_mirrors):
        system.add(PartialMirror(
            name=f"mirror_{i}",
            position=first_pos - i * mirror_offset,
            normal=mirror_normal.copy(),
            width=mirror_x_width,
            height=mirror_y_width,
            reflectance=refl_table[i],
            wavelengths=mirror_wavelengths,
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

    return system
