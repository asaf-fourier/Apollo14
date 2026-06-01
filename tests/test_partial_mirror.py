"""Energy-conservation and golden-path tests for partial mirrors.

These lock the core cascade physics: a partial mirror splits an incident
ray into a reflected fraction ``r`` and a transmitted fraction ``1 - r``
that together conserve energy, and the reflected geometry matches a
hand-computed golden case.
"""

import jax.numpy as jnp

from apollo14.elements.partial_mirror import (
    PartialMirror,
    mirror_reflect_one,
    mirror_transmit_one,
)
from apollo14.geometry import normalize
from apollo14.materials import air
from apollo14.ray import Ray
from apollo14.route import REFLECT, TRANSMIT
from apollo14.units import nm


def _mirror(reflectance=0.3):
    """A 10×10 mirror on the y=0 plane, normal +Y, flat reflectance."""
    return PartialMirror(
        name="m",
        position=jnp.array([0.0, 0.0, 0.0]),
        normal=jnp.array([0.0, 1.0, 0.0]),
        width=10.0,
        height=10.0,
        reflectance=reflectance,
        wavelengths=jnp.array([400.0, 550.0, 700.0]) * nm,
    )


def _incident_ray(intensity=1.0):
    """45° down-right ray that strikes the y=0 plane at the origin."""
    return Ray(
        pos=jnp.array([-5.0, 5.0, 0.0]),
        dir=normalize(jnp.array([1.0, -1.0, 0.0])),
        intensity=jnp.asarray(intensity),
    )


def test_energy_conservation_reflect_plus_transmit():
    mirror = _mirror(reflectance=0.3)
    reflect_seg, _ = mirror.build_segment(air, REFLECT)
    transmit_seg, _ = mirror.build_segment(air, TRANSMIT)
    ray = _incident_ray(intensity=1.0)
    wavelength = 550.0 * nm

    reflected, _, reflect_valid = mirror_reflect_one(reflect_seg, ray, wavelength)
    transmitted, _, transmit_valid = mirror_transmit_one(
        transmit_seg, ray, wavelength)

    assert bool(reflect_valid) and bool(transmit_valid)
    # r + (1 - r) = 1 — no energy created or lost at the coating.
    total = float(reflected.intensity) + float(transmitted.intensity)
    assert abs(total - 1.0) < 1e-5
    assert abs(float(reflected.intensity) - 0.3) < 1e-5
    assert abs(float(transmitted.intensity) - 0.7) < 1e-5


def test_energy_conservation_scales_with_input():
    """Split fractions hold for an arbitrary incoming intensity."""
    mirror = _mirror(reflectance=0.25)
    reflect_seg, _ = mirror.build_segment(air, REFLECT)
    transmit_seg, _ = mirror.build_segment(air, TRANSMIT)
    ray = _incident_ray(intensity=0.4)

    reflected, _, _ = mirror_reflect_one(reflect_seg, ray, 550.0 * nm)
    transmitted, _, _ = mirror_transmit_one(transmit_seg, ray, 550.0 * nm)
    assert abs(float(reflected.intensity)
               + float(transmitted.intensity) - 0.4) < 1e-5


def test_golden_reflected_ray():
    mirror = _mirror(reflectance=0.3)
    reflect_seg, _ = mirror.build_segment(air, REFLECT)
    reflected, hit, _ = mirror_reflect_one(reflect_seg, _incident_ray(),
                                           550.0 * nm)
    # Hand-computed: hits the y=0 plane at the origin; a 45° down-right ray
    # reflects to up-right off a +Y normal.
    assert jnp.allclose(hit, jnp.array([0.0, 0.0, 0.0]), atol=1e-5)
    assert jnp.allclose(reflected.dir, normalize(jnp.array([1.0, 1.0, 0.0])),
                        atol=1e-5)


def test_transmit_preserves_direction():
    mirror = _mirror(reflectance=0.3)
    transmit_seg, _ = mirror.build_segment(air, TRANSMIT)
    transmitted, _, _ = mirror_transmit_one(transmit_seg, _incident_ray(),
                                            550.0 * nm)
    # A thin coating in one medium doesn't bend the transmitted ray.
    assert jnp.allclose(transmitted.dir, _incident_ray().dir, atol=1e-6)
