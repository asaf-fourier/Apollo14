"""Energy-conservation and golden-path tests for partial mirrors.

These lock the core cascade physics: a partial mirror splits an incident
ray into a reflected fraction ``r`` and a transmitted fraction ``1 - r``
that together conserve energy, and the reflected geometry matches a
hand-computed golden case.
"""

import jax
import jax.numpy as jnp

from apollo14.elements.partial_mirror import (
    PartialMirror,
    mirror_reflect_one,
    mirror_transmit_one,
)
from apollo14.geometry import normalize
from apollo14.materials import air
from apollo14.ray import Ray
from apollo14.route import REFLECT, TRANSMIT, Route, _group_mirror_runs
from apollo14.system import OpticalSystem
from apollo14.export.prescription import build_prescription
from apollo14.export.zosapi_script import build_script_text
from apollo14.trace import prepare_route
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


def _prepared_segments(mirror, wavelength):
    reflect_seg, _ = mirror.build_segment(air, REFLECT)
    transmit_seg, _ = mirror.build_segment(air, TRANSMIT)
    transmit_stack, = _group_mirror_runs([transmit_seg])

    prepared_reflect, = prepare_route(
        Route(segments=(reflect_seg,)), wavelength).segments
    prepared_stack, = prepare_route(
        Route(segments=(transmit_stack,)), wavelength).segments
    prepared_transmit = jax.tree.map(lambda value: value[0], prepared_stack)
    return prepared_reflect, prepared_transmit


def test_energy_conservation_reflect_plus_transmit():
    mirror = _mirror(reflectance=0.3)
    ray = _incident_ray(intensity=1.0)
    wavelength = 550.0 * nm
    reflect_seg, transmit_seg = _prepared_segments(mirror, wavelength)

    reflected, _, reflect_valid = mirror_reflect_one(reflect_seg, ray)
    transmitted, _, transmit_valid = mirror_transmit_one(transmit_seg, ray)

    assert bool(reflect_valid) and bool(transmit_valid)
    # r + (1 - r) = 1 — no energy created or lost at the coating.
    total = float(reflected.intensity) + float(transmitted.intensity)
    assert abs(total - 1.0) < 1e-5
    assert abs(float(reflected.intensity) - 0.3) < 1e-5
    assert abs(float(transmitted.intensity) - 0.7) < 1e-5


def test_energy_conservation_scales_with_input():
    """Split fractions hold for an arbitrary incoming intensity."""
    mirror = _mirror(reflectance=0.25)
    reflect_seg, transmit_seg = _prepared_segments(mirror, 550.0 * nm)
    ray = _incident_ray(intensity=0.4)

    reflected, _, _ = mirror_reflect_one(reflect_seg, ray)
    transmitted, _, _ = mirror_transmit_one(transmit_seg, ray)
    assert abs(float(reflected.intensity)
               + float(transmitted.intensity) - 0.4) < 1e-5


def test_golden_reflected_ray():
    mirror = _mirror(reflectance=0.3)
    reflect_seg, _ = _prepared_segments(mirror, 550.0 * nm)
    reflected, hit, _ = mirror_reflect_one(reflect_seg, _incident_ray())
    # Hand-computed: hits the y=0 plane at the origin; a 45° down-right ray
    # reflects to up-right off a +Y normal.
    assert jnp.allclose(hit, jnp.array([0.0, 0.0, 0.0]), atol=1e-5)
    assert jnp.allclose(reflected.dir, normalize(jnp.array([1.0, 1.0, 0.0])),
                        atol=1e-5)


def test_transmit_preserves_direction():
    mirror = _mirror(reflectance=0.3)
    _, transmit_seg = _prepared_segments(mirror, 550.0 * nm)
    transmitted, _, _ = mirror_transmit_one(transmit_seg, _incident_ray())
    # A thin coating in one medium doesn't bend the transmitted ray.
    assert jnp.allclose(transmitted.dir, _incident_ray().dir, atol=1e-6)


def _missing_ray(intensity=1.0):
    """Ray parallel to the incident one but offset far outside the 10×10
    mirror, so it crosses the mirror plane well beyond the rectangle."""
    return Ray(
        pos=jnp.array([-5.0, 5.0, 100.0]),
        dir=normalize(jnp.array([1.0, -1.0, 0.0])),
        intensity=jnp.asarray(intensity),
    )


def test_transmit_miss_passes_through_unchanged():
    """A live ray that misses the rectangle is NOT terminated — it keeps its
    intensity and direction and continues (only its interaction is skipped)."""
    mirror = _mirror(reflectance=0.3)
    _, transmit_seg = _prepared_segments(mirror, 550.0 * nm)
    ray = _missing_ray(intensity=0.8)

    transmitted, _, valid = mirror_transmit_one(transmit_seg, ray)

    assert not bool(valid)                                    # no interaction
    assert abs(float(transmitted.intensity) - 0.8) < 1e-6     # intensity kept
    assert jnp.allclose(transmitted.dir, ray.dir, atol=1e-6)  # direction kept


def test_transmit_dead_ray_stays_dead():
    """A ray that already died upstream (intensity 0) is never resurrected."""
    mirror = _mirror(reflectance=0.3)
    _, transmit_seg = _prepared_segments(mirror, 550.0 * nm)
    ray = _incident_ray(intensity=0.0)  # hits the rectangle, but is a corpse

    transmitted, _, valid = mirror_transmit_one(transmit_seg, ray)

    assert not bool(valid)
    assert float(transmitted.intensity) == 0.0


def test_partial_mirror_exports_as_rectangular_volume_with_face_coating():
    system = OpticalSystem(env_material=air)
    system.add(_mirror(reflectance=0.3))

    prescription = build_prescription(
        system,
        chassis_pivot=jnp.array([0.0, 0.0, 0.0]),
        chassis_tilt_deg=0.0,
        sources=[],
        trace_wavelengths=jnp.array([550.0]) * nm,
        glass_names={},
        coating_names={"m": "MIRROR_COAT"},
    )

    assert prescription.objects[0]["type"] == "rectangular_volume"
    assert prescription.objects[0]["face_coatings"] == {"1": "MIRROR_COAT"}
    assert prescription.objects[0]["data"]["z_length"] == 0.01


def test_each_mirror_volume_keeps_its_own_coating():
    system = OpticalSystem(env_material=air)
    mirror_a = _mirror(reflectance=0.2)
    mirror_a.name = "m_a"
    mirror_b = _mirror(reflectance=0.4)
    mirror_b.name = "m_b"
    system.add(mirror_a)
    system.add(mirror_b)

    prescription = build_prescription(
        system,
        chassis_pivot=jnp.array([0.0, 0.0, 0.0]),
        chassis_tilt_deg=0.0,
        sources=[],
        trace_wavelengths=jnp.array([550.0]) * nm,
        glass_names={},
        coating_names={"m_a": "COAT_A", "m_b": "COAT_B"},
    )

    mirrors = {entry["comment"]: entry for entry in prescription.objects}
    assert mirrors["m_a"]["face_coatings"] == {"1": "COAT_A"}
    assert mirrors["m_b"]["face_coatings"] == {"1": "COAT_B"}


def test_generated_build_script_selects_the_coating_file():
    script = build_script_text()
    assert "SystemData.Files.CoatingFile" in script
    assert "SystemData.Files.ReloadFiles()" in script
