"""Atlas integration: wavelength units, dispersion, and JAX preparation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from atlas import Material, Materials

from apollo14.materials import air, refractive_index
from apollo14.perseus import build_perseus_geometry, build_perseus_system
from apollo14.route import build_route
from apollo14.system import OpticalSystem
from apollo14.trace import prepare_route
from apollo14.units import nm
from helios.combiner_params import CombinerParams
from helios.perseus_params import build_parametrized_perseus


def test_atlas_index_converts_millimeters_to_meters():
    glass = Material.from_nk("test glass", [400, 600], [1.7, 1.5], [0.01, 0.02])
    wavelengths = jnp.array([400, 500, 600]) * nm
    indices = jax.jit(lambda wl: refractive_index(glass, wl))(wavelengths)
    np.testing.assert_allclose(indices, [1.7, 1.6, 1.5], rtol=1e-6)
    assert not jnp.iscomplexobj(indices)


def test_custom_atlas_glass_is_used_by_perseus_and_prepared_routes():
    glass = Material.from_nk("custom glass", [400, 600], [1.7, 1.5], [0, 0])
    geometry = build_perseus_geometry(spacings=jnp.array([2.0]), glass_material=glass)
    assert geometry.glass_material is glass
    assert geometry.chassis.material is glass
    system = OpticalSystem()
    system.add(geometry.chassis)
    route = build_route(system, [("chassis", "back"), ("chassis", "front")])
    indices = jax.jit(jax.vmap(lambda wl: prepare_route(route, wl).segments[0].n2))(
        jnp.array([400, 500, 600]) * nm
    )
    np.testing.assert_allclose(indices, [1.7, 1.6, 1.5], rtol=1e-6)
    prepared = prepare_route(route, 500 * nm)
    np.testing.assert_allclose(prepared.segments[1].n1, 1.6, rtol=1e-6)
    np.testing.assert_allclose(prepared.segments[1].n2, refractive_index(air, 500 * nm), rtol=1e-6)


def test_serialized_atlas_and_legacy_material_names():
    from atlas import Materials

    from apollo14.materials import agc_m074, material_from_name

    for material in (air, agc_m074, Materials.moveon.MR10, Materials.moveon.TiO2,
                     Materials.moveon.SiO2, Materials.pld.TiO2, Materials.pld.Al2O3):
        assert material_from_name(material.name) is material
    assert material_from_name("air") is air
    assert material_from_name("agc_m074") is agc_m074


@pytest.mark.parametrize("parametrized", [False, True])
def test_system_builders_require_and_propagate_explicit_atlas_glass(parametrized):
    params = CombinerParams.initial(num_mirrors=3)
    if parametrized:
        with pytest.raises(TypeError, match="glass_material"):
            build_parametrized_perseus(params)
        system = build_parametrized_perseus(params, glass_material=Materials.moveon.MR10)
    else:
        with pytest.raises(TypeError, match="glass_material"):
            build_perseus_system(num_mirrors=3)
        system = build_perseus_system(num_mirrors=3, glass_material=Materials.moveon.MR10)
    glass = Materials.moveon.MR10
    assert system.resolve(("chassis", "back"))._block_material is glass
    assert system.resolve(("chassis", "front"))._block_material is glass
    route = build_route(system, [("chassis", "back"), ("chassis", "front")])
    wavelengths = jnp.array([450.0, 550.0, 650.0]) * nm
    indices = jax.jit(jax.vmap(lambda wl: prepare_route(route, wl).segments[0].n2))(
        wavelengths
    )
    np.testing.assert_allclose(indices, jnp.real(glass.compute_nk(wavelengths * 1e-3)),
                               rtol=1e-6)
