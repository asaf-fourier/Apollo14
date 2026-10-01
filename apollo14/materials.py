"""Atlas materials and index evaluation in Apollo14's millimeter units."""

import jax.numpy as jnp
from atlas import Material, Materials

air = Materials.Air
agc_m074 = Materials.AGC_M074


def refractive_index(material: Material, wavelength):
    """Evaluate the real refractive index; Atlas expects wavelengths in meters."""
    return jnp.real(material.compute_nk(jnp.asarray(wavelength) * 1e-3))


def material_from_name(name: str) -> Material:
    """Resolve serialized Atlas catalog names, including legacy Apollo14 aliases."""
    aliases = {"air": air, "agc_m074": agc_m074}
    if name in aliases:
        return aliases[name]
    namespace = Materials
    attribute = name
    for prefix, group in (("moveon_", "moveon"), ("PLD_", "pld")):
        if name.startswith(prefix):
            namespace = getattr(Materials, group)
            attribute = name.removeprefix(prefix)
            break
    material = getattr(namespace, attribute, None) if not attribute.startswith("_") else None
    if not isinstance(material, Material) or material.name != name:
        raise ValueError(f"Unknown Atlas catalog material: {name!r}")
    return material
