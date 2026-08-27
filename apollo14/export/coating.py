"""Write a Zemax coating file (``.DAT``) for the partial-mirror stack.

Apollo14 models a mirror as a scalar reflectance sampled over wavelength and
applied regardless of incidence angle or polarization. A Zemax coating can carry
much more, and stepping up that ladder is the point of exporting at all — each
rung isolates one assumption:

``ideal``
    One number per mirror. Angle- and wavelength-independent, matching the
    tracer's model at a single wavelength. A mismatch here is a geometry or
    bookkeeping bug, not physics.
``flat_table``
    The optimizer's ``R(λ)``, still angle-independent. Isolates spectral effects.
``atlas_table``
    ``R(λ, θ)`` measured off the coating Atlas actually designed. Adds the angle
    dependence the tracer assumes away.
``atlas_stack``
    The physical film stack, so OpticStudio runs its own thin-film calculation
    and resolves s and p separately.

``atlas_table`` and ``atlas_stack`` are complementary rather than ranked:
the table carries Atlas's full material dispersion but averages polarization
(Atlas designs unpolarized, so ``Rs`` and ``Rp`` are written equal), while the
stack resolves polarization but exports each film with the single index Atlas
recorded at its reference wavelength, i.e. non-dispersive. Running both and
comparing brackets the polarization/dispersion uncertainty.

File syntax (OpticStudio coating file)::

    MATE <name>
    <wavelength µm> <n> <k>
    ...

    IDEAL <name> <transmitted intensity> <reflected intensity>

    TABLE <name>
    ANGL <degrees>
    WAVE <µm> <Rs> <Rp> <Ts> <Tp>
    ...

    COAT <name>
    <material> <thickness> <is_absolute>
    ...

Absorption is implied: OpticStudio computes ``A = 1 − R − T`` for ``IDEAL``, and
for ``TABLE`` whatever the four intensities leave over. Wavelengths are in
micrometres and both angles and wavelengths must ascend.
"""

from typing import NamedTuple

import numpy as np

# Layer thicknesses are written as absolute physical thickness. The alternative
# (is_absolute = 0) expresses thickness in waves at the coating reference
# wavelength, which would silently re-scale Atlas's nanometre designs.
ABSOLUTE_THICKNESS = 1

# Coating wavelengths are micrometres throughout a Zemax coating file, so film
# thicknesses written as absolute values are micrometres too.
NM_PER_UM = 1000.0

# Angles padded onto a measured R(λ, θ) table so every incidence angle a ray can
# arrive at has a defined value. The padded rows repeat the nearest computed
# angle — explicit clamping in place of whatever OpticStudio would extrapolate
# from a table covering only the design range.
PAD_ANGLES_DEG = (0.0, 89.0)


class CoatingBlock(NamedTuple):
    """One named coating definition, plus the materials it needs defined."""
    name: str
    definition: str
    material_definitions: dict[str, str]   # material name → MATE block


def _format_wave_row(wavelength_um: float, reflectance: float) -> str:
    """One ``WAVE`` row with s and p set equal and no absorption.

    Atlas designs against unpolarized reflectance, so there is no s/p split to
    export; transmission takes the remainder. A film stack with real absorption
    is better exported via :func:`stack_coating`, where OpticStudio computes the
    split itself.
    """
    reflected = float(np.clip(reflectance, 0.0, 1.0))
    transmitted = 1.0 - reflected
    return (f"WAVE {wavelength_um:.6f} {reflected:.6f} {reflected:.6f} "
            f"{transmitted:.6f} {transmitted:.6f}")


def ideal_coating(name: str, reflectance: float) -> CoatingBlock:
    """A constant reflectance, independent of wavelength and angle."""
    reflected = float(np.clip(reflectance, 0.0, 1.0))
    transmitted = 1.0 - reflected
    definition = (f"IDEAL {name} {transmitted:.6f} {reflected:.6f}")
    return CoatingBlock(name=name, definition=definition,
                        material_definitions={})


def table_coating(name: str, wavelengths_nm, angles_deg, reflectance,
                  *, pad_angles: bool = True) -> CoatingBlock:
    """A tabulated ``R(λ, θ)`` coating.

    ``reflectance`` is indexed ``[wavelength, angle]``. Wavelengths and angles
    are sorted ascending as the format requires; when ``pad_angles`` is set the
    table is extended to 0° and 89° by repeating the nearest computed angle.
    """
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    angles_deg = np.asarray(angles_deg, dtype=float)
    reflectance = np.asarray(reflectance, dtype=float)

    if reflectance.shape != (wavelengths_nm.size, angles_deg.size):
        raise ValueError(
            f"Coating {name!r}: reflectance shape {reflectance.shape} does not "
            f"match {wavelengths_nm.size} wavelengths × {angles_deg.size} "
            "angles.")

    wavelength_order = np.argsort(wavelengths_nm)
    angle_order = np.argsort(angles_deg)
    wavelengths_nm = wavelengths_nm[wavelength_order]
    angles_deg = angles_deg[angle_order]
    reflectance = reflectance[np.ix_(wavelength_order, angle_order)]

    if pad_angles:
        angles_deg, reflectance = _pad_angle_range(angles_deg, reflectance)

    lines = [f"TABLE {name}"]
    for angle_index, angle in enumerate(angles_deg):
        lines.append(f"ANGL {angle:.6f}")
        for wavelength_index, wavelength_nm in enumerate(wavelengths_nm):
            lines.append(_format_wave_row(
                wavelength_nm / NM_PER_UM,
                reflectance[wavelength_index, angle_index]))

    return CoatingBlock(name=name, definition="\n".join(lines),
                        material_definitions={})


def _pad_angle_range(angles_deg, reflectance):
    """Clamp-extend an angle grid to the full 0°–89° range."""
    low_pad, high_pad = PAD_ANGLES_DEG
    padded_angles = list(angles_deg)
    padded_columns = [reflectance[:, index]
                      for index in range(angles_deg.size)]

    if angles_deg[0] > low_pad:
        padded_angles.insert(0, low_pad)
        padded_columns.insert(0, reflectance[:, 0])
    if angles_deg[-1] < high_pad:
        padded_angles.append(high_pad)
        padded_columns.append(reflectance[:, -1])

    return np.array(padded_angles), np.stack(padded_columns, axis=1)


def flat_table_coating(name: str, wavelengths_nm, reflectance) -> CoatingBlock:
    """A tabulated ``R(λ)`` replicated across all angles.

    Reproduces Apollo14's own model: the spectral shape the optimizer produced,
    applied at every incidence angle.
    """
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    reflectance = np.asarray(reflectance, dtype=float).reshape(-1, 1)
    angles_deg = np.array(PAD_ANGLES_DEG, dtype=float)
    replicated = np.repeat(reflectance, angles_deg.size, axis=1)
    return table_coating(name, wavelengths_nm, angles_deg, replicated,
                         pad_angles=False)


def material_block(material_name: str, index_real: float, index_imaginary: float,
                   wavelength_range_nm: tuple[float, float]) -> str:
    """A ``MATE`` block holding one non-dispersive film material.

    Two rows bracket the band with identical values rather than a single row, so
    the material is well defined across the whole range whatever OpticStudio
    does at a table's edge.
    """
    minimum_nm, maximum_nm = wavelength_range_nm
    return "\n".join([
        f"MATE {material_name}",
        f"{minimum_nm / NM_PER_UM:.6f} {index_real:.6f} {index_imaginary:.6f}",
        f"{maximum_nm / NM_PER_UM:.6f} {index_real:.6f} {index_imaginary:.6f}",
    ])


def stack_coating(name: str, layers, wavelength_range_nm,
                  material_prefix: str = "A14") -> CoatingBlock:
    """A physical film stack, letting OpticStudio run the thin-film maths.

    ``layers`` are Atlas layer dicts (``material``, ``thickness_nm``, ``n``,
    ``k``) ordered as Atlas designed them. Each distinct (material, n, k) triple
    becomes its own ``MATE`` entry, because Atlas tunes the index of a film
    within a material's allowed range — two ``PLD_TiO2`` layers in the same
    stack can legitimately carry different indices and must not collide on one
    catalog name.
    """
    material_definitions: dict[str, str] = {}
    layer_lines = [f"COAT {name}"]

    for layer_index, layer in enumerate(layers):
        index_real = float(layer["n"])
        index_imaginary = float(layer["k"])
        thickness_nm = float(layer["thickness_nm"])
        material_name = _material_name(
            material_prefix, layer["material"], index_real, index_imaginary)

        material_definitions[material_name] = material_block(
            material_name, index_real, index_imaginary, wavelength_range_nm)
        layer_lines.append(
            f"{material_name} {thickness_nm / NM_PER_UM:.8f} "
            f"{ABSOLUTE_THICKNESS}"
            f"   ! layer {layer_index + 1}: {layer['material']}, "
            f"{thickness_nm:.2f} nm")

    return CoatingBlock(name=name, definition="\n".join(layer_lines),
                        material_definitions=material_definitions)


def _material_name(prefix: str, material: str, index_real: float,
                   index_imaginary: float) -> str:
    """A catalog-safe name that keeps distinct indices distinct.

    The real index is encoded to three decimals, finer than the 0.005 tolerance
    Atlas tunes within; the extinction coefficient to six, since real film ``k``
    values run down to ~1e-5 and a coarser encoding would map two genuinely
    different films onto one name. A residual collision is still caught by
    :func:`format_coating_file`, which refuses to merge one name onto two
    definitions.
    """
    stem = "".join(character for character in material
                   if character.isalnum())[:10].upper()
    absorption = ("" if index_imaginary <= 0
                  else f"K{round(index_imaginary * 1e6):06d}")
    return f"{prefix}_{stem}_{round(index_real * 1000):04d}{absorption}"


def format_coating_file(blocks, title: str) -> str:
    """Assemble coating blocks into one coating file.

    Materials are emitted ahead of every coating that references them, which is
    the order the file format expects.
    """
    blocks = list(blocks)

    material_definitions: dict[str, str] = {}
    for block in blocks:
        for material_name, definition in block.material_definitions.items():
            existing = material_definitions.get(material_name)
            if existing is not None and existing != definition:
                raise ValueError(
                    f"Two different definitions for coating material "
                    f"{material_name!r}.")
            material_definitions[material_name] = definition

    lines = [f"! {title}",
             "! Generated by apollo14.export — do not edit by hand.",
             "! Wavelengths in micrometres; layer thicknesses absolute (µm).",
             ""]

    if material_definitions:
        lines.append("! ── Film materials ──")
        for material_name in sorted(material_definitions):
            lines.append(material_definitions[material_name])
            lines.append("")

    lines.append("! ── Coatings ──")
    for block in blocks:
        lines.append(block.definition)
        lines.append("")

    return "\n".join(lines)
