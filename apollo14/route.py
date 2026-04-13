"""Route construction for the sequential single-path tracer.

A route is a flat stacked ``Surface`` — one ``lax.scan`` worth of surfaces,
built once, wavelength-agnostic. This module owns everything from element
reference parsing to stacked-surface construction; ``trace.py`` owns the
beam preparation and the tracer itself.
"""

from typing import Sequence, Union

import jax
import jax.numpy as jnp

from apollo14.surface import Surface, TRANSMIT, REFLECT, ABSORB
from apollo14.system import OpticalSystem


# An element reference is either a plain string ("mirror_0") or a
# (block, face) tuple (("chassis", "back")). A path entry is that
# reference, optionally wrapped with a mode: (ref, mode).
ElementRef = Union[str, tuple]
PathEntry = Union[ElementRef, tuple]


# ── Path entry helpers ──────────────────────────────────────────────────────
# Terse constructors for mode-tagged path entries. Plain strings / tuples
# default to TRANSMIT, so ``transmit(...)`` is only needed for readability.

def transmit(ref: ElementRef) -> PathEntry:
    """Mark an element reference as a transmit step (the default)."""
    return (ref, TRANSMIT)


def reflect(ref: ElementRef) -> PathEntry:
    """Mark an element reference as a reflect step."""
    return (ref, REFLECT)


def absorb(ref: ElementRef) -> PathEntry:
    """Mark an element reference as an absorbing terminal step."""
    return (ref, ABSORB)


def _parse_entry(entry: PathEntry) -> tuple:
    """Split a path entry into ``(ref, mode)``.

    Accepted forms:
        "mirror_0"                         → transmit
        ("chassis", "back")                → transmit (block+face ref)
        ("mirror_0", REFLECT)              → explicit mode on plain ref
        (("chassis", "back"), REFLECT)     → explicit mode on block+face ref
    """
    if isinstance(entry, str):
        return entry, TRANSMIT
    if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], int):
        return entry[0], int(entry[1])
    return entry, TRANSMIT


def _ref_name(entry: PathEntry) -> str:
    """Return the bare element name of a path entry (ignoring mode wrapping)."""
    ref, _ = _parse_entry(entry)
    return ref if isinstance(ref, str) else ref[0]


def _stack_surfaces(states: Sequence[Surface]) -> Surface:
    """Stack per-surface states into one with a leading (N,) dim on every leaf."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)


def build_route(system: OpticalSystem, path: Sequence[PathEntry]) -> Surface:
    """Build a stacked wavelength-agnostic ``Surface`` from a flat path.

    Each entry is either a plain element name (``"mirror_0"``), a
    ``(block, face)`` tuple (``("chassis", "back")``), or a ``(ref, mode)``
    pair where ``mode`` is one of ``TRANSMIT``/``REFLECT``/``ABSORB``.

    The builder tracks the current medium as a ``Material`` instance
    (starting from ``system.env_material``) and hands it to each element's
    ``to_generic_surface`` — no wavelength involved.
    """
    current = system.env_material
    surfaces: list[Surface] = []
    for entry in path:
        name, mode = _parse_entry(entry)
        elem = system.resolve(name)
        surf, current = elem.to_generic_surface(current, mode)
        surfaces.append(surf)

    return _stack_surfaces(surfaces)


def branch_path(main_path: Sequence[PathEntry], at: str,
                tail: Sequence[PathEntry], mode: int = REFLECT) -> list:
    """Splice a branch onto a main path.

    Returns the main path's prefix up to (but not including) the element
    ``at``, followed by ``(at, mode)`` and the ``tail`` entries. Use it to
    express a reflected side-path without retyping the shared prefix:

        branch = branch_path(
            main_path, at="mirror_0",
            tail=[("chassis", "front"), absorb("pupil")],
        )

    Branching is physical — the ray splits at ``at``. Tracing the main
    path and each branch route independently gives you every outgoing
    beam; their intensities are complementary via ``reflectance`` /
    ``1 - reflectance`` at the split.
    """
    prefix = []
    for entry in main_path:
        if _ref_name(entry) == at:
            break
        prefix.append(entry)
    return prefix + [(at, mode)] + list(tail)


# ── Combiner helper ──────────────────────────────────────────────────────────

def combiner_main_path(system: OpticalSystem,
                       entry_face: str = "back",
                       exit_face: str = "front") -> Surface:
    """Build the straight-through main path for a Talos-style combiner.

    The path is: aperture (if present) → entry chassis face → each partial
    mirror in order (all transmit) → exit chassis face. No pupil, no
    reflected branches — branching is authored as separate routes.
    """
    chassis = next(e for e in system.elements if hasattr(e, "get_face"))
    mirrors = [e for e in system.elements
               if hasattr(e, "reflectance") and hasattr(e, "name")
               and getattr(e, "name", "").startswith("mirror")]
    apertures = [e for e in system.elements
                 if getattr(e, "name", None) == "aperture"]

    path: list[PathEntry] = []
    if apertures:
        path.append(apertures[0].name)
    path.append((chassis.name, entry_face))
    path.extend(m.name for m in mirrors)
    path.append((chassis.name, exit_face))

    return build_route(system, path)
