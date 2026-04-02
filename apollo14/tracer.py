from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import jax.numpy as jnp

from apollo14.system import OpticalSystem
from apollo14.elements.surface import PartialMirror
from apollo14.elements.glass_block import GlassBlock
from apollo14.elements.pupil import Pupil
from apollo14.units import EPSILON


@dataclass
class TraceHit:
    """Record of a ray hitting an element. Forms a tree via children."""
    element_name: str
    point: jnp.ndarray       # (3,)
    normal: jnp.ndarray      # (3,)
    direction: jnp.ndarray   # (3,) incoming ray direction
    intensity: float
    interaction: str          # "reflected", "transmitted", "refracted", "absorbed", etc.
    children: List['TraceHit'] = field(default_factory=list)


@dataclass
class TraceResult:
    """Full trace of a ray through the system.

    ``hits`` contains root-level TraceHit nodes.  Each node may have
    ``children`` forming a ray tree.  Use ``flat_hits()`` to iterate
    every node in the tree.
    """
    hits: List[TraceHit] = field(default_factory=list)
    pupil_hit: Optional[TraceHit] = None

    def flat_hits(self) -> List[TraceHit]:
        """Return all hits in the tree as a flat list (pre-order DFS)."""
        out: List[TraceHit] = []
        stack = list(reversed(self.hits))
        while stack:
            node = stack.pop()
            out.append(node)
            stack.extend(reversed(node.children))
        return out


def trace_nonsequential(system: OpticalSystem, origin, direction, wavelength,
                     intensity=1.0, max_depth=50) -> TraceResult:
    """Non-sequential trace following all ray children (DFS).

    Builds a tree of TraceHit nodes via the ``children`` field.
    At each interaction (e.g. a mirror split), one TraceHit node is created
    per outgoing ray, each with ``children`` holding downstream hits.
    ``TraceResult.hits`` contains the root-level nodes.

    Use ``result.flat_hits()`` to iterate all nodes in the tree.

    Add a Stage to the system to catch stray rays at the simulation boundary.

    For the sequential optimization path, use trace_mirrors_sequential instead.
    """
    result = TraceResult()

    def _trace(org, d, inten, depth) -> List[TraceHit]:
        """Trace a ray and return a list of TraceHit nodes for this interaction."""
        if depth >= max_depth or inten < EPSILON:
            return []

        hit_result = system.find_closest_intersection(org, d)
        if hit_result is None:
            return []

        elem, dist, hit_point, hit_normal = hit_result

        if isinstance(elem, Pupil):
            hit = TraceHit(
                element_name=elem.name, point=hit_point, normal=hit_normal,
                direction=d, intensity=inten, interaction="absorbed",
            )
            if result.pupil_hit is None:
                result.pupil_hit = hit
            return [hit]

        if isinstance(elem, GlassBlock):
            children = elem.interact(org, d, inten, wavelength, system.env_material)
        else:
            children = elem.interact(org, d, inten)

        if not children:
            return [TraceHit(
                element_name=elem.name, point=hit_point, normal=hit_normal,
                direction=d, intensity=inten, interaction="absorbed",
            )]

        nodes = []
        for child_origin, child_dir, child_intensity, interaction_type in children:
            node = TraceHit(
                element_name=elem.name, point=hit_point, normal=hit_normal,
                direction=d, intensity=child_intensity, interaction=interaction_type,
            )
            node.children = _trace(child_origin, child_dir, child_intensity, depth + 1)
            nodes.append(node)

        return nodes

    result.hits = _trace(origin, direction, intensity, 0)
    return result


def trace_mirrors_sequential(system: OpticalSystem, origin, direction, wavelength,
                             intensity=1.0, max_depth=50) -> TraceResult:
    """Sequential trace that follows the primary path through the combiner.

    At each PartialMirror, follows the transmitted ray (primary path continues
    through the mirror stack). Reflected rays toward the pupil are recorded
    but not followed further.

    This is the fast path for optimization — single ray, no branching.
    """
    result = TraceResult()
    org, d, inten = origin, direction, intensity

    for _ in range(max_depth):
        if inten < EPSILON:
            break

        hit_result = system.find_closest_intersection(org, d)
        if hit_result is None:
            break

        elem, dist, hit_point, hit_normal = hit_result

        if isinstance(elem, Pupil):
            result.hits.append(TraceHit(
                element_name=elem.name, point=hit_point, normal=hit_normal,
                direction=d, intensity=inten, interaction="absorbed",
            ))
            break

        if isinstance(elem, GlassBlock):
            children = elem.interact(org, d, inten, wavelength, system.env_material)
            if not children:
                break
            child_origin, child_dir, child_intensity, interaction = children[0]
            result.hits.append(TraceHit(
                element_name=elem.name, point=hit_point, normal=hit_normal,
                direction=d, intensity=child_intensity, interaction=interaction,
            ))
            org, d, inten = child_origin, child_dir, child_intensity

        elif isinstance(elem, PartialMirror):
            children = elem.interact(org, d, inten)
            for child_origin, child_dir, child_intensity, interaction in children:
                result.hits.append(TraceHit(
                    element_name=elem.name, point=hit_point, normal=hit_normal,
                    direction=d, intensity=child_intensity, interaction=interaction,
                ))
                if interaction == "transmitted":
                    # Follow the transmitted ray (primary path)
                    org, d, inten = child_origin, child_dir, child_intensity

        else:
            # Aperture or other absorbing element
            result.hits.append(TraceHit(
                element_name=elem.name, point=hit_point, normal=hit_normal,
                direction=d, intensity=inten, interaction="absorbed",
            ))
            break

    return result
