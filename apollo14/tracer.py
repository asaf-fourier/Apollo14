from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import jax.numpy as jnp
import numpy as np

from apollo14.system import OpticalSystem
from apollo14.elements.glass_block import GlassBlock
from apollo14.elements.surface import PartialMirror
from apollo14.elements.pupil import Pupil, RectangularPupil
from apollo14.interaction import Interaction
from apollo14.geometry import normalize, snell_refract, reflect, ray_plane_intersection
from apollo14.units import EPSILON


@dataclass
class TraceHit:
    """Record of a ray hitting an element. Forms a tree via children."""
    element_name: str
    point: jnp.ndarray       # (3,)
    normal: jnp.ndarray      # (3,)
    direction: jnp.ndarray   # (3,) incoming ray direction
    intensity: float
    interaction: Interaction
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

    For the differentiable optimization path, use the JAX tracer (jax_tracer.py).
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

        if isinstance(elem, (Pupil, RectangularPupil)):
            hit = TraceHit(
                element_name=elem.name, point=hit_point, normal=hit_normal,
                direction=d, intensity=inten, interaction=Interaction.ABSORBED,
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
                direction=d, intensity=inten, interaction=Interaction.ABSORBED,
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


def jax_to_trace_result(
    origin, direction,
    endpoints, intensities, valid, main_hits, branch_hits,
    system: OpticalSystem, wavelength: float,
) -> TraceResult:
    """Convert JAX tracer output to a TraceResult tree.

    Takes the 5-tuple output from ``trace_ray``/``trace_combiner_ray``
    (or a single-ray slice from a batched trace) and builds the
    ``TraceHit`` tree that the visualizer expects.

    Tree structure::

        entry_hit (ENTERING chassis, point=entry_point)
          └─ mirror_0 (point=main_hits[0])
               ├─ exit_0 (EXITING, point=branch_hits[0,0])
               │    └─ pupil_0 (ABSORBED, point=branch_hits[0,1]) if valid
               └─ mirror_1 (point=main_hits[1])
                    ├─ exit_1 → pupil_1
                    └─ mirror_2 → ...

    Args:
        origin: (3,) ray start position.
        direction: (3,) ray direction in air (normalized).
        endpoints: (M, 3) pupil hit points per mirror.
        intensities: (M,) reflected intensity per mirror.
        valid: (M,) bool — True if reflection reached the pupil.
        main_hits: (M, 3) hit point on each mirror surface.
        branch_hits: (M, B, 3) intermediate points along each branch.
        system: OpticalSystem with chassis, mirrors, and pupil.
        wavelength: trace wavelength (for computing n_glass).

    Returns:
        TraceResult with the ray tree and pupil_hit set.
    """
    origin = np.asarray(origin)
    direction = np.asarray(direction)
    main_hits = np.asarray(main_hits)
    branch_hits = np.asarray(branch_hits)
    intensities = np.asarray(intensities)
    valid = np.asarray(valid)

    mirrors = [e for e in system.elements if isinstance(e, PartialMirror)]
    chassis = next(e for e in system.elements if isinstance(e, GlassBlock))
    pupil_elem = next(
        e for e in system.elements if isinstance(e, (Pupil, RectangularPupil))
    )

    n_glass = float(chassis.material.n(wavelength))

    # Find entry face — the face whose inward normal most aligns with direction
    best_face = max(
        chassis.faces,
        key=lambda f: float(jnp.dot(direction, -np.asarray(f.normal))),
    )
    entry_normal = np.asarray(normalize(best_face.normal))
    entry_pos = np.asarray(best_face.position)

    # Compute entry point on chassis face
    denom = float(np.dot(direction, entry_normal))
    if abs(denom) > 1e-12:
        t_entry = float(np.dot(entry_pos - origin, entry_normal) / denom)
    else:
        t_entry = 0.0
    entry_point = origin + max(t_entry, 0.0) * direction

    # Refracted direction inside glass
    d_glass = np.asarray(snell_refract(
        jnp.asarray(direction), jnp.asarray(entry_normal), 1.0, n_glass
    )[0])

    M = main_hits.shape[0]
    B = branch_hits.shape[1]

    # Transmitted intensity arriving at each mirror
    trans_intensity = np.empty(M)
    remaining = 1.0
    for i in range(M):
        trans_intensity[i] = remaining
        remaining -= float(intensities[i])

    result = TraceResult()

    # Build bottom-up: last mirror → first, chaining via transmitted path
    next_mirror_node = None

    for i in range(M - 1, -1, -1):
        mirror = mirrors[i] if i < len(mirrors) else None
        mirror_name = mirror.name if mirror else f"mirror_{i}"
        mirror_normal = np.asarray(mirror.normal) if mirror else np.array([0, 0, 1.0])

        # Reflected branch: exit face → pupil
        branch_children = []
        if B >= 1:
            exit_node = TraceHit(
                element_name=chassis.name,
                point=jnp.asarray(branch_hits[i, 0]),
                normal=jnp.asarray(entry_normal),
                direction=jnp.asarray(d_glass),
                intensity=float(intensities[i]),
                interaction=Interaction.EXITING,
            )
            if B >= 2 and valid[i]:
                pupil_node = TraceHit(
                    element_name=pupil_elem.name,
                    point=jnp.asarray(branch_hits[i, -1]),
                    normal=jnp.asarray(pupil_elem.normal),
                    direction=jnp.asarray(direction),
                    intensity=float(intensities[i]),
                    interaction=Interaction.ABSORBED,
                )
                exit_node.children = [pupil_node]
                if result.pupil_hit is None:
                    result.pupil_hit = pupil_node
            branch_children.append(exit_node)

        # Mirror node: children = reflected branch + next mirror (transmitted)
        children = branch_children[:]
        if next_mirror_node is not None:
            children.append(next_mirror_node)

        mirror_node = TraceHit(
            element_name=mirror_name,
            point=jnp.asarray(main_hits[i]),
            normal=jnp.asarray(mirror_normal),
            direction=jnp.asarray(d_glass),
            intensity=float(trans_intensity[i]),
            interaction=Interaction.REFLECTED,
            children=children,
        )
        next_mirror_node = mirror_node

    # Entry hit — root of the tree
    entry_hit = TraceHit(
        element_name=chassis.name,
        point=jnp.asarray(entry_point),
        normal=jnp.asarray(entry_normal),
        direction=jnp.asarray(direction),
        intensity=1.0,
        interaction=Interaction.ENTERING,
        children=[next_mirror_node] if next_mirror_node is not None else [],
    )

    result.hits = [entry_hit]
    return result
