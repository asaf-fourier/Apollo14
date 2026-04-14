"""Route construction for the segmented single-path tracer.

A ``Route`` is an ordered tuple of typed segment pytrees — the "compiled
IR" of an optical path. Each segment is a ``NamedTuple`` that lives next
to its element (``ApertureSeg`` in ``aperture.py``, ``FaceSeg`` in
``glass_block.py``, ``MirrorStackSeg``/``ReflectMirrorSeg`` in
``partial_mirror.py``, ``PupilSeg`` in ``pupil.py``). The tracer walks
``route.segments`` in Python and dispatches each entry to the matching
element's ``_interact`` function; consecutive transmit mirrors are fused
into one ``MirrorStackSeg`` so the hot loop runs under ``lax.scan``
instead of unrolling per-mirror.

Wavelength is baked into ``FaceSeg.n1``/``n2`` by ``prepare_route`` —
routes returned by ``build_route`` still hold ``MaterialData`` there,
which lets the same route be prepared at many wavelengths.

Typical usage::

    # 1. Build a system once.
    system = build_default_system()

    # 2. Describe the optical path as a flat list of element names.
    #    Strings default to TRANSMIT; (ref, mode) pairs override.
    main = build_route(system, [
        "aperture",
        ("chassis", "back"),          # entry face — enters glass
        "mirror_0", "mirror_1", "mirror_2",
        "mirror_3", "mirror_4", "mirror_5",
        ("chassis", "front"),          # exit face — back into air
    ])

    # 3. Express a reflected branch as a splice on the main path.
    pupil_via_m2 = build_route(system, branch_path(
        main_path=["aperture", ("chassis", "back"),
                   "mirror_0", "mirror_1", "mirror_2",
                   "mirror_3", "mirror_4", "mirror_5",
                   ("chassis", "front")],
        at="mirror_2",
        tail=[("chassis", "front"), absorb("pupil")],
    ))

    # 4. Resolve wavelength, then trace.
    route = prepare_route(main, 550e-6)
    result = trace(route, Ray(pos=o, dir=d, intensity=1.0))
"""

from typing import NamedTuple, Sequence, Union

import jax
import jax.numpy as jnp


# ── Mode tags ────────────────────────────────────────────────────────────────
# Path-level tags for ``build_route`` — they pick which ``build_segment``
# branch an element takes, not a runtime dispatch inside the kernel.
#
# Defined here (not in any element module) so elements can import them
# without pulling in the other elements' segment types.

TRANSMIT = 0
REFLECT = 1
ABSORB = 2


from apollo14.system import OpticalSystem  # noqa: E402


class Route(NamedTuple):
    """Ordered tuple of heterogeneous segment pytrees.

    Wrapped in a ``NamedTuple`` (rather than a bare tuple) so JAX treats
    the whole route as a single pytree: ``jit`` uses its structure as a
    cache key, and ``jax.grad`` can flow through any traced leaf inside.

    The ``segments`` field is a Python tuple (not a list) to keep the
    pytree hashable/static — mutating it would invalidate the jit cache.
    """
    segments: tuple


# ── Path entry helpers ──────────────────────────────────────────────────────
# A "path" is a list of entries, where each entry identifies one element
# in the system and optionally tags it with a mode.
#
# Accepted shapes:
#   "mirror_0"                   — plain element name, defaults to TRANSMIT
#   ("chassis", "back")          — (block, face) resolve, defaults to TRANSMIT
#   ("mirror_0", REFLECT)        — element with explicit mode
#   (("chassis", "back"), TRANSMIT) — block face with explicit mode
#
# The ``transmit``/``reflect``/``absorb`` helpers produce the tagged form
# without having to spell out the mode integer.

ElementRef = Union[str, tuple]
PathEntry = Union[ElementRef, tuple]


def transmit(ref: ElementRef) -> PathEntry:
    """Tag ``ref`` as a TRANSMIT path entry.

    Rarely needed directly — plain strings and ``(block, face)`` tuples
    already default to TRANSMIT. Use this when you want the mode to be
    explicit for documentation, e.g. next to a nearby ``reflect(...)``.

    Example::

        build_route(system, [
            transmit("aperture"),
            transmit(("chassis", "back")),
            reflect("mirror_2"),
            transmit(("chassis", "front")),
            absorb("pupil"),
        ])
    """
    return (ref, TRANSMIT)


def reflect(ref: ElementRef) -> PathEntry:
    """Tag ``ref`` as a REFLECT path entry — ray bounces off the element.

    For a partial mirror this selects the reflecting kernel
    (``ReflectMirrorSeg`` + ``mirror_reflect_one``) instead of the
    transmit kernel. This is the fork point of a branch route.

    Example — build the "light off mirror_3 reaches the pupil" branch::

        route = build_route(system, [
            "aperture",
            ("chassis", "back"),
            "mirror_0", "mirror_1", "mirror_2",
            reflect("mirror_3"),           # fork here
            ("chassis", "front"),
            absorb("pupil"),
        ])
    """
    return (ref, REFLECT)


def absorb(ref: ElementRef) -> PathEntry:
    """Tag ``ref`` as a terminal ABSORB entry — the ray ends here.

    Use this on a detector element like a pupil. The element's
    ``build_segment`` is still called in ABSORB mode; the tracer records
    the final hit and stops advancing the ray.

    Example::

        build_route(system, [..., absorb("pupil")])
    """
    return (ref, ABSORB)


def _parse_entry(entry: PathEntry) -> tuple:
    """Normalize any accepted path entry into a ``(ref, mode)`` pair.

    Three input shapes collapse to one output shape so ``build_route``
    doesn't need to special-case them:

    - ``"mirror_0"``              → ``("mirror_0", TRANSMIT)``
    - ``("chassis", "back")``     → ``(("chassis", "back"), TRANSMIT)``
    - ``("mirror_0", REFLECT)``   → ``("mirror_0", REFLECT)``

    The middle case is ambiguous with the last one — we distinguish by
    checking whether the second element is an ``int`` (a mode tag) vs a
    ``str`` (a face name).
    """
    if isinstance(entry, str):
        return entry, TRANSMIT
    if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], int):
        return entry[0], int(entry[1])
    return entry, TRANSMIT


def _ref_name(entry: PathEntry) -> str:
    """Return the element name referenced by ``entry``, regardless of shape.

    Used by ``branch_path`` to locate the fork point without caring
    whether the entry is a bare string, a ``(block, face)`` tuple, or
    already mode-tagged. For ``("chassis", "back")`` the name is
    ``"chassis"`` — i.e. the block name, not the face.
    """
    ref, _ = _parse_entry(entry)
    return ref if isinstance(ref, str) else ref[0]


# ── Route building ──────────────────────────────────────────────────────────

def _group_mirror_runs(raw_segments: list) -> list:
    """Collapse consecutive ``_SingleMirror`` entries into a ``MirrorStackSeg``.

    **Why fuse.** Each ``PartialMirror.build_segment`` (transmit branch)
    emits one ``_SingleMirror`` with per-mirror ``(3,)``-shaped leaves.
    If we left them as-is, the tracer's Python loop would emit one copy
    of the mirror-interact code per mirror into the jaxpr — compile
    time and code size grow linearly with mirror count. Stacking them
    into a single ``MirrorStackSeg`` (leaves ``(M, 3)``) lets
    ``jax.lax.scan`` compile the body *once* and iterate over M at
    runtime, so 6 mirrors cost the same as 1 at compile time.

    **What it does.** Walks ``raw_segments`` left to right; each maximal
    run of adjacent ``_SingleMirror`` entries is stacked along a new
    leading axis via ``jax.tree_util.tree_map``. Non-mirror segments
    (aperture, face, pupil, reflect-mirror) pass through unchanged, and
    a mirror run broken by a non-mirror seg becomes two separate stacks
    — reflect mirrors and faces act as natural stack boundaries.

    Example (schematic, shapes in comments)::

        raw = [
            ApertureSeg(...),         #                    not a mirror
            FaceSeg(...),             #                    not a mirror
            _SingleMirror(...),       # (3,) per leaf      ┐
            _SingleMirror(...),       # (3,) per leaf      ├ fused
            _SingleMirror(...),       # (3,) per leaf      ┘
            FaceSeg(...),             #                    not a mirror
        ]
        grouped = _group_mirror_runs(raw)
        # → [ApertureSeg, FaceSeg, MirrorStackSeg((3, 3) leaves), FaceSeg]

    **Import note.** ``_SingleMirror``/``MirrorStackSeg`` live in
    ``elements/partial_mirror.py`` which imports the mode tags from
    *this* module at load time. The reverse edge (route ← partial_mirror)
    is deferred to call time to avoid a circular import.
    """
    from apollo14.elements.partial_mirror import _SingleMirror, MirrorStackSeg

    def stack(run):
        stacked = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=0), *run)
        return MirrorStackSeg(**stacked._asdict())

    out = []
    i = 0
    n = len(raw_segments)
    while i < n:
        seg = raw_segments[i]
        if isinstance(seg, _SingleMirror):
            run = [seg]
            j = i + 1
            while j < n and isinstance(raw_segments[j], _SingleMirror):
                run.append(raw_segments[j])
                j += 1
            out.append(stack(run))
            i = j
        else:
            out.append(seg)
            i += 1
    return out


def build_route(system: OpticalSystem, path: Sequence[PathEntry]) -> Route:
    """Build a ``Route`` from a flat path specification.

    This is the main entry point for authoring an optical path. Each
    ``path`` entry identifies one element in ``system`` and optionally
    tags it with a mode (see :func:`_parse_entry` for accepted shapes).

    **Medium tracking.** The builder walks the path in order, threading
    a ``current_material`` through ``elem.build_segment(current, mode)``.
    Each element returns its segment plus the material the ray is in
    *after* the interaction — glass faces flip it between air and glass,
    everything else passes it through. Elements use this to fill
    ``FaceSeg.n1``/``n2`` correctly regardless of which direction the
    path crosses the face.

    **No wavelength resolution.** ``FaceSeg.n1``/``n2`` are still
    ``MaterialData`` after ``build_route`` — resolving them to scalars
    happens later in ``prepare_route(route, wavelength)``. This lets the
    same route be prepared for many wavelengths without rebuilding.

    **Mirror fusion.** Consecutive ``_SingleMirror`` entries (transmit
    partial mirrors) are fused into one ``MirrorStackSeg`` via
    :func:`_group_mirror_runs` so the tracer can scan over them.

    Example — main path through the combiner::

        route = build_route(system, [
            "aperture",
            ("chassis", "back"),          # enters glass
            "mirror_0", "mirror_1", "mirror_2",
            "mirror_3", "mirror_4", "mirror_5",
            ("chassis", "front"),         # back to air
        ])
        # route.segments ==
        #   (ApertureSeg, FaceSeg(air→glass), MirrorStackSeg(M=6),
        #    FaceSeg(glass→air))

    Example — reflected branch ending at the pupil::

        route = build_route(system, [
            "aperture",
            ("chassis", "back"),
            "mirror_0", "mirror_1",
            reflect("mirror_2"),           # fork here
            ("chassis", "front"),
            absorb("pupil"),
        ])
    """
    current = system.env_material
    raw: list = []
    for entry in path:
        ref, mode = _parse_entry(entry)
        elem = system.resolve(ref)
        seg, current = elem.build_segment(current, mode)
        raw.append(seg)

    return Route(segments=tuple(_group_mirror_runs(raw)))


def branch_path(main_path: Sequence[PathEntry], at: str,
                tail: Sequence[PathEntry], mode: int = REFLECT) -> list:
    """Splice a branch onto a main path at element ``at``.

    A combiner has one main transmit path plus N reflected branches —
    one per mirror — that all share the same prefix up to their fork
    point. This helper builds each branch by pointing at the main path
    and naming the fork element, so the prefix is never typed twice.

    The returned list is:

    1. everything in ``main_path`` *before* the first entry whose
       element name matches ``at``,
    2. ``(at, mode)`` — the fork entry, with the requested mode
       (``REFLECT`` by default),
    3. all of ``tail``.

    Note that ``at`` is matched by element *name* via :func:`_ref_name`,
    so ``("chassis", "back")`` matches ``at="chassis"``.

    Example — derive the mirror_3 reflected branch from the main path::

        main = [
            "aperture", ("chassis", "back"),
            "mirror_0", "mirror_1", "mirror_2",
            "mirror_3", "mirror_4", "mirror_5",
            ("chassis", "front"),
        ]
        tail = [("chassis", "front"), absorb("pupil")]
        branch = branch_path(main, at="mirror_3", tail=tail)
        # → ["aperture", ("chassis", "back"),
        #    "mirror_0", "mirror_1", "mirror_2",
        #    ("mirror_3", REFLECT),
        #    ("chassis", "front"), ("pupil", ABSORB)]
        route = build_route(system, branch)
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
                       exit_face: str = "front") -> Route:
    """Build the straight-through main path for a Talos-style combiner.

    Convenience wrapper that introspects the system and assembles the
    conventional main-path entry list — aperture (if any), chassis
    entry face, every mirror named ``"mirror_*"`` in system order, then
    the chassis exit face. All entries are TRANSMIT.

    This is the path the projector's beam takes when light is passing
    *through* the combiner. Reflected branches that reach the pupil are
    built separately using :func:`branch_path` on top of the same list.

    Use this when the system follows the Talos conventions (one chassis
    with named faces, mirrors named ``mirror_0…mirror_N``). For anything
    non-standard, call :func:`build_route` directly with an explicit
    path.

    Example::

        system = build_default_system()
        main = combiner_main_path(system)
        route = prepare_route(main, 550e-6)
        result = trace(route, Ray(pos=o, dir=d, intensity=1.0))
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
