"""Perseus visualization — full system, ray-traced.

Perseus is a new combiner layout built up incrementally. The scene holds
the **projector** (housing box), the beam-defining **aperture** (opaque
frame with a clear opening), the **combiner** (a glass chassis with a
cascade of tilted partial mirrors), and the **target pupil** (the eyebox
detector plane where the eye samples the image).

The projector beam is **ray-traced** through the system: the main
transmitted path (aperture → chassis back → mirror cascade → chassis
front) plus one reflected branch per mirror (reflect → out the eye-side
face → onto the pupil). Traced rays are drawn in place of the beam.

Everything tunable lives in the PARAMS block at the top of this file.

Rotations are **counter-clockwise about the world-x axis** (``+`` swings
toward +z, the eye side; ``0`` is straight down / untilted, matching the
flat / Talos system):

- the projector leans ``PROJECTOR_TILT`` from straight down;
- the aperture is oriented parallel to the combiner's prism (top / entry)
  face, sitting ``APERTURE_OFFSET`` downstream of the projector plus
  ``APERTURE_DROP_Y`` lower in y;
- the combiner has a ``COMBINER_PANTOSCOPIC_TILT`` — its normal (the
  eye-facing +z slab face, untilted) leans that far CCW, and the whole
  chassis + mirror stack rotate with it.

The mirror normals sit ``MIRROR_TO_COMBINER_NORMAL_ANGLE`` off the
combiner normal; a top-side ``COMBINER_PRISM_ANGLE`` wedge is baked in as
a chassis z-skew. The chassis is ``COMBINER_DEPTH`` thick, but the mirror
core spans only the central ``MIRROR_Z_EXTENT`` — the outer glass is
index-matched cover, so the whole thing is one block (no internal
interface to refract). Each element's normal is drawn as a short line.

Run::

    python examples/visualize_perseus.py
"""

import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go

from apollo14.elements.aperture import RectangularAperture
from apollo14.elements.glass_block import GlassBlock
from apollo14.elements.partial_mirror import PartialMirror
from apollo14.elements.pupil import RectangularPupil
from apollo14.geometry import normalize, rotate_points, rotate_vectors
from apollo14.materials import agc_m074, air
from apollo14.projector import PlayNitrideLed
from apollo14.route import combiner_main_path
from apollo14.system import OpticalSystem
from apollo14.trace import prepare_route, trace_rays
from apollo14.units import deg, mm, nm
from apollo14.visualizer import plot_system

from helios.merit import build_combiner_branch_routes

# ── PARAMS ───────────────────────────────────────────────────────────────────
# Projector placement & orientation
PROJECTOR_POSITION = jnp.array([7.0 * mm, 31.0 * mm, 1.0 * mm])
PROJECTOR_TILT = 13.422 * deg    # CCW tilt about world-x from straight down

# Emitted beam
BEAM_WIDTH = 10.0 * mm           # beam cross-section, local x
BEAM_HEIGHT = 2.0 * mm           # beam cross-section, local y
BEAM_RAYS_X = 9                  # ray-grid columns (drawing density)
BEAM_RAYS_Y = 3                  # ray-grid rows

# Aperture (beam-defining stop) — opaque frame with a clear opening.
# Its normal follows the combiner's prism face (see COMBINER_PRISM_ANGLE).
APERTURE_OFFSET = 0.2 * mm       # downstream of the projector along the beam
APERTURE_DROP_Y = 0.6 * mm       # extra drop, straight down the y axis
APERTURE_OUTER_WIDTH = 14.0 * mm
APERTURE_OUTER_HEIGHT = 6.0 * mm
APERTURE_INNER_WIDTH = 10.0 * mm   # clear opening (passes the beam)
APERTURE_INNER_HEIGHT = 2.0 * mm

# Combiner (glass chassis + cascade of tilted partial mirrors)
COMBINER_PANTOSCOPIC_TILT = 7.0 * deg          # CCW about world-x (normal tilt)
COMBINER_NUM_MIRRORS = 10
COMBINER_PRISM_ANGLE = 10.578 * deg            # top-side wedge (chassis z-skew)
MIRROR_TO_COMBINER_NORMAL_ANGLE = 34.3 * deg  # mirror normal vs combiner normal
MIRROR_Y_SPACING = 1.0 * mm                    # mirror-center spacing along y
COMBINER_DEPTH = 3.0 * mm                      # full chassis thickness (z)
MIRROR_Z_EXTENT = 1.2 * mm                     # mirror core, centered in the depth
COMBINER_MATERIAL = agc_m074                   # glass; env outside is air
# The mirror core (MIRROR_Z_EXTENT) sits at the center of the COMBINER_DEPTH
# block, so the outer (COMBINER_DEPTH - MIRROR_Z_EXTENT)/2 = 0.9 mm on each side
# is index-matched cover glass — modeled as part of the one block (no interface).
# Placement / chassis extent — NOT given in the spec, sensible defaults:
COMBINER_CENTER = jnp.array([7.0 * mm, 24.0 * mm, 1.6 * mm])
COMBINER_WIDTH = 14.0 * mm                     # chassis x
COMBINER_HEIGHT = 12.0 * mm                    # chassis y (holds the stack)

# Target pupil (eyebox detector) — the plane where the eye samples the image.
# z is the EYE_RELIEF measured from the combiner's (untilted) eye-side edge:
#   pupil_z = combiner_center_z + depth/2 + eye_relief = 1.6 + 1.5 + 16 = 19.1 mm.
# x/y placement + size are not yet specified — sensible defaults.
EYE_RELIEF = 16.0 * mm
PUPIL_CENTER = jnp.array([
    7.0 * mm,
    14.5 * mm,
    COMBINER_CENTER[2] + COMBINER_DEPTH / 2 + EYE_RELIEF,
])
PUPIL_NORMAL = jnp.array([0.0, 0.0, -1.0])     # faces back toward the combiner
PUPIL_WIDTH = 14.0 * mm
PUPIL_HEIGHT = 18.0 * mm

# Visualization
NORMAL_DRAW_LENGTH = 2.0 * mm    # length of each normal line
REPORT_PATH = Path("examples/reports/visualize_perseus.html")

# ── Derived ──────────────────────────────────────────────────────────────────
WORLD_X = jnp.array([1.0, 0.0, 0.0])
STRAIGHT_DOWN = jnp.array([0.0, -1.0, 0.0])
# The tilts are CCW about world-x — the opposite sense of rotate_vectors'
# right-handed rotation (whose positive angle swings toward -z), so apply the
# negative angle. The results lean toward +z (the eye side).
PROJECTOR_DIRECTION = normalize(rotate_vectors(STRAIGHT_DOWN, WORLD_X, -PROJECTOR_TILT))
# The aperture is oriented parallel to the combiner's prism (top / entry) face,
# with its normal pointing downstream (-y, into the combiner): the prism face's
# inward normal [0, -cos(prism), -sin(prism)] before the combiner's pantoscopic
# tilt, so apply the same tilt to it.
APERTURE_NORMAL = normalize(rotate_vectors(
    jnp.array([0.0, -math.cos(float(COMBINER_PRISM_ANGLE)),
               -math.sin(float(COMBINER_PRISM_ANGLE))]),
    WORLD_X, -COMBINER_PANTOSCOPIC_TILT))
# The aperture sits APERTURE_OFFSET downstream of the projector along the beam,
# then dropped APERTURE_DROP_Y lower in y.
APERTURE_POSITION = (PROJECTOR_POSITION
                     + APERTURE_OFFSET * PROJECTOR_DIRECTION
                     - jnp.array([0.0, APERTURE_DROP_Y, 0.0]))

# Combiner normal: the eye-facing +z slab face, leaned CCW by the pantoscopic
# tilt. The mirror normals sit MIRROR_TO_COMBINER_NORMAL_ANGLE off this.
COMBINER_NORMAL_UNTILTED = jnp.array([0.0, 0.0, 1.0])
COMBINER_NORMAL = normalize(
    rotate_vectors(COMBINER_NORMAL_UNTILTED, WORLD_X, -COMBINER_PANTOSCOPIC_TILT))


def build_projector() -> PlayNitrideLed:
    """The Perseus projector, from the PARAMS above."""
    return PlayNitrideLed.create_broadband(
        position=PROJECTOR_POSITION, direction=PROJECTOR_DIRECTION,
        beam_width=BEAM_WIDTH, beam_height=BEAM_HEIGHT,
        nx=BEAM_RAYS_X, ny=BEAM_RAYS_Y,
    )


def trace_beam(system, projector, wavelength) -> list:
    """Ray-trace the projector beam through the combiner.

    Traces the main transmitted path (aperture → chassis back → the mirror
    cascade → chassis front) plus one reflected branch per mirror (reflect
    off mirror i → out the eye-side face → onto the pupil). Returns a flat
    list of ``TraceResult`` for ``plot_system`` to draw.
    """
    beam = projector.generate_rays(direction=PROJECTOR_DIRECTION,
                                   wavelength=wavelength)
    main_route = prepare_route(combiner_main_path(system), wavelength)
    branch_routes = [
        prepare_route(route, wavelength)
        for route in build_combiner_branch_routes(
            system, num_mirrors=COMBINER_NUM_MIRRORS, chassis_name="combiner")
    ]
    routes = [main_route, *branch_routes]
    return [trace_rays(route, beam, wavelength=wavelength) for route in routes]


def build_aperture() -> RectangularAperture:
    """The beam-defining aperture, from the PARAMS above."""
    return RectangularAperture(
        name="aperture", position=APERTURE_POSITION, normal=APERTURE_NORMAL,
        width=APERTURE_OUTER_WIDTH, height=APERTURE_OUTER_HEIGHT,
        inner_width=APERTURE_INNER_WIDTH, inner_height=APERTURE_INNER_HEIGHT,
    )


def build_combiner() -> tuple[GlassBlock, list[PartialMirror]]:
    """The combiner chassis + cascaded partial mirrors, from the PARAMS.

    Built in the untilted frame (combiner normal = +z), then the whole
    assembly is rotated CCW by ``COMBINER_PANTOSCOPIC_TILT`` about world-x
    through the combiner center. Returns ``(chassis, mirrors)``.
    """
    # Mirror plane orientation: its normal sits the given angle off the
    # combiner normal (+z, untilted). Mirrors span the full chassis width in
    # x and only the MIRROR_Z_EXTENT core in z (height = extent / sin(angle)).
    angle = float(MIRROR_TO_COMBINER_NORMAL_ANGLE)
    mirror_normal = jnp.array([0.0, math.sin(angle), math.cos(angle)])
    mirror_width = float(COMBINER_WIDTH)
    mirror_height = float(MIRROR_Z_EXTENT / math.sin(angle))

    # Chassis with the top-side prism baked in as a z-skew wedge.
    z_skew = float(COMBINER_DEPTH * math.tan(float(COMBINER_PRISM_ANGLE)))
    chassis = GlassBlock.create_chassis(
        name="combiner", x=float(COMBINER_WIDTH), y=float(COMBINER_HEIGHT),
        z=float(COMBINER_DEPTH), material=COMBINER_MATERIAL, z_skew=z_skew,
    ).translate(COMBINER_CENTER)

    # Mirror centers: equal MIRROR_Y_SPACING gaps in y, stack centered on the
    # chassis; all share the chassis x/z center (they only step down in y).
    stack_span = (COMBINER_NUM_MIRRORS - 1) * MIRROR_Y_SPACING
    top_y = COMBINER_CENTER[1] + stack_span / 2
    mirror_y = top_y - jnp.arange(COMBINER_NUM_MIRRORS) * MIRROR_Y_SPACING
    mirror_positions = jnp.stack([
        jnp.full((COMBINER_NUM_MIRRORS,), COMBINER_CENTER[0]),
        mirror_y,
        jnp.full((COMBINER_NUM_MIRRORS,), COMBINER_CENTER[2]),
    ], axis=1)  # (M, 3)

    # Apply the pantoscopic tilt (CCW about world-x) about the combiner center.
    tilt = -COMBINER_PANTOSCOPIC_TILT
    chassis = chassis.rotate(WORLD_X, tilt, COMBINER_CENTER)
    mirror_positions = rotate_points(mirror_positions, WORLD_X, tilt, COMBINER_CENTER)
    mirror_normal = normalize(rotate_vectors(mirror_normal, WORLD_X, tilt))

    mirrors = [
        PartialMirror(
            name=f"mirror_{mirror_idx}",
            position=mirror_positions[mirror_idx],
            normal=mirror_normal.copy(),
            width=mirror_width, height=mirror_height,
        )
        for mirror_idx in range(COMBINER_NUM_MIRRORS)
    ]
    return chassis, mirrors


def build_pupil() -> RectangularPupil:
    """The target pupil (eyebox detector), from the PARAMS above."""
    return RectangularPupil(
        name="pupil", position=PUPIL_CENTER, normal=PUPIL_NORMAL,
        width=PUPIL_WIDTH, height=PUPIL_HEIGHT,
    )


def normal_trace(position, direction, name, color) -> go.Scatter3d:
    """A short line along ``direction`` from ``position`` (an element normal).

    Length is ``NORMAL_DRAW_LENGTH``.
    """
    start = np.asarray(position)
    end = start + NORMAL_DRAW_LENGTH * np.asarray(direction)
    return go.Scatter3d(
        x=[float(start[0]), float(end[0])],
        y=[float(start[1]), float(end[1])],
        z=[float(start[2]), float(end[2])],
        mode="lines",
        line=dict(color=color, width=6),
        name=name, hoverinfo="name",
    )


def main():
    projector = build_projector()

    print("── Perseus projector ──")
    print(f"position : {np.round(np.asarray(PROJECTOR_POSITION) / mm, 2)} mm")
    print(f"tilt     : {float(PROJECTOR_TILT) / deg:.3f} deg CCW (about world-x)")
    print(f"normal   : {np.round(np.asarray(PROJECTOR_DIRECTION), 4)}")
    print(f"beam     : {BEAM_WIDTH / mm:.1f} x {BEAM_HEIGHT / mm:.1f} mm, "
          f"{BEAM_RAYS_X} x {BEAM_RAYS_Y} rays")
    print("── Perseus aperture ──")
    print(f"position : {np.round(np.asarray(APERTURE_POSITION) / mm, 3)} mm")
    print(f"normal   : {np.round(np.asarray(APERTURE_NORMAL), 4)}  "
          f"(parallel to the combiner prism face)")
    print(f"frame    : {APERTURE_OUTER_WIDTH / mm:.1f} x {APERTURE_OUTER_HEIGHT / mm:.1f} mm, "
          f"opening {APERTURE_INNER_WIDTH / mm:.1f} x {APERTURE_INNER_HEIGHT / mm:.1f} mm")

    chassis, mirrors = build_combiner()
    print("── Perseus combiner ──")
    print(f"center   : {np.round(np.asarray(COMBINER_CENTER) / mm, 2)} mm")
    print(f"pantoscopic tilt : {float(COMBINER_PANTOSCOPIC_TILT) / deg:.3f} deg CCW")
    print(f"normal   : {np.round(np.asarray(COMBINER_NORMAL), 4)}")
    print(f"mirrors  : {COMBINER_NUM_MIRRORS}, "
          f"{MIRROR_Y_SPACING / mm:.2f} mm y-spacing, "
          f"normal {float(MIRROR_TO_COMBINER_NORMAL_ANGLE) / deg:.3f} deg off combiner normal")
    print(f"prism    : {float(COMBINER_PRISM_ANGLE) / deg:.3f} deg  "
          f"depth {COMBINER_DEPTH / mm:.2f} mm (mirror core {MIRROR_Z_EXTENT / mm:.2f} mm, "
          f"cover {(COMBINER_DEPTH - MIRROR_Z_EXTENT) / 2 / mm:.2f} mm/side)  "
          f"material {COMBINER_MATERIAL.name}")
    print(f"mirror normal: {np.round(np.asarray(mirrors[0].normal), 4)}  "
          f"size {float(mirrors[0].width) / mm:.1f} x {float(mirrors[0].height) / mm:.2f} mm")

    pupil = build_pupil()
    print("── Perseus target pupil (eyebox detector) ──")
    print(f"center   : {np.round(np.asarray(PUPIL_CENTER) / mm, 2)} mm")
    print(f"normal   : {np.round(np.asarray(PUPIL_NORMAL), 4)}")
    print(f"size     : {PUPIL_WIDTH / mm:.1f} x {PUPIL_HEIGHT / mm:.1f} mm")

    # System holds aperture + combiner + pupil; the projector draws its own box.
    system = OpticalSystem(env_material=air)
    system.add(build_aperture())
    system.add(chassis)
    for mirror in mirrors:
        system.add(mirror)
    system.add(pupil)

    # Ray-trace the beam through the combiner (main path + reflected branches).
    # Flat mirrors → geometry is wavelength-independent; trace at the projector
    # spectral peak (the glass adds only negligible dispersion).
    spec_wls, spec_rad = projector.spectrum
    trace_wavelength = float(spec_wls[int(jnp.argmax(spec_rad))])
    trace_results = trace_beam(system, projector, trace_wavelength)
    pupil_rays = sum(int(np.asarray(tr.valids[:, -1]).sum())
                     for tr in trace_results[1:])   # branches end on the pupil
    rays_per_beam = BEAM_RAYS_X * BEAM_RAYS_Y
    print("── Ray trace ──")
    print(f"wavelength : {trace_wavelength / nm:.0f} nm (projector peak)")
    print(f"routes     : main + {COMBINER_NUM_MIRRORS} reflected branches, "
          f"{rays_per_beam} rays each")
    print(f"branch rays reaching the pupil: {pupil_rays} / "
          f"{COMBINER_NUM_MIRRORS * rays_per_beam}")

    fig = plot_system(system, trace_results=trace_results,
                      projector=projector, show=False)
    fig.add_trace(normal_trace(
        PROJECTOR_POSITION, PROJECTOR_DIRECTION, "projector normal", "red"))
    fig.add_trace(normal_trace(
        APERTURE_POSITION, APERTURE_NORMAL, "aperture normal", "orange"))
    fig.add_trace(normal_trace(
        COMBINER_CENTER, COMBINER_NORMAL, "combiner normal", "lime"))
    fig.update_layout(title="Perseus — ray trace")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(REPORT_PATH))
    print(f"\nSaved: {REPORT_PATH}")
    fig.show()


if __name__ == "__main__":
    main()
