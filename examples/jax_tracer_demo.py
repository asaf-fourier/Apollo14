"""JAX tracer demo — build, prepare, trace, and visualize the main path.

Builds the combiner's straight-through main path, prepares a ``Beam`` at
the default wavelength, sweeps the projector across the FOV, and renders
the traced rays in a 3D ``plot_system`` view with a per-angle slider.

Reflected branches (the physics that actually reaches the pupil) are
deferred — this demo exercises the current linear tracer end-to-end.
"""

import numpy as np

from apollo14.combiner import (
    build_default_system,
    DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, DEFAULT_WAVELENGTH,
)
from apollo14.trace import (
    build_route, branch_path, combiner_main_path, prepare_beam, trace_beam,
)
from apollo14.surface import ABSORB
from apollo14.projector import Projector, scan_directions
from apollo14.visualizer import plot_system
from apollo14.units import mm, deg


# ── Build system + beam ─────────────────────────────────────────────────────

system = build_default_system()

# Two equivalent ways to declare the same main path:
#   (a) via the helper
route = combiner_main_path(system)
#   (b) as an explicit list — plain names for single elements,
#       (block, face) tuples for addressable sub-elements.
explicit_path = [
    "aperture",
    ("chassis", "back"),
    "mirror_0",
    "mirror_1",
    "mirror_2",
    "mirror_3",
    "mirror_4",
    "mirror_5",
    ("chassis", "front"),
]

explicit_route = build_route(system, explicit_path)
beam = prepare_beam(explicit_route, DEFAULT_WAVELENGTH)

# Branch route: everything up to mirror_0, reflect off it, then through
# the exit face and into the pupil. A branch is just another linear route.
branch_0_path = branch_path(
    explicit_path, at="mirror_0",
    tail=[("chassis", "front"), ("pupil", ABSORB)],
)
branch_0_route = build_route(system, branch_0_path)
branch_0_beam = prepare_beam(branch_0_route, DEFAULT_WAVELENGTH)

print(f"System route length: {route.position.shape[0]} "
      f"(explicit: {explicit_route.position.shape[0]}, "
      f"branch_0: {branch_0_route.position.shape[0]})")

# ── Projector + scan grid ───────────────────────────────────────────────────

x_fov = 8.0 * deg
y_fov = 8.0 * deg
step = 2.0 * deg
num_x = int(x_fov / step) + 1
num_y = int(y_fov / step) + 1

scan_dirs, scan_angles = scan_directions(
    DEFAULT_LIGHT_DIRECTION, x_fov, y_fov, num_x, num_y,
)

projector = Projector.uniform(
    position=DEFAULT_LIGHT_POSITION,
    direction=DEFAULT_LIGHT_DIRECTION,
    beam_width=4.0 * mm,
    beam_height=2.0 * mm,
    wavelength=DEFAULT_WAVELENGTH,
    nx=5, ny=5,
)

print(f"Scan: {num_x}x{num_y} angles, FOV {x_fov/deg:.0f}x{y_fov/deg:.0f} deg")
print(f"Beam: {projector.nx}x{projector.ny} rays, "
      f"{projector.beam_width/mm:.0f}x{projector.beam_height/mm:.0f} mm")

# ── Trace one batched beam per scan angle ──────────────────────────────────

print("\n── Tracing main path + branch_0 at each scan angle ──")

trace_results = []
for iy in range(num_y):
    for ix in range(num_x):
        direction = scan_dirs[iy, ix]
        origins, _, _, _ = projector.generate_rays(direction=direction)
        tr_main = trace_beam(beam, origins, direction, color_idx=0)
        tr_branch = trace_beam(branch_0_beam, origins, direction, color_idx=0)
        trace_results.append(tr_main)
        trace_results.append(tr_branch)

# ── 3D visualization with angle slider ─────────────────────────────────────

print("── Rendering 3D view ──")

fig = plot_system(system, trace_results=trace_results,
                  scan_angles=np.asarray(scan_angles), show=False)
fig.write_html("jax_tracer_demo.html")
print("Saved: jax_tracer_demo.html")

print("\nDone.")
