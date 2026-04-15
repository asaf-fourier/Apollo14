"""JAX tracer demo — main path + all six reflected branches in RGB.

Builds the combiner's main transmitted path plus one reflected branch per
mirror (each terminating on the pupil), prepares beams at three wavelengths
(R/G/B), sweeps the projector across the FOV, and renders all traced rays
in a 3D ``plot_system`` view with a per-angle slider.

Per-color intensity reaching the pupil across the FOV is printed as a
summary. Chromatic effects come from two places: the per-color reflectance
on each partial mirror and the per-wavelength refractive index resolved by
``prepare_route``.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

import apollo14
from apollo14.combiner import (
    build_default_system,
    DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
)
from apollo14.route import build_route, branch_path, absorb
from apollo14.trace import prepare_route, trace_rays
from apollo14.projector import Projector, scan_directions
from apollo14.visualizer import plot_system, plot_pupil_fill
from apollo14.units import mm, nm, deg


# ── Build system + routes ──────────────────────────────────────────────────

system = build_default_system()

# Explicit main path — plain strings for top-level elements, (block, face)
# tuples for addressable sub-elements.
main_path = [
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

# One reflected branch per mirror: everything up to that mirror, reflect,
# then out through the top of the chassis and into the pupil. Each branch
# is just another linear route, traced independently.
#
# Exit face is "top" (not "front"): after reflecting off a ~48°-tilted
# partial mirror, the ray direction is mostly +z, so it leaves the
# chassis through the top slab.
branch_tail = [("chassis", "top"), absorb("pupil")]
branch_paths = {
    f"mirror_{i}": branch_path(main_path, at=f"mirror_{i}", tail=branch_tail)
    for i in range(6)
}

main_route = build_route(system, main_path)
branch_routes = {name: build_route(system, p) for name, p in branch_paths.items()}

def _describe(route):
    kinds = [type(s).__name__ for s in route.segments]
    return f"{len(route.segments)} segments [{', '.join(kinds)}]"

print(f"Main route: {_describe(main_route)}")
for name, r in branch_routes.items():
    print(f"  branch {name}: {_describe(r)}")

# ── RGB beams ──────────────────────────────────────────────────────────────

RGB_WAVELENGTHS = {
    "R": 650.0 * nm,
    "G": 550.0 * nm,
    "B": 450.0 * nm,
}
RGB_COLOR_IDX = {"R": 0, "G": 1, "B": 2}

main_routes = {c: prepare_route(main_route, wl) for c, wl in RGB_WAVELENGTHS.items()}
branch_routes_rgb = {
    c: {name: prepare_route(r, wl) for name, r in branch_routes.items()}
    for c, wl in RGB_WAVELENGTHS.items()
}

# ── Projector + scan grid ──────────────────────────────────────────────────

x_fov = 8.0 * deg
y_fov = 8.0 * deg
step = 2.0 * deg
num_x = int(x_fov / step) + 1
num_y = int(y_fov / step) + 1

scan_dirs, scan_angles = scan_directions(
    DEFAULT_LIGHT_DIRECTION, x_fov, y_fov, num_x, num_y,
)

PROJECTOR_CSV = (
    Path(apollo14.__file__).parent / "data/projector"
    / "PlayNitride_(-1-3)_APL05prc.csv"
)
projector = Projector.from_csv(
    PROJECTOR_CSV,
    position=DEFAULT_LIGHT_POSITION,
    direction=DEFAULT_LIGHT_DIRECTION,
    beam_width=4.0 * mm,
    beam_height=2.0 * mm,
    nx=5, ny=5,
)


print(f"\nScan: {num_x}x{num_y} angles, FOV {x_fov/deg:.0f}x{y_fov/deg:.0f} deg")
print(f"Projector: {projector.nx}x{projector.ny} rays, "
      f"{projector.beam_width/mm:.0f}x{projector.beam_height/mm:.0f} mm")

# ── Trace main + branches per color, per angle ─────────────────────────────

print("\n── Tracing RGB across FOV (main + 6 branches) ──")

# Intensity reaching the pupil per (color, angle), summed over branches.
result = np.zeros((num_y, num_x, 3))

# Visualization traces — just one color so the 3D view isn't overwhelming.
viz_traces = []

# Per-angle green-branch traces for the pupil-fill plot: one list of
# 6 branch TraceResults per scan angle, row-major over (iy, ix).
pupil_traces_per_angle = []

for iy in range(num_y):
    for ix in range(num_x):
        direction = scan_dirs[iy, ix]

        green_branch_traces = []

        for c, ci in RGB_COLOR_IDX.items():
            # Per-color Ray batch — intensity comes from the projector's
            # measured spectrum sampled at this wavelength.
            ray = projector.generate_rays(direction=direction,
                                          wavelength=RGB_WAVELENGTHS[c])

            # Main path (doesn't reach the pupil — recorded for completeness).
            _ = trace_rays(main_routes[c], ray, color_idx=ci)

            # Each reflected branch contributes to the pupil.
            total = 0.0
            for name, broute in branch_routes_rgb[c].items():
                tr = trace_rays(broute, ray, color_idx=ci)
                last_valid = tr.valids[..., -1]
                total += float(
                    jnp.where(last_valid, tr.final_intensity, 0.0).sum()
                )
                if c == "G":
                    viz_traces.append(tr)
                    green_branch_traces.append(tr)
            result[iy, ix, ci] = total

        pupil_traces_per_angle.append(green_branch_traces)

        # Main-path viz trace for the green channel (one per angle).
        ray_g = projector.generate_rays(direction=direction,
                                        wavelength=RGB_WAVELENGTHS["G"])
        tr_main = trace_rays(main_routes["G"], ray_g,
                             color_idx=RGB_COLOR_IDX["G"])
        viz_traces.append(tr_main)

# ── Summary ────────────────────────────────────────────────────────────────

print(f"\nResults shape: {result.shape}  (num_y, num_x, 3)")
print(f"Total pupil intensity — "
      f"R: {result[:,:,0].sum():.3f}, "
      f"G: {result[:,:,1].sum():.3f}, "
      f"B: {result[:,:,2].sum():.3f}")

cy, cx = num_y // 2, num_x // 2
print(f"Center angle intensity — "
      f"R: {result[cy, cx, 0]:.3f}, "
      f"G: {result[cy, cx, 1]:.3f}, "
      f"B: {result[cy, cx, 2]:.3f}")

# ── 3D visualization ───────────────────────────────────────────────────────

print("\n── Rendering 3D view (green channel) ──")

# viz_traces has 7 entries per scan angle (6 branches + main).
fig = plot_system(system, trace_results=viz_traces,
                  scan_angles=np.asarray(scan_angles),
                  projector=projector, show=False)
fig.show()
fig.write_html("examples/reports/jax_tracer_demo.html")
print("Saved: jax_tracer_demo.html")

# ── Pupil fill visualization ───────────────────────────────────────────────

print("\n── Rendering pupil fill (green channel) ──")

pupil_element = system.resolve("pupil")

fig_pupil = plot_pupil_fill(
    pupil_traces_per_angle,
    scan_angles=np.asarray(scan_angles),
    pupil_element=pupil_element,
    pixel_size=0.5,
    show=False,
)
fig_pupil.write_html("examples/reports/jax_tracer_demo_pupil.html")
print("Saved: jax_tracer_demo_pupil.html")

print("\nDone.")
