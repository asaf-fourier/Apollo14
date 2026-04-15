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

import time

import jax
import jax.numpy as jnp
import numpy as np

from apollo14.combiner import (
    build_default_system,
    DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
)
from apollo14.route import build_route, branch_path, absorb
from apollo14.trace import prepare_route, trace_rays

# JIT-cached tracer: route structure is stable across wavelengths (only
# per-face n1/n2 leaves and the scalar mirror wavelength change), so one
# compile per route is reused across the full wavelength scan.
_trace_rays_jit = jax.jit(trace_rays)
from apollo14.projector import PlayNitrideLed, scan_directions
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

# ── RGB projectors (PlayNitride micro-LEDs) ───────────────────────────────

RGB_COLOR_IDX = {"R": 0, "G": 1, "B": 2}

projectors = {
    c: PlayNitrideLed.create(
        position=DEFAULT_LIGHT_POSITION,
        direction=DEFAULT_LIGHT_DIRECTION,
        beam_width=4.0 * mm,
        beam_height=2.0 * mm,
        nx=5, ny=5,
        color=c,
    )
    for c in RGB_COLOR_IDX
}


def _peak_wavelength(proj):
    wls, rad = proj.spectrum
    return float(wls[int(jnp.argmax(rad))])


# 30 nm window in 1 nm steps, centred on each channel's measured peak.
SCAN_OFFSETS = jnp.arange(-15, 16) * nm  # 31 samples
scan_wavelengths = {
    c: [_peak_wavelength(p) + float(off) for off in SCAN_OFFSETS]
    for c, p in projectors.items()
}
peak_wavelengths = {c: _peak_wavelength(p) for c, p in projectors.items()}

for c in "RGB":
    lo = scan_wavelengths[c][0] / nm
    hi = scan_wavelengths[c][-1] / nm
    print(f"{c} peak {peak_wavelengths[c]/nm:.0f} nm — "
          f"scan {lo:.0f}..{hi:.0f} nm ({len(scan_wavelengths[c])} samples)")

# Prepare one (main, branches) route set per (color, wavelength).
main_routes = {
    c: [prepare_route(main_route, wl) for wl in scan_wavelengths[c]]
    for c in RGB_COLOR_IDX
}
branch_routes_rgb = {
    c: [{name: prepare_route(r, wl) for name, r in branch_routes.items()}
        for wl in scan_wavelengths[c]]
    for c in RGB_COLOR_IDX
}

# Extra single-wavelength route set for the 3D visualisation (green peak).
viz_main_route = prepare_route(main_route, peak_wavelengths["G"])
viz_branch_routes = {
    name: prepare_route(r, peak_wavelengths["G"])
    for name, r in branch_routes.items()
}

# ── Scan grid ──────────────────────────────────────────────────────────────

x_fov = 8.0 * deg
y_fov = 8.0 * deg
step = 2.0 * deg
num_x = int(x_fov / step) + 1
num_y = int(y_fov / step) + 1

scan_dirs, scan_angles = scan_directions(
    DEFAULT_LIGHT_DIRECTION, x_fov, y_fov, num_x, num_y,
)

print(f"\nScan: {num_x}x{num_y} angles, FOV {x_fov/deg:.0f}x{y_fov/deg:.0f} deg")
print(f"Projector: {projectors['R'].nx}x{projectors['R'].ny} rays, "
      f"{projectors['R'].beam_width/mm:.0f}x"
      f"{projectors['R'].beam_height/mm:.0f} mm")


# Blue channel has a linear angular falloff of ~2% per degree in both
# x and y (applied as a scalar on ray intensity — the tracer is linear).
BLUE_FALLOFF_PER_DEG = 0.02


def blue_angular_gain(ax_rad, ay_rad):
    ax_deg = float(ax_rad) / deg
    ay_deg = float(ay_rad) / deg
    gain = (1.0 - BLUE_FALLOFF_PER_DEG * abs(ax_deg)) \
         * (1.0 - BLUE_FALLOFF_PER_DEG * abs(ay_deg))
    return max(gain, 0.0)

# ── Trace main + branches per color, per wavelength, per angle ────────────

print("\n── Tracing RGB across FOV (main + 6 branches, 31 wavelengths/color) ──")

# Intensity reaching the pupil per (color, angle), summed over wavelengths
# and branches.
result = np.zeros((num_y, num_x, 3))

# Visualization traces — green peak only, so the 3D view isn't overwhelming.
viz_traces = []

# Per-angle green-branch traces (peak wavelength) for the pupil-fill plot.
pupil_traces_per_angle = []

trace_t0 = time.perf_counter()

for iy in range(num_y):
    for ix in range(num_x):
        direction = scan_dirs[iy, ix]
        ax_rad, ay_rad = scan_angles[iy, ix]

        for c, ci in RGB_COLOR_IDX.items():
            proj = projectors[c]
            gain = blue_angular_gain(ax_rad, ay_rad) if c == "B" else 1.0

            total = 0.0
            for wi, wl in enumerate(scan_wavelengths[c]):
                ray = proj.generate_rays(direction=direction, wavelength=wl)
                if gain != 1.0:
                    ray = ray._replace(intensity=ray.intensity * gain)

                # Main path (doesn't reach the pupil — recorded for completeness).
                _ = _trace_rays_jit(main_routes[c][wi], ray, wavelength=wl)

                # Each reflected branch contributes to the pupil.
                for name, broute in branch_routes_rgb[c][wi].items():
                    tr = _trace_rays_jit(broute, ray, wavelength=wl)
                    last_valid = tr.valids[..., -1]
                    total += float(
                        jnp.where(last_valid, tr.final_intensity, 0.0).sum()
                    )
            result[iy, ix, ci] = total

        # Green-peak traces for visualisation + pupil fill (single wavelength).
        ray_g = projectors["G"].generate_rays(
            direction=direction, wavelength=peak_wavelengths["G"])
        green_branch_traces = []
        for name, broute in viz_branch_routes.items():
            tr = _trace_rays_jit(broute, ray_g,
                                 wavelength=peak_wavelengths["G"])
            viz_traces.append(tr)
            green_branch_traces.append(tr)
        pupil_traces_per_angle.append(green_branch_traces)

        tr_main = _trace_rays_jit(viz_main_route, ray_g,
                                  wavelength=peak_wavelengths["G"])
        viz_traces.append(tr_main)

trace_elapsed = time.perf_counter() - trace_t0
n_angles = num_x * num_y
print(f"\nTrace step elapsed: {trace_elapsed:.2f} s "
      f"({1e3*trace_elapsed/n_angles:.1f} ms/angle, "
      f"{n_angles} angles)")

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
                  projector=projectors["G"], show=False)
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
