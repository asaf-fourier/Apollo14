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
# Kept as device-side (W,) arrays so the scan can be vmapped.
SCAN_OFFSETS = jnp.arange(-15, 16) * nm  # 31 samples
peak_wavelengths = {c: _peak_wavelength(p) for c, p in projectors.items()}
scan_wavelengths = {
    c: jnp.asarray([peak_wavelengths[c] + float(off) for off in SCAN_OFFSETS],
                    dtype=jnp.float32)
    for c in projectors
}

for c in "RGB":
    lo = float(scan_wavelengths[c][0]) / nm
    hi = float(scan_wavelengths[c][-1]) / nm
    print(f"{c} peak {peak_wavelengths[c]/nm:.0f} nm — "
          f"scan {lo:.0f}..{hi:.0f} nm ({scan_wavelengths[c].shape[0]} samples)")

# Single-wavelength prepared routes for the 3D visualisation (green peak).
# The hot-loop tracer works from raw (pre-prepare_route) routes so it can
# vmap over wavelength; the viz path only needs one wavelength so we keep
# it on the simpler prepared-route API.
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

# Flatten to (A, 3) / (A, 2) so vmap only sees one batched axis.
flat_dirs = scan_dirs.reshape(-1, 3)        # (A, 3)
flat_angles = scan_angles.reshape(-1, 2)    # (A, 2), radians
A = flat_dirs.shape[0]

print(f"\nScan: {num_x}x{num_y} angles, FOV {x_fov/deg:.0f}x{y_fov/deg:.0f} deg")
print(f"Projector: {projectors['R'].nx}x{projectors['R'].ny} rays, "
      f"{projectors['R'].beam_width/mm:.0f}x"
      f"{projectors['R'].beam_height/mm:.0f} mm")


# Blue channel has a linear angular falloff of ~2% per degree in both
# x and y. Precomputed on-device as a per-angle gain vector so it can
# multiply ray intensity inside the vmapped trace without a host sync.
BLUE_FALLOFF_PER_DEG = 0.02
_ax_deg = flat_angles[:, 0] / deg
_ay_deg = flat_angles[:, 1] / deg
blue_gain_per_angle = jnp.clip(
    (1.0 - BLUE_FALLOFF_PER_DEG * jnp.abs(_ax_deg))
    * (1.0 - BLUE_FALLOFF_PER_DEG * jnp.abs(_ay_deg)),
    0.0, 1.0,
)  # (A,)
gain_per_angle_by_color = {
    "R": jnp.ones(A, dtype=jnp.float32),
    "G": jnp.ones(A, dtype=jnp.float32),
    "B": blue_gain_per_angle.astype(jnp.float32),
}

# ── Vmapped trace kernel: (wavelength × angle × ray) per branch ───────────
#
# One JIT-compiled function per (color, branch_route_shape) pair. Closes
# over the color-specific projector, wavelength grid, and angular gain so
# the compiled kernel has no Python dispatch overhead — 31×25 = 775
# (wavelength, angle) combinations run as one XLA program per branch.

def make_branch_tracer(projector, wavelengths, directions, gain_per_angle):
    """Build a jitted function that sums pupil intensity per angle for one
    branch route, marginalised over all scan wavelengths.

    The returned function takes a raw (pre-``prepare_route``) route and
    returns an ``(A,)`` per-angle total summed over the wavelength scan.
    """
    def per_angle(prepared, wavelength, direction, gain):
        ray = projector.generate_rays(direction=direction, wavelength=wavelength)
        ray = ray._replace(intensity=ray.intensity * gain)
        tr = trace_rays(prepared, ray, wavelength=wavelength)
        last_valid = tr.valids[..., -1]
        return jnp.where(last_valid, tr.final_intensity, 0.0).sum()

    def per_wavelength(raw_route, wavelength):
        prepared = prepare_route(raw_route, wavelength)
        return jax.vmap(per_angle, in_axes=(None, None, 0, 0))(
            prepared, wavelength, directions, gain_per_angle)  # (A,)

    def trace_branch(raw_route):
        per_wl = jax.vmap(per_wavelength, in_axes=(None, 0))(
            raw_route, wavelengths)  # (W, A)
        return per_wl.sum(axis=0)  # (A,)

    return jax.jit(trace_branch)


branch_tracers = {
    c: make_branch_tracer(
        projectors[c], scan_wavelengths[c], flat_dirs,
        gain_per_angle_by_color[c],
    )
    for c in "RGB"
}

print("\n── Tracing RGB across FOV "
      "(6 branches × 31 wavelengths × 25 angles, vmapped) ──")


def run_once():
    """One full RGB sweep, returning per-color (A,) arrays, on-device."""
    out = {c: jnp.zeros(A, dtype=jnp.float32) for c in "RGB"}
    for c in "RGB":
        for name, raw_branch in branch_routes.items():
            out[c] = out[c] + branch_tracers[c](raw_branch)
    return out


# Warm-up: first call pays JIT compilation (one compile per branch shape,
# per color — ~18 compiles of the vmapped kernel).
warm_t0 = time.perf_counter()
warm = run_once()
for v in warm.values():
    v.block_until_ready()
warm_elapsed = time.perf_counter() - warm_t0
print(f"Warm-up (JIT compile + first run): {warm_elapsed:.2f} s")

# Timed run: all kernels cached, this is the actual execution cost.
trace_t0 = time.perf_counter()
result_on_device = run_once()
result_per_color = {c: np.asarray(v.block_until_ready())
                     for c, v in result_on_device.items()}
trace_elapsed = time.perf_counter() - trace_t0

result = np.zeros((num_y, num_x, 3))
for c, ci in RGB_COLOR_IDX.items():
    result[:, :, ci] = result_per_color[c].reshape(num_y, num_x)

print(f"\nTrace step elapsed: {trace_elapsed:.2f} s "
      f"({1e3*trace_elapsed/A:.1f} ms/angle, {A} angles)")

# ── Visualisation traces (green peak only, single wavelength path) ────────

viz_traces = []
pupil_traces_per_angle = []

for iy in range(num_y):
    for ix in range(num_x):
        direction = scan_dirs[iy, ix]
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
