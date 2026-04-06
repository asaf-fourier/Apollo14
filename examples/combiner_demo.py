"""
Combiner demo — equivalent to Apollo13's main_talos.py

Builds the default combiner system (6 cascaded partial mirrors in a glass chassis),
traces rays from a projector through it, and visualizes the result.
"""

import jax.numpy as jnp

from apollo14.combiner import (
    build_default_system,
    DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, DEFAULT_WAVELENGTH,
    DEFAULT_BEAM_WIDTH, DEFAULT_BEAM_HEIGHT, DEFAULT_X_FOV, DEFAULT_Y_FOV,
    DEFAULT_NUM_X_STEPS, DEFAULT_NUM_Y_STEPS,
)
from apollo14.elements.aperture import RectangularAperture
from apollo14.elements.glass_block import GlassBlock
from apollo14.elements.pupil import RectangularPupil
from apollo14.projector import Projector, scan_directions
from apollo14.tracer import trace_nonsequential, TraceResult
from apollo14.visualizer import plot_system, plot_pupil_fill
from apollo14.units import mm, nm, deg

# ── Build the system ──────────────────────────────────────────────────────────

system = build_default_system()

print("System elements:")
for elem in system.elements:
    print(f"  {elem.name}")

# ── Extract elements for projector/merit setup ──────────────────────────────

aperture = next(e for e in system.elements if isinstance(e, RectangularAperture))
pupil = next(e for e in system.elements if isinstance(e, RectangularPupil))

# ── Create a projector ────────────────────────────────────────────────────────

projector = Projector.uniform(
    position=DEFAULT_LIGHT_POSITION,
    direction=DEFAULT_LIGHT_DIRECTION,
    beam_width=DEFAULT_BEAM_WIDTH,
    beam_height=DEFAULT_BEAM_HEIGHT,
    wavelength=DEFAULT_WAVELENGTH,
    nx=5, ny=3,
    intensity=1.0,
)

# ── Trace on-axis (single scan angle) ────────────────────────────────────────

print("\n── On-axis trace ──")
origins, directions, intensities, pixels = projector.generate_rays()

results: list[TraceResult] = []
for i in range(origins.shape[0]):
    result = trace_nonsequential(
        system, origins[i], directions[i], DEFAULT_WAVELENGTH,
        intensity=float(intensities[i]),
    )
    results.append(result)

# Summarize pupil hits
pupil_hits = [r.pupil_hit for r in results if r.pupil_hit is not None]
print(f"Rays: {origins.shape[0]}, Pupil hits: {len(pupil_hits)}")
if pupil_hits:
    total_intensity = sum(float(h.intensity) for h in pupil_hits)
    print(f"Total pupil intensity: {total_intensity:.4f}")

# ── Angular scan across FOV ──────────────────────────────────────────────────

print("\n── Angular scan ──")
scan_dirs, scan_angles = scan_directions(
    DEFAULT_LIGHT_DIRECTION,
    x_fov=DEFAULT_X_FOV,
    y_fov=DEFAULT_Y_FOV,
    num_x=DEFAULT_NUM_X_STEPS,
    num_y=DEFAULT_NUM_Y_STEPS,
)

all_results: list[TraceResult] = []
for iy in range(scan_dirs.shape[0]):
    for ix in range(scan_dirs.shape[1]):
        d = scan_dirs[iy, ix]
        origins, directions, intensities, pixels = projector.generate_rays(direction=d)

        for i in range(origins.shape[0]):
            result = trace_nonsequential(
                system, origins[i], directions[i], DEFAULT_WAVELENGTH,
                intensity=float(intensities[i]),
            )
            all_results.append(result)

pupil_hits = [r.pupil_hit for r in all_results if r.pupil_hit is not None]
total_rays = len(all_results)
print(f"Total rays: {total_rays}, Pupil hits: {len(pupil_hits)}")
if pupil_hits:
    avg_intensity = sum(float(h.intensity) for h in pupil_hits) / len(pupil_hits)
    print(f"Average pupil hit intensity: {avg_intensity:.4f}")

# ── Per-mirror reflectance summary (JAX tracer) ─────────────────────────────

print("\n── Per-mirror reflectance (on-axis, single ray) ──")
from apollo14.jax_tracer import trace_ray, params_from_system

params = params_from_system(system, DEFAULT_WAVELENGTH)
chassis = next(e for e in system.elements if isinstance(e, GlassBlock))
n_glass = float(chassis.material.n(DEFAULT_WAVELENGTH))
num_mirrors = len([e for e in system.elements if hasattr(e, 'reflection_ratio') and hasattr(e, 'transmission_ratio')])

pupil_pts, pupil_ints, pupil_valid = trace_ray(
    DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, n_glass, params,
)

for i in range(num_mirrors):
    status = "hit pupil" if pupil_valid[i] else "missed"
    print(f"  mirror_{i}: reflected={float(pupil_ints[i]):.4f}  {status}")

# ── Merit function evaluation ─────────────────────────────────────────────────

print("\n── Merit function (D65 white balance) ──")

from helios.merit import MeritConfig, evaluate_merit, D65_WEIGHTS, DEFAULT_WAVELENGTHS

merit_config = MeritConfig(
    wavelengths=DEFAULT_WAVELENGTHS,   # R=630nm, G=525nm, B=460nm microLED peaks
    d65_weights=D65_WEIGHTS,
    target_efficiency=0.10,            # 10% of projector light to pupil
    pupil_nx=3, pupil_ny=3,            # 3x3 pupil sample grid
    angle_nx=3, angle_ny=3,            # 3x3 angular samples within FOV
)

# Use a beam that fits through the aperture
merit_proj = Projector.uniform(
    position=DEFAULT_LIGHT_POSITION,
    direction=DEFAULT_LIGHT_DIRECTION,
    beam_width=aperture.width * 0.8,
    beam_height=aperture.height * 0.8,
    wavelength=DEFAULT_WAVELENGTH,
    nx=3, ny=3,
)

mse, simulated, target = evaluate_merit(
    system, merit_proj,
    pupil.position, pupil.normal, pupil.width / 2,
    x_fov=DEFAULT_X_FOV, y_fov=DEFAULT_Y_FOV,
    config=merit_config,
)

print(f"MSE: {mse:.6f}")
print(f"Target efficiency: {merit_config.target_efficiency}")
print(f"D65 weights (R/G/B): {D65_WEIGHTS}")
print(f"Simulated intensity (summed): {float(simulated.sum()):.4f}")
print(f"Target intensity (summed): {float(target.sum()):.4f}")

# Per-color breakdown at center pupil, on-axis
center_py, center_px = merit_config.pupil_ny // 2, merit_config.pupil_nx // 2
center_ay, center_ax = merit_config.angle_ny // 2, merit_config.angle_nx // 2
sim_center = simulated[center_py, center_px, center_ay, center_ax, :]
tgt_center = target[center_py, center_px, center_ay, center_ax, :]
print(f"\nCenter pupil, on-axis:")
for ci, color in enumerate(["Red (630nm)", "Green (525nm)", "Blue (460nm)"]):
    print(f"  {color}: simulated={float(sim_center[ci]):.4f}  target={float(tgt_center[ci]):.4f}")

# ── Visualize 3D with angular slider ────────────────────────────────────────

print("\n── Rendering 3D view (Plotly) ──")

# Trace rays for all scan angles for the interactive slider
viz_proj = Projector.uniform(
    position=DEFAULT_LIGHT_POSITION,
    direction=DEFAULT_LIGHT_DIRECTION,
    beam_width=DEFAULT_BEAM_WIDTH,
    beam_height=DEFAULT_BEAM_HEIGHT,
    wavelength=DEFAULT_WAVELENGTH,
    nx=3, ny=3,
)

viz_scan_dirs, viz_scan_angles = scan_directions(
    DEFAULT_LIGHT_DIRECTION,
    x_fov=DEFAULT_X_FOV,
    y_fov=DEFAULT_Y_FOV,
    num_x=DEFAULT_NUM_X_STEPS,
    num_y=DEFAULT_NUM_Y_STEPS,
)

viz_results: list[TraceResult] = []
for iy in range(viz_scan_dirs.shape[0]):
    for ix in range(viz_scan_dirs.shape[1]):
        d = viz_scan_dirs[iy, ix]
        origins, directions, intensities, _ = viz_proj.generate_rays(direction=d)
        for i in range(origins.shape[0]):
            viz_results.append(trace_nonsequential(
                system, origins[i], directions[i], DEFAULT_WAVELENGTH,
                intensity=float(intensities[i]),
            ))

fig = plot_system(system, trace_results=viz_results, scan_angles=viz_scan_angles)
fig.write_html("examples/combiner_demo.html")
print("Saved: examples/combiner_demo.html")

# ── Pupil fill per scan angle ─────────────────────────────────────────────────

print("\n── Pupil fill heatmaps ──")
from apollo14.elements.pupil import Pupil as PupilElement

pupil_elem = [e for e in system.elements if isinstance(e, PupilElement)][0]

fill_proj = Projector.uniform(
    position=DEFAULT_LIGHT_POSITION,
    direction=DEFAULT_LIGHT_DIRECTION,
    beam_width=aperture.width * 0.8,
    beam_height=aperture.height * 0.8,
    wavelength=DEFAULT_WAVELENGTH,
    nx=5, ny=5,
)

fig2 = plot_pupil_fill(
    system, fill_proj, pupil_elem,
    x_fov=DEFAULT_X_FOV,
    y_fov=DEFAULT_Y_FOV,
    num_x_angles=5,
    num_y_angles=5,
    wavelength=DEFAULT_WAVELENGTH,
    pixel_size=0.5,
)
fig2.write_html("examples/pupil_fill.html")
print("Saved: examples/pupil_fill.html")
