"""
Combiner demo — equivalent to Apollo13's main_talos.py

Builds the default combiner system (6 cascaded partial mirrors in a glass chassis),
traces rays from a projector through it, and visualizes the result.
"""

import jax.numpy as jnp

from apollo14.combiner import CombinerConfig, build_system
from apollo14.projector import Projector, scan_directions
from apollo14.tracer import trace_sequential, trace_mirrors_sequential, TraceResult
from apollo14.visualizer import plot_system
from apollo14.units import mm, nm, deg

# ── Build the system ──────────────────────────────────────────────────────────

config = CombinerConfig.default()
system = build_system(config)

print("System elements:")
for elem in system.elements:
    print(f"  {elem.name}")

# ── Create a projector ────────────────────────────────────────────────────────

projector = Projector.uniform(
    position=config.light.position,
    direction=config.light.direction,
    beam_width=config.light.beam_width,
    beam_height=config.light.beam_height,
    wavelength=config.light.wavelength,
    nx=5, ny=3,
    intensity=1.0,
)

# ── Trace on-axis (single scan angle) ────────────────────────────────────────

print("\n── On-axis trace ──")
origins, directions, intensities, pixels = projector.generate_rays()

results: list[TraceResult] = []
for i in range(origins.shape[0]):
    result = trace_sequential(
        system, origins[i], directions[i], config.light.wavelength,
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
    config.light.direction,
    x_fov=config.light.x_fov,
    y_fov=config.light.y_fov,
    num_x=config.light.num_x_steps,
    num_y=config.light.num_y_steps,
)

all_results: list[TraceResult] = []
for iy in range(scan_dirs.shape[0]):
    for ix in range(scan_dirs.shape[1]):
        d = scan_dirs[iy, ix]
        origins, directions, intensities, pixels = projector.generate_rays(direction=d)

        for i in range(origins.shape[0]):
            result = trace_sequential(
                system, origins[i], directions[i], config.light.wavelength,
                intensity=float(intensities[i]),
            )
            all_results.append(result)

pupil_hits = [r.pupil_hit for r in all_results if r.pupil_hit is not None]
total_rays = len(all_results)
print(f"Total rays: {total_rays}, Pupil hits: {len(pupil_hits)}")
if pupil_hits:
    avg_intensity = sum(float(h.intensity) for h in pupil_hits) / len(pupil_hits)
    print(f"Average pupil hit intensity: {avg_intensity:.4f}")

# ── Per-mirror reflectance summary ───────────────────────────────────────────

print("\n── Per-mirror reflectance (on-axis, single ray) ──")
result = trace_mirrors_sequential(
    system,
    origin=config.light.position,
    direction=config.light.direction,
    wavelength=config.light.wavelength,
)

for hit in result.hits:
    if hit.element_name.startswith("mirror_"):
        print(f"  {hit.element_name}: {hit.interaction:12s}  intensity={hit.intensity:.4f}")

# ── Visualize ─────────────────────────────────────────────────────────────────

print("\n── Rendering 3D view ──")

# Trace a sparse set of rays for visualization
viz_results = []
viz_proj = Projector.uniform(
    position=config.light.position,
    direction=config.light.direction,
    beam_width=config.light.beam_width,
    beam_height=config.light.beam_height,
    wavelength=config.light.wavelength,
    nx=3, ny=3,
)
origins, directions, intensities, _ = viz_proj.generate_rays()
for i in range(origins.shape[0]):
    viz_results.append(trace_sequential(
        system, origins[i], directions[i], config.light.wavelength,
        intensity=float(intensities[i]),
    ))

fig, ax = plot_system(system, trace_results=viz_results)
ax.set_title("Combiner — 6 cascaded partial mirrors")

import matplotlib.pyplot as plt
fig.savefig("examples/combiner_demo.png", dpi=150, bbox_inches='tight')
print("Saved: examples/combiner_demo.png")
plt.show()
