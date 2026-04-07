"""
JAX tracer demo — RGB color scanning across FOV.

Uses the CombinerPath API: the full optical path (aperture, entry face,
mirrors, exit face, pupil) is specified once. No manual path construction.
"""

import jax.numpy as jnp
import numpy as np

from apollo14.combiner import (
    build_default_system,
    DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, DEFAULT_WAVELENGTH,
)
from apollo14.jax_tracer import (
    combiner_path_from_system, trace_combiner_beam,
)
from apollo14.projector import Projector, scan_directions
from apollo14.visualizer import plot_system
from apollo14.units import mm, deg

# ── Build system and path ────────────────────────────────────────────────────

system = build_default_system()
path = combiner_path_from_system(system, DEFAULT_WAVELENGTH)

print(f"System: {path.mirror_positions.shape[0]} mirrors, "
      f"n_glass={float(path.n_glass):.3f}, aperture={bool(path.has_aperture)}")

# ── Projector setup ──────────────────────────────────────────────────────────

x_fov = 8.0 * deg
y_fov = 8.0 * deg
step = 1.0 * deg
num_x = int(x_fov / step) + 1  # 9 angles
num_y = int(y_fov / step) + 1  # 9 angles

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

# ── Trace RGB across all angles ──────────────────────────────────────────────

print("\n── Tracing R/G/B beams across FOV ──")

result = np.zeros((num_y, num_x, 3))

for iy in range(num_y):
    for ix in range(num_x):
        direction = scan_dirs[iy, ix]
        beam_origins, _, _, _ = projector.generate_rays(direction=direction)

        for ci in range(3):
            _, ints, valid, _, _ = trace_combiner_beam(
                beam_origins, direction, path, color_idx=ci)
            result[iy, ix, ci] = float(jnp.where(valid, ints, 0.0).sum())

# ── Print summary ────────────────────────────────────────────────────────────

print(f"\nResults shape: {result.shape}  (num_y, num_x, 3)")
print(f"Total intensity — R: {result[:,:,0].sum():.4f}, "
      f"G: {result[:,:,1].sum():.4f}, B: {result[:,:,2].sum():.4f}")

center_y, center_x = num_y // 2, num_x // 2
print(f"Center angle intensity — R: {result[center_y, center_x, 0]:.4f}, "
      f"G: {result[center_y, center_x, 1]:.4f}, "
      f"B: {result[center_y, center_x, 2]:.4f}")

# ── Visualize: per-color FOV heatmaps ───────────────────────────────────────

print("\n── Rendering FOV heatmaps ──")

import plotly.graph_objects as go
from plotly.subplots import make_subplots

ax_deg = np.array(scan_angles[:, :, 0]) * 180 / np.pi
ay_deg = np.array(scan_angles[:, :, 1]) * 180 / np.pi

color_names = ['Red', 'Green', 'Blue']
color_scales = ['Reds', 'Greens', 'Blues']

fig = make_subplots(
    rows=1, cols=4,
    subplot_titles=color_names + ['Total (RGB)'],
    horizontal_spacing=0.06,
)

vmax = float(result.max()) if result.max() > 0 else 1.0

for ci in range(3):
    fig.add_trace(go.Heatmap(
        z=result[:, :, ci],
        x=ax_deg[0, :],
        y=ay_deg[:, 0],
        zmin=0, zmax=vmax,
        colorscale=color_scales[ci],
        showscale=(ci == 0),
    ), row=1, col=ci + 1)

fig.add_trace(go.Heatmap(
    z=result.sum(axis=2),
    x=ax_deg[0, :],
    y=ay_deg[:, 0],
    colorscale='Viridis',
    showscale=True,
    colorbar=dict(x=1.02),
), row=1, col=4)

fig.update_layout(
    title='RGB FOV scan — per-color pupil intensity',
    height=400, width=1200,
)
for col in range(1, 5):
    fig.update_xaxes(title_text="x angle (deg)", row=1, col=col)
    fig.update_yaxes(title_text="y angle (deg)", row=1, col=col)

fig.write_html("jax_tracer_demo.html")
print("Saved: jax_tracer_demo.html")

# ── 3D system visualization with traced rays ─────────────────────────────────

print("\n── Rendering 3D system with rays ──")

from apollo14.tracer import jax_to_trace_result

trace_results = []
for iy in range(num_y):
    for ix in range(num_x):
        direction = scan_dirs[iy, ix]
        beam_origins, _, _, _ = projector.generate_rays(direction=direction)

        pts, ints, valid, main_hits, branch_hits = trace_combiner_beam(
            beam_origins, direction, path, color_idx=0)

        for ri in range(beam_origins.shape[0]):
            trace_results.append(jax_to_trace_result(
                beam_origins[ri], direction,
                pts[ri], ints[ri], valid[ri], main_hits[ri], branch_hits[ri],
                system, DEFAULT_WAVELENGTH,
            ))

fig3d = plot_system(system, trace_results=trace_results,
                    scan_angles=scan_angles, show=False)

fig3d.update_layout(title='JAX Tracer — 3D Ray Visualization')
fig3d.show()
fig3d.write_html("jax_tracer_3d.html")
print("Saved: jax_tracer_3d.html")
print("\nDone.")
