"""
JAX tracer demo — explicit path construction with RGB color scanning.

Builds the complete optical path from elements: entry face → mirrors → exit face → pupil.
Scans a projector beam across the FOV in R/G/B at 1-degree steps.
"""

import jax
import jax.numpy as jnp
import numpy as np

from apollo14.combiner import (
    build_default_system,
    DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, DEFAULT_WAVELENGTH,
)
from apollo14.elements.glass_block import GlassBlock
from apollo14.elements.pupil import RectangularPupil
from apollo14.elements.surface import PartialMirror
from apollo14.jax_tracer import (
    partial, refract, target, stack_path, trace_path,
)
from apollo14.projector import Projector, scan_directions
from apollo14.visualizer import plot_system
from apollo14.units import mm, deg

# ── Build system ─────────────────────────────────────────────────────────────

system = build_default_system()
chassis = next(e for e in system.elements if isinstance(e, GlassBlock))
n_glass = float(chassis.material.n(DEFAULT_WAVELENGTH))

# ── Extract elements from the built system ───────────────────────────────────

mirrors = [e for e in system.elements if isinstance(e, PartialMirror)]
pupil = next(e for e in system.elements if isinstance(e, RectangularPupil))
entry_face = next(f for f in chassis.faces if f.name == "back")
exit_face = next(f for f in chassis.faces if f.name == "top")

print(f"System: {len(mirrors)} mirrors, chassis ({chassis.name}), pupil ({pupil.name})")
print(f"Entry face: {entry_face.name} (normal={entry_face.normal})")
print(f"Exit face: {exit_face.name} (normal={exit_face.normal})")

# ── Build complete optical path from elements ────────────────────────────────
#
# Main path: entry face (air→glass) → mirror_0 → mirror_1 → ... → mirror_5
# Branch per step:
#   - entry face (REFRACT): dummy branch (no reflection, 0 intensity)
#   - each mirror (PARTIAL): exit face (glass→air) → pupil (TARGET)

main_list = (
    [refract(entry_face, n1=1.0, n2=n_glass)]
    + [partial(m) for m in mirrors]
)

mirror_branch = [refract(exit_face, n1=n_glass, n2=1.0), target(pupil)]
# The entry step is REFRACT, so its branch produces 0 intensity — reuse same structure
branch_lists = [mirror_branch] * len(main_list)

main_steps, branch_steps = stack_path(main_list, branch_lists)

# Output of trace_path: (M+1,) arrays where index 0 = entry step (always 0 intensity),
# indices 1..M = mirror branches.

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

print(f"\nScan: {num_x}x{num_y} angles ({num_x * num_y} total), "
      f"FOV {x_fov/deg:.0f}x{y_fov/deg:.0f} deg, step {step/deg:.0f} deg")
print(f"Beam: {projector.nx}x{projector.ny} rays, "
      f"{projector.beam_width/mm:.0f}x{projector.beam_height/mm:.0f} mm")
print(f"Colors: R, G, B (3 channels)")

# ── Trace RGB across all angles ──────────────────────────────────────────────

print("\n── Tracing R/G/B beams across FOV ──")

result = np.zeros((num_y, num_x, 3))

for iy in range(num_y):
    for ix in range(num_x):
        direction = scan_dirs[iy, ix]
        beam_origins, _, _, _ = projector.generate_rays(direction=direction)

        def _trace_single(origin, color_idx):
            endpoints, intensities, valid = trace_path(
                origin, direction, jnp.array(1.0),
                main_steps, branch_steps, color_idx)
            # Skip index 0 (entry face REFRACT step — always 0 intensity)
            return jnp.where(valid[1:], intensities[1:], 0.0).sum()

        for ci in range(3):
            trace_color = jax.vmap(lambda o: _trace_single(o, ci))
            total = float(trace_color(beam_origins).sum())
            result[iy, ix, ci] = total

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
    title='RGB FOV scan — per-color pupil intensity (explicit paths)',
    height=400, width=1200,
)
for col in range(1, 5):
    fig.update_xaxes(title_text="x angle (deg)", row=1, col=col)
    fig.update_yaxes(title_text="y angle (deg)", row=1, col=col)

fig.write_html("jax_tracer_demo.html")
print("Saved: jax_tracer_demo.html")

# ── 3D system visualization with traced rays ─────────────────────────────────

print("\n── Rendering 3D system with rays ──")

fig3d = plot_system(system, show=False)
static_count = len(fig3d.data)

angle_labels = []
for iy in range(num_y):
    for ix in range(num_x):
        direction = scan_dirs[iy, ix]
        beam_origins, _, _, _ = projector.generate_rays(direction=direction)

        # Trace all rays — full path including entry refraction
        def _trace_full(origin):
            endpoints, intensities, valid = trace_path(
                origin, direction, jnp.array(1.0),
                main_steps, branch_steps, color_idx=0)
            return endpoints, valid

        all_pts, all_valid = jax.vmap(_trace_full)(beam_origins)
        # all_pts: (N, M+1, 3), all_valid: (N, M+1)
        # index 0 = entry face branch endpoint (dummy), 1..M = mirror branches

        x_coords, y_coords, z_coords = [], [], []

        for ri in range(beam_origins.shape[0]):
            o = beam_origins[ri]

            # Find entry point: intersection of ray with entry face plane
            entry_n = jnp.asarray(entry_face.normal)
            entry_p = jnp.asarray(entry_face.position)
            denom = jnp.dot(direction, entry_n)
            t_entry = jnp.dot(entry_p - o, entry_n) / denom
            ep = o + jnp.maximum(t_entry, 0.0) * direction

            # Incident ray: projector origin → chassis entry
            x_coords.extend([float(o[0]), float(ep[0]), None])
            y_coords.extend([float(o[1]), float(ep[1]), None])
            z_coords.extend([float(o[2]), float(ep[2]), None])

            # For each valid mirror hit, draw entry → mirror → pupil
            for mi in range(len(mirrors)):
                if all_valid[ri, mi + 1]:  # +1 because index 0 is entry step
                    pupil_pt = all_pts[ri, mi + 1]
                    # Compute mirror hit point (ray-plane intersection in glass)
                    m_pos = jnp.asarray(mirrors[mi].position)
                    m_normal = jnp.asarray(mirrors[mi].normal)
                    # Refracted direction inside glass (approximate from entry normal)
                    from apollo14.geometry import snell_refract
                    d_glass, _ = snell_refract(direction, entry_n, 1.0, n_glass)
                    d_m = jnp.dot(d_glass, m_normal)
                    t_m = jnp.dot(m_pos - ep, m_normal) / d_m
                    mhit = ep + jnp.maximum(t_m, 0.0) * d_glass

                    x_coords.extend([float(ep[0]), float(mhit[0]),
                                     float(pupil_pt[0]), None])
                    y_coords.extend([float(ep[1]), float(mhit[1]),
                                     float(pupil_pt[1]), None])
                    z_coords.extend([float(ep[2]), float(mhit[2]),
                                     float(pupil_pt[2]), None])

        ax = float(scan_angles[iy, ix, 0]) * 180 / np.pi
        ay = float(scan_angles[iy, ix, 1]) * 180 / np.pi
        label = f"({ax:.1f}, {ay:.1f}) deg"
        angle_labels.append(label)

        fig3d.add_trace(go.Scatter3d(
            x=x_coords, y=y_coords, z=z_coords,
            mode='lines',
            line=dict(color='rgba(0,100,255,0.7)', width=1),
            name=label,
            hoverinfo='name',
            visible=(iy == 0 and ix == 0),
        ))

# Angle slider
n_angles = len(angle_labels)
steps = []
for i, label in enumerate(angle_labels):
    vis = [True] * static_count + [False] * n_angles
    vis[static_count + i] = True
    steps.append(dict(args=[{'visible': vis}], label=label, method='restyle'))

fig3d.update_layout(
    title='JAX Tracer — 3D Ray Visualization (explicit paths)',
    sliders=[dict(
        pad=dict(b=10, t=60), len=0.9, x=0.1, y=0,
        steps=steps,
        currentvalue=dict(prefix="Angle: "),
    )],
)

fig3d.show()
fig3d.write_html("jax_tracer_3d.html")
print("Saved: jax_tracer_3d.html")
print("\nDone.")
