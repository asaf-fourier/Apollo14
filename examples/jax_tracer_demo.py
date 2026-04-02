"""
JAX tracer demo — fast, differentiable ray tracing with visualization.

Demonstrates all three JAX tracer entry points (trace_ray, trace_batch,
trace_beam), computes gradients of pupil intensity w.r.t. mirror spacing,
and visualizes the traced rays overlaid on the optical system.
"""

import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apollo14.combiner import CombinerConfig, build_system
from apollo14.jax_tracer import trace_ray, trace_batch, trace_beam, params_from_config
from apollo14.projector import Projector, scan_directions
from apollo14.visualizer import plot_system, plot_jax_pupil_fill
from apollo14.units import mm

# ── Build system ─────────────────────────────────────────────────────────────

config = CombinerConfig.default()
params = params_from_config(config)
n_glass = float(config.chassis.material.n(config.light.wavelength))

# ── 1. Single ray (trace_ray) ────────────────────────────────────────────────

print("── trace_ray: single on-axis ray ──")
pupil_pts, pupil_ints, pupil_valid = trace_ray(
    config.light.position, config.light.direction, n_glass, params,
)

for i in range(config.num_mirrors):
    status = "hit" if pupil_valid[i] else "miss"
    print(f"  mirror_{i}: intensity={float(pupil_ints[i]):.4f}  [{status}]")

total = float(jnp.where(pupil_valid, pupil_ints, 0.0).sum())
print(f"  Total pupil intensity: {total:.4f}")

# ── 2. Batched angular scan (trace_batch) ────────────────────────────────────

print("\n── trace_batch: angular scan across FOV ──")
scan_dirs, scan_angles = scan_directions(
    config.light.direction,
    x_fov=config.light.x_fov,
    y_fov=config.light.y_fov,
    num_x=config.light.num_x_steps,
    num_y=config.light.num_y_steps,
)

# Flatten scan directions into a batch of rays (all from the same origin)
flat_dirs = scan_dirs.reshape(-1, 3)
n_scan = flat_dirs.shape[0]
origins = jnp.tile(config.light.position, (n_scan, 1))

batch_pts, batch_ints, batch_valid = trace_batch(origins, flat_dirs, n_glass, params)
# batch_pts: (N, M, 3), batch_ints: (N, M), batch_valid: (N, M)

valid_total = float(jnp.where(batch_valid, batch_ints, 0.0).sum())
n_hits = int(batch_valid.sum())
print(f"  Scan rays: {n_scan}, Mirror hits reaching pupil: {n_hits}")
print(f"  Total pupil intensity (all angles): {valid_total:.4f}")

# ── 3. Beam trace (trace_beam) ───────────────────────────────────────────────

print("\n── trace_beam: projector beam (shared direction) ──")
projector = Projector.uniform(
    position=config.light.position,
    direction=config.light.direction,
    beam_width=config.light.beam_width,
    beam_height=config.light.beam_height,
    wavelength=config.light.wavelength,
    nx=5, ny=3,
)

beam_origins, beam_dirs, beam_intensities, _ = projector.generate_rays()
beam_pts, beam_ints, beam_valid = trace_beam(
    beam_origins, config.light.direction, n_glass, params,
)
# beam_pts: (N, M, 3), beam_ints: (N, M), beam_valid: (N, M)

beam_total = float(jnp.where(beam_valid, beam_ints, 0.0).sum())
beam_hits = int(beam_valid.sum())
print(f"  Beam rays: {beam_origins.shape[0]}, Pupil hits: {beam_hits}")
print(f"  Total pupil intensity: {beam_total:.4f}")

# ── 4. Gradient demo — d(intensity)/d(mirror_spacing) ───────────────────────

print("\n── Gradient: d(total_pupil_intensity)/d(reflectances) ──")


def total_intensity_from_reflectances(reflectances):
    """Trace a single ray with custom per-mirror reflectances."""
    new_params = params._replace(mirror_reflectances=reflectances)
    pts, ints, valid = trace_ray(
        config.light.position, config.light.direction, n_glass, new_params,
    )
    return jnp.where(valid, ints, 0.0).sum()


grad_fn = jax.grad(total_intensity_from_reflectances)
current_refl = params.mirror_reflectances
grad_val = grad_fn(current_refl)
print(f"  Current reflectances: {[f'{float(r):.4f}' for r in current_refl]}")
print(f"  Gradients (d_total/d_refl_i): {[f'{float(g):.4f}' for g in grad_val]}")
print(f"  (Shows how each mirror's reflectance affects total pupil intensity)")

# ── 5. Visualize JAX-traced rays on the 3D system ───────────────────────────

print("\n── Rendering 3D visualization ──")
system = build_system(config)

# Trace a beam for several scan angles using JAX tracer, then visualize
viz_scan_dirs, viz_scan_angles = scan_directions(
    config.light.direction,
    x_fov=config.light.x_fov,
    y_fov=config.light.y_fov,
    num_x=5, num_y=5,
)

viz_proj = Projector.uniform(
    position=config.light.position,
    direction=config.light.direction,
    beam_width=config.light.beam_width,
    beam_height=config.light.beam_height,
    wavelength=config.light.wavelength,
    nx=3, ny=3,
)

# Build Plotly figure with system elements (no rays yet)
fig = plot_system(system, show=False)
static_count = len(fig.data)

# For each scan angle, trace with JAX and add ray segments
num_y, num_x = viz_scan_dirs.shape[:2]
angle_labels = []

for iy in range(num_y):
    for ix in range(num_x):
        d = viz_scan_dirs[iy, ix]
        ray_origins, _, _, _ = viz_proj.generate_rays(direction=d)

        # Use trace_beam for speed
        pts, ints, valid = trace_beam(ray_origins, d, n_glass, params)
        # pts: (N, M, 3), valid: (N, M)

        # Entry point: approximate as the ray origin (close enough for viz)
        # For each ray, draw: origin → each valid mirror's pupil point
        x_coords, y_coords, z_coords = [], [], []

        # Compute entry points for proper visualization
        from apollo14.jax_tracer import _box_entry
        from apollo14.geometry import snell_refract

        for ri in range(ray_origins.shape[0]):
            o = ray_origins[ri]
            t_entry, entry_normal = _box_entry(o, d, params.chassis_min, params.chassis_max)
            entry_pt = o + t_entry * d
            d_glass, _ = snell_refract(d, entry_normal, 1.0, n_glass)

            # Draw incident ray: origin → entry point
            x_coords.extend([float(o[0]), float(entry_pt[0]), None])
            y_coords.extend([float(o[1]), float(entry_pt[1]), None])
            z_coords.extend([float(o[2]), float(entry_pt[2]), None])

            # For each mirror hit that reaches the pupil, draw reflected ray
            for mi in range(config.num_mirrors):
                if valid[ri, mi]:
                    pupil_pt = pts[ri, mi]
                    # We need the mirror hit point — compute it from the scan
                    m_pos = params.mirror_positions[mi]
                    m_normal = params.mirror_normals[mi]
                    denom = jnp.dot(d_glass, m_normal)
                    t_mirror = jnp.where(
                        jnp.abs(denom) > 1e-12,
                        jnp.dot(m_pos - entry_pt, m_normal) / denom,
                        jnp.inf,
                    )
                    mirror_hit = entry_pt + jnp.maximum(t_mirror, 0.0) * d_glass

                    # Draw: entry → mirror hit → pupil point
                    x_coords.extend([float(mirror_hit[0]), float(pupil_pt[0]), None])
                    y_coords.extend([float(mirror_hit[1]), float(pupil_pt[1]), None])
                    z_coords.extend([float(mirror_hit[2]), float(pupil_pt[2]), None])

        ax_deg = float(viz_scan_angles[iy, ix, 0]) * 180 / np.pi
        ay_deg = float(viz_scan_angles[iy, ix, 1]) * 180 / np.pi
        label = f"({ax_deg:.1f}, {ay_deg:.1f}) deg"
        angle_labels.append(label)

        fig.add_trace(go.Scatter3d(
            x=x_coords, y=y_coords, z=z_coords,
            mode='lines',
            line=dict(color='rgba(0,100,255,0.7)', width=1),
            name=label,
            hoverinfo='name',
            visible=(iy == 0 and ix == 0),
        ))

# Build slider
n_angles = len(angle_labels)
steps = []
for i, label in enumerate(angle_labels):
    vis = [True] * static_count + [False] * n_angles
    vis[static_count + i] = True
    steps.append(dict(args=[{'visible': vis}], label=label, method='restyle'))

fig.update_layout(
    title='JAX Tracer — Ray Visualization',
    sliders=[dict(
        pad=dict(b=10, t=60), len=0.9, x=0.1, y=0,
        steps=steps,
        currentvalue=dict(prefix="Angle: "),
    )],
)

fig.write_html("examples/jax_tracer_demo.html")
print("Saved: examples/jax_tracer_demo.html")

# ── 6. Per-mirror intensity heatmap across FOV ───────────────────────────────

print("\n── Per-mirror pupil intensity across FOV ──")

# Use trace_batch: one ray per scan angle
flat_scan = viz_scan_dirs.reshape(-1, 3)
n_s = flat_scan.shape[0]
scan_origins = jnp.tile(config.light.position, (n_s, 1))

s_pts, s_ints, s_valid = trace_batch(scan_origins, flat_scan, n_glass, params)
# s_ints: (n_angles, M), s_valid: (n_angles, M)

valid_ints = jnp.where(s_valid, s_ints, 0.0)
per_angle_total = valid_ints.sum(axis=1).reshape(num_y, num_x)

# Angle axes in degrees
ax_deg = np.array(viz_scan_angles[:, :, 0]) * 180 / np.pi
ay_deg = np.array(viz_scan_angles[:, :, 1]) * 180 / np.pi

fig2 = make_subplots(
    rows=2, cols=4,
    subplot_titles=[f"Mirror {i}" for i in range(config.num_mirrors)] + ["Total"],
    horizontal_spacing=0.05, vertical_spacing=0.15,
)

vmax = float(valid_ints.max()) if float(valid_ints.max()) > 0 else 1.0

for mi in range(config.num_mirrors):
    row, col = divmod(mi, 4)
    grid = np.array(jnp.where(s_valid[:, mi], s_ints[:, mi], 0.0).reshape(num_y, num_x))
    fig2.add_trace(go.Heatmap(
        z=grid,
        x=np.array(ax_deg[0, :]),
        y=np.array(ay_deg[:, 0]),
        zmin=0, zmax=vmax,
        colorscale='Viridis',
        showscale=(mi == 0),
    ), row=row + 1, col=col + 1)

# Total intensity
fig2.add_trace(go.Heatmap(
    z=np.array(per_angle_total),
    x=np.array(ax_deg[0, :]),
    y=np.array(ay_deg[:, 0]),
    colorscale='Inferno',
    showscale=True,
    colorbar=dict(x=1.02),
), row=2, col=4)

fig2.update_layout(
    title='Per-mirror pupil intensity across FOV (JAX tracer)',
    height=500, width=1100,
)
for i in range(1, 9):
    fig2.update_xaxes(title_text="x angle (deg)", row=(i - 1) // 4 + 1, col=(i - 1) % 4 + 1)
    fig2.update_yaxes(title_text="y angle (deg)", row=(i - 1) // 4 + 1, col=(i - 1) % 4 + 1)

fig2.write_html("examples/jax_tracer_fov.html")
print("Saved: examples/jax_tracer_fov.html")

# ── 7. Pupil fill — 2D grid intensity per scan angle ────────────────────────

print("\n── Pupil fill (JAX tracer) ──")

fill_proj = Projector.uniform(
    position=config.light.position,
    direction=config.light.direction,
    beam_width=config.aperture.width * 0.8,
    beam_height=config.aperture.height * 0.8,
    wavelength=config.light.wavelength,
    nx=7, ny=5,
)

fig3 = plot_jax_pupil_fill(
    fill_proj, params, n_glass,
    x_fov=config.light.x_fov,
    y_fov=config.light.y_fov,
    num_x_angles=7, num_y_angles=7,
    pixel_size=0.5 * mm,
    show=False,
)

fig3.write_html("examples/jax_tracer_pupil.html")
print("Saved: examples/jax_tracer_pupil.html")

print("\nDone.")
