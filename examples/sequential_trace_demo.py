"""
Sequential trace demo — uses trace_mirrors_sequential to analyze
the combiner's primary transmitted path through the mirror stack.

Shows per-mirror reflectance/transmission breakdown across wavelengths
and scan angles.
"""

import numpy as np
import plotly.graph_objects as go

from apollo14.combiner import CombinerConfig, build_system
from apollo14.projector import Projector, scan_directions
from apollo14.interaction import Interaction
from apollo14.tracer import trace_mirrors_sequential, TraceResult
from apollo14.visualizer import plot_system
from apollo14.units import nm


def _sequential_ray_coords(results: list[TraceResult], pupil_center):
    """Build line coords from sequential (flat) trace results.

    Returns two sets of coords:
      - primary path (transmitted/refracted segments connecting mirrors)
      - reflected rays (from each mirror hit to the pupil center)
    """
    px, py, pz = [], [], []  # primary path
    rx, ry, rz = [], [], []  # reflected rays to pupil

    pupil = np.array(pupil_center)

    for tr in results:
        # Primary path: connect consecutive non-reflected hits
        path_pts = [np.array(h.point) for h in tr.hits if h.interaction != Interaction.REFLECTED]
        for i in range(len(path_pts) - 1):
            px.extend([float(path_pts[i][0]), float(path_pts[i + 1][0]), None])
            py.extend([float(path_pts[i][1]), float(path_pts[i + 1][1]), None])
            pz.extend([float(path_pts[i][2]), float(path_pts[i + 1][2]), None])

        # Reflected rays: draw from mirror hit point to pupil center
        for h in tr.hits:
            if h.interaction == Interaction.REFLECTED:
                pt = np.array(h.point)
                rx.extend([float(pt[0]), float(pupil[0]), None])
                ry.extend([float(pt[1]), float(pupil[1]), None])
                rz.extend([float(pt[2]), float(pupil[2]), None])

    return (px, py, pz), (rx, ry, rz)

# ── Build the system ──────────────────────────────────────────────────────────

config = CombinerConfig.default()
system = build_system(config)

print("System elements:")
for elem in system.elements:
    print(f"  {elem.name}")

# ── Single on-axis ray ───────────────────────────────────────────────────────

print("\n── On-axis sequential trace (single ray) ──")
result = trace_mirrors_sequential(
    system,
    origin=config.light.position,
    direction=config.light.direction,
    wavelength=config.light.wavelength,
)

print(f"Total hits: {len(result.hits)}")
for hit in result.hits:
    print(f"  {hit.element_name:20s}  {hit.interaction.value:12s}  "
          f"intensity={hit.intensity:.4f}  point={hit.point}")

# ── Per-mirror reflected vs transmitted intensity ────────────────────────────

print("\n── Per-mirror reflected / transmitted (on-axis) ──")
mirror_reflected = {}
mirror_transmitted = {}
for hit in result.hits:
    name = hit.element_name
    if hit.interaction == "reflected":
        mirror_reflected[name] = hit.intensity
    elif hit.interaction == Interaction.TRANSMITTED:
        mirror_transmitted[name] = hit.intensity

for name in mirror_reflected:
    r = mirror_reflected[name]
    t = mirror_transmitted.get(name, 0.0)
    print(f"  {name:20s}  R={r:.4f}  T={t:.4f}  R/(R+T)={r/(r+t):.2%}")

# ── Multi-wavelength comparison ──────────────────────────────────────────────

print("\n── Wavelength comparison (on-axis) ──")
wavelengths = [460 * nm, 525 * nm, 630 * nm]
labels = ["Blue (460nm)", "Green (525nm)", "Red (630nm)"]

for wl, label in zip(wavelengths, labels):
    res = trace_mirrors_sequential(
        system,
        origin=config.light.position,
        direction=config.light.direction,
        wavelength=wl,
    )
    reflected = [h for h in res.hits if h.interaction == Interaction.REFLECTED]
    total_reflected = sum(float(h.intensity) for h in reflected)
    print(f"  {label:20s}  mirrors hit: {len(reflected)}  "
          f"total reflected intensity: {total_reflected:.4f}")

# ── Multi-ray on-axis ───────────────────────────────────────────────────────

print("\n── Multi-ray on-axis (5x3 grid) ──")
projector = Projector.uniform(
    position=config.light.position,
    direction=config.light.direction,
    beam_width=config.light.beam_width,
    beam_height=config.light.beam_height,
    wavelength=config.light.wavelength,
    nx=5, ny=3,
    intensity=1.0,
)

origins, directions, intensities, pixels = projector.generate_rays()

results: list[TraceResult] = []
for i in range(origins.shape[0]):
    res = trace_mirrors_sequential(
        system, origins[i], directions[i], config.light.wavelength,
        intensity=float(intensities[i]),
    )
    results.append(res)

reflected_total = 0.0
for res in results:
    reflected_total += sum(float(h.intensity)
                          for h in res.hits if h.interaction == Interaction.REFLECTED)

print(f"Rays traced: {origins.shape[0]}")
print(f"Total reflected intensity (all mirrors, all rays): {reflected_total:.4f}")
print(f"Average reflected per ray: {reflected_total / origins.shape[0]:.4f}")

# ── Angular scan ─────────────────────────────────────────────────────────────

print("\n── Angular scan (sequential) ──")
scan_dirs, scan_angles = scan_directions(
    config.light.direction,
    x_fov=config.light.x_fov,
    y_fov=config.light.y_fov,
    num_x=config.light.num_x_steps,
    num_y=config.light.num_y_steps,
)

for iy in range(scan_dirs.shape[0]):
    for ix in range(scan_dirs.shape[1]):
        d = scan_dirs[iy, ix]
        res = trace_mirrors_sequential(
            system,
            origin=config.light.position,
            direction=d,
            wavelength=config.light.wavelength,
        )
        reflected = [h for h in res.hits if h.interaction == Interaction.REFLECTED]
        total_r = sum(float(h.intensity) for h in reflected)
        angle_x = float(scan_angles[iy, ix, 0])
        angle_y = float(scan_angles[iy, ix, 1])
        print(f"  angle=({angle_x:+6.2f}, {angle_y:+6.2f}) deg  "
              f"mirrors hit: {len(reflected)}  total reflected: {total_r:.4f}")

# ── 3D visualization ────────────────────────────────────────────────────────

print("\n── Rendering 3D view (sequential rays) ──")

# Trace rays at each scan angle for the slider
viz_proj = Projector.uniform(
    position=config.light.position,
    direction=config.light.direction,
    beam_width=config.light.beam_width,
    beam_height=config.light.beam_height,
    wavelength=config.light.wavelength,
    nx=3, ny=3,
)

viz_scan_dirs, viz_scan_angles = scan_directions(
    config.light.direction,
    x_fov=config.light.x_fov,
    y_fov=config.light.y_fov,
    num_x=config.light.num_x_steps,
    num_y=config.light.num_y_steps,
)

# Build the base figure (system elements only, no rays)
fig = plot_system(system, show=False)

# Add sequential ray traces per angle as slider groups
num_y, num_x = viz_scan_angles.shape[:2]
dynamic_traces = []

for iy in range(num_y):
    for ix in range(num_x):
        d = viz_scan_dirs[iy, ix]
        origins_v, directions_v, intensities_v, _ = viz_proj.generate_rays(direction=d)

        angle_results = []
        for i in range(origins_v.shape[0]):
            angle_results.append(trace_mirrors_sequential(
                system, origins_v[i], directions_v[i], config.light.wavelength,
                intensity=float(intensities_v[i]),
            ))

        (px, py, pz), (rx, ry, rz) = _sequential_ray_coords(
            angle_results, config.pupil.center,
        )
        ax_deg = float(viz_scan_angles[iy, ix, 0]) * 180 / np.pi
        ay_deg = float(viz_scan_angles[iy, ix, 1]) * 180 / np.pi
        label = f"({ax_deg:.1f}, {ay_deg:.1f}) deg"

        # Primary transmitted path (green)
        dynamic_traces.append(go.Scatter3d(
            x=px, y=py, z=pz,
            mode='lines',
            line=dict(color='rgba(0,200,80,0.7)', width=2),
            name=f"{label} primary",
            hoverinfo='name',
            visible=(iy == 0 and ix == 0),
        ))
        # Reflected rays to pupil (orange)
        dynamic_traces.append(go.Scatter3d(
            x=rx, y=ry, z=rz,
            mode='lines',
            line=dict(color='rgba(255,140,0,0.5)', width=1),
            name=f"{label} reflected",
            hoverinfo='name',
            visible=(iy == 0 and ix == 0),
        ))

fig.add_traces(dynamic_traces)

# Build slider — each angle has 2 traces (primary + reflected)
num_static = len(fig.data) - len(dynamic_traces)
n_angles = len(dynamic_traces) // 2
steps = []
for i in range(n_angles):
    vis = [True] * num_static + [False] * len(dynamic_traces)
    vis[num_static + i * 2] = True      # primary
    vis[num_static + i * 2 + 1] = True  # reflected
    iy, ix = divmod(i, num_x)
    ax_deg = float(viz_scan_angles[iy, ix, 0]) * 180 / np.pi
    ay_deg = float(viz_scan_angles[iy, ix, 1]) * 180 / np.pi
    steps.append(dict(
        args=[{'visible': vis}],
        label=f"({ax_deg:.1f}, {ay_deg:.1f})",
        method='restyle',
    ))

fig.update_layout(
    title='Sequential Trace Visualization',
    sliders=[dict(
        pad=dict(b=10, t=60),
        len=0.9, x=0.1, y=0,
        steps=steps,
        currentvalue=dict(prefix="Angle: "),
    )],
)

fig.write_html("examples/sequential_trace_demo.html")
print("Saved: examples/sequential_trace_demo.html")
