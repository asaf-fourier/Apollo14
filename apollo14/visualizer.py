from collections import defaultdict

import numpy as np
import plotly.graph_objects as go

from apollo14.system import OpticalSystem
from apollo14.elements.surface import PartialMirror
from apollo14.elements.glass_block import GlassBlock, GlassFace
from apollo14.elements.aperture import RectangularAperture
from apollo14.elements.pupil import Pupil
from apollo14.elements.boundary import BoundaryPlane
from apollo14.tracer import TraceResult
from apollo14.geometry import compute_local_axes


def plot_system(system: OpticalSystem, trace_results: list[TraceResult] = None,
                scan_angles=None, show=True):
    """Render the optical system in 3D with Plotly.

    When *scan_angles* is provided it must be a (num_y, num_x, 2) array of
    (angle_x, angle_y) radian pairs matching the angular scan grid.
    *trace_results* is then expected to contain rays grouped by angle
    (num_y * num_x consecutive groups, each group having the same number of
    rays).  A slider lets you step through angles.

    Without *scan_angles*, all rays are shown at once (no slider).

    Returns the Plotly Figure.
    """
    static_traces = []
    dynamic_traces = []

    # ── static element traces ────────────────────────────────────────────
    for elem in system.elements:
        if isinstance(elem, BoundaryPlane):
            pass  # infinite planes — not rendered, but they catch stray rays
        elif isinstance(elem, GlassBlock):
            _add_glass_block(static_traces, elem)
        elif isinstance(elem, PartialMirror):
            _add_mirror(static_traces, elem)
        elif isinstance(elem, RectangularAperture):
            _add_aperture(static_traces, elem)
        elif isinstance(elem, Pupil):
            _add_pupil(static_traces, elem)

    # ── dynamic ray traces (grouped per angle) ───────────────────────────
    if trace_results and scan_angles is not None:
        num_y, num_x = scan_angles.shape[:2]
        n_angles = num_y * num_x
        rays_per_angle = len(trace_results) // n_angles

        for angle_idx in range(n_angles):
            start = angle_idx * rays_per_angle
            end = start + rays_per_angle
            group = trace_results[start:end]

            x, y, z = [], [], []
            for tr in group:
                _collect_ray_coords(tr, x, y, z)

            iy, ix = divmod(angle_idx, num_x)
            ax_deg = float(scan_angles[iy, ix, 0]) * 180 / np.pi
            ay_deg = float(scan_angles[iy, ix, 1]) * 180 / np.pi
            label = f"({ax_deg:.1f}, {ay_deg:.1f}) deg"

            dynamic_traces.append(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color='rgba(0,100,255,0.7)', width=1),
                name=label,
                hoverinfo='name',
                visible=(angle_idx == 0),
            ))

    elif trace_results:
        # No angular scan — show all rays in a single trace
        x, y, z = [], [], []
        for tr in trace_results:
            _collect_ray_coords(tr, x, y, z)
        if x:
            static_traces.append(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color='rgba(255,0,0,0.6)', width=1),
                name="Rays",
                hoverinfo='name',
            ))

    # ── build figure ─────────────────────────────────────────────────────
    fig = go.Figure()
    fig.add_traces(static_traces)

    sliders = None
    if dynamic_traces:
        fig.add_traces(dynamic_traces)
        sliders = _build_slider(static_traces, dynamic_traces, scan_angles)

    fig.update_layout(
        title='System Visualization',
        scene=dict(
            xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)',
            aspectmode='data',
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        sliders=sliders,
        scene_camera=dict(
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.5, y=-1.5, z=1.0),
        ),
    )

    if show:
        fig.show()
    return fig


def plot_pupil_fill(system: OpticalSystem, projector, pupil_element,
                    x_fov, y_fov, num_x_angles, num_y_angles,
                    wavelength, pixel_size=0.1, show=True):
    """Plot pupil intensity heatmaps with a slider to step through scan angles.

    Returns a Plotly Figure.
    """
    from apollo14.projector import scan_directions
    from apollo14.tracer import trace_nonsequential

    scan_dirs, scan_angles = scan_directions(
        projector.direction, x_fov, y_fov, num_x_angles, num_y_angles,
    )

    pupil_pos = np.array(pupil_element.position)
    pupil_normal = pupil_element.normal
    pupil_r = pupil_element.radius
    lx, ly = compute_local_axes(pupil_normal)
    lx, ly = np.array(lx), np.array(ly)

    n_bins = int(np.ceil(2 * pupil_r / pixel_size))
    bin_edges = np.linspace(-pupil_r, pupil_r, n_bins + 1)

    # Trace all angles and collect grids
    grids = []
    labels = []
    for iy in range(num_y_angles):
        for ix in range(num_x_angles):
            d = scan_dirs[iy, ix]
            origins, directions, intensities, _ = projector.generate_rays(direction=d)

            grid = np.zeros((n_bins, n_bins))
            for ri in range(origins.shape[0]):
                tr = trace_nonsequential(
                    system, origins[ri], directions[ri], wavelength,
                    intensity=float(intensities[ri]),
                )
                if tr.pupil_hit is None:
                    continue
                hit_pt = np.array(tr.pupil_hit.point)
                hit_intensity = float(tr.pupil_hit.intensity)
                delta = hit_pt - pupil_pos
                px = float(np.dot(delta, lx))
                py = float(np.dot(delta, ly))
                bx = np.searchsorted(bin_edges, px) - 1
                by = np.searchsorted(bin_edges, py) - 1
                if 0 <= bx < n_bins and 0 <= by < n_bins:
                    grid[by, bx] += hit_intensity

            grids.append(grid)
            angle_x_deg = float(scan_angles[iy, ix, 0]) * 180 / np.pi
            angle_y_deg = float(scan_angles[iy, ix, 1]) * 180 / np.pi
            labels.append(f"({angle_x_deg:.1f}, {angle_y_deg:.1f}) deg")

    vmax = max(g.max() for g in grids) if grids else 1.0
    if vmax == 0:
        vmax = 1.0

    # Pupil boundary circle
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = pupil_r * np.cos(theta)
    circle_y = pupil_r * np.sin(theta)

    fig = go.Figure()

    for i, (grid, label) in enumerate(zip(grids, labels)):
        fig.add_trace(go.Heatmap(
            z=grid,
            x0=-pupil_r, dx=2 * pupil_r / n_bins,
            y0=-pupil_r, dy=2 * pupil_r / n_bins,
            zmin=0, zmax=vmax,
            colorscale='Viridis',
            name=label,
            visible=(i == 0),
            colorbar=dict(title="Intensity"),
        ))

    # Pupil boundary (always visible on all steps)
    fig.add_trace(go.Scatter(
        x=np.concatenate([circle_x, [circle_x[0]]]).tolist(),
        y=np.concatenate([circle_y, [circle_y[0]]]).tolist(),
        mode='lines',
        line=dict(color='red', dash='dash', width=1),
        name='Pupil boundary',
        visible=True,
    ))

    n_heatmaps = len(grids)
    boundary_idx = n_heatmaps  # index of the circle trace

    steps = []
    for i, label in enumerate(labels):
        vis = [False] * n_heatmaps + [True]  # boundary always on
        vis[i] = True
        steps.append(dict(
            args=[{'visible': vis}],
            label=label,
            method='restyle',
        ))

    fig.update_layout(
        title='Pupil fill per scan angle',
        xaxis_title='mm',
        yaxis_title='mm',
        yaxis_scaleanchor='x',
        sliders=[dict(
            pad=dict(b=10, t=60),
            len=0.9, x=0.1, y=0,
            steps=steps,
            currentvalue=dict(prefix="Angle: "),
        )] if steps else None,
    )

    if show:
        fig.show()
    return fig


# ── element renderers ────────────────────────────────────────────────────────

def _add_glass_block(traces, block: GlassBlock):
    for face in block.faces:
        verts = np.array(face.vertices)
        n = len(verts)
        if n < 3:
            continue
        x = verts[:, 0].tolist()
        y = verts[:, 1].tolist()
        z = verts[:, 2].tolist()
        i_idx = [0] * (n - 2)
        j_idx = list(range(1, n - 1))
        k_idx = list(range(2, n))
        traces.append(go.Mesh3d(
            x=x, y=y, z=z,
            i=i_idx, j=j_idx, k=k_idx,
            name=f"{block.name}",
            opacity=0.08,
            color='cyan',
            hoverinfo='name',
        ))


def _add_mirror(traces, mirror: PartialMirror):
    pos = np.array(mirror.position)
    lx, ly = compute_local_axes(mirror.normal)
    lx, ly = np.array(lx), np.array(ly)
    hw, hh = mirror.width / 2.0, mirror.height / 2.0

    corners = np.array([
        pos - hw * lx - hh * ly,
        pos + hw * lx - hh * ly,
        pos + hw * lx + hh * ly,
        pos - hw * lx + hh * ly,
    ])
    x, y, z = corners[:, 0].tolist(), corners[:, 1].tolist(), corners[:, 2].tolist()
    traces.append(go.Mesh3d(
        x=x, y=y, z=z,
        i=[0, 0], j=[1, 2], k=[2, 3],
        name=mirror.name,
        opacity=0.7,
        color='gold',
        hoverinfo='name+x+y+z',
    ))

    # Normal vector arrow
    n = np.array(mirror.normal)
    n = n / np.linalg.norm(n)
    scale = min(mirror.width, mirror.height) * 0.3
    tip = pos + n * scale
    traces.append(go.Scatter3d(
        x=[pos[0], tip[0]], y=[pos[1], tip[1]], z=[pos[2], tip[2]],
        mode='lines',
        line=dict(color='darkorange', width=3),
        name=f"{mirror.name} normal",
        hoverinfo='name',
    ))


def _add_aperture(traces, aperture: RectangularAperture):
    pos = np.array(aperture.position)
    lx, ly = compute_local_axes(aperture.normal)
    lx, ly = np.array(lx), np.array(ly)
    hw, hh = aperture.width / 2.0, aperture.height / 2.0

    corners = np.array([
        pos - hw * lx - hh * ly,
        pos + hw * lx - hh * ly,
        pos + hw * lx + hh * ly,
        pos - hw * lx + hh * ly,
    ])
    x, y, z = corners[:, 0].tolist(), corners[:, 1].tolist(), corners[:, 2].tolist()
    traces.append(go.Mesh3d(
        x=x, y=y, z=z,
        i=[0, 0], j=[1, 2], k=[2, 3],
        name=aperture.name,
        opacity=0.5,
        color='gray',
        hoverinfo='name+x+y+z',
    ))


def _add_pupil(traces, pupil: Pupil):
    pos = np.array(pupil.position)
    lx, ly = compute_local_axes(pupil.normal)
    lx, ly = np.array(lx), np.array(ly)

    num_steps = 32
    x_coords = [pos[0]]
    y_coords = [pos[1]]
    z_coords = [pos[2]]

    for i in range(num_steps):
        angle = (i / num_steps) * 2 * np.pi
        pt = pos + pupil.radius * (np.cos(angle) * lx + np.sin(angle) * ly)
        x_coords.append(float(pt[0]))
        y_coords.append(float(pt[1]))
        z_coords.append(float(pt[2]))

    i_idx, j_idx, k_idx = [], [], []
    for i in range(num_steps):
        i_idx.append(0)
        j_idx.append(i + 1)
        k_idx.append((i + 1) % num_steps + 1)

    traces.append(go.Mesh3d(
        x=x_coords, y=y_coords, z=z_coords,
        i=i_idx, j=j_idx, k=k_idx,
        name=pupil.name,
        opacity=0.6,
        color='darkslateblue',
        hoverinfo='name+x+y+z',
    ))

    # Normal vector
    n = np.array(pupil.normal)
    n = n / np.linalg.norm(n)
    scale = pupil.radius * 0.5
    tip = pos + n * scale
    traces.append(go.Scatter3d(
        x=[pos[0], tip[0]], y=[pos[1], tip[1]], z=[pos[2], tip[2]],
        mode='lines',
        line=dict(color='cyan', width=3),
        name=f"{pupil.name} normal",
        hoverinfo='name',
    ))


# ── ray path helpers ─────────────────────────────────────────────────────────

def _collect_ray_coords(trace: TraceResult, x, y, z):
    """Walk the TraceHit tree and draw each parent→child segment."""
    for root in trace.hits:
        _walk_hit_tree(root, x, y, z)


def _walk_hit_tree(hit, x, y, z):
    """Recursively draw segments from this hit to each child."""
    parent_pt = np.array(hit.point)
    for child in hit.children:
        child_pt = np.array(child.point)
        x.extend([float(parent_pt[0]), float(child_pt[0]), None])
        y.extend([float(parent_pt[1]), float(child_pt[1]), None])
        z.extend([float(parent_pt[2]), float(child_pt[2]), None])
        _walk_hit_tree(child, x, y, z)


# ── slider builder ───────────────────────────────────────────────────────────

def _build_slider(static_traces, dynamic_traces, scan_angles):
    num_static = len(static_traces)
    num_dynamic = len(dynamic_traces)
    num_y, num_x = scan_angles.shape[:2]

    steps = []
    for i in range(num_dynamic):
        vis = [True] * num_static + [False] * num_dynamic
        vis[num_static + i] = True

        iy, ix = divmod(i, num_x)
        ax_deg = float(scan_angles[iy, ix, 0]) * 180 / np.pi
        ay_deg = float(scan_angles[iy, ix, 1]) * 180 / np.pi

        steps.append(dict(
            args=[{'visible': vis}],
            label=f"({ax_deg:.1f}, {ay_deg:.1f})",
            method='restyle',
        ))

    return [dict(
        pad=dict(b=10, t=60),
        len=0.9, x=0.1, y=0,
        steps=steps,
        currentvalue=dict(prefix="Angle: "),
    )]
