from collections import defaultdict

import numpy as np
import plotly.graph_objects as go

from apollo14.system import OpticalSystem
from apollo14.elements.partial_mirror import PartialMirror
from apollo14.elements.glass_block import GlassBlock, GlassFace
from apollo14.elements.aperture import RectangularAperture
from apollo14.elements.pupil import Pupil, RectangularPupil
from apollo14.trace import TraceResult
from apollo14.binning import bin_hits_to_grid_np
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
        if isinstance(elem, GlassBlock):
            _add_glass_block(static_traces, elem)
        elif isinstance(elem, PartialMirror):
            _add_mirror(static_traces, elem)
        elif isinstance(elem, RectangularAperture):
            _add_aperture(static_traces, elem)
        elif isinstance(elem, (Pupil, RectangularPupil)):
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


def plot_pupil_fill(beams, projector, pupil_element,
                    x_fov, y_fov, num_x_angles, num_y_angles,
                    color_idx: int = 0, pixel_size: float = 0.5,
                    show: bool = True):
    """Plot pupil intensity heatmaps with a slider to step through scan angles.

    Traces each ``Beam`` in ``beams`` at every scan angle and sums their
    pupil hits into a 2D grid. Use one beam per reflected branch to see
    the combined pupil fill from the whole combiner stack.

    Args:
        beams: iterable of ``Beam`` — each is assumed to terminate on the
            pupil (e.g. via ``absorb(pupil.name)``).
        projector: ``Projector`` — generates origins and sweeps the FOV.
        pupil_element: ``RectangularPupil`` — pupil geometry for binning.
        x_fov, y_fov: scan extents in radians.
        num_x_angles, num_y_angles: scan grid.
        color_idx: which per-color reflectance channel to use.
        pixel_size: bin size on the pupil plane in mm.

    Returns:
        Plotly Figure with one heatmap per scan angle and an angle slider.
    """
    from apollo14.trace import trace_beam
    from apollo14.projector import scan_directions

    beams = list(beams)

    pupil_center = np.asarray(pupil_element.position)
    pupil_normal = np.asarray(pupil_element.normal)
    pupil_lx, pupil_ly = compute_local_axes(pupil_normal)
    pupil_lx, pupil_ly = np.asarray(pupil_lx), np.asarray(pupil_ly)
    pupil_hw = pupil_element.width / 2
    pupil_hh = pupil_element.height / 2

    pupil_r = max(float(pupil_hw), float(pupil_hh))
    n_bins = int(np.ceil(2 * pupil_r / pixel_size))
    bin_edges = np.linspace(-pupil_r, pupil_r, n_bins + 1)
    bin_centers_x = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_centers_y = (bin_edges[:-1] + bin_edges[1:]) / 2

    scan_dirs, scan_angles = scan_directions(
        projector.direction, x_fov, y_fov, num_x_angles, num_y_angles,
    )

    grids = []
    labels = []

    for iy in range(num_y_angles):
        for ix in range(num_x_angles):
            d = scan_dirs[iy, ix]
            ray_origins, _, _, _ = projector.generate_rays(direction=d)

            grid = np.zeros((n_bins, n_bins))
            for beam in beams:
                tr = trace_beam(beam, ray_origins, d, color_idx=color_idx)
                grid += bin_hits_to_grid_np(
                    tr, pupil_center, pupil_lx, pupil_ly,
                    bin_edges, bin_edges,
                )

            grids.append(grid)
            a_x = float(scan_angles[iy, ix, 0]) * 180 / np.pi
            a_y = float(scan_angles[iy, ix, 1]) * 180 / np.pi
            labels.append(f"({a_x:.1f}, {a_y:.1f}) deg")

    vmax = max(g.max() for g in grids) if grids else 1.0
    if vmax == 0:
        vmax = 1.0

    # Pupil boundary rectangle
    rect_x = [-pupil_hw, pupil_hw, pupil_hw, -pupil_hw, -pupil_hw]
    rect_y = [-pupil_hh, -pupil_hh, pupil_hh, pupil_hh, -pupil_hh]

    fig = go.Figure()

    for i, (grid, label) in enumerate(zip(grids, labels)):
        fig.add_trace(go.Heatmap(
            z=grid,
            x=bin_centers_x.tolist(),
            y=bin_centers_y.tolist(),
            zmin=0, zmax=vmax,
            colorscale='Viridis',
            name=label,
            visible=(i == 0),
            colorbar=dict(title="Intensity"),
        ))

    fig.add_trace(go.Scatter(
        x=[float(x) for x in rect_x],
        y=[float(y) for y in rect_y],
        mode='lines',
        line=dict(color='red', dash='dash', width=1),
        name='Pupil boundary',
        visible=True,
    ))

    n_heatmaps = len(grids)
    steps = []
    for i, label in enumerate(labels):
        vis = [False] * n_heatmaps + [True]
        vis[i] = True
        steps.append(dict(args=[{'visible': vis}], label=label, method='restyle'))

    fig.update_layout(
        title='Pupil fill per scan angle',
        xaxis_title='mm',
        yaxis_title='mm',
        yaxis_scaleanchor='x',
        sliders=[dict(
            pad=dict(b=10, t=60), len=0.9, x=0.1, y=0,
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


def _add_pupil(traces, pupil):
    from apollo14.elements.pupil import RectangularPupil

    pos = np.array(pupil.position)
    lx, ly = compute_local_axes(pupil.normal)
    lx, ly = np.array(lx), np.array(ly)

    if isinstance(pupil, RectangularPupil):
        hw, hh = pupil.width / 2, pupil.height / 2
        corners = [
            pos - hw * lx - hh * ly,
            pos + hw * lx - hh * ly,
            pos + hw * lx + hh * ly,
            pos - hw * lx + hh * ly,
        ]
        x_coords = [float(c[0]) for c in corners]
        y_coords = [float(c[1]) for c in corners]
        z_coords = [float(c[2]) for c in corners]

        traces.append(go.Mesh3d(
            x=x_coords, y=y_coords, z=z_coords,
            i=[0, 0], j=[1, 2], k=[2, 3],
            name=pupil.name,
            opacity=0.6,
            color='darkslateblue',
            hoverinfo='name+x+y+z',
        ))
        scale = max(hw, hh) * 0.5
    else:
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
        scale = pupil.radius * 0.5

    # Normal vector
    n = np.array(pupil.normal)
    n = n / np.linalg.norm(n)
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
    """Draw a polyline through each ray's valid hit points.

    Handles both a single-ray TraceResult (``hits`` shape ``(N, 3)``) and
    a beam-batched one (``(R, N, 3)``).
    """
    hits = np.asarray(trace.hits)
    valids = np.asarray(trace.valids)
    if hits.ndim == 2:
        hits = hits[None]
        valids = valids[None]

    for ray_hits, ray_valids in zip(hits, valids):
        pts = ray_hits[ray_valids.astype(bool)]
        for i in range(len(pts) - 1):
            x.extend([float(pts[i, 0]), float(pts[i + 1, 0]), None])
            y.extend([float(pts[i, 1]), float(pts[i + 1, 1]), None])
            z.extend([float(pts[i, 2]), float(pts[i + 1, 2]), None])


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
