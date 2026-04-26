
import numpy as np
import plotly.graph_objects as go

from apollo14.binning import bin_hits_to_grid_np
from apollo14.elements.aperture import RectangularAperture
from apollo14.elements.glass_block import GlassBlock
from apollo14.elements.partial_mirror import PartialMirror
from apollo14.elements.pupil import Pupil, RectangularPupil
from apollo14.geometry import compute_local_axes
from apollo14.system import OpticalSystem
from apollo14.trace import TraceResult
from apollo14.units import mm


def plot_system(system: OpticalSystem,
                trace_results: list[TraceResult] | None = None,
                scan_angles=None, projector=None, show=True):
    """Render the optical system in 3D with Plotly.

    When *scan_angles* is provided it must be a (num_y, num_x, 2) array of
    (angle_x, angle_y) radian pairs matching the angular scan grid.
    *trace_results* is then expected to contain rays grouped by angle
    (num_y * num_x consecutive groups, each group having the same number of
    rays).  A slider lets you step through angles.

    Without *scan_angles*, all rays are shown at once (no slider).

    Returns the Plotly Figure.
    """
    static_traces = _static_element_traces(system, projector)
    dynamic_traces: list = []

    if trace_results and scan_angles is not None:
        dynamic_traces = _dynamic_ray_traces(trace_results, scan_angles)
    elif trace_results:
        single = _static_ray_trace(trace_results)
        if single is not None:
            static_traces.append(single)

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


def _static_element_traces(system: OpticalSystem, projector) -> list:
    """Build the per-element Plotly traces (geometry, normals, projector box)."""
    traces: list = []
    for elem in system.elements:
        if isinstance(elem, GlassBlock):
            _add_glass_block(traces, elem)
        elif isinstance(elem, PartialMirror):
            _add_mirror(traces, elem)
        elif isinstance(elem, RectangularAperture):
            _add_aperture(traces, elem)
        elif isinstance(elem, (Pupil, RectangularPupil)):
            _add_pupil(traces, elem)
    if projector is not None:
        _add_projector(traces, projector)
    return traces


def _dynamic_ray_traces(trace_results, scan_angles) -> list:
    """Build one Plotly Scatter3d per scan angle, all but the first hidden."""
    num_y, num_x = scan_angles.shape[:2]
    n_angles = num_y * num_x
    rays_per_angle = len(trace_results) // n_angles

    traces = []
    for angle_idx in range(n_angles):
        start = angle_idx * rays_per_angle
        group = trace_results[start:start + rays_per_angle]
        x, y, z = _gather_ray_coords(group)

        iy, ix = divmod(angle_idx, num_x)
        ax_deg = float(scan_angles[iy, ix, 0]) * 180 / np.pi
        ay_deg = float(scan_angles[iy, ix, 1]) * 180 / np.pi

        traces.append(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color='rgba(0,100,255,0.7)', width=1),
            name=f"({ax_deg:.1f}, {ay_deg:.1f}) deg",
            hoverinfo='name',
            visible=(angle_idx == 0),
        ))
    return traces


def _static_ray_trace(trace_results):
    """Build one combined Plotly Scatter3d for all rays — used without scan."""
    x, y, z = _gather_ray_coords(trace_results)
    if not x:
        return None
    return go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(color='rgba(255,0,0,0.6)', width=1),
        name="Rays",
        hoverinfo='name',
    )


def _gather_ray_coords(trace_results):
    """Collect (x, y, z) polyline coordinates across many ``TraceResult``s."""
    x, y, z = [], [], []
    for tr in trace_results:
        _collect_ray_coords(tr, x, y, z)
    return x, y, z


def plot_pupil_fill(trace_results_per_angle, scan_angles, pupil_element,
                    pixel_size: float = 0.5, show: bool = True):
    """Plot pupil intensity heatmaps with a slider to step through scan angles.

    Pure renderer — takes already-computed trace results and bins their
    final hits onto the pupil plane. The caller is responsible for running
    the tracer.

    Args:
        trace_results_per_angle: nested list ``[angle_idx][route_idx]`` of
            ``TraceResult``. Every ``TraceResult`` is expected to end on
            the pupil (the last interaction is a ``PupilSeg``). Sums over
            the inner route dimension give one heatmap per angle.
        scan_angles: ``(num_y, num_x, 2)`` array of ``(angle_x, angle_y)``
            in radians, matching the order of ``trace_results_per_angle``
            when traversed row-major.
        pupil_element: ``RectangularPupil`` — pupil geometry for binning.
        pixel_size: bin size on the pupil plane in mm.

    Returns:
        Plotly Figure with one heatmap per scan angle and an angle slider.
    """
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

    scan_angles_np = np.asarray(scan_angles)
    num_y_angles, num_x_angles = scan_angles_np.shape[:2]

    grids = []
    labels = []
    idx = 0
    for iy in range(num_y_angles):
        for ix in range(num_x_angles):
            grid = np.zeros((n_bins, n_bins))
            for tr in trace_results_per_angle[idx]:
                grid += bin_hits_to_grid_np(
                    tr, pupil_center, pupil_lx, pupil_ly,
                    bin_edges, bin_edges,
                )
            grids.append(grid)
            a_x = float(scan_angles_np[iy, ix, 0]) * 180 / np.pi
            a_y = float(scan_angles_np[iy, ix, 1]) * 180 / np.pi
            labels.append(f"({a_x:.1f}, {a_y:.1f}) deg")
            idx += 1

    vmax = max(g.max() for g in grids) if grids else 1.0
    if vmax == 0:
        vmax = 1.0

    # Pupil boundary rectangle
    rect_x = [-pupil_hw, pupil_hw, pupil_hw, -pupil_hw, -pupil_hw]
    rect_y = [-pupil_hh, -pupil_hh, pupil_hh, pupil_hh, -pupil_hh]

    fig = go.Figure()

    for i, (grid, label) in enumerate(zip(grids, labels, strict=True)):
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
    ihw, ihh = aperture.inner_width / 2.0, aperture.inner_height / 2.0

    # 8 vertices — 4 outer corners then 4 inner corners, both CCW starting BL.
    verts = np.array([
        pos - hw * lx - hh * ly,   # 0: outer BL
        pos + hw * lx - hh * ly,   # 1: outer BR
        pos + hw * lx + hh * ly,   # 2: outer TR
        pos - hw * lx + hh * ly,   # 3: outer TL
        pos - ihw * lx - ihh * ly, # 4: inner BL
        pos + ihw * lx - ihh * ly, # 5: inner BR
        pos + ihw * lx + ihh * ly, # 6: inner TR
        pos - ihw * lx + ihh * ly, # 7: inner TL
    ])

    # Frame = 4 quad strips (bottom, right, top, left), each 2 triangles.
    i = [0, 0, 1, 1, 2, 2, 3, 3]
    j = [1, 5, 2, 6, 3, 7, 0, 4]
    k = [5, 4, 6, 5, 7, 6, 4, 7]

    x, y, z = verts[:, 0].tolist(), verts[:, 1].tolist(), verts[:, 2].tolist()
    traces.append(go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
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


def _add_projector(traces, projector):
    """Render the projector as a thin rectangular box.

    Box extents: ``(beam_width + 1 mm) × (beam_height + 1 mm)`` in the beam
    cross-section plane, ``1 mm`` thick along the emission direction.
    Centered on ``projector.position`` with its front face pointing in
    ``projector.direction``.
    """
    pos = np.array(projector.position)
    d = np.array(projector.direction)
    d = d / np.linalg.norm(d)
    lx, ly = projector._compute_basis(projector.direction)
    lx, ly = np.array(lx), np.array(ly)

    hw = (projector.beam_width + 1.0 * mm) / 2.0
    hh = (projector.beam_height + 1.0 * mm) / 2.0
    ht = (.5 * mm) / 2.0  # half-thickness along emission axis

    # 8 box corners. Back face (−d side) is where the emitter sits; front
    # face (+d side) is where rays exit.
    signs = [
        (-1, -1, -1), ( 1, -1, -1), ( 1,  1, -1), (-1,  1, -1),
        (-1, -1,  1), ( 1, -1,  1), ( 1,  1,  1), (-1,  1,  1),
    ]
    verts = np.array([
        pos + sx * hw * lx + sy * hh * ly + sz * ht * d
        for sx, sy, sz in signs
    ])

    # 12 triangles — 2 per face, 6 faces.
    i = [0, 0, 1, 1, 2, 2, 4, 4, 0, 0, 3, 3]
    j = [1, 2, 5, 6, 3, 7, 5, 6, 3, 4, 2, 6]
    k = [2, 3, 6, 2, 7, 6, 6, 7, 4, 7, 6, 7]

    x, y, z = verts[:, 0].tolist(), verts[:, 1].tolist(), verts[:, 2].tolist()
    traces.append(go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        name="projector",
        opacity=0.7,
        color='goldenrod',
        hoverinfo='name+x+y+z',
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

    for ray_hits, ray_valids in zip(hits, valids, strict=True):
        pts = ray_hits[ray_valids.astype(bool)]
        for i in range(len(pts) - 1):
            x.extend([float(pts[i, 0]), float(pts[i + 1, 0]), None])
            y.extend([float(pts[i, 1]), float(pts[i + 1, 1]), None])
            z.extend([float(pts[i, 2]), float(pts[i + 1, 2]), None])


# ── slider builder ───────────────────────────────────────────────────────────

def _build_slider(static_traces, dynamic_traces, scan_angles):
    num_static = len(static_traces)
    num_dynamic = len(dynamic_traces)
    num_x = scan_angles.shape[1]

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
