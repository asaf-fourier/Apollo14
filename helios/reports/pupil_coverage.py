"""Pupil coverage report — 2D heatmap of average intensity across FOV.

Traces a beam at each FOV angle, bins pupil hits into a spatial grid,
and shows the average intensity per grid cell across all angles.
"""

import numpy as np
import plotly.graph_objects as go

from apollo14.system import OpticalSystem
from apollo14.elements.glass_block import GlassBlock
from apollo14.elements.pupil import RectangularPupil
from apollo14.geometry import compute_local_axes, normalize
from apollo14.jax_tracer import trace_combiner_beam, combiner_path_from_system
from apollo14.projector import Projector, scan_directions


def pupil_coverage_report(
    system: OpticalSystem,
    projector: Projector,
    wavelength: float,
    x_fov: float,
    y_fov: float,
    num_x_angles: int,
    num_y_angles: int,
    cell_size: float = 2.0,
    show: bool = True,
) -> go.Figure:
    """Generate a pupil coverage heatmap averaged over the full FOV.

    For each scan angle, traces a beam of rays through the system and
    bins pupil hits into a spatial grid. Each grid cell shows the
    average intensity across all FOV angles.

    Args:
        system: OpticalSystem with chassis, mirrors, and pupil.
        projector: Projector for beam generation.
        wavelength: Trace wavelength (nm).
        x_fov: Horizontal half-FOV (radians).
        y_fov: Vertical half-FOV (radians).
        num_x_angles: Number of horizontal FOV steps.
        num_y_angles: Number of vertical FOV steps.
        cell_size: Grid cell size in mm. Default 2.0.
        show: Whether to call fig.show(). Default True.

    Returns:
        Plotly Figure with the heatmap.
    """
    import jax.numpy as jnp

    path = combiner_path_from_system(system, wavelength)

    pupil_elem = next(
        e for e in system.elements if isinstance(e, RectangularPupil)
    )
    pupil_center = np.asarray(pupil_elem.position)
    pupil_normal = normalize(np.asarray(pupil_elem.normal))
    pupil_lx, pupil_ly = compute_local_axes(pupil_normal)
    pupil_lx, pupil_ly = np.asarray(pupil_lx), np.asarray(pupil_ly)
    pupil_hw = pupil_elem.width / 2
    pupil_hh = pupil_elem.height / 2

    # Grid setup
    nx_cells = max(1, int(np.ceil(pupil_elem.width / cell_size)))
    ny_cells = max(1, int(np.ceil(pupil_elem.height / cell_size)))
    x_edges = np.linspace(-pupil_hw, pupil_hw, nx_cells + 1)
    y_edges = np.linspace(-pupil_hh, pupil_hh, ny_cells + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    scan_dirs, scan_angles = scan_directions(
        projector.direction, x_fov, y_fov, num_x_angles, num_y_angles,
    )

    n_angles = num_x_angles * num_y_angles
    grid_sum = np.zeros((ny_cells, nx_cells))
    grid_count = np.zeros((ny_cells, nx_cells))

    for iy in range(num_y_angles):
        for ix in range(num_x_angles):
            d = scan_dirs[iy, ix]
            ray_origins, _, _, _ = projector.generate_rays(direction=d)

            # Trace all 3 colors and average (approximate white)
            angle_grid = np.zeros((ny_cells, nx_cells))
            for ci in range(3):
                pts, ints, valid, _, _ = trace_combiner_beam(
                    ray_origins, d, path, color_idx=ci)

                pts_np = np.asarray(pts)
                ints_np = np.asarray(ints)
                valid_np = np.asarray(valid)

                for ri in range(pts_np.shape[0]):
                    for mi in range(pts_np.shape[1]):
                        if not valid_np[ri, mi]:
                            continue
                        delta = pts_np[ri, mi] - pupil_center
                        px = float(np.dot(delta, pupil_lx))
                        py = float(np.dot(delta, pupil_ly))
                        bx = np.searchsorted(x_edges, px) - 1
                        by = np.searchsorted(y_edges, py) - 1
                        if 0 <= bx < nx_cells and 0 <= by < ny_cells:
                            angle_grid[by, bx] += float(ints_np[ri, mi])

            angle_grid /= 3.0  # average over R/G/B
            grid_sum += angle_grid
            grid_count += (angle_grid > 0).astype(float)

    # Average across angles that contributed to each cell
    avg_grid = np.where(grid_count > 0, grid_sum / n_angles, 0.0)

    # Pupil boundary rectangle
    rect_x = [-pupil_hw, pupil_hw, pupil_hw, -pupil_hw, -pupil_hw]
    rect_y = [-pupil_hh, -pupil_hh, pupil_hh, pupil_hh, -pupil_hh]

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=avg_grid,
        x=x_centers.tolist(),
        y=y_centers.tolist(),
        colorscale='Viridis',
        colorbar=dict(title="Avg intensity"),
    ))

    fig.add_trace(go.Scatter(
        x=rect_x, y=rect_y,
        mode='lines',
        line=dict(color='red', dash='dash', width=1.5),
        name='Pupil boundary',
    ))

    # Annotate cells with values
    for iy_c in range(ny_cells):
        for ix_c in range(nx_cells):
            val = avg_grid[iy_c, ix_c]
            if val > 0:
                fig.add_annotation(
                    x=float(x_centers[ix_c]),
                    y=float(y_centers[iy_c]),
                    text=f"{val:.4f}",
                    showarrow=False,
                    font=dict(size=9, color='white'),
                )

    total_cells = nx_cells * ny_cells
    filled_cells = int(np.sum(grid_count > 0))
    coverage_pct = filled_cells / total_cells * 100 if total_cells > 0 else 0
    avg_intensity = float(avg_grid[avg_grid > 0].mean()) if np.any(avg_grid > 0) else 0
    uniformity = float(avg_grid[avg_grid > 0].std() / avg_intensity) if avg_intensity > 0 else 0

    fig.update_layout(
        title=(f'Pupil Coverage — {cell_size:.0f}x{cell_size:.0f} mm grid, '
               f'{num_x_angles}x{num_y_angles} FOV angles<br>'
               f'<sub>Coverage: {coverage_pct:.0f}% | '
               f'Avg intensity: {avg_intensity:.4f} | '
               f'Non-uniformity (CV): {uniformity:.1%}</sub>'),
        xaxis_title='x (mm)',
        yaxis_title='y (mm)',
        yaxis_scaleanchor='x',
        width=600,
        height=550,
    )

    if show:
        fig.show()
    return fig
