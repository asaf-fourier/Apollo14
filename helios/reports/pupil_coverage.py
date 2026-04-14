"""Pupil coverage report — 2D heatmap of average intensity across FOV.

Traces a beam at each FOV angle, bins pupil hits into a spatial grid,
and shows the average intensity per grid cell across all angles.
"""

import numpy as np
import plotly.graph_objects as go

from apollo14.system import OpticalSystem
from apollo14.elements.pupil import RectangularPupil
from apollo14.trace import trace_rays
from apollo14.binning import make_pupil_grid, bin_hits_to_pupil_grid
from apollo14.projector import Projector, scan_directions

from helios.merit import build_combiner_pupil_routes


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
    routes = build_combiner_pupil_routes(system, [wavelength])[0]

    pupil_elem = next(
        e for e in system.elements if isinstance(e, RectangularPupil)
    )
    grid = make_pupil_grid(pupil_elem, cell_size)

    scan_dirs, _ = scan_directions(
        projector.direction, x_fov, y_fov, num_x_angles, num_y_angles,
    )

    n_angles = num_x_angles * num_y_angles
    grid_sum = np.zeros((grid.ny, grid.nx))
    grid_count = np.zeros((grid.ny, grid.nx))

    for iy in range(num_y_angles):
        for ix in range(num_x_angles):
            d = scan_dirs[iy, ix]
            ray_origins, _, _, _ = projector.generate_rays(direction=d)

            # Single wavelength — sum over branches.
            angle_grid = np.zeros((grid.ny, grid.nx))
            for route in routes:
                tr = trace_rays(route, ray_origins, d, color_idx=1)
                angle_grid += bin_hits_to_pupil_grid(tr, grid)
            grid_sum += angle_grid
            grid_count += (angle_grid > 0).astype(float)

    # Average across angles that contributed to each cell
    avg_grid = np.where(grid_count > 0, grid_sum / n_angles, 0.0)

    # Pupil boundary rectangle
    hw, hh = grid.half_width, grid.half_height
    rect_x = [-hw, hw, hw, -hw, -hw]
    rect_y = [-hh, -hh, hh, hh, -hh]
    x_centers = grid.centers_x
    y_centers = grid.centers_y
    nx_cells, ny_cells = grid.nx, grid.ny

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
