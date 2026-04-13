"""Spatial binning utilities for pupil hit points.

JAX-differentiable nearest-neighbor binning and NumPy grid binning
for visualization/diagnostics.
"""

import jax
import jax.numpy as jnp
import numpy as np

from apollo14.trace import TraceResult


def bin_hits_to_nearest(trace_result: TraceResult, grid_points, stop_grad=True):
    """Bin traced pupil hits to the nearest grid point. Pure JAX, differentiable.

    Flattens (rays x mirrors) hits and assigns each valid hit to its
    nearest grid point via one-hot binning. Gradients flow through
    intensities; spatial assignment is fixed.

    Args:
        trace_result: TraceResult with (R, M, ...) arrays.
        grid_points: (S, 3) target grid positions.
        stop_grad: If True, stop gradients on the argmin (default).
            Use True for optimization (prevents discrete assignment
            from producing zero gradients). Use False when the
            binning result itself needs no gradient.

    Returns:
        binned: (S,) total intensity at each grid point.
    """
    S = grid_points.shape[0]
    pts_flat = trace_result.pupil_points.reshape(-1, 3)
    ints_flat = trace_result.intensities.reshape(-1)
    valid_flat = trace_result.valid.reshape(-1)

    delta = pts_flat[:, None, :] - grid_points[None, :, :]  # (N, S, 3)
    dist_sq = jnp.sum(delta ** 2, axis=-1)                  # (N, S)
    nearest = jnp.argmin(dist_sq, axis=-1)                   # (N,)
    if stop_grad:
        nearest = jax.lax.stop_gradient(nearest)

    one_hot = jax.nn.one_hot(nearest, S)                     # (N, S)
    weighted = jnp.where(valid_flat[:, None], ints_flat[:, None] * one_hot, 0.0)
    return jnp.sum(weighted, axis=0)                          # (S,)


def bin_hits_to_grid_np(trace_result: TraceResult, center, local_x, local_y,
                        bin_edges_x, bin_edges_y):
    """Bin traced pupil hits into a 2D spatial grid. NumPy, for visualization.

    Args:
        trace_result: TraceResult with (R, M, ...) arrays.
        center: (3,) pupil center (numpy array).
        local_x: (3,) first local axis on pupil plane.
        local_y: (3,) second local axis on pupil plane.
        bin_edges_x: (nx+1,) bin edge positions along local_x.
        bin_edges_y: (ny+1,) bin edge positions along local_y.

    Returns:
        grid: (ny, nx) intensity summed into each spatial bin.
    """
    pts_np = np.asarray(trace_result.pupil_points)
    ints_np = np.asarray(trace_result.intensities)
    valid_np = np.asarray(trace_result.valid)
    center = np.asarray(center)
    local_x = np.asarray(local_x)
    local_y = np.asarray(local_y)

    nx = len(bin_edges_x) - 1
    ny = len(bin_edges_y) - 1
    grid = np.zeros((ny, nx))

    for ri in range(pts_np.shape[0]):
        for mi in range(pts_np.shape[1]):
            if not valid_np[ri, mi]:
                continue
            delta = pts_np[ri, mi] - center
            px = float(np.dot(delta, local_x))
            py = float(np.dot(delta, local_y))
            bx = np.searchsorted(bin_edges_x, px) - 1
            by = np.searchsorted(bin_edges_y, py) - 1
            if 0 <= bx < nx and 0 <= by < ny:
                grid[by, bx] += float(ints_np[ri, mi])

    return grid
