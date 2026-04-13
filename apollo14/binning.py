"""Spatial binning utilities for ray exit hits.

JAX-differentiable nearest-neighbor binning and NumPy grid binning
for visualization/diagnostics. Operates on the ``TraceResult`` produced
by the single-path tracer: one exit hit (``final_pos``) per ray, with a
scalar ``final_intensity`` and the last-step validity flag.
"""

import jax
import jax.numpy as jnp
import numpy as np

from apollo14.trace import TraceResult


def _ray_final(trace_result: TraceResult):
    """Return ``(points, intensities, valid)`` per ray, flattening batch dims."""
    pts = trace_result.final_pos.reshape(-1, 3)
    ints = trace_result.final_intensity.reshape(-1)
    # Validity of the whole trace = validity of the last step for that ray.
    valid = trace_result.valids[..., -1].reshape(-1)
    return pts, ints, valid


def bin_hits_to_nearest(trace_result: TraceResult, grid_points, stop_grad=True):
    """Bin per-ray exit hits to the nearest grid point. Pure JAX, differentiable.

    Args:
        trace_result: TraceResult with leading batch dim (R, ...).
        grid_points: (S, 3) target grid positions.
        stop_grad: If True, stop gradients on the argmin (default).

    Returns:
        binned: (S,) total intensity at each grid point.
    """
    S = grid_points.shape[0]
    pts_flat, ints_flat, valid_flat = _ray_final(trace_result)

    delta = pts_flat[:, None, :] - grid_points[None, :, :]  # (R, S, 3)
    dist_sq = jnp.sum(delta ** 2, axis=-1)                  # (R, S)
    nearest = jnp.argmin(dist_sq, axis=-1)                  # (R,)
    if stop_grad:
        nearest = jax.lax.stop_gradient(nearest)

    one_hot = jax.nn.one_hot(nearest, S)                    # (R, S)
    weighted = jnp.where(valid_flat[:, None],
                         ints_flat[:, None] * one_hot, 0.0)
    return jnp.sum(weighted, axis=0)                        # (S,)


def bin_hits_to_grid_np(trace_result: TraceResult, center, local_x, local_y,
                        bin_edges_x, bin_edges_y):
    """Bin per-ray exit hits into a 2D spatial grid. NumPy, for visualization.

    Args:
        trace_result: TraceResult with leading batch dim (R, ...).
        center: (3,) pupil/plane center.
        local_x, local_y: (3,) in-plane axes.
        bin_edges_x, bin_edges_y: bin edges along local_x / local_y.

    Returns:
        grid: (ny, nx) intensity summed into each spatial bin.
    """
    pts_np = np.asarray(trace_result.final_pos).reshape(-1, 3)
    ints_np = np.asarray(trace_result.final_intensity).reshape(-1)
    valid_np = np.asarray(trace_result.valids[..., -1]).reshape(-1)
    center = np.asarray(center)
    local_x = np.asarray(local_x)
    local_y = np.asarray(local_y)

    nx = len(bin_edges_x) - 1
    ny = len(bin_edges_y) - 1
    grid = np.zeros((ny, nx))

    for ri in range(pts_np.shape[0]):
        if not valid_np[ri]:
            continue
        delta = pts_np[ri] - center
        px = float(np.dot(delta, local_x))
        py = float(np.dot(delta, local_y))
        bx = np.searchsorted(bin_edges_x, px) - 1
        by = np.searchsorted(bin_edges_y, py) - 1
        if 0 <= bx < nx and 0 <= by < ny:
            grid[by, bx] += float(ints_np[ri])

    return grid
