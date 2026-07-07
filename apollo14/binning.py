"""Spatial binning utilities for ray exit hits.

JAX-differentiable nearest-neighbor binning and NumPy grid binning
for visualization/diagnostics. Operates on the ``TraceResult`` produced
by the single-path tracer: one exit hit (``final_pos``) per ray, with a
scalar ``final_intensity`` and the last-step validity flag.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from apollo14.geometry import compute_local_axes
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


def bin_hits_soft(trace_result: TraceResult, grid_points, sigma):
    """Soft Gaussian-kernel binning — fully differentiable w.r.t. ray positions.

    Each ray distributes its intensity to all grid points with weights
    proportional to ``exp(-dist² / (2·sigma²))``, normalized so the
    weights sum to 1 per ray. Gradients flow through both intensity
    and spatial position, enabling optimization of mirror spacings.

    Args:
        trace_result: TraceResult with leading batch dim (R, ...).
        grid_points: (S, 3) target grid positions.
        sigma: Gaussian kernel width. Should be ~0.5× the grid spacing
            for smooth gradients without excessive blurring.

    Returns:
        binned: (S,) total intensity at each grid point.
    """
    pts_flat, ints_flat, valid_flat = _ray_final(trace_result)

    delta = pts_flat[:, None, :] - grid_points[None, :, :]   # (R, S, 3)
    dist_sq = jnp.sum(delta ** 2, axis=-1)                    # (R, S)
    weights = jax.nn.softmax(-dist_sq / (2.0 * sigma ** 2), axis=-1)  # (R, S)
    weighted = jnp.where(valid_flat[:, None],
                         ints_flat[:, None] * weights, 0.0)
    return jnp.sum(weighted, axis=0)                           # (S,)


class SampleLattice(NamedTuple):
    """A regular, cell-centered 2-D sample grid on a plane.

    Unlike a flat ``(S, 3)`` point cloud, this carries the lattice structure
    (frame + pitch) that :func:`bin_hits_bilinear` needs to map a ray's hit
    to fractional cell coordinates. The grid spans ``[-half_x, half_x] ×
    [-half_y, half_y]`` in ``(local_x, local_y)`` as ``nx × ny`` equal cells,
    with cell centers at ``-half + pitch·(i + ½)`` — matching
    :func:`apollo14.geometry.planar_grid_points` with ``cell_centered=True``,
    so the flattened output aligns index-for-index with that grid.
    """
    center: jnp.ndarray    # (3,) grid center (world)
    local_x: jnp.ndarray   # (3,) unit in-plane axis
    local_y: jnp.ndarray   # (3,) unit in-plane axis
    half_x: float          # half-extent along local_x
    half_y: float          # half-extent along local_y
    nx: int
    ny: int


def make_sample_lattice(center, normal, half_x, half_y, nx, ny) -> SampleLattice:
    """Build a :class:`SampleLattice` whose frame matches ``planar_grid_points``.

    Uses the same :func:`compute_local_axes` as ``planar_grid_points`` so a
    ``bin_hits_bilinear`` result lines up cell-for-cell with the flattened
    ``planar_grid_points(center, normal, half_x, half_y, nx, ny,
    cell_centered=True)`` a caller bins against elsewhere.
    """
    local_x, local_y = compute_local_axes(jnp.asarray(normal, dtype=jnp.float32))
    return SampleLattice(
        center=jnp.asarray(center, dtype=jnp.float32),
        local_x=local_x, local_y=local_y,
        half_x=float(half_x), half_y=float(half_y), nx=int(nx), ny=int(ny))


def bin_hits_bilinear(trace_result: TraceResult,
                      lattice: SampleLattice) -> jnp.ndarray:
    """Bilinear (tent-kernel) splat of ray exit hits onto ``lattice``.

    A faithful, differentiable *deposit*: each ray's hit is projected onto the
    lattice plane and its intensity is spread to the (up to) four surrounding
    cell centers by the separable linear-B-spline weights
    ``max(0, 1 − |frac_index − i|)``. Properties that fix ``bin_hits_soft``'s
    softmax:

    * **Distance-limited.** The tent has ±1-cell support, so weight *fades*
      with distance instead of being renormalized back to 1. A ray landing
      well outside the lattice (beyond any evaluated pupil's reach) deposits
      ≈ 0 — off-grid weight simply has no cell to land on. No more dumping a
      far ray's full intensity onto the nearest edge cell.
    * **Energy-conserving for in-grid rays.** The tent weights are a partition
      of unity, so a ray strictly inside the lattice deposits exactly its
      intensity (same total as softmax gave every ray) — only the spurious
      out-of-grid credit is removed.
    * **Differentiable w.r.t. ray position.** Piecewise-linear in the hit
      coordinate, so gradients still flow ``spacing → mirror pos → hit → cell``
      for spacing optimization (the reason soft binning existed).

    Returns ``(ny*nx,)`` in row-major ``(iy, ix)`` order — index ``iy*nx + ix``
    — matching ``planar_grid_points``.
    """
    pts_flat, ints_flat, valid_flat = _ray_final(trace_result)

    delta = pts_flat - lattice.center[None, :]                     # (R, 3)
    coord_x = delta @ lattice.local_x                              # (R,) along x
    coord_y = delta @ lattice.local_y                              # (R,) along y

    pitch_x = 2.0 * lattice.half_x / lattice.nx
    pitch_y = 2.0 * lattice.half_y / lattice.ny
    # Fractional cell-center index: cell i center sits at -half + pitch·(i+½).
    frac_x = (coord_x + lattice.half_x) / pitch_x - 0.5            # (R,)
    frac_y = (coord_y + lattice.half_y) / pitch_y - 0.5

    cells_x = jnp.arange(lattice.nx)                               # (nx,)
    cells_y = jnp.arange(lattice.ny)                               # (ny,)
    weight_x = jnp.clip(1.0 - jnp.abs(frac_x[:, None] - cells_x[None, :]),
                        0.0, 1.0)                                  # (R, nx)
    weight_y = jnp.clip(1.0 - jnp.abs(frac_y[:, None] - cells_y[None, :]),
                        0.0, 1.0)                                  # (R, ny)

    # Per-ray (ny, nx) separable weight, gated by ray validity and intensity.
    weight = weight_y[:, :, None] * weight_x[:, None, :]           # (R, ny, nx)
    contribution = jnp.where(valid_flat[:, None, None],
                             ints_flat[:, None, None] * weight, 0.0)
    return jnp.sum(contribution, axis=0).reshape(-1)               # (ny*nx,)


class PupilGrid(NamedTuple):
    """Precomputed pupil-plane binning grid.

    Built once from a ``RectangularPupil`` and reused across many
    trace results (e.g. one per FOV angle). Holds the in-plane frame,
    bin edges, and the outer half-extents used to draw the pupil boundary.
    """
    center: np.ndarray        # (3,)
    local_x: np.ndarray       # (3,)
    local_y: np.ndarray       # (3,)
    edges_x: np.ndarray       # (nx+1,)
    edges_y: np.ndarray       # (ny+1,)
    half_width: float
    half_height: float

    @property
    def nx(self) -> int:
        return len(self.edges_x) - 1

    @property
    def ny(self) -> int:
        return len(self.edges_y) - 1

    @property
    def centers_x(self) -> np.ndarray:
        return (self.edges_x[:-1] + self.edges_x[1:]) / 2

    @property
    def centers_y(self) -> np.ndarray:
        return (self.edges_y[:-1] + self.edges_y[1:]) / 2


def make_pupil_grid(pupil_element, cell_size: float) -> PupilGrid:
    """Build a ``PupilGrid`` from a ``RectangularPupil`` at ``cell_size`` mm."""
    half_w = float(pupil_element.width) / 2
    half_h = float(pupil_element.height) / 2
    nx = max(1, int(np.ceil(pupil_element.width / cell_size)))
    ny = max(1, int(np.ceil(pupil_element.height / cell_size)))
    lx, ly = compute_local_axes(np.asarray(pupil_element.normal))
    return PupilGrid(
        center=np.asarray(pupil_element.position),
        local_x=np.asarray(lx),
        local_y=np.asarray(ly),
        edges_x=np.linspace(-half_w, half_w, nx + 1),
        edges_y=np.linspace(-half_h, half_h, ny + 1),
        half_width=half_w,
        half_height=half_h,
    )


def bin_hits_to_pupil_grid(trace_result: TraceResult,
                           grid: PupilGrid) -> np.ndarray:
    """Bin per-ray exit hits into a prebuilt ``PupilGrid``. NumPy."""
    return bin_hits_to_grid_np(
        trace_result, grid.center, grid.local_x, grid.local_y,
        grid.edges_x, grid.edges_y,
    )


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
