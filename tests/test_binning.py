"""Tests for spatial binning of ray exit hits.

Locks the two binning operators the eyebox merit relies on:
``bin_hits_to_nearest`` (hard, argmin) and ``bin_hits_soft`` (softmax,
differentiable). Covers weight normalization, dead-ray handling, and
gradient flow through ray position (the path spacing gradients take).
"""

import jax
import jax.numpy as jnp

from apollo14.binning import (
    bin_hits_bilinear,
    bin_hits_soft,
    bin_hits_to_nearest,
    make_sample_lattice,
)
from apollo14.trace import TraceResult

GRID = jnp.array([[0.0, 0.0, 0.0],
                  [2.0, 0.0, 0.0],
                  [0.0, 2.0, 0.0]])


def _result(points, intensities, valid):
    """Minimal TraceResult: one exit hit + intensity + validity per ray."""
    points = jnp.asarray(points, dtype=jnp.float32)
    intensities = jnp.asarray(intensities, dtype=jnp.float32)
    valid = jnp.asarray(valid)
    num_rays = points.shape[0]
    return TraceResult(
        hits=jnp.zeros((num_rays, 1, 3)),
        valids=valid[:, None],          # last column = per-ray validity
        final_pos=points,
        final_dir=jnp.zeros((num_rays, 3)),
        final_intensity=intensities,
    )


def test_soft_binning_weights_sum_to_one():
    # Softmax weights sum to 1 per ray, so a single valid ray of intensity 1
    # deposits exactly 1 unit of intensity across the whole grid.
    result = _result([[0.3, 0.1, 0.0]], [1.0], [True])
    binned = bin_hits_soft(result, GRID, sigma=0.5)
    assert jnp.allclose(jnp.sum(binned), 1.0, atol=1e-5)


def test_nearest_binning_assigns_full_intensity_to_nearest_cell():
    result = _result([[0.1, 0.1, 0.0]], [1.0], [True])   # closest to GRID[0]
    binned = bin_hits_to_nearest(result, GRID, stop_grad=True)
    assert jnp.allclose(binned[0], 1.0, atol=1e-6)
    assert jnp.allclose(jnp.sum(binned[1:]), 0.0, atol=1e-6)


def test_dead_rays_contribute_zero():
    result = _result([[0.0, 0.0, 0.0]], [1.0], [False])   # invalid ray
    assert jnp.allclose(jnp.sum(bin_hits_soft(result, GRID, sigma=0.5)), 0.0)
    assert jnp.allclose(jnp.sum(bin_hits_to_nearest(result, GRID)), 0.0)


def test_soft_binning_gradient_flows_through_position():
    # The intensity at a single cell depends on where the ray lands, so the
    # gradient w.r.t. ray position must be finite and non-zero — this is the
    # only path mirror-spacing gradients reach the merit.
    def cell0_intensity(position):
        result = _result(position[None, :], [1.0], [True])
        return bin_hits_soft(result, GRID, sigma=0.5)[0]

    grads = jax.grad(cell0_intensity)(jnp.array([0.5, 0.5, 0.0]))
    assert jnp.all(jnp.isfinite(grads))
    assert jnp.any(grads != 0.0)


# ── Bilinear splat (bin_hits_bilinear) ──────────────────────────────────────
# 4×4 cell-centered lattice on z=0 spanning [-2, 2]²; pitch 1, cell centers at
# ±0.5 and ±1.5. Index = iy*4 + ix.
LATTICE = make_sample_lattice(
    center=jnp.zeros(3), normal=jnp.array([0.0, 0.0, 1.0]),
    half_x=2.0, half_y=2.0, nx=4, ny=4)


def test_bilinear_in_grid_ray_conserves_intensity():
    # Tent weights are a partition of unity, so a ray strictly inside the
    # lattice deposits *exactly* its intensity — no more, no less.
    result = _result([[0.3, -0.7, 0.0]], [1.0], [True])
    binned = bin_hits_bilinear(result, LATTICE)
    assert jnp.allclose(jnp.sum(binned), 1.0, atol=1e-5)


def test_bilinear_ray_at_cell_center_is_a_delta():
    # A ray exactly on a cell center puts all of its intensity in that cell.
    result = _result([[0.5, 0.5, 0.0]], [1.0], [True])   # cell (iy=2, ix=2)
    binned = bin_hits_bilinear(result, LATTICE)
    assert jnp.allclose(binned[2 * 4 + 2], 1.0, atol=1e-5)
    assert jnp.allclose(jnp.sum(binned), 1.0, atol=1e-5)


def test_bilinear_out_of_grid_ray_deposits_nothing():
    # The C4 fix: a ray far outside the lattice (beyond any pupil's reach)
    # fades to ~0 instead of dumping its full intensity onto the edge cell.
    # softmax soft binning, by contrast, still credits the whole unit.
    far = _result([[10.0, 10.0, 0.0]], [1.0], [True])
    assert jnp.allclose(jnp.sum(bin_hits_bilinear(far, LATTICE)), 0.0, atol=1e-6)
    # Contrast: the old softmax path over-credits the same far ray.
    grid_pts = jnp.array([[0.5, 0.5, 0.0], [-0.5, 0.5, 0.0],
                          [0.5, -0.5, 0.0], [-0.5, -0.5, 0.0]])
    assert float(jnp.sum(bin_hits_soft(far, grid_pts, sigma=0.5))) > 0.99


def test_bilinear_edge_ray_fades_over_one_cell():
    # A ray just past the outermost cell center (1.5) deposits a *partial*
    # weight that decays linearly and hits 0 one cell out — not a full dump.
    result = _result([[1.7, 0.5, 0.0]], [1.0], [True])   # 0.2 past center 1.5
    binned = bin_hits_bilinear(result, LATTICE)
    total = float(jnp.sum(binned))
    assert 0.0 < total < 1.0
    assert jnp.allclose(total, 0.8, atol=1e-5)  # tent: 1 - |0.2|


def test_bilinear_dead_ray_contributes_zero():
    result = _result([[0.3, 0.3, 0.0]], [1.0], [False])
    assert jnp.allclose(jnp.sum(bin_hits_bilinear(result, LATTICE)), 0.0)


def test_bilinear_gradient_flows_through_position():
    # Position gradients must survive — this is why bilinear replaces the
    # non-differentiable hard binner for spacing optimization.
    def cell_intensity(position):
        result = _result(position[None, :], [1.0], [True])
        return bin_hits_bilinear(result, LATTICE)[2 * 4 + 1]  # cell (2, 1)

    grads = jax.grad(cell_intensity)(jnp.array([-0.2, 0.3, 0.0]))
    assert jnp.all(jnp.isfinite(grads))
    assert jnp.any(grads != 0.0)
