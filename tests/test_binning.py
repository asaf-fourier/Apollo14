"""Tests for spatial binning of ray exit hits.

Locks the two binning operators the eyebox merit relies on:
``bin_hits_to_nearest`` (hard, argmin) and ``bin_hits_soft`` (softmax,
differentiable). Covers weight normalization, dead-ray handling, and
gradient flow through ray position (the path spacing gradients take).
"""

import jax
import jax.numpy as jnp

from apollo14.binning import bin_hits_soft, bin_hits_to_nearest
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
