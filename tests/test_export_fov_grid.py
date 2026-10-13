"""Tests for the clean float64 FOV grid used by external exports."""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from apollo14.export.zemax.fov_grid import (
    build_export_fov_samples,
    rounded_fov_axis_deg,
)
from apollo14.projector import FovGrid, Projector


def test_rounded_axis_values_drive_fractional_steps_without_float_residue():
    assert rounded_fov_axis_deg(math.radians(1.000001), 3) == [-0.5, 0.0, 0.5]
    assert rounded_fov_axis_deg(math.radians(0.500001), 3) == [-0.25, 0.0, 0.25]
    assert rounded_fov_axis_deg(math.radians(2.0), 4) == [-1.0, -0.333, 0.333, 1.0]


def test_export_samples_derive_directions_and_frames_from_clean_angles():
    samples = build_export_fov_samples(
        [0.0, -1.0, 0.0],
        math.radians(1.000001),
        math.radians(0.500001),
        3,
        3,
    )

    assert [(sample.angle_x_deg, sample.angle_y_deg) for sample in samples] == [
        (-0.5, -0.25),
        (0.0, -0.25),
        (0.5, -0.25),
        (-0.5, 0.0),
        (0.0, 0.0),
        (0.5, 0.0),
        (-0.5, 0.25),
        (0.0, 0.25),
        (0.5, 0.25),
    ]
    for sample in samples:
        assert sample.direction.dtype == np.float64
        assert np.isclose(np.linalg.norm(sample.direction), 1.0)
        assert np.isclose(np.linalg.norm(sample.beam_local_x), 1.0)
        assert np.isclose(np.linalg.norm(sample.beam_local_y), 1.0)
        assert np.isclose(np.dot(sample.direction, sample.beam_local_x), 0.0)
        assert np.isclose(np.dot(sample.direction, sample.beam_local_y), 0.0)


def test_float64_export_grid_preserves_the_projector_grid_convention():
    base_direction = np.array([0.0, -0.97, 0.23])
    x_fov = math.radians(10.0)
    y_fov = math.radians(12.0)
    exported = build_export_fov_samples(base_direction, x_fov, y_fov, 3, 3)
    traced = FovGrid(jnp.asarray(base_direction), x_fov, y_fov, 3, 3)
    projector = Projector.uniform(
        position=jnp.zeros(3),
        direction=jnp.asarray(base_direction),
        beam_width=1.0,
        beam_height=1.0,
        nx=1,
        ny=1,
    )

    assert np.allclose(
        np.asarray([sample.direction for sample in exported]),
        np.asarray(traced.flat_directions),
        atol=5e-7,
    )
    for sample in exported:
        jax_x, jax_y = projector._compute_basis(jnp.asarray(sample.direction))
        assert np.allclose(sample.beam_local_x, np.asarray(jax_x), atol=2e-7)
        assert np.allclose(sample.beam_local_y, np.asarray(jax_y), atol=2e-7)


@pytest.mark.parametrize("count", [0, -1, 1.5])
def test_export_grid_rejects_invalid_axis_counts(count):
    with pytest.raises(ValueError, match="positive integer"):
        rounded_fov_axis_deg(math.radians(1.0), count)


def test_export_grid_rejects_samples_that_collapse_after_rounding():
    with pytest.raises(ValueError, match="collapse together"):
        rounded_fov_axis_deg(math.radians(0.0001), 3, decimals=3)
