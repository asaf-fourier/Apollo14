"""Functional test: generic and Talos tracers produce identical results.

Exercises the full simulation pipeline — multiple beam positions, all 3 color
channels, multiple FOV angles — and verifies both tracers agree on every
output field to within floating-point tolerance.
"""

import jax.numpy as jnp
import pytest

from apollo14.combiner import (
    build_default_system,
    DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, DEFAULT_WAVELENGTH,
    DEFAULT_BEAM_WIDTH, DEFAULT_BEAM_HEIGHT, DEFAULT_X_FOV, DEFAULT_Y_FOV,
)
from apollo14.route import display_route as generic_display_route
from apollo14.talos_route import display_route as talos_display_route
from apollo14.trace import trace_ray as generic_trace_ray
from apollo14.trace import trace_beam as generic_trace_beam
from apollo14.trace import trace_batch as generic_trace_batch
from apollo14.talos_trace import trace_ray as talos_trace_ray
from apollo14.talos_trace import trace_beam as talos_trace_beam
from apollo14.talos_trace import trace_batch as talos_trace_batch
from apollo14.projector import Projector, scan_directions
from apollo14.units import nm, mm, deg


ATOL = 1e-4  # absolute tolerance for position comparisons
ITOL = 1e-5  # absolute tolerance for intensity comparisons


@pytest.fixture(scope="module")
def system():
    return build_default_system()


@pytest.fixture(scope="module")
def generic_route(system):
    return generic_display_route(system, DEFAULT_WAVELENGTH)


@pytest.fixture(scope="module")
def talos_route(system):
    return talos_display_route(system, DEFAULT_WAVELENGTH)


@pytest.fixture(scope="module")
def projector():
    return Projector.uniform(
        position=DEFAULT_LIGHT_POSITION,
        direction=DEFAULT_LIGHT_DIRECTION,
        beam_width=DEFAULT_BEAM_WIDTH,
        beam_height=DEFAULT_BEAM_HEIGHT,
        wavelength=DEFAULT_WAVELENGTH,
        nx=3, ny=3,
    )


@pytest.fixture(scope="module")
def scan_dirs():
    dirs, _ = scan_directions(
        DEFAULT_LIGHT_DIRECTION, DEFAULT_X_FOV, DEFAULT_Y_FOV, 3, 3)
    return dirs.reshape(-1, 3)  # (9, 3)


# ── Single ray, all colors ───────────────────────────────────────────────────

class TestSingleRayAllColors:
    """On-axis ray traced through both tracers for R, G, B."""

    @pytest.mark.parametrize("color_idx", [0, 1, 2])
    def test_valid_mask(self, generic_route, talos_route, color_idx):
        tr_g = generic_trace_ray(DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
                                  generic_route, color_idx)
        tr_t = talos_trace_ray(DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
                                talos_route, color_idx)
        assert jnp.array_equal(tr_g.valid, tr_t.valid)

    @pytest.mark.parametrize("color_idx", [0, 1, 2])
    def test_intensities(self, generic_route, talos_route, color_idx):
        tr_g = generic_trace_ray(DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
                                  generic_route, color_idx)
        tr_t = talos_trace_ray(DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
                                talos_route, color_idx)
        assert jnp.allclose(tr_g.intensities, tr_t.intensities, atol=ITOL)

    @pytest.mark.parametrize("color_idx", [0, 1, 2])
    def test_pupil_points(self, generic_route, talos_route, color_idx):
        tr_g = generic_trace_ray(DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
                                  generic_route, color_idx)
        tr_t = talos_trace_ray(DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
                                talos_route, color_idx)
        for i in range(tr_g.valid.shape[0]):
            if tr_g.valid[i]:
                assert jnp.allclose(tr_g.pupil_points[i], tr_t.pupil_points[i],
                                    atol=ATOL)

    @pytest.mark.parametrize("color_idx", [0, 1, 2])
    def test_main_hits(self, generic_route, talos_route, color_idx):
        tr_g = generic_trace_ray(DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
                                  generic_route, color_idx)
        tr_t = talos_trace_ray(DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
                                talos_route, color_idx)
        for i in range(tr_g.valid.shape[0]):
            if tr_g.valid[i]:
                assert jnp.allclose(tr_g.main_hits[i], tr_t.main_hits[i],
                                    atol=ATOL)


# ── Beam across FOV angles ──────────────────────────────────────────────────

class TestBeamAcrossFOV:
    """Trace a 3x3 beam at multiple FOV angles and compare both tracers."""

    @pytest.mark.parametrize("angle_idx", range(9))
    def test_beam_intensities_per_angle(self, generic_route, talos_route,
                                         projector, scan_dirs, angle_idx):
        d = scan_dirs[angle_idx]
        origins, _, _, _ = projector.generate_rays(direction=d)

        tr_g = generic_trace_beam(origins, d, generic_route, color_idx=0)
        tr_t = talos_trace_beam(origins, d, talos_route, color_idx=0)

        assert jnp.allclose(tr_g.intensities, tr_t.intensities, atol=ITOL)

    @pytest.mark.parametrize("angle_idx", range(9))
    def test_beam_valid_per_angle(self, generic_route, talos_route,
                                    projector, scan_dirs, angle_idx):
        d = scan_dirs[angle_idx]
        origins, _, _, _ = projector.generate_rays(direction=d)

        tr_g = generic_trace_beam(origins, d, generic_route, color_idx=0)
        tr_t = talos_trace_beam(origins, d, talos_route, color_idx=0)

        assert jnp.array_equal(tr_g.valid, tr_t.valid)

    @pytest.mark.parametrize("angle_idx", range(9))
    def test_beam_total_intensity_per_angle(self, generic_route, talos_route,
                                             projector, scan_dirs, angle_idx):
        d = scan_dirs[angle_idx]
        origins, _, _, _ = projector.generate_rays(direction=d)

        tr_g = generic_trace_beam(origins, d, generic_route, color_idx=0)
        tr_t = talos_trace_beam(origins, d, talos_route, color_idx=0)

        assert jnp.allclose(tr_g.total_intensity, tr_t.total_intensity, atol=ITOL)


# ── Full RGB scan ────────────────────────────────────────────────────────────

class TestFullRGBScan:
    """Trace R/G/B beams across all FOV angles. Compare total collected intensity."""

    def test_rgb_intensity_grid_matches(self, generic_route, talos_route,
                                         projector, scan_dirs):
        """Sum total intensity across all rays, all angles, all colors."""
        for ci in range(3):
            total_g = 0.0
            total_t = 0.0
            for ai in range(scan_dirs.shape[0]):
                d = scan_dirs[ai]
                origins, _, _, _ = projector.generate_rays(direction=d)

                tr_g = generic_trace_beam(origins, d, generic_route, color_idx=ci)
                tr_t = talos_trace_beam(origins, d, talos_route, color_idx=ci)

                total_g += float(tr_g.total_intensity.sum())
                total_t += float(tr_t.total_intensity.sum())

            assert abs(total_g - total_t) < ITOL * 100, \
                f"color {ci}: generic={total_g:.6f}, talos={total_t:.6f}"

    def test_per_mirror_intensity_profile(self, generic_route, talos_route):
        """On-axis beam: per-mirror intensity profile must match exactly."""
        origins = jnp.stack([
            DEFAULT_LIGHT_POSITION + jnp.array([dx, 0, 0])
            for dx in jnp.linspace(-1.0, 1.0, 5) * mm
        ])
        for ci in range(3):
            tr_g = generic_trace_beam(origins, DEFAULT_LIGHT_DIRECTION,
                                       generic_route, color_idx=ci)
            tr_t = talos_trace_beam(origins, DEFAULT_LIGHT_DIRECTION,
                                     talos_route, color_idx=ci)

            # Per-ray, per-mirror intensities
            assert jnp.allclose(tr_g.intensities, tr_t.intensities, atol=ITOL), \
                f"color {ci}: per-mirror intensity mismatch"


# ── trace_batch equivalence ──────────────────────────────────────────────────

class TestBatchEquivalence:
    """trace_batch (per-ray directions) produces same results in both tracers."""

    def test_batch_varied_directions(self, generic_route, talos_route):
        origins = jnp.stack([DEFAULT_LIGHT_POSITION] * 5)
        directions = jnp.array([
            [0.0, -1.0, 0.0],
            [0.05, -1.0, 0.0],
            [-0.05, -1.0, 0.0],
            [0.0, -1.0, 0.05],
            [0.0, -1.0, -0.05],
        ])
        directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)

        tr_g = generic_trace_batch(origins, directions, generic_route, color_idx=0)
        tr_t = talos_trace_batch(origins, directions, talos_route, color_idx=0)

        assert jnp.array_equal(tr_g.valid, tr_t.valid)
        assert jnp.allclose(tr_g.intensities, tr_t.intensities, atol=ITOL)
        assert jnp.allclose(tr_g.total_intensity, tr_t.total_intensity, atol=ITOL)

        for ri in range(5):
            for mi in range(tr_g.valid.shape[1]):
                if tr_g.valid[ri, mi]:
                    assert jnp.allclose(tr_g.pupil_points[ri, mi],
                                        tr_t.pupil_points[ri, mi], atol=ATOL)
                    assert jnp.allclose(tr_g.main_hits[ri, mi],
                                        tr_t.main_hits[ri, mi], atol=ATOL)


# ── Different wavelengths ────────────────────────────────────────────────────

class TestWavelengthEquivalence:
    """Both tracers agree at different wavelengths (different n_glass)."""

    @pytest.mark.parametrize("wavelength", [460.0 * nm, 550.0 * nm, 630.0 * nm])
    def test_wavelength(self, system, wavelength):
        g_route = generic_display_route(system, wavelength)
        t_route = talos_display_route(system, wavelength)

        tr_g = generic_trace_ray(DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
                                  g_route, color_idx=0)
        tr_t = talos_trace_ray(DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
                                t_route, color_idx=0)

        assert jnp.array_equal(tr_g.valid, tr_t.valid)
        assert jnp.allclose(tr_g.intensities, tr_t.intensities, atol=ITOL)
        assert jnp.allclose(tr_g.total_intensity, tr_t.total_intensity, atol=ITOL)
