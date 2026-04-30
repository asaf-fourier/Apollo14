"""Smoke + behavioral tests for the pupil report figures.

Each chart module gets two kinds of test:

1. Smoke: build a small synthetic ``response`` and verify the figure
   is produced without errors and has the expected trace shapes.
2. Behavioral: assert the figure responds correctly to known inputs
   (e.g., a uniform D65 response gives ΔD65 ≈ 0 everywhere; a
   half-bright cell shows up as half on the brightness map).
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
import pytest

from helios.merit import D65_WEIGHTS
from helios.reports.composer import (
    coefficient_of_variation_over_angles,
    d65_distance_per_cell_per_angle,
    luminance_per_cell_per_angle,
    luminance_weights_for_response,
    mean_over_angles,
    radiance_per_cell_per_angle,
)
from helios.reports.figures.eyebox_cdf import eyebox_quality_cdf_figure
from helios.reports.figures.global_fov import fov_global_figures
from helios.reports.figures.mirrors import mirror_reflectance_figure
from helios.reports.figures.overview import (
    pupil_brightness_figure,
    pupil_d65_distance_figure,
)
from helios.reports.figures.per_cell import per_cell_d65_fov_figure
from helios.reports.figures.projector import (
    mirror_input_spectrum_figure,
    projector_spectrum_figure,
)
from helios.reports.figures.visible_color import per_cell_visible_color_figure
from helios.reports.headline import (
    compute_headline_numbers,
    headline_numbers_html,
)
from helios.reports.pupil_report import render_pupil_report

# ── Fixtures ────────────────────────────────────────────────────────────────


def _synthetic_response(ny=3, nx=3, n_fov_y=2, n_fov_x=2,
                        wavelengths_nm=None, brightness=1.0,
                        d65_balanced=True):
    """Build an ``(S, A, K)`` response with controllable shape & color."""
    if wavelengths_nm is None:
        wavelengths_nm = np.array([446.0, 545.0, 627.0])
    K = len(wavelengths_nm)
    S = ny * nx
    A = n_fov_y * n_fov_x

    if d65_balanced:
        d65 = np.asarray(D65_WEIGHTS) if K == 3 else np.ones(K) / K
        per_channel = brightness * d65
    else:
        per_channel = np.zeros(K)
        per_channel[-1] = brightness   # all in last channel — strongly tinted

    return np.broadcast_to(per_channel, (S, A, K)).copy()


def _scan_angles(n_fov_y=2, n_fov_x=2, half_fov_deg=3.5):
    radians = np.deg2rad(half_fov_deg)
    ax = np.linspace(-radians, radians, n_fov_x)
    ay = np.linspace(-radians, radians, n_fov_y)
    grid = np.zeros((n_fov_y, n_fov_x, 2))
    for iy in range(n_fov_y):
        for ix in range(n_fov_x):
            grid[iy, ix] = [ax[ix], ay[iy]]
    return grid


def _pupil_axes(ny=3, nx=3, half_extent_mm=4.0):
    return (np.linspace(-half_extent_mm, half_extent_mm, nx),
            np.linspace(-half_extent_mm, half_extent_mm, ny))


# ── Composer ────────────────────────────────────────────────────────────────


class TestComposer:

    def test_luminance_weights_returns_none_for_short_axes(self):
        assert luminance_weights_for_response(None) is None
        assert luminance_weights_for_response(np.array([555.0])) is None

    def test_luminance_weights_nonzero_for_visible_band(self):
        weights = luminance_weights_for_response(
            np.array([446.0, 545.0, 627.0]))
        # All three should be positive; green should dominate.
        assert np.all(weights > 0)
        assert weights[1] > weights[0]
        assert weights[1] > weights[2]

    def test_luminance_per_cell_shape(self):
        response = _synthetic_response(ny=3, nx=4, n_fov_y=2, n_fov_x=5)
        wls = np.array([446.0, 545.0, 627.0])
        weights = luminance_weights_for_response(wls)
        x, y = _pupil_axes(3, 4)
        result = luminance_per_cell_per_angle(response, x, y, weights)
        assert result.shape == (3, 4, 10)

    def test_d65_distance_zero_for_d65_response(self):
        wls = np.array([446.0, 545.0, 627.0])
        response = _synthetic_response(d65_balanced=True, brightness=1.0)
        x, y = _pupil_axes(3, 3)
        d = d65_distance_per_cell_per_angle(response, x, y, wls)
        assert float(d.max()) < 1e-5

    def test_d65_distance_positive_for_tinted_response(self):
        wls = np.array([446.0, 545.0, 627.0])
        response = _synthetic_response(d65_balanced=False, brightness=1.0)
        x, y = _pupil_axes(3, 3)
        d = d65_distance_per_cell_per_angle(response, x, y, wls)
        assert float(d.min()) > 0.1

    def test_cv_zero_for_flat_fov(self):
        per_angle = np.full((3, 3, 5), 7.0)
        cv = coefficient_of_variation_over_angles(per_angle)
        assert np.allclose(cv, 0.0)

    def test_cv_positive_for_varying_fov(self):
        per_angle = np.linspace(0.5, 1.5, 5)[None, None, :]
        per_angle = np.broadcast_to(per_angle, (3, 3, 5))
        cv = coefficient_of_variation_over_angles(per_angle)
        assert float(cv[0, 0]) > 0.1


# ── Overview figures ────────────────────────────────────────────────────────


class TestOverview:

    def test_brightness_figure_with_wavelengths(self):
        wls = np.array([446.0, 545.0, 627.0])
        response = _synthetic_response()
        x, y = _pupil_axes()
        fig = pupil_brightness_figure(response, x, y, wls,
                                      threshold_nits=10.0)
        assert isinstance(fig, go.Figure)
        # 1 heatmap + 1 contour for the threshold
        assert len(fig.data) == 2

    def test_brightness_figure_falls_back_without_wavelengths(self):
        response = _synthetic_response()
        x, y = _pupil_axes()
        fig = pupil_brightness_figure(response, x, y, wavelengths_nm=None)
        assert "radiance" in fig.layout.title.text.lower() or \
               "intensity" in fig.layout.title.text.lower()

    def test_d65_figure_with_tolerance_adds_contour(self):
        response = _synthetic_response()
        x, y = _pupil_axes()
        fig = pupil_d65_distance_figure(response, x, y,
                                        np.array([446.0, 545.0, 627.0]),
                                        tolerance=0.05)
        assert len(fig.data) == 2  # heatmap + contour


# ── CDF ─────────────────────────────────────────────────────────────────────


class TestEyeboxCdf:

    def test_two_curves_when_d65_tolerance_set(self):
        wls = np.array([446.0, 545.0, 627.0])
        response = _synthetic_response()
        x, y = _pupil_axes()
        fig = eyebox_quality_cdf_figure(response, x, y, wls,
                                        d65_tolerance=0.05)
        assert len(fig.data) == 2

    def test_one_curve_without_d65_tolerance(self):
        wls = np.array([446.0, 545.0, 627.0])
        response = _synthetic_response()
        x, y = _pupil_axes()
        fig = eyebox_quality_cdf_figure(response, x, y, wls,
                                        d65_tolerance=None)
        assert len(fig.data) == 1

    def test_uniform_response_drops_at_threshold(self):
        """A uniform-brightness response should give a sharp drop in
        the CDF at exactly the cell brightness value."""
        wls = np.array([446.0, 545.0, 627.0])
        response = _synthetic_response(brightness=2.0)
        x, y = _pupil_axes()
        fig = eyebox_quality_cdf_figure(response, x, y, wls,
                                        d65_tolerance=None)
        fractions = np.asarray(fig.data[0].y)
        # Below the cell brightness all cells qualify; above, none.
        assert fractions[0] == 1.0
        assert fractions[-1] == 0.0


# ── Per-cell ────────────────────────────────────────────────────────────────


class TestPerCell:

    def test_d65_fov_figure_has_slider_with_n_steps(self):
        wls = np.array([446.0, 545.0, 627.0])
        response = _synthetic_response(ny=3, nx=3, n_fov_y=2, n_fov_x=2)
        x, y = _pupil_axes(3, 3)
        sa = _scan_angles(2, 2)
        fig = per_cell_d65_fov_figure(response, sa, x, y, wls)
        slider = fig.layout.sliders[0]
        assert len(slider.steps) == 9   # 3×3 cells

    def test_d65_fov_figure_two_traces_per_cell(self):
        wls = np.array([446.0, 545.0, 627.0])
        response = _synthetic_response(ny=2, nx=2, n_fov_y=2, n_fov_x=2)
        x, y = _pupil_axes(2, 2)
        sa = _scan_angles(2, 2)
        fig = per_cell_d65_fov_figure(response, sa, x, y, wls)
        # 4 cells × (heatmap + contour) = 8 traces
        assert len(fig.data) == 8

    def test_visible_color_image_per_cell(self):
        wls = np.array([446.0, 545.0, 627.0])
        response = _synthetic_response(ny=2, nx=2, n_fov_y=4, n_fov_x=4)
        x, y = _pupil_axes(2, 2)
        sa = _scan_angles(4, 4)
        fig = per_cell_visible_color_figure(response, sa, x, y, wls)
        assert len(fig.data) == 4         # one image per cell
        # Each trace is a Plotly Image
        for trace in fig.data:
            assert trace.type == "image"


# ── Global FOV ──────────────────────────────────────────────────────────────


class TestGlobalFov:

    def test_returns_two_figures(self):
        response = _synthetic_response(n_fov_y=3, n_fov_x=4)
        sa = _scan_angles(3, 4)
        figs = fov_global_figures(response, sa)
        assert len(figs) == 2


# ── Mirrors ─────────────────────────────────────────────────────────────────


class TestMirrors:

    def test_no_mirrors_returns_empty_chart(self):
        manifest = {"system": {"elements": []}}
        fig = mirror_reflectance_figure(manifest)
        assert len(fig.data) == 0

    def test_one_line_per_mirror(self):
        manifest = {
            "system": {
                "elements": [
                    {"type": "GaussianMirror", "name": "mirror_0",
                     "wavelengths": [4.46e-4, 5.45e-4, 6.27e-4],
                     "reflectance": [0.05, 0.06, 0.04]},
                    {"type": "GaussianMirror", "name": "mirror_1",
                     "wavelengths": [4.46e-4, 5.45e-4, 6.27e-4],
                     "reflectance": [0.07, 0.08, 0.05]},
                    {"type": "RectangularPupil", "name": "pupil"},  # ignored
                ]
            }
        }
        fig = mirror_reflectance_figure(manifest)
        assert len(fig.data) == 2
        # Trace names should match mirror names
        assert {trace.name for trace in fig.data} == {"mirror_0", "mirror_1"}


# ── Projector spectrum + per-mirror residual ────────────────────────────────


class TestProjectorFigures:

    def _manifest_with_projector(self, mirror_reflectances):
        """Manifest with a 5%-flat projector spectrum and the given mirrors.
        Wavelengths are in apollo14 internal units (nm = 1e-6)."""
        wls = [4.46e-4, 5.45e-4, 6.27e-4]
        elements = [
            {"type": "GaussianMirror", "name": f"mirror_{i}",
             "wavelengths": wls, "reflectance": list(refl)}
            for i, refl in enumerate(mirror_reflectances)
        ]
        return {
            "system": {"elements": elements},
            "projector": {"spectrum": {"wavelengths": wls,
                                        "radiance": [1.0, 1.0, 1.0]}},
        }

    def test_projector_spectrum_renders(self):
        manifest = self._manifest_with_projector([[0.05, 0.05, 0.05]])
        fig = projector_spectrum_figure(manifest)
        assert len(fig.data) == 1

    def test_projector_spectrum_empty_when_missing(self):
        fig = projector_spectrum_figure({"system": {"elements": []}})
        assert len(fig.data) == 0

    def test_mirror_input_spectrum_one_curve_per_mirror_plus_exit(self):
        manifest = self._manifest_with_projector([
            [0.05, 0.05, 0.05],
            [0.10, 0.10, 0.10],
        ])
        fig = mirror_input_spectrum_figure(manifest)
        # 2 mirror inputs + 1 exit residual = 3 curves
        assert len(fig.data) == 3

    def test_mirror_input_first_curve_is_projector(self):
        # Mirror 0 sees the projector untouched.
        manifest = self._manifest_with_projector([
            [0.05, 0.05, 0.05],
            [0.10, 0.10, 0.10],
        ])
        fig = mirror_input_spectrum_figure(manifest)
        first_y = np.asarray(fig.data[0].y)
        assert np.allclose(first_y, [1.0, 1.0, 1.0])

    def test_mirror_input_residual_after_uniform_5pct_mirror(self):
        # After one 5% mirror the residual should be 0.95× projector.
        manifest = self._manifest_with_projector([[0.05, 0.05, 0.05]])
        fig = mirror_input_spectrum_figure(manifest)
        # 1 mirror input + 1 exit
        assert len(fig.data) == 2
        exit_y = np.asarray(fig.data[1].y)
        assert np.allclose(exit_y, [0.95, 0.95, 0.95])

    def test_mirror_input_empty_when_projector_missing(self):
        manifest = {"system": {"elements": [
            {"type": "GaussianMirror", "name": "mirror_0",
             "wavelengths": [4.46e-4, 5.45e-4, 6.27e-4],
             "reflectance": [0.05, 0.05, 0.05]},
        ]}}
        fig = mirror_input_spectrum_figure(manifest)
        assert len(fig.data) == 0


# ── Headline numbers ────────────────────────────────────────────────────────


class TestHeadlineNumbers:

    def _full_manifest(self, mirror_reflectances, nx=4, ny=4):
        wls_internal = [4.46e-4, 5.45e-4, 6.27e-4]   # apollo14 internal units
        elements = [
            {"type": "GaussianMirror", "name": f"mirror_{i}",
             "wavelengths": wls_internal, "reflectance": list(refl)}
            for i, refl in enumerate(mirror_reflectances)
        ]
        return {
            "system": {"elements": elements},
            "projector": {
                "nx": nx, "ny": ny,
                "spectrum": {"wavelengths": wls_internal,
                             "radiance": [1.0, 1.0, 1.0]},
            },
        }

    def test_per_mirror_avg_reflectance(self):
        manifest = self._full_manifest([
            [0.05, 0.07, 0.09],
            [0.10, 0.10, 0.10],
        ])
        # Need a non-empty response with 3 wavelengths.
        response = np.zeros((4, 2, 3))
        nums = compute_headline_numbers(manifest, response,
                                        wavelengths_nm=np.array([446., 545., 627.]))
        per_mirror = nums["per_mirror_avg_reflectance"]
        assert [m["name"] for m in per_mirror] == ["mirror_0", "mirror_1"]
        assert abs(per_mirror[0]["avg_reflectance"] - 0.07) < 1e-6
        assert abs(per_mirror[1]["avg_reflectance"] - 0.10) < 1e-6

    def test_ambient_transparency_uses_per_mirror_average(self):
        # Ambient light crosses one mirror's region, not the cascade.
        # Two uniform 5% mirrors ⇒ avg per-mirror reflectance = 5%,
        # ambient transparency = 95% (NOT 0.95² as a stack product).
        manifest = self._full_manifest([
            [0.05, 0.05, 0.05],
            [0.05, 0.05, 0.05],
        ])
        response = np.zeros((4, 2, 3))
        nums = compute_headline_numbers(manifest, response,
                                        wavelengths_nm=np.array([446., 545., 627.]))
        assert abs(nums["avg_per_mirror_reflectance"] - 0.05) < 1e-6
        assert abs(nums["avg_ambient_transparency"] - 0.95) < 1e-6

    def test_ambient_reflectance_averages_across_mirrors(self):
        # Mirrors at 5% and 15% ⇒ average per-mirror reflectance = 10%.
        manifest = self._full_manifest([
            [0.05, 0.05, 0.05],
            [0.15, 0.15, 0.15],
        ])
        response = np.zeros((4, 2, 3))
        nums = compute_headline_numbers(manifest, response,
                                        wavelengths_nm=np.array([446., 545., 627.]))
        assert abs(nums["avg_per_mirror_reflectance"] - 0.10) < 1e-6
        assert abs(nums["avg_ambient_transparency"] - 0.90) < 1e-6

    def test_eyebox_efficiency_zero_for_zero_response(self):
        manifest = self._full_manifest([[0.05, 0.05, 0.05]])
        response = np.zeros((4, 2, 3))
        nums = compute_headline_numbers(manifest, response,
                                        wavelengths_nm=np.array([446., 545., 627.]))
        assert nums["eyebox_efficiency_pct"] == 0.0

    def test_eyebox_efficiency_unit_response(self):
        # response[s, a, k] = 1 for all (S=4, A=2, K=3) ⇒ total deposited = 24.
        # input = num_directions(2) × num_rays(16) × Σ_λ spectrum (1+1+1=3) = 96.
        # Efficiency = 24/96 = 25%.
        manifest = self._full_manifest([[0.05, 0.05, 0.05]], nx=4, ny=4)
        response = np.ones((4, 2, 3))
        nums = compute_headline_numbers(manifest, response,
                                        wavelengths_nm=np.array([446., 545., 627.]))
        assert abs(nums["eyebox_efficiency_pct"] - 25.0) < 1e-6

    def test_returns_none_when_projector_missing(self):
        manifest = {"system": {"elements": [
            {"type": "GaussianMirror", "name": "mirror_0",
             "wavelengths": [4.46e-4, 5.45e-4, 6.27e-4],
             "reflectance": [0.05, 0.05, 0.05]},
        ]}}
        response = np.zeros((4, 2, 3))
        nums = compute_headline_numbers(
            manifest, response, wavelengths_nm=np.array([446., 545., 627.]))
        assert nums["eyebox_efficiency_pct"] is None
        assert nums["mean_cell_brightness_pct"] is None
        # Mirror-derived numbers still populated.
        assert nums["avg_per_mirror_reflectance"] is not None
        assert nums["avg_ambient_transparency"] is not None

    def test_html_renders_without_crash(self):
        nums = {
            "eyebox_efficiency_pct": 12.3,
            "mean_cell_brightness_pct": 0.20,
            "min_cell_brightness_pct": 0.05,
            "max_cell_brightness_pct": 0.40,
            "avg_per_mirror_reflectance": 0.07,
            "avg_ambient_transparency": 0.93,
            "per_mirror_avg_reflectance": [
                {"name": "mirror_0", "avg_reflectance": 0.07},
            ],
        }
        html = headline_numbers_html(nums)
        assert "Headline numbers" in html
        assert "12.3" in html or "12.30%" in html
        assert "mirror_0" in html
        assert "ambient transparency" in html.lower()

    def test_html_handles_missing_values_with_dash(self):
        nums = {
            "eyebox_efficiency_pct": None,
            "mean_cell_brightness_pct": None,
            "min_cell_brightness_pct": None,
            "max_cell_brightness_pct": None,
            "avg_per_mirror_reflectance": None,
            "avg_ambient_transparency": None,
            "per_mirror_avg_reflectance": [],
        }
        html = headline_numbers_html(nums)
        assert "Headline numbers" in html
        assert "—" in html


# ── End-to-end orchestrator ────────────────────────────────────────────────


class TestRenderPupilReport:

    def test_writes_html(self, tmp_path):
        # Build a minimal manifest + response.npz that the renderer accepts.
        manifest = {
            "git_sha": "abcdef0",
            "timestamp": "2026-04-26T00:00:00",
            "scan": {"x_fov": 0.122, "y_fov": 0.122, "num_x": 2, "num_y": 2},
            "system": {"elements": [
                {"type": "GaussianMirror", "name": "mirror_0",
                 "wavelengths": [4.46e-4, 5.45e-4, 6.27e-4],
                 "reflectance": [0.05, 0.05, 0.05]},
            ]},
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))

        response = _synthetic_response(ny=2, nx=2, n_fov_y=2, n_fov_x=2)
        np.savez(
            tmp_path / "response.npz",
            response=response,
            pupil_x_mm=np.array([-2.0, 2.0]),
            pupil_y_mm=np.array([-2.0, 2.0]),
            scan_angles=_scan_angles(2, 2),
            wavelengths_nm=np.array([446.0, 545.0, 627.0]),
        )

        out = render_pupil_report(
            tmp_path,
            eyebox_threshold_nits=10.0,
            d65_tolerance=0.05,
        )
        assert out.exists()
        text = out.read_text()
        assert "Apollo14 pupil report" in text
        assert "Per-cell" in text or "drill-down" in text
        assert "Headline numbers" in text
