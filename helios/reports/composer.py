"""Data-prep helpers for the pupil report.

Pure numpy. Produces the per-cell, per-(cell, angle) tensors that the
chart modules consume — keeps the visual code free of math.
"""

from __future__ import annotations

import numpy as np

from apollo14.units import nm
from helios.merit import D65_WEIGHTS, d65_weights_at
from helios.photometry import luminance_weights_np

# ── Reshape ─────────────────────────────────────────────────────────────────


def reshape_pupil(response: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """``(S, A, K) → (ny, nx, A, K)``."""
    S, A, K = response.shape
    if ny * nx != S:
        raise ValueError(f"S={S} does not match pupil grid {ny}×{nx}")
    return response.reshape(ny, nx, A, K)


def reshape_fov(response: np.ndarray, n_fov_y: int, n_fov_x: int) -> np.ndarray:
    """``(S, A, K) → (S, n_fov_y, n_fov_x, K)``."""
    S, A, K = response.shape
    if n_fov_y * n_fov_x != A:
        raise ValueError(f"A={A} does not match FOV grid {n_fov_y}×{n_fov_x}")
    return response.reshape(S, n_fov_y, n_fov_x, K)


# ── Spectral references ─────────────────────────────────────────────────────


def d65_ratios(wavelengths_nm: np.ndarray | None) -> np.ndarray:
    """``(K,)`` D65 simplex weights at the response's sampled wavelengths.

    Falls back to the precomputed 3-channel ``D65_WEIGHTS`` (R/G/B) when
    no wavelength axis is provided.
    """
    if wavelengths_nm is None or len(wavelengths_nm) == 3:
        d65 = np.asarray(D65_WEIGHTS)
    else:
        import jax.numpy as jnp
        wls = jnp.asarray(wavelengths_nm) * nm
        d65 = np.asarray(d65_weights_at(wls))
    return d65 / d65.sum()


def luminance_weights_for_response(
    wavelengths_nm: np.ndarray | None,
) -> np.ndarray | None:
    """``(K,)`` per-wavelength weights for converting response → luminance.

    Returns ``None`` when wavelengths are unavailable — callers should
    fall back to a unit-weight (radiometric) sum in that case.
    """
    if wavelengths_nm is None or len(wavelengths_nm) < 2:
        return None
    return luminance_weights_np(wavelengths_nm)


# ── Per-cell aggregations ───────────────────────────────────────────────────


def luminance_per_cell_per_angle(
    response: np.ndarray,
    pupil_x_mm: np.ndarray,
    pupil_y_mm: np.ndarray,
    luminance_weights: np.ndarray,
) -> np.ndarray:
    """``(ny, nx, A)`` luminance in nits for each pupil cell and FOV angle.

    Sum of per-wavelength response values weighted by ``K_m·V(λ)·Δλ``.
    """
    ny, nx = len(pupil_y_mm), len(pupil_x_mm)
    grid = reshape_pupil(response, ny, nx)            # (ny, nx, A, K)
    return np.sum(grid * luminance_weights[None, None, None, :], axis=-1)


def radiance_per_cell_per_angle(
    response: np.ndarray,
    pupil_x_mm: np.ndarray,
    pupil_y_mm: np.ndarray,
) -> np.ndarray:
    """``(ny, nx, A)`` raw summed radiance per (cell, angle)."""
    ny, nx = len(pupil_y_mm), len(pupil_x_mm)
    grid = reshape_pupil(response, ny, nx)
    return np.sum(grid, axis=-1)


def d65_distance_per_cell_per_angle(
    response: np.ndarray,
    pupil_x_mm: np.ndarray,
    pupil_y_mm: np.ndarray,
    wavelengths_nm: np.ndarray | None = None,
) -> np.ndarray:
    """``(ny, nx, A)`` simplex L2 distance from D65 per (cell, angle).

    Per (cell, angle), the K channels are normalized to a simplex
    (summing to 1) and compared against the D65 simplex via L2 distance.
    Result is dimensionless; 0 = perfect D65, ≈ √2 = pathologically
    one-channel response.
    """
    d65 = d65_ratios(wavelengths_nm)
    ny, nx = len(pupil_y_mm), len(pupil_x_mm)
    grid = reshape_pupil(response, ny, nx)            # (ny, nx, A, K)
    total = grid.sum(axis=-1, keepdims=True) + 1e-12
    ratios = grid / total                             # (ny, nx, A, K)
    return np.linalg.norm(ratios - d65[None, None, None, :], axis=-1)


# ── Reductions across FOV angles ────────────────────────────────────────────


def mean_over_angles(per_angle: np.ndarray) -> np.ndarray:
    """``(ny, nx, A) → (ny, nx)`` mean across the FOV axis."""
    return per_angle.mean(axis=-1)


def coefficient_of_variation_over_angles(per_angle: np.ndarray) -> np.ndarray:
    """``(ny, nx, A) → (ny, nx)`` ``std/mean`` per cell — FOV uniformity.

    0 → perfectly flat across FOV; 0.5 → 50% std as a fraction of mean
    (mountainous). Returns 0 where mean is 0 (avoid divide-by-zero).
    """
    mean = per_angle.mean(axis=-1)
    std = per_angle.std(axis=-1)
    return np.where(mean > 0, std / np.maximum(mean, 1e-12), 0.0)


def worst_over_angles(per_angle: np.ndarray) -> np.ndarray:
    """``(ny, nx, A) → (ny, nx)`` worst (minimum) across FOV per cell."""
    return per_angle.min(axis=-1)
