"""Partial mirror — splits light into reflected and transmitted fractions.

Reflectance is stored as a sampled spectral curve: a shared ``wavelengths``
grid and a per-mirror ``reflectance`` array. At trace time the effective
reflectance is ``jnp.interp(wavelength, wavelengths, reflectance)`` —
differentiable w.r.t. the control values and continuous in wavelength.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp

from apollo14.elements.planar import PlanarElement
from apollo14.geometry import ray_intersect_planar_seg, reflect
from apollo14.ray import Ray
from apollo14.route import REFLECT
from apollo14.units import nm

# Default spectral sample grid: R/G/B micro-LED peaks, sorted ascending so
# ``jnp.interp`` can consume it directly. Every new mirror inherits this
# grid unless the caller overrides it.
DEFAULT_MIRROR_WAVELENGTHS = jnp.array([460.0, 525.0, 630.0],
                                        dtype=jnp.float32) * nm


class _SingleMirror(NamedTuple):
    """Unstacked transmit-mirror emitted by ``PartialMirror.build_segment``.

    The route builder's grouper fuses runs of these into one
    ``MirrorStackSeg``. It never reaches the tracer directly.
    """
    position: jnp.ndarray       # (3,)
    normal: jnp.ndarray         # (3,)
    local_x: jnp.ndarray        # (3,)
    local_y: jnp.ndarray        # (3,)
    half_extents: jnp.ndarray   # (2,)
    wavelengths: jnp.ndarray    # (K,) spectral sample grid (ascending)
    reflectance: jnp.ndarray    # (K,) reflectance at each sample


class MirrorStackSeg(NamedTuple):
    """Stacked partial mirrors, all transmit, same medium — traced via scan."""
    position: jnp.ndarray       # (M, 3)
    normal: jnp.ndarray         # (M, 3)
    local_x: jnp.ndarray        # (M, 3)
    local_y: jnp.ndarray        # (M, 3)
    half_extents: jnp.ndarray   # (M, 2)
    wavelengths: jnp.ndarray    # (M, K) — each row identical in practice
    reflectance: jnp.ndarray    # (M, K)


class ReflectMirrorSeg(NamedTuple):
    """Single reflecting mirror (the fork point of a branch route)."""
    position: jnp.ndarray       # (3,)
    normal: jnp.ndarray
    local_x: jnp.ndarray
    local_y: jnp.ndarray
    half_extents: jnp.ndarray
    wavelengths: jnp.ndarray    # (K,)
    reflectance: jnp.ndarray    # (K,)


@dataclass
class PartialMirror(PlanarElement):
    reflectance: jnp.ndarray = None   # (K,) sampled reflectance curve
    wavelengths: jnp.ndarray = None   # (K,) wavelength grid (ascending)

    def __post_init__(self):
        super().__post_init__()
        if self.wavelengths is None:
            self.wavelengths = DEFAULT_MIRROR_WAVELENGTHS
        else:
            self.wavelengths = jnp.asarray(self.wavelengths, dtype=jnp.float32)

        k = self.wavelengths.shape[0]
        if self.reflectance is None:
            self.reflectance = jnp.full((k,), 0.05, dtype=jnp.float32)
        else:
            r = jnp.asarray(self.reflectance, dtype=jnp.float32)
            self.reflectance = jnp.broadcast_to(r, (k,)).copy()

    def build_segment(self, current_material, mode):
        if mode == REFLECT:
            seg = ReflectMirrorSeg(
                position=self.position,
                normal=self.normal,
                local_x=self._local_x,
                local_y=self._local_y,
                half_extents=self.half_extents,
                wavelengths=self.wavelengths,
                reflectance=self.reflectance,
            )
            return seg, current_material
        # TRANSMIT — emit one ``_SingleMirror``; ``_group_mirror_runs``
        # fuses adjacent ones into a single ``MirrorStackSeg``.
        seg = _SingleMirror(
            position=self.position,
            normal=self.normal,
            local_x=self._local_x,
            local_y=self._local_y,
            half_extents=self.half_extents,
            wavelengths=self.wavelengths,
            reflectance=self.reflectance,
        )
        return seg, current_material


@dataclass
class GaussianMirror(PartialMirror):
    """Partial mirror whose reflectance is a sum of per-color Gaussians.

    Each color channel contributes one Gaussian bump centered at its
    projector primary wavelength. The effective reflectance at any
    wavelength is the sum of all three::

        r(λ) = Σ_c amplitude[c] · exp( −(λ − center[c])² / (2·sigma[c]²) )

    The ``amplitude`` and ``sigma`` arrays are the design variables for
    optimization (6 per mirror). The ``centers`` are fixed at the
    projector primaries and are *not* optimized.

    The dense curve is evaluated at ``probe_wavelengths`` and stored on
    the parent ``PartialMirror``'s ``wavelengths``/``reflectance`` fields.
    The trace's ``jnp.interp`` then linearly interpolates between dense
    samples — visually and numerically a smooth Gaussian sum, provided
    ``probe_wavelengths`` is fine enough relative to the smallest σ.

    Args:
        amplitude: ``(C,)`` per-color peak reflectance — design variable.
        sigma: ``(C,)`` per-color Gaussian width (in length units —
            typically mm, since the project uses ``apollo14.units.nm``)
            — design variable.
        centers: ``(C,)`` Gaussian centers, fixed (e.g. R/G/B projector
            peaks). Defaults to ``DEFAULT_MIRROR_WAVELENGTHS``.
        probe_wavelengths: ``(K,)`` dense sample grid where the curve is
            evaluated before being stored. Defaults to ``centers``
            (3 points — back-compat; pass a dense grid like the trace
            wavelengths to get a smooth Gaussian shape).
    """
    amplitude: jnp.ndarray = None          # (C,) per-color peak reflectance
    sigma: jnp.ndarray = None              # (C,) per-color Gaussian width
    centers: jnp.ndarray = None            # (C,) Gaussian centers (fixed)
    probe_wavelengths: jnp.ndarray = None  # (K,) dense sample grid

    def __post_init__(self):
        if self.centers is None:
            self.centers = DEFAULT_MIRROR_WAVELENGTHS
        else:
            self.centers = jnp.asarray(self.centers, dtype=jnp.float32)

        if self.probe_wavelengths is None:
            self.probe_wavelengths = self.centers
        else:
            self.probe_wavelengths = jnp.asarray(self.probe_wavelengths,
                                                  dtype=jnp.float32)

        if self.amplitude is None:
            self.amplitude = jnp.full((self.centers.shape[0],), 0.05,
                                       dtype=jnp.float32)
        else:
            self.amplitude = jnp.asarray(self.amplitude, dtype=jnp.float32)

        if self.sigma is None:
            self.sigma = jnp.full((self.centers.shape[0],), 20.0 * nm,
                                   dtype=jnp.float32)
        else:
            self.sigma = jnp.asarray(self.sigma, dtype=jnp.float32)

        self.wavelengths = self.probe_wavelengths
        self.reflectance = self._build_reflectance_curve()
        super().__post_init__()

    def _build_reflectance_curve(self) -> jnp.ndarray:
        """Evaluate sum-of-Gaussians at each probe wavelength.

        ``offset[c, k] = probe_wavelengths[k] − centers[c]``.
        """
        offset = (self.probe_wavelengths[None, :]
                  - self.centers[:, None])                       # (C, K)
        exponent = -(offset ** 2) / (
            2.0 * self.sigma[:, None] ** 2 + 1e-18)
        per_color = self.amplitude[:, None] * jnp.exp(exponent)  # (C, K)
        return jnp.sum(per_color, axis=0)                        # (K,)


def _interp_reflectance(wavelengths, reflectance, wavelength):
    """``jnp.interp`` with a consistent float32 output."""
    return jnp.interp(wavelength, wavelengths, reflectance)


def mirror_transmit_one(mirror_params, ray: Ray, wavelength):
    """Transmit through one partial mirror (used inside ``lax.scan``).

    The mirror coating is a thin layer inside a single medium, so the
    transmitted direction is unchanged — only the intensity attenuates.
    """
    alive_in = ray.intensity > 0

    hit, _, in_bounds = ray_intersect_planar_seg(ray, mirror_params)

    r = _interp_reflectance(
        mirror_params.wavelengths, mirror_params.reflectance, wavelength)
    new_intensity = ray.intensity * (1.0 - r)

    valid = in_bounds & alive_in
    out_intensity = jnp.where(valid, new_intensity, 0.0)
    out_pos = jnp.where(valid, hit, ray.pos)
    return Ray(pos=out_pos, dir=ray.dir, intensity=out_intensity), hit, valid


def mirror_reflect_one(seg: ReflectMirrorSeg, ray: Ray, wavelength):
    """Reflect off one partial mirror (branch fork point)."""
    alive_in = ray.intensity > 0

    hit, _, in_bounds = ray_intersect_planar_seg(ray, seg)

    facing = jnp.where(jnp.dot(ray.dir, seg.normal) < 0,
                       seg.normal, -seg.normal)
    reflected = reflect(ray.dir, facing)
    r = _interp_reflectance(seg.wavelengths, seg.reflectance, wavelength)
    new_intensity = ray.intensity * r

    valid = in_bounds & alive_in
    out_intensity = jnp.where(valid, new_intensity, 0.0)
    out_pos = jnp.where(valid, hit, ray.pos)
    out_dir = jnp.where(valid, reflected, ray.dir)
    return Ray(pos=out_pos, dir=out_dir, intensity=out_intensity), hit, valid
