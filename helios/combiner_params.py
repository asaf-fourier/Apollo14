"""Parametrized Talos combiner — design variables for optimization.

This module exposes the Talos reference combiner as a **function of its
design variables**: inter-mirror spacings and a parametric reflectance
curve per mirror.  Everything else (chassis geometry, mirror tilt,
aperture, pupil position) is held fixed and lifted from
:func:`apollo14.combiner.build_default_system`.

Usage::

    from helios.combiner_params import CombinerParams, build_parametrized_system
    params = CombinerParams.initial(num_mirrors=6)
    system = build_parametrized_system(params)

The resulting :class:`OpticalSystem` carries JAX arrays inside its
elements, so it can be passed straight into
``helios.merit.build_combiner_pupil_routes`` inside a JIT-compiled
loss function — gradients flow back to ``params`` via the reflectance
and mirror-position fields.

## Design-variable layout

- ``spacings``  ``(M-1,)``  — distance between consecutive mirrors, in
  the same units as the chassis (``mm``). Cumulative sum gives the
  offset of each mirror from ``mirror_0``.
- ``curves``    a batched :class:`apollo14.spectral.SumOfGaussiansCurve`
  with leaves shaped ``(M, B)``.  Each mirror's reflectance is the sum
  of ``B`` Gaussians at fixed ``centers``; the optimizer tunes the
  per-basis ``amplitude`` and ``sigma``.

For ``M=6`` and ``B=3`` (R/G/B primaries) that's ``5 + 2·6·3 = 41``
variables.  Switching ``B`` (e.g. denser sampling, or a different basis
type entirely) just means handing a different curve to ``initial`` —
nothing else in the optimizer or tracer needs to change.

## Compensation — left to the optimizer

``build_default_system`` pre-compensates reflectances so every mirror
delivers equal absolute intensity. This parametrization does **not** —
the optimizer sees the raw, uncompensated gain of each mirror and must
learn compensation itself. This is intentional: compensation is a
policy that depends on the merit function's definition of "good"
(equal brightness, D65-balanced, above threshold), and hard-coding a
specific rule would bias the search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from apollo14.combiner import (
    DEFAULT_LIGHT_DIRECTION,
    DEFAULT_LIGHT_POSITION,
)
from apollo14.elements.aperture import RectangularAperture
from apollo14.elements.glass_block import GlassBlock
from apollo14.elements.partial_mirror import PartialMirror
from apollo14.elements.pupil import RectangularPupil
from apollo14.materials import agc_m074, air
from apollo14.spectral import SumOfGaussiansCurve
from apollo14.system import OpticalSystem
from apollo14.units import deg, mm, nm
from helios.merit import DEFAULT_WAVELENGTHS

# ── Fixed geometry (not optimized) ──────────────────────────────────────────
# All constants are plain Python floats so they remain concrete under JIT.
# Only arrays that must participate in JAX tracing are kept as jnp arrays.

NUM_MIRRORS = 6
CHASSIS_X = 14.0 * mm
CHASSIS_Y = 20.0 * mm
CHASSIS_Z = 2.0 * mm
CHASSIS_CENTER = jnp.array([CHASSIS_X / 2, CHASSIS_Y, CHASSIS_Z / 2])
SKEW_ANGLE = 6.0 * deg
MIRROR_ANGLE = 48.0 * deg
EYE_RELIEF = 15.0 * mm
FIRST_MIRROR_OFFSET_Y = 5.0 * mm

_NORMAL_ANGLE = math.pi / 2 - MIRROR_ANGLE
_MIRROR_NORMAL = jnp.array([
    0.0,
    math.sin(_NORMAL_ANGLE),
    math.cos(_NORMAL_ANGLE),
])
_MIRROR_X_WIDTH = CHASSIS_X
_MIRROR_Y_WIDTH = CHASSIS_Z / math.cos(MIRROR_ANGLE)
_MIRROR_EDGE_TO_CENTER_Y = 0.5 * math.sqrt(
    _MIRROR_Y_WIDTH ** 2 - CHASSIS_Z ** 2
)
_UNIT_OFFSET = jnp.array([0.0, 1.0 / math.sin(_NORMAL_ANGLE), 0.0])
_Z_SKEW = CHASSIS_Z * math.tan(SKEW_ANGLE)

_FIRST_MIRROR_CENTER = CHASSIS_CENTER + jnp.array([0.0, FIRST_MIRROR_OFFSET_Y, 0.0])
_FIRST_MIRROR_POS = _FIRST_MIRROR_CENTER - jnp.array(
    [0.0, _MIRROR_EDGE_TO_CENTER_Y, 0.0]
)


# ── Design variables ────────────────────────────────────────────────────────


class CombinerParams(NamedTuple):
    """Design variables for the Talos combiner optimization.

    Registered as a :class:`NamedTuple` so JAX treats it as a pytree —
    ``jax.grad`` and optimizers like Adam can consume it directly.
    """
    spacings: jnp.ndarray             # (M-1,) inter-mirror spacing, mm
    curves: SumOfGaussiansCurve       # batched, leaves shape (M, B)

    @classmethod
    def initial(
        cls,
        num_mirrors: int = NUM_MIRRORS,
        spacing_mm: float = 1.47,
        amplitude: float = 0.05,
        width_nm: float = 20.0,
        centers: jnp.ndarray | None = None,
    ) -> "CombinerParams":
        """Reasonable starting point for optimization.

        Flat per-basis reflectance per mirror, narrow Gaussians so basis
        bumps are initially decoupled, and even spacing throughout the
        chassis. ``centers`` defaults to
        :data:`helios.merit.DEFAULT_WAVELENGTHS` (R/G/B primaries) — pass
        a different array (any length ``B``) to optimize against a
        different basis.
        """
        if centers is None:
            centers = DEFAULT_WAVELENGTHS
        return cls(
            spacings=jnp.full((num_mirrors - 1,), spacing_mm * mm),
            curves=SumOfGaussiansCurve.uniform(
                centers=centers,
                amplitude=amplitude,
                sigma=width_nm * nm,
                num_mirrors=num_mirrors,
            ),
        )


# ── Bounds & reparametrization ──────────────────────────────────────────────


# FWHM = 2·sqrt(2·ln2)·σ for a Gaussian. Bounds are expressed in FWHM
# (intuitive optical-design units), but ``params.curves.sigma`` carries σ
# (what the Gaussian math consumes), so we convert at clip time.
_FWHM_OVER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


def fwhm_to_sigma(fwhm):
    """FWHM → Gaussian σ. Works on scalars and JAX/NumPy arrays."""
    return fwhm / _FWHM_OVER_SIGMA


def sigma_to_fwhm(sigma):
    """Gaussian σ → FWHM. Works on scalars and JAX/NumPy arrays."""
    return sigma * _FWHM_OVER_SIGMA


@dataclass
class ParamBounds:
    """Hard bounds for post-step clipping.

    Not a reparametrization — the optimizer sees raw values and we clip
    after each Adam step to keep the design physical. The chassis is
    ``chassis_y = 20 mm``; the sum of spacings must fit inside it.

    Width bounds are given as FWHM (nm); ``params.curves.sigma`` stores σ.
    Bounds are specific to :class:`SumOfGaussiansCurve`; a different
    curve type would call for its own bounds object.
    """
    spacing_min_mm: float = 0.5
    spacing_max_mm: float = 3.0
    amplitude_min: float = 0.0
    amplitude_max: float = 0.20
    fwhm_min_nm: float = 20.0
    fwhm_max_nm: float = 50.0
    chassis_usable_mm: float = 18.0  # margin below 20 mm

    def clip(self, params: CombinerParams) -> CombinerParams:
        clipped_spacings = jnp.clip(params.spacings,
                                    self.spacing_min_mm * mm,
                                    self.spacing_max_mm * mm)
        total_spacing = jnp.sum(clipped_spacings)
        usable_length = self.chassis_usable_mm * mm
        rescale = jnp.where(total_spacing > usable_length,
                            usable_length / total_spacing, 1.0)
        clipped_spacings = clipped_spacings * rescale
        sigma_min = fwhm_to_sigma(self.fwhm_min_nm * nm)
        sigma_max = fwhm_to_sigma(self.fwhm_max_nm * nm)
        clipped_curves = SumOfGaussiansCurve(
            amplitude=jnp.clip(params.curves.amplitude,
                                self.amplitude_min, self.amplitude_max),
            sigma=jnp.clip(params.curves.sigma, sigma_min, sigma_max),
            centers=params.curves.centers,
        )
        return CombinerParams(spacings=clipped_spacings, curves=clipped_curves)


# ── System builder ──────────────────────────────────────────────────────────


def build_parametrized_system(
    params: CombinerParams,
    probe_wavelengths: jnp.ndarray | None = None,
) -> OpticalSystem:
    """Build the Talos combiner using ``params`` as the design variables.

    All JAX arrays inside ``params`` propagate into the resulting
    system's elements, so differentiating a downstream merit function
    w.r.t. ``params`` produces gradients on spacings and the curve's
    leaves (amplitude/sigma).

    Args:
        params: :class:`CombinerParams` holding spacings + a batched
            :class:`SumOfGaussiansCurve`.
        probe_wavelengths: ``(K,)`` dense sample grid where each
            mirror's curve is evaluated before being stored. The
            tracer's ``jnp.interp`` then interpolates between dense
            samples — pass the same wavelength grid the trace uses for
            an effectively-exact curve. Defaults to the curve's
            ``centers`` (B points — back-compat; gives a piecewise-
            linear curve through the basis peaks).

    Returns:
        :class:`OpticalSystem` with chassis, aperture, mirrors, pupil.
    """
    if probe_wavelengths is None:
        # Default probe grid = the curve's centers (back-compat with the
        # 3-point R/G/B grid). ``stop_gradient`` keeps the centers from
        # picking up a gradient via the wavelength axis — they're fixed
        # by design and the curve's own ``sample`` already wraps them.
        probe_wavelengths = jax.lax.stop_gradient(params.curves.centers[0])
    probe_wavelengths = jnp.asarray(probe_wavelengths)

    system = OpticalSystem(env_material=air)

    # Chassis — fixed geometry (all dimensions are plain floats, JIT-safe)
    chassis = GlassBlock.create_chassis(
        name="chassis",
        x=CHASSIS_X, y=CHASSIS_Y, z=CHASSIS_Z,
        material=agc_m074,
        z_skew=_Z_SKEW,
    ).translate(CHASSIS_CENTER)
    system.add(chassis)

    # Aperture — fixed
    system.add(RectangularAperture(
        name="aperture",
        position=DEFAULT_LIGHT_POSITION - jnp.array([0.0, 0.5 * mm, 0.0]),
        normal=DEFAULT_LIGHT_DIRECTION,
        width=6.0 * mm,
        height=3.0 * mm,
        inner_width=4.0 * mm,
        inner_height=1.0 * mm,
    ))

    # Mirrors — positions from cumulative spacings, reflectance from the curve.
    cumulative_offset = jnp.concatenate(
        [jnp.zeros(1), jnp.cumsum(params.spacings)]
    )  # (M,)
    mirror_positions = (
        _FIRST_MIRROR_POS[None, :]
        - cumulative_offset[:, None] * _UNIT_OFFSET[None, :]
    )  # (M, 3)

    num_mirrors = params.curves.amplitude.shape[0]
    for mirror_idx in range(num_mirrors):
        mirror_curve = params.curves.at(mirror_idx)
        system.add(PartialMirror(
            name=f"mirror_{mirror_idx}",
            position=mirror_positions[mirror_idx],
            normal=_MIRROR_NORMAL.copy(),
            width=_MIRROR_X_WIDTH,
            height=_MIRROR_Y_WIDTH,
            wavelengths=probe_wavelengths,
            curve=mirror_curve,
        ))

    # Pupil — fixed
    system.add(RectangularPupil(
        name="pupil",
        position=jnp.array([
            CHASSIS_CENTER[0],
            CHASSIS_CENTER[1] - 2 * mm,
            EYE_RELIEF + CHASSIS_Z,
        ]),
        normal=jnp.array([0.0, 0.0, -1.0]),
        width=10.0 * mm,
        height=14.0 * mm,
    ))

    return system
