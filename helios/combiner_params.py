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
CHASSIS_Z = 2.0 * mm              # canonical Talos glass thickness (z); matches
                                 # apollo14.combiner.build_default_system. Pass
                                 # ``chassis_z=`` to build_parametrized_system to
                                 # override per build (e.g. a thinner combiner).
SKEW_ANGLE = 6.0 * deg
MIRROR_ANGLE = 48.0 * deg
EYE_RELIEF = 15.0 * mm
FIRST_MIRROR_OFFSET_Y = 5.0 * mm

# Pupil center, expressed as offsets from the chassis center.
# ``PUPIL_OFFSET_Y`` is hand-tuned to roughly center the pupil under where
# light lands after the chassis-skew shift. To refine it, trace the design
# and compute the FOV-averaged light centroid at the pupil plane, then set
# the offsets so the pupil sits on that centroid.
PUPIL_OFFSET_X = 0.0 * mm
PUPIL_OFFSET_Y = -2.38 * mm

_NORMAL_ANGLE = math.pi / 2 - MIRROR_ANGLE
_MIRROR_NORMAL = jnp.array([
    0.0,
    math.sin(_NORMAL_ANGLE),
    math.cos(_NORMAL_ANGLE),
])
_MIRROR_X_WIDTH = CHASSIS_X                              # z-independent
_UNIT_OFFSET = jnp.array([0.0, 1.0 / math.sin(_NORMAL_ANGLE), 0.0])


class _ChassisGeometry(NamedTuple):
    """Geometry that scales with the chassis thickness ``chassis_z``."""
    chassis_center: jnp.ndarray    # (3,) chassis center; z = chassis_z / 2
    mirror_y_width: float          # mirror height = chassis_z / cos(tilt)
    z_skew: float                  # chassis z-skew shear
    first_mirror_pos: jnp.ndarray  # (3,) position of mirror_0


def _derive_chassis_geometry(chassis_z: float) -> _ChassisGeometry:
    """Derive all chassis-thickness-dependent geometry from ``chassis_z``.

    The mirror height (``chassis_z / cos θ``), the chassis z-skew, the
    chassis center, and the first mirror's position all scale with the
    glass thickness. Collecting them here lets a caller rebuild the
    combiner at any thickness — e.g. to make it as thin as possible — by
    threading a single number through ``build_parametrized_system``.
    """
    chassis_center = jnp.array([CHASSIS_X / 2, CHASSIS_Y, chassis_z / 2])
    mirror_y_width = chassis_z / math.cos(MIRROR_ANGLE)
    mirror_edge_to_center_y = 0.5 * math.sqrt(
        mirror_y_width ** 2 - chassis_z ** 2
    )
    z_skew = chassis_z * math.tan(SKEW_ANGLE)
    first_mirror_center = chassis_center + jnp.array(
        [0.0, FIRST_MIRROR_OFFSET_Y, 0.0]
    )
    first_mirror_pos = first_mirror_center - jnp.array(
        [0.0, mirror_edge_to_center_y, 0.0]
    )
    return _ChassisGeometry(
        chassis_center=chassis_center,
        mirror_y_width=mirror_y_width,
        z_skew=z_skew,
        first_mirror_pos=first_mirror_pos,
    )


# Default-thickness geometry. ``CHASSIS_CENTER`` is a module constant reused
# below for the usable-stack-height budget and the pupil placement (its x/y
# are z-independent).
_DEFAULT_GEOMETRY = _derive_chassis_geometry(CHASSIS_Z)
CHASSIS_CENTER = _DEFAULT_GEOMETRY.chassis_center

# Largest y-footprint the mirror stack may span while every mirror stays fully
# inside the chassis: the drop from mirror_0's center down to the chassis
# bottom face, less one mirror half-height so the last mirror's lower edge
# doesn't poke through. ``mirror_half_height_y`` is recovered from the default
# geometry (mirror_0 center sits one half-height below its top corner at
# CHASSIS_CENTER_Y + FIRST_MIRROR_OFFSET_Y). Evaluated at the default
# thickness — a thinner chassis has shorter mirrors (more room), so this is a
# safe, never-too-loose budget for any thickness ≤ CHASSIS_Z. A *thicker*
# chassis would need a smaller budget: override ParamBounds.chassis_usable_mm.
_MIRROR_HALF_HEIGHT_Y = float(
    CHASSIS_CENTER[1] + FIRST_MIRROR_OFFSET_Y
    - _DEFAULT_GEOMETRY.first_mirror_pos[1])
USABLE_STACK_Y_MM = (
    float(_DEFAULT_GEOMETRY.first_mirror_pos[1])
    - (float(CHASSIS_CENTER[1]) - CHASSIS_Y / 2.0)
    - _MIRROR_HALF_HEIGHT_Y) / mm


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
    after each Adam step to keep the design physical. ``chassis_usable_mm``
    caps the mirror stack's *y-footprint* (see :data:`USABLE_STACK_Y_MM`)
    so the last mirror stays inside the chassis.

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
    chassis_usable_mm: float = USABLE_STACK_Y_MM  # mirror-stack y-footprint cap

    def clip(self, params: CombinerParams) -> CombinerParams:
        clipped_spacings = jnp.clip(params.spacings,
                                    self.spacing_min_mm * mm,
                                    self.spacing_max_mm * mm)
        # ``spacings`` are perpendicular mirror-plane gaps; each projects onto
        # the chassis y-axis by 1/sin(_NORMAL_ANGLE), so the stack's real
        # y-footprint is sum(spacings)/sin(_NORMAL_ANGLE). Bound *that* against
        # the usable y-budget — bounding the raw sum (as before) undercounts
        # the footprint by 1/sin ≈ 1.5× and lets the last mirror slide out the
        # bottom of the chassis.
        stack_y_extent = jnp.sum(clipped_spacings) / math.sin(_NORMAL_ANGLE)
        usable_length = self.chassis_usable_mm * mm
        rescale = jnp.where(stack_y_extent > usable_length,
                            usable_length / stack_y_extent, 1.0)
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
    chassis_z: float | None = None,
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
        chassis_z: Glass thickness (z) in internal length units. All
            thickness-dependent geometry — mirror height, chassis
            z-skew, chassis center, first-mirror position, and the pupil
            plane (kept at a constant ``EYE_RELIEF`` from the front
            face) — is rederived from it. Defaults to :data:`CHASSIS_Z`.
            Pass a smaller value to make the combiner thinner.

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

    if chassis_z is None:
        chassis_z = CHASSIS_Z
    geometry = _derive_chassis_geometry(chassis_z)
    chassis_center = geometry.chassis_center

    system = OpticalSystem(env_material=air)

    # Chassis — geometry scales with the chosen thickness (all dimensions
    # are plain floats, JIT-safe).
    chassis = GlassBlock.create_chassis(
        name="chassis",
        x=CHASSIS_X, y=CHASSIS_Y, z=chassis_z,
        material=agc_m074,
        z_skew=geometry.z_skew,
    ).translate(chassis_center)
    system.add(chassis)

    # Aperture — fixed, must match ``apollo14.combiner.build_default_system``
    # (fixed geometry is lifted from the reference build). Beam-defining stop:
    # opaque 14×6 mm frame (larger than the 10×2 mm beam) with a 10×2 mm
    # clear-aperture hole matching the beam — it passes the full beam and
    # absorbs only stray outside it. The previous 6×3 mm frame was *smaller*
    # than the beam, so wing rays at |x|>3 leaked past instead of being clipped.
    system.add(RectangularAperture(
        name="aperture",
        position=DEFAULT_LIGHT_POSITION - jnp.array([0.0, 0.5 * mm, 0.0]),
        normal=DEFAULT_LIGHT_DIRECTION,
        width=14.0 * mm,
        height=6.0 * mm,
        inner_width=10.0 * mm,
        inner_height=2.0 * mm,
    ))

    # Mirrors — positions from cumulative spacings, reflectance from the curve.
    cumulative_offset = jnp.concatenate(
        [jnp.zeros(1), jnp.cumsum(params.spacings)]
    )  # (M,)
    mirror_positions = (
        geometry.first_mirror_pos[None, :]
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
            height=geometry.mirror_y_width,
            wavelengths=probe_wavelengths,
            curve=mirror_curve,
        ))

    # Pupil — sized 14×18 mm (vs eyebox 8×10 mm) so there's slack for the
    # ``PUPIL_OFFSET_X / Y`` recentering without clipping rays at the
    # aperture. Center is at ``CHASSIS_CENTER`` shifted by the offsets
    # (defined at the top of this module).
    system.add(RectangularPupil(
        name="pupil",
        position=jnp.array([
            chassis_center[0] + PUPIL_OFFSET_X,
            chassis_center[1] + PUPIL_OFFSET_Y,
            EYE_RELIEF + chassis_z,
        ]),
        normal=jnp.array([0.0, 0.0, -1.0]),
        width=14.0 * mm,
        height=18.0 * mm,
    ))

    return system
