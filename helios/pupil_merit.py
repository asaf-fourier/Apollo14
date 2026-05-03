"""Pupil merit — drive every cell to a target brightness, D65-flat.

Two decoupled terms:

1. **Target** — every cell's mean-over-FOV brightness should equal a
   single target value ``target_relative · input_flux``. Symmetric
   squared error pulls cells toward the target from both sides, so the
   final design has minimal inter-cell variance.

2. **Shape (D65 + FOV uniformity)** — at every cell, the angular response
   should be (a) flat across FOV and (b) D65 white-balanced across R/G/B.
   Captured by a single scale-invariant squared error against the
   template ``D65[c] · Ī(s)``, where ``Ī(s)`` is the cell's own mean
   brightness. Different cells may sit at different absolute brightness
   levels — only their *shape* is compared.

## Tensor conventions

- ``response``  ``(S, A, K)`` — intensity at pupil sample ``s``, FOV
  angle ``a``, wavelength/color ``c``.
- ``cell_mask`` ``(S,)`` — 1.0 for samples inside the target eyebox
  region, 0.0 outside. Pass ``None`` to use all samples.
- ``input_flux`` scalar — reference flux. Same units as the brightness
  sum controlled by ``config.luminance_weights``.

## Math

Per-cell scales::

    radiance(s, a)    = Σ_c response[s, a, c]
    brightness(s, a)  = Σ_c response[s, a, c] · luminance_weights[c]
    Ī_radiance(s)     = mean_a radiance(s, a)
    Ī_brightness(s)   = mean_a brightness(s, a)

**Target** (scale-invariant per cell — at-target ⇒ 0, dark or 2× over ⇒ 1).
With ``asymmetric_target=True``, only the under-target deficit counts; cells
brighter than the target contribute zero::

    L_target = mean_s ( ( Ī_brightness(s) − τ·I_in ) / (τ·I_in) )²       [symmetric]
    L_target = mean_s ( min(0, Ī_brightness(s) − τ·I_in) / (τ·I_in) )²   [asymmetric]

**Shape** (radiance space, scale-invariant per cell)::

    err(s)   = mean_{a,c} ( response[s,a,c] − D65[c]·Ī_radiance(s) )² /
               ( Ī_radiance(s)² + ε·I_in² )
    L_shape  = mean_s err(s)

**Total**::

    L = w_target · L_target + w_shape · L_shape
"""

from dataclasses import dataclass

import jax.numpy as jnp

from helios.merit import D65_WEIGHTS


@dataclass
class PupilMeritConfig:
    """Hyperparameters for :func:`pupil_merit`.

    Attributes:
        target_relative: Per-cell brightness target as a fraction of
            ``input_flux``. The merit pulls every cell's mean-over-FOV
            brightness toward ``target_relative · input_flux``.
        d65_weights: ``(K,)`` target color ratios in radiance space.
        luminance_weights: ``(K,)`` per-wavelength weights for the
            brightness sum. ``None`` → ``ones`` (radiometric brightness).
        shape_floor_epsilon: Floor on the shape denominator, as a
            fraction of the brightest cell's ``mean_radiance²``. Prevents
            divide-by-zero on dark cells. Cells dimmer than
            ``√epsilon · max_cell`` contribute weakly to shape error.
        weight_target: Weight for the target (squared-error-to-target)
            term.
        weight_shape: Weight for the shape (D65 + FOV uniformity) term.
        asymmetric_target: When True, the target term penalizes only
            cells that fall *below* ``target_relative · input_flux``;
            cells above target contribute zero. Useful when the design
            is brightness-bound and there's no reason to drag the
            brightest cells back down to match the dim corners — only
            the dim cells need pulling up.
    """

    target_relative: float = 0.05
    d65_weights: jnp.ndarray = None
    luminance_weights: jnp.ndarray | None = None
    shape_floor_epsilon: float = 1e-3
    weight_target: float = 1.0
    weight_shape: float = 1.0
    asymmetric_target: bool = False

    def __post_init__(self):
        if self.d65_weights is None:
            self.d65_weights = D65_WEIGHTS


def _compute_terms(
    response: jnp.ndarray,
    input_flux: float,
    config: PupilMeritConfig,
    cell_mask: jnp.ndarray | None,
) -> dict:
    num_channels = response.shape[-1]
    d65_weights = config.d65_weights.reshape(1, 1, num_channels)

    if config.luminance_weights is None:
        luminance_weights = jnp.ones((num_channels,))
    else:
        luminance_weights = jnp.asarray(config.luminance_weights)
    luminance_weights = luminance_weights.reshape(1, 1, num_channels)

    radiance_per_angle = jnp.sum(response, axis=-1)                     # (S, A)
    brightness_per_angle = jnp.sum(response * luminance_weights,
                                   axis=-1)                             # (S, A)
    mean_radiance = jnp.mean(radiance_per_angle, axis=-1)               # (S,)
    mean_brightness = jnp.mean(brightness_per_angle, axis=-1)           # (S,)

    if cell_mask is None:
        cell_mask = jnp.ones_like(mean_brightness)
    num_target_cells = jnp.sum(cell_mask) + 1e-8

    # Shape (radiance space, scale-invariant). The ε floor must be in
    # radiance units to match ``mean_radiance²`` — using ``input_flux``
    # would mix radiance and (V-weighted) luminance and silently disable
    # the term. Reference against the brightest cell so ε keeps its
    # intent ("dark cells contribute weakly to shape").
    d65_template = d65_weights * mean_radiance[:, None, None]           # (S,A,K)
    shape_diff = response - d65_template
    # Floor the *squared* reference (squaring tiny floats underflows in
    # float32). Guards a fully-zero response from producing 0/0 NaN that
    # would contaminate ``loss_shape`` even when ``weight_shape == 0``.
    radiance_reference_sq = jnp.maximum(jnp.max(mean_radiance) ** 2, 1e-30)
    shape_denom = (mean_radiance ** 2
                   + config.shape_floor_epsilon * radiance_reference_sq)
    shape_err_per_cell = (jnp.mean(shape_diff ** 2, axis=(1, 2))
                          / shape_denom)
    loss_shape = (jnp.sum(cell_mask * shape_err_per_cell)
                  / num_target_cells)

    # Target (squared error to ``target_relative · input_flux``, normalized
    # by the target itself so loss is O(1) — comparable to the shape term
    # when used together). When ``asymmetric_target`` is set, only the
    # under-target deficit is penalized — over-target cells get zero cost.
    brightness_target = config.target_relative * input_flux
    deviation = mean_brightness - brightness_target
    if config.asymmetric_target:
        deviation = jnp.minimum(deviation, 0.0)
    loss_target = (jnp.sum(cell_mask * deviation ** 2)
                   / num_target_cells / (brightness_target ** 2))

    total = (config.weight_target * loss_target
             + config.weight_shape * loss_shape)

    return {
        "shape": loss_shape,
        "target": loss_target,
        "total": total,
        "mean_brightness": mean_brightness,
        "cell_mask": cell_mask,
        "num_target_cells": num_target_cells,
    }


def pupil_merit(
    response: jnp.ndarray,
    input_flux: float,
    config: PupilMeritConfig | None = None,
    cell_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Scalar merit: lower is better. See module docstring."""
    if config is None:
        config = PupilMeritConfig()
    return _compute_terms(response, input_flux, config, cell_mask)["total"]


def merit_breakdown(
    response: jnp.ndarray,
    input_flux: float,
    config: PupilMeritConfig | None = None,
    cell_mask: jnp.ndarray | None = None,
) -> dict:
    """Return each merit term separately, plus diagnostics."""
    if config is None:
        config = PupilMeritConfig()
    terms = _compute_terms(response, input_flux, config, cell_mask)
    masked_brightness = jnp.where(terms["cell_mask"] > 0,
                                  terms["mean_brightness"], jnp.nan)
    return {
        "shape": terms["shape"],
        "target": terms["target"],
        "total": terms["total"],
        "min_brightness_rel": (jnp.min(
            jnp.where(terms["cell_mask"] > 0,
                      terms["mean_brightness"], jnp.inf))
            / input_flux),
        "max_brightness_rel": (jnp.max(
            jnp.where(terms["cell_mask"] > 0,
                      terms["mean_brightness"], -jnp.inf))
            / input_flux),
        "mean_brightness_rel": jnp.nanmean(masked_brightness) / input_flux,
        "brightness_std_rel": jnp.nanstd(masked_brightness) / input_flux,
    }
