"""Pupil merit for combiner optimization — shape + threshold formulation.

This merit evaluates a combiner design against three decoupled goals:

1. **Coverage** — every cell in a pre-defined target eyebox region on the
   pupil plane should receive total brightness **above a threshold**
   ``I_thresh``. The threshold is expressed relative to the projector's
   input flux, so it is invariant to the number of rays traced.

2. **Shape (D65 + FOV uniformity)** — at every eyebox cell, the angular
   response should be (a) flat across the FOV and (b) D65 white-balanced
   across R/G/B. Both are captured by a single scale-invariant squared
   error against the template ``D65[c] · Ī(s)``, where ``Ī(s)`` is the
   cell's own mean brightness. Different cells may sit at different
   absolute brightness levels — only their *shape* is compared.

3. **Efficiency (optional soft cap)** — an optional soft upper bound
   ``I_cap`` discourages wasting projector light once a cell is already
   well above threshold.

The variables the merit is designed to be differentiated against are the
per-mirror **Gaussian reflectance parameters** (amplitude, width per
color) and the **inter-mirror spacings**. Both are passed through the
apollo14 tracer, which then produces the ``(S, A, 3)`` response tensor
this merit consumes. Nothing in this file assumes a specific variable
layout — it only operates on the response tensor plus a scalar input
flux reference.

## Tensor conventions

- ``response``  ``(S, A, 3)`` — intensity at pupil sample ``s``, FOV
  angle ``a``, color ``c``. Produced by
  ``helios.eyebox.compute_eyebox_response`` (or any compatible tracer).
- ``cell_mask`` ``(S,)``     — 1.0 for samples inside the pre-defined
  target eyebox region, 0.0 outside. Lets the merit ignore samples that
  are on the pupil plane but outside the region we're optimizing for.
  Pass ``None`` to use all samples.
- ``I_in``      scalar       — reference "input brightness": the total
  radiometric flux one FOV direction delivers into the combiner from the
  projector, *before* any combiner losses. All thresholds are relative
  to this so the merit is invariant to ray-count changes.

## Math

Per-sample total brightness and mean brightness across angles::

    I(s, a) = Σ_c response[s, a, c]                       # (S, A)
    Ī(s)    = mean_a I(s, a)                              # (S,)

**Shape error** — scale-invariant L2 against the D65-flat template::

    err(s) = mean_{a,c} ( response[s,a,c] − D65[c]·Ī(s) )² / (Ī(s)² + ε·I_in²)

The ``ε·I_in²`` floor prevents a dark cell from blowing up the gradient
when ``Ī(s) ≈ 0``. Without it, the optimizer would see an almost-zero
denominator during warm-up.

**Coverage** — soft indicator that ``Ī(s)`` exceeds the threshold, then
a soft-min over samples so the *weakest* cell dominates the gradient
(that's the one actually limiting the usable eyebox)::

    above(s) = σ( k · (Ī(s) − τ·I_in) / (τ·I_in) )        # (S,)  in (0,1)
    soft_min = −(1/β) · log( mean_s exp(−β · above(s)) )  # scalar in (0,1)
    L_cov    = 1 − soft_min

Soft-min (rather than plain ``mean``) matches the physical reality that
"the eyebox is only as big as its weakest cell". When every cell is
safely above threshold, ``L_cov → 0``.

**Warm-up** — a relu-style hinge that keeps a non-zero gradient while
``above(s)`` is still saturated at 0 for some cells::

    L_warm = mean_s relu(τ · I_in − Ī(s))² / I_in²

This term is what pulls cells up out of the "dead zone" during the first
few optimizer steps. Its weight should be small; it's scaffolding.

**Optional soft cap**::

    L_cap = mean_s relu(Ī(s) − κ · I_in)² / I_in²         # κ > τ

Set ``w_cap = 0.0`` to disable it.

**Total merit**::

    L = w_shape · mean_s ( above(s) · err(s) )            # shape where usable
      + w_cov   · L_cov                                   # fill the eyebox
      + w_warm  · L_warm                                  # warm-up gradient
      + w_cap   · L_cap                                   # optional efficiency

Multiplying ``err(s)`` by ``above(s)`` means the optimizer ignores
whitening of cells that aren't usable yet — they only need to rise. Once
they cross the threshold, shape starts to matter. This gating is what
makes the two objectives ("fill it" and "make it D65") cooperate
instead of fighting.

## Gradient properties

- ``err`` — always differentiable; the ε floor keeps it bounded.
- ``L_cov`` via soft-min — provides strong gradient on the worst cell
  even when the mean over cells is high.
- ``L_warm`` — provides linear gradient to any cell still below
  threshold, in the regime where ``above(s)`` is saturated at 0.
- Together, every cell and every color has a non-vanishing gradient
  path back to the variables, whether the current design is near or
  far from a solution.

## Usage

>>> from helios.pupil_merit import PupilMeritConfig, pupil_merit
>>> response, _ = compute_eyebox_response(...)            # (S, A, 3)
>>> cfg = PupilMeritConfig(threshold_rel=0.05, cap_rel=0.15)
>>> loss = pupil_merit(response, I_in=1.0, config=cfg)    # scalar
>>> grad = jax.grad(lambda p: pupil_merit(trace(p), 1.0, cfg))(params)
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from helios.merit import D65_WEIGHTS

# ── Configuration ───────────────────────────────────────────────────────────


@dataclass
class PupilMeritConfig:
    """Hyperparameters for :func:`pupil_merit`.

    All thresholds are **relative to the reference input flux** ``I_in``
    so the merit is invariant to the number of rays traced.

    Brightness can be either radiometric (default — sum of per-wavelength
    response values) or **photometric** (V(λ)-weighted sum, proportional
    to luminance). Set ``luminance_weights`` to switch:

    - ``None`` → radiometric: every wavelength contributes equally per
      watt. ``input_flux`` should be ``N_rays × Σ_λ spectrum(λ)``.
    - ``photopic_v(trace_wavelengths) × Δλ × K_m`` (or any other per-λ
      weighting) → photometric: blue counts ~3% of green per watt, etc.
      ``input_flux`` should be ``N_rays × Σ_λ spectrum(λ) × weights[λ]``.

    The shape term is **always** computed in radiance space — its template
    ``D65 × cell_radiance_scale`` must share units with ``response``,
    which is radiance. Only the threshold-driven terms (coverage, warm-up,
    cap) consume the photometric scale.

    Attributes:
        threshold_relative: Minimum acceptable cell brightness as a
            fraction of ``input_flux``. A cell is "in the eyebox" when
            its mean-over-FOV brightness exceeds
            ``threshold_relative · input_flux``. Default 0.05 (5% of
            input flux reaches each eyebox cell).
        cap_relative: Optional upper bound as a fraction of
            ``input_flux``. Set to ``None`` or a very large number to
            disable. Default ``None``.
        d65_weights: ``(K,)`` target color ratios in radiance space.
            Default: D65 weights at the standard R/G/B wavelengths from
            :mod:`helios.merit`.
        luminance_weights: ``(K,)`` per-wavelength weights for the
            brightness sum. ``None`` → ``ones`` (radiometric brightness,
            current default). For photometric brightness, pass
            ``helios.photometry.luminance_weights(trace_wavelengths)``.
        sigmoid_steepness: Steepness of the "above threshold" sigmoid,
            in units of *fractional excess over threshold* — i.e., the
            sigmoid argument is ``k · (mean − threshold) / threshold``,
            so ``k=10`` means a 10% over/undershoot saturates near 0 / 1.
            Larger → sharper threshold, noisier gradient. Default 50.0.
        soft_min_temperature: Temperature for the soft-min over cells in
            the coverage term. Larger → closer to a hard min (focus on
            the worst cell). Default 20.0.
        shape_floor_epsilon: Floor added to the per-cell denominator in
            the shape error, as a fraction of ``input_flux²``. Prevents
            divide-by-zero on dark cells during warm-up. Default 1e-3.
        weight_shape: Weight for the shape (D65 + FOV uniformity) term.
        weight_coverage: Weight for the coverage (soft-min above
            threshold) term.
        weight_warmup: Weight for the warm-up hinge term. Small —
            scaffolding.
        weight_cap: Weight for the optional upper-bound cap term.
    """

    threshold_relative: float = 0.05
    cap_relative: float | None = None
    d65_weights: jnp.ndarray = None
    luminance_weights: jnp.ndarray | None = None
    sigmoid_steepness: float = 8.0
    soft_min_temperature: float = 20.0
    shape_floor_epsilon: float = 1e-3
    weight_shape: float = 1.0
    weight_coverage: float = 2.0
    weight_warmup: float = 0.5
    weight_cap: float = 0.0

    def __post_init__(self):
        if self.d65_weights is None:
            self.d65_weights = D65_WEIGHTS


# ── Merit ───────────────────────────────────────────────────────────────────


def _compute_terms(
    response: jnp.ndarray,
    input_flux: float,
    config: PupilMeritConfig,
    cell_mask: jnp.ndarray | None,
) -> dict:
    """Compute every loss term and diagnostic — shared by merit + breakdown.

    Two per-cell scales are computed:

    - ``mean_radiance`` — raw sum of ``response`` over wavelengths, mean
      over angles. Used by the shape term because the shape template
      ``D65 × scale`` must share units with ``response`` (radiance).
    - ``mean_brightness`` — V-weighted (or ones-weighted) sum, mean over
      angles. Used by the threshold-driven terms (coverage / warm-up /
      cap). ``input_flux`` must be expressed in the same units.

    When ``config.luminance_weights is None``, ``mean_brightness`` ≡
    ``mean_radiance`` and behavior is identical to the pre-photometric
    merit.
    """
    num_channels = response.shape[-1]
    d65_weights = config.d65_weights.reshape(1, 1, num_channels)

    if config.luminance_weights is None:
        luminance_weights = jnp.ones((num_channels,))
    else:
        luminance_weights = jnp.asarray(config.luminance_weights)
    luminance_weights = luminance_weights.reshape(1, 1, num_channels)

    # ── Per-cell scales ──
    radiance_per_angle = jnp.sum(response, axis=-1)                     # (S, A)
    brightness_per_angle = jnp.sum(response * luminance_weights,
                                   axis=-1)                             # (S, A)
    mean_radiance = jnp.mean(radiance_per_angle, axis=-1)               # (S,)
    mean_brightness = jnp.mean(brightness_per_angle, axis=-1)           # (S,)

    # ── Shape error: in radiance units (template uses mean_radiance) ──
    d65_template = d65_weights * mean_radiance[:, None, None]           # (S,A,K)
    shape_diff = response - d65_template
    shape_denom = (mean_radiance ** 2
                   + config.shape_floor_epsilon * (input_flux ** 2))
    shape_err_per_cell = (jnp.mean(shape_diff ** 2, axis=(1, 2))
                          / shape_denom)                                # (S,)

    # ── Above-threshold soft indicator (uses brightness scale) ──
    # Sigmoid argument is normalized by ``brightness_threshold`` (not
    # ``input_flux``), so ``sigmoid_steepness`` is the slope per unit
    # *fractional excess over threshold*. Without this rescaling, the
    # natural span of (mean − threshold) is ``threshold_relative · input_flux``,
    # which makes the sigmoid sit in its linear regime over the entire
    # useful range and effectively turns the coverage term into a linear
    # function of brightness instead of a threshold indicator.
    brightness_threshold = config.threshold_relative * input_flux
    above_threshold = jax.nn.sigmoid(
        config.sigmoid_steepness
        * (mean_brightness - brightness_threshold) / brightness_threshold
    )                                                                   # (S,)

    # ── Cell mask handling ──
    if cell_mask is None:
        cell_mask = jnp.ones_like(mean_brightness)
    num_target_cells = jnp.sum(cell_mask) + 1e-8

    # ── Shape term: gated on "cell is usable" ──
    loss_shape = (jnp.sum(cell_mask * above_threshold * shape_err_per_cell)
                  / num_target_cells)

    # ── Coverage via soft-min over masked cells ──
    # soft_min(above) = -(1/β) log( Σ mask·exp(-β·above) / Σ mask )
    # Focuses gradient on the *weakest* cell, which defines the eyebox
    # boundary. Masked-out cells contribute 0 to both sums.
    softmin_temp = config.soft_min_temperature
    cell_weights = cell_mask / num_target_cells
    soft_min_above = -(1.0 / softmin_temp) * jnp.log(
        jnp.sum(cell_weights * jnp.exp(-softmin_temp * above_threshold))
        + 1e-12
    )
    loss_coverage = 1.0 - soft_min_above

    # ── Warm-up hinge: linear gradient for cells still below threshold ──
    shortfall = jax.nn.relu(brightness_threshold - mean_brightness)
    loss_warmup = (jnp.sum(cell_mask * shortfall ** 2)
                   / num_target_cells / (input_flux ** 2))

    # ── Optional soft upper-bound cap ──
    if config.cap_relative is not None and config.weight_cap > 0.0:
        brightness_cap = config.cap_relative * input_flux
        overshoot = jax.nn.relu(mean_brightness - brightness_cap)
        loss_cap = (jnp.sum(cell_mask * overshoot ** 2)
                    / num_target_cells / (input_flux ** 2))
    else:
        loss_cap = jnp.array(0.0)

    total = (config.weight_shape * loss_shape
             + config.weight_coverage * loss_coverage
             + config.weight_warmup * loss_warmup
             + config.weight_cap * loss_cap)

    return {
        "shape": loss_shape,
        "coverage": loss_coverage,
        "warmup": loss_warmup,
        "cap": loss_cap,
        "total": total,
        "above_threshold": above_threshold,
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
    """Scalar merit: lower is better.

    Combines four terms as described in the module docstring:

    1. Shape error (D65 + FOV-flat, scale-invariant per cell), gated on
       whether the cell is already above threshold. Always in radiance
       units — independent of ``config.luminance_weights``.
    2. Coverage (soft-min-over-cells of a soft above-threshold indicator).
       Uses photometric brightness when ``config.luminance_weights`` is
       set, radiometric otherwise.
    3. Warm-up hinge to give dark cells a non-zero gradient. Same scale.
    4. Optional soft upper-bound cap (off by default). Same scale.

    Args:
        response: ``(S, A, K)`` tensor of intensities at ``S`` eyebox
            samples × ``A`` FOV angles × ``K`` wavelengths/colors.
        input_flux: Reference input flux. **Must use the same units as
            the brightness sum** controlled by ``config.luminance_weights``
            — i.e., either ``N_rays × Σ_λ spectrum(λ)`` (radiometric) or
            ``N_rays × Σ_λ spectrum(λ) × luminance_weights[λ]``
            (photometric). Mismatched units silently miscalibrate the
            threshold/cap percentages.
        config: :class:`PupilMeritConfig` (defaults used if ``None``).
        cell_mask: Optional ``(S,)`` mask of 1/0 selecting which samples
            belong to the target eyebox region. When ``None``, all
            samples contribute.

    Returns:
        Scalar JAX array. Reaches 0 when every target-eyebox cell is
        above threshold, perfectly FOV-flat, and perfectly D65 balanced.
    """
    if config is None:
        config = PupilMeritConfig()
    return _compute_terms(response, input_flux, config, cell_mask)["total"]


# ── Diagnostics ─────────────────────────────────────────────────────────────


def merit_breakdown(
    response: jnp.ndarray,
    input_flux: float,
    config: PupilMeritConfig | None = None,
    cell_mask: jnp.ndarray | None = None,
) -> dict:
    """Return each merit term separately for logging and debugging.

    Useful during optimization to see which term is dominating — if
    coverage loss is stuck but shape loss is already small, the optimizer
    is "painting" instead of "spreading", and you may want to raise
    ``weight_coverage`` or lower ``threshold_relative``.

    Returns:
        Dict with keys ``shape``, ``coverage``, ``warmup``, ``cap``,
        ``total``, ``active_fraction`` (plain mean of above-threshold
        indicator, for human-readable reporting), and
        ``min_brightness_rel``.
    """
    if config is None:
        config = PupilMeritConfig()
    terms = _compute_terms(response, input_flux, config, cell_mask)
    return {
        "shape": terms["shape"],
        "coverage": terms["coverage"],
        "warmup": terms["warmup"],
        "cap": terms["cap"],
        "total": terms["total"],
        "active_fraction": (jnp.sum(terms["cell_mask"] * terms["above_threshold"])
                            / terms["num_target_cells"]),
        "min_brightness_rel": (jnp.min(
            jnp.where(terms["cell_mask"] > 0,
                      terms["mean_brightness"], jnp.inf))
            / input_flux),
    }
