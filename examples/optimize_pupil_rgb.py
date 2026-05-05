"""Pupil optimization driver — single broadband W projector with a
threshold-based wavelength sampling and a spectrum-preserving merit.

How this differs from :mod:`examples.optimize_pupil`
----------------------------------------------------
The original broadband driver compared the per-wavelength response to a
*continuous D65 reference SPD*. That target is unsatisfiable for a 3-band
micro-LED whose W spectrum has deep valleys between R/G/B peaks: the merit
penalizes the design for not emitting at wavelengths the projector itself
can't supply, and phase 2 plateaus at ~0.5.

This driver fixes that by changing the merit's shape target. Instead of
"deliver D65", it asks the design to "preserve the projector's spectrum"
— i.e. ``response(s, a, λ) ∝ W(λ)`` at every (cell, FOV). Since the
panel's W column **is** the manufacturer's calibrated D65 white, matching
the projector's shape is equivalent to producing D65 at the eye, but it
is satisfiable: at valley wavelengths both the response and the shape
target are small, so shape_diff naturally vanishes. The mirrors only need
to apply approximately wavelength-uniform reflectance to satisfy the
shape term — a target that lives inside the design space.

Wavelength sampling
-------------------
Tracing wavelengths span ``W``'s above-``SPECTRAL_THRESHOLD`` band, sampled
uniformly. ``SPECTRAL_THRESHOLD = 0.10`` is the recommended default — it
trims the dead zones in the deep blue/red wings while keeping the three
LED peaks contiguous, so ``mean(diff)`` gives the correct Δλ for the
report's V(λ) integration. ``0.05`` keeps essentially the whole spectrum
including valleys at a slightly higher loss floor; ``0.15`` splits the B
sub-pixel into two parts (its measured spectrum has a dual peak), avoid.

Run::

    python examples/optimize_pupil_rgb.py
"""

from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp

from apollo14.units import mm, nm, deg
from apollo14.combiner import (
    DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
)
from apollo14.elements.pupil import RectangularPupil
from apollo14.geometry import planar_grid_points
from apollo14.projector import PlayNitrideLed, FovGrid
from apollo14.spectral import SumOfGaussiansCurve

from helios.combiner_params import (
    CombinerParams, ParamBounds, build_parametrized_system, NUM_MIRRORS,
    fwhm_to_sigma,
)
from apollo14.trace import prepare_route
from helios.merit import build_combiner_branch_routes
from helios.eyebox import trace_branch_over_fov
from helios.photometry import luminance_weights as photopic_luminance_weights
from helios.pupil_merit import (
    PupilMeritConfig, pupil_merit, merit_breakdown,
)
from helios.adam import AdamConfig, adam_init, adam_step
from helios.io import save_optimization_report, save_run, ScanConfig
from helios.reports.pupil_report import render_pupil_report
from helios.jax_cache import enable_jax_compilation_cache

# Persistent JIT cache survives across runs so successive optimizations
# skip the multi-second compile. Cache dir auto-detects from
# ``$XDG_CACHE_HOME`` or ``~/.cache``, so the script is portable.
enable_jax_compilation_cache()

# ── Eyebox target region (pre-defined, fixed) ───────────────────────────────

EYEBOX_HALF_X = 4.0 * mm         # 8 mm full width on x
EYEBOX_HALF_Y = 5.0 * mm         # 10 mm full width on y
EYEBOX_NX, EYEBOX_NY = 8, 10   # 80 cells, exactly 1×1 mm each (cell-centered)

X_FOV = 4.0 * deg
Y_FOV = 4.0 * deg

# ── Single broadband projector (panel's calibrated white) ─────────────────

PROJECTOR_NX, PROJECTOR_NY = 50, 10
# PROJECTOR_NX, PROJECTOR_NY = 25, 5
ANGULAR_STEPS_X, ANGULAR_STEPS_Y = 8, 8

PROJECTOR = PlayNitrideLed.create_broadband(
    position=DEFAULT_LIGHT_POSITION, direction=DEFAULT_LIGHT_DIRECTION,
    beam_width=10.0 * mm, beam_height=2.0 * mm,
    nx=PROJECTOR_NX, ny=PROJECTOR_NY,
)

# ── Wavelength sampling ─────────────────────────────────────────────────────
# Span W's above-threshold band uniformly. Threshold 0.10 gives a clean
# three-peak envelope without splitting any LED's measured spectrum, and
# excludes ~7 % of the panel's luminance — recoverable post-build via the
# panel's RGB drive trim.

SPECTRAL_THRESHOLD = 0.10
SPECTRAL_SAMPLES = 100

_w_lo, _w_hi = PROJECTOR.spectral_band(threshold=SPECTRAL_THRESHOLD)
TRACE_WAVELENGTHS = jnp.linspace(_w_lo, _w_hi, SPECTRAL_SAMPLES)

# ── Spectrum-preserving shape target ───────────────────────────────────────
# The merit's ``d65_weights`` should be the per-wavelength shape the
# response should match. Using the projector's own spectrum (peak-
# normalized, then re-normalized to sum=1 across the trace samples) makes
# the shape term ask the design to *preserve* the projector's emission
# shape — which is already calibrated to D65. Result: at valley
# wavelengths both response and target are small, shape_diff stays
# bounded, and the merit is satisfiable.

_spec_wls, _spec_rad = PROJECTOR.spectrum
_W_AT_TRACE = jnp.interp(TRACE_WAVELENGTHS, _spec_wls, _spec_rad)
SHAPE_TARGET = _W_AT_TRACE / _W_AT_TRACE.sum()

# Photopic V·Δλ·K_m weights at the trace wavelengths (uniform spacing,
# so ``mean(diff)`` inside ``photopic_luminance_weights`` is the actual
# per-sample Δλ — the report's report-time V-weighting matches).
LUMINANCE_TRACE_WEIGHTS = photopic_luminance_weights(TRACE_WAVELENGTHS)

# ── Per-cell brightness target ─────────────────────────────────────────────

NUM_EYEBOX_CELLS = EYEBOX_NX * EYEBOX_NY
EYEBOX_TARGET = 0.08
PER_CELL_TARGET = EYEBOX_TARGET / NUM_EYEBOX_CELLS


# ── Merit & tracer configuration ────────────────────────────────────────────

FOV_GRID = FovGrid(DEFAULT_LIGHT_DIRECTION, X_FOV, Y_FOV, num_x=ANGULAR_STEPS_X, num_y=ANGULAR_STEPS_Y)

# Phase 1: pull every cell to the brightness target *exactly* — symmetric
# squared error, so over-target cells get pulled down and under-target cells
# get pulled up. With a tight target this keeps the optimizer working on
# tightening the cell-to-cell spread instead of stopping the moment all cells
# clear the threshold. Shape term is off — color balance is phase 2's job.
merit_cfg_phase1 = PupilMeritConfig(
    target_relative=PER_CELL_TARGET,
    d65_weights=SHAPE_TARGET,
    luminance_weights=LUMINANCE_TRACE_WEIGHTS,
    weight_target=1.0,
    weight_shape=0.0,
    asymmetric_target=False,
)

# Phase 2: hold every cell *at or above* target (asymmetric — over-target is
# fine) while polishing spectrum-preservation across (cell, FOV, λ).
merit_cfg_phase2 = PupilMeritConfig(
    target_relative=PER_CELL_TARGET,
    d65_weights=SHAPE_TARGET,
    luminance_weights=LUMINANCE_TRACE_WEIGHTS,
    weight_target=1.0,
    weight_shape=1.0,
    asymmetric_target=False,
)

bounds = ParamBounds(amplitude_max=0.20, fwhm_max_nm=120, fwhm_min_nm=10)

# When False, mirror inter-spacing is held at its initial value and only
# the Gaussian reflectance curves are tuned. Hard nearest-neighbor binning
# is then sufficient — see optimize_pupil.py for the reasoning.
OPTIMIZE_SPACINGS = False


# ── Reference input flux (photometric, matches luminance_weights) ──────────

NUM_RAYS = PROJECTOR_NX * PROJECTOR_NY
INPUT_FLUX = float(NUM_RAYS * jnp.sum(_W_AT_TRACE * LUMINANCE_TRACE_WEIGHTS))

# ── Eye-pupil moving-window aggregation ───────────────────────────────────
# Each "merit cell" represents what a real eye-pupil (~3 mm diameter) would
# see if positioned at that point in the eyebox. Implemented as a 2-D mean
# convolution of the fine-grid binned response with a (K × K)-cell box
# kernel — physically meaningful, and provides spatial smoothness without
# soft binning.
#
# The fine sample grid is enlarged by ``KERNEL_SIZE_CELLS // 2`` cells on
# every side so that ``mode="valid"`` convolution outputs *exactly* the
# original (EYEBOX_NY, EYEBOX_NX) shape — no boundary truncation, every
# eyebox cell sees full kernel support.
#
# Setting ``KERNEL_SIZE_CELLS = 1`` falls back to point sampling (no
# window).

KERNEL_SIZE_CELLS = 3
PADDING_CELLS = KERNEL_SIZE_CELLS // 2

_ref_system = build_parametrized_system(
    CombinerParams.initial(), probe_wavelengths=TRACE_WAVELENGTHS)
_pupil = next(e for e in _ref_system.elements if isinstance(e, RectangularPupil))

# Cell-centered convention: NX × NY cells of equal size exactly tile
# the half-extent box without overhang. Pitch is ``2·half / N``.
_CELL_PITCH_X = 2 * EYEBOX_HALF_X / EYEBOX_NX
_CELL_PITCH_Y = 2 * EYEBOX_HALF_Y / EYEBOX_NY

SAMPLE_HALF_X = EYEBOX_HALF_X + PADDING_CELLS * _CELL_PITCH_X
SAMPLE_HALF_Y = EYEBOX_HALF_Y + PADDING_CELLS * _CELL_PITCH_Y
SAMPLE_NX = EYEBOX_NX + 2 * PADDING_CELLS
SAMPLE_NY = EYEBOX_NY + 2 * PADDING_CELLS

# The points the tracer bins onto live on the *sample* grid (eyebox + 1
# padding cell on every side, cell-centered). After binning we convolve
# with the (KERNEL × KERNEL) mean kernel and the result lives on the
# inner (EYEBOX_NY, EYEBOX_NX) eyebox grid — that's what the merit
# consumes. ``cell_centered=True`` puts every grid point at the center
# of a 1 × 1 mm cell, so the EYEBOX_NX × EYEBOX_NY merit grid exactly
# tiles the nominal eyebox without ½-cell overhang at the boundary.
EYEBOX_POINTS = planar_grid_points(
    _pupil.position, _pupil.normal,
    SAMPLE_HALF_X, SAMPLE_HALF_Y, SAMPLE_NX, SAMPLE_NY,
    cell_centered=True,
)   # (SAMPLE_NX * SAMPLE_NY, 3)
CELL_MASK = jnp.ones(EYEBOX_NX * EYEBOX_NY)

# σ ≈ ½ the smaller fine-cell pitch — for soft binning, when used. With
# the moving-window aggregation in place soft binning is redundant for
# spatial smoothness, but it's still the only path to spacing gradients.
BINNING_SIGMA = 0.5 * min(_CELL_PITCH_X, _CELL_PITCH_Y)


def _window_mean(arr: jnp.ndarray, ny: int, nx: int,
                 kernel_size: int) -> jnp.ndarray:
    """Apply a (kernel × kernel) mean filter along the leading spatial axis.

    Args:
        arr: ``(ny * nx, …)`` row-major flat spatial axis. Trailing dims are
            preserved untouched.
        ny, nx: shape of the spatial axis before flattening.
        kernel_size: side length of the square mean kernel, in cells.

    Returns:
        ``((ny - K + 1) * (nx - K + 1), …)`` — same trailing dims, the
        spatial axis convolved with ``mode="valid"``.
    """
    if kernel_size <= 1:
        return arr
    other_shape = arr.shape[1:]
    arr_2d = arr.reshape(ny, nx, *other_shape)
    window_dims = (kernel_size, kernel_size) + (1,) * len(other_shape)
    strides = (1, 1) + (1,) * len(other_shape)
    summed = jax.lax.reduce_window(
        arr_2d, 0.0, jax.lax.add, window_dims, strides, "VALID")
    averaged = summed / (kernel_size * kernel_size)
    out_ny = ny - kernel_size + 1
    out_nx = nx - kernel_size + 1
    return averaged.reshape(out_ny * out_nx, *other_shape)


# ── Loss function ───────────────────────────────────────────────────────────


def _compute_spectral_response(params: CombinerParams) -> jnp.ndarray:
    """Trace the W projector at each wavelength, scanned over wavelengths.

    Returns ``(S, A, N)`` per-wavelength radiance — same shape the merit
    and the report both expect.
    """
    system = build_parametrized_system(
        params, probe_wavelengths=TRACE_WAVELENGTHS)
    branch_routes = build_combiner_branch_routes(
        system, num_mirrors=NUM_MIRRORS,
    )
    directions = FOV_GRID.flat_directions  # (A, 3)

    def trace_one_wavelength(_, wavelength):
        binned = jnp.zeros((directions.shape[0], EYEBOX_POINTS.shape[0]))
        for route in branch_routes:
            prepared = prepare_route(route, wavelength)
            binned = binned + trace_branch_over_fov(
                prepared, PROJECTOR, EYEBOX_POINTS, wavelength,
                directions,
                sigma=BINNING_SIGMA if OPTIMIZE_SPACINGS else None,
                vmap_directions=True)  # (A, S_sample); A=64 fits comfortably
        # (S_sample, A) → moving-window mean → (S_eyebox, A). Windowing
        # *inside* the scan body keeps the per-iteration activation tape
        # at eyebox size rather than sample size — the convolution is
        # cheap and the memory saving carries through ``jax.checkpoint``
        # below.
        windowed = _window_mean(
            binned.T, SAMPLE_NY, SAMPLE_NX, KERNEL_SIZE_CELLS)
        return None, windowed  # (S_eyebox, A)

    # ``jax.checkpoint`` rematerializes each wavelength's forward during
    # backward instead of saving the full per-(wavelength, direction, ray,
    # cell) activation tape — without it value_and_grad needs ~200 GB.
    _, all_responses = jax.lax.scan(
        jax.checkpoint(trace_one_wavelength), None, TRACE_WAVELENGTHS)  # (N, S_eyebox, A)
    return jnp.transpose(all_responses, (1, 2, 0))  # (S_eyebox, A, N)


def loss_fn_phase1(params: CombinerParams) -> jnp.ndarray:
    response = _compute_spectral_response(params)
    return pupil_merit(response, INPUT_FLUX, merit_cfg_phase1, cell_mask=CELL_MASK)


def loss_fn_phase2(params: CombinerParams) -> jnp.ndarray:
    response = _compute_spectral_response(params)
    return pupil_merit(response, INPUT_FLUX, merit_cfg_phase2, cell_mask=CELL_MASK)


def breakdown_fn(params: CombinerParams, merit_cfg: PupilMeritConfig) -> dict:
    response = _compute_spectral_response(params)
    return merit_breakdown(response, INPUT_FLUX, merit_cfg, cell_mask=CELL_MASK)


value_and_grad_phase1 = jax.jit(jax.value_and_grad(loss_fn_phase1))
value_and_grad_phase2 = jax.jit(jax.value_and_grad(loss_fn_phase2))


def _mask_frozen(grad: CombinerParams) -> CombinerParams:
    """Zero out gradient on frozen design variables before the Adam update."""
    if OPTIMIZE_SPACINGS:
        return grad
    return grad._replace(spacings=jnp.zeros_like(grad.spacings))


# ── Adam optimizer ──────────────────────────────────────────────────────────

PHASE1_STEPS = 100
PHASE2_STEPS = 100

adam_cfg_phase1 = AdamConfig(peak_lr=3e-3, warmup_steps=20, num_steps=PHASE1_STEPS)
# Phase 2 enters with the target term already mostly satisfied, so the
# optimizer is polishing the shape term in a flat region. Adam's
# 1/sqrt(variance) normalization makes the effective step ~ lr·sign(grad)
# there — at peak_lr=2e-3 the trajectory dithers instead of descending.
# Dropping by 4× lets the same number of steps actually polish residue.
adam_cfg_phase2 = AdamConfig(peak_lr=5e-4, warmup_steps=10, num_steps=PHASE2_STEPS)


# ── Run ─────────────────────────────────────────────────────────────────────

def _print_breakdown(label, bd):
    print(f"\n{label}: {float(bd['total']):.5f}")
    print(f"  target={float(bd['target']):.5f}  "
          f"shape={float(bd['shape']):.5f}")
    print(f"  brightness mean_rel={float(bd['mean_brightness_rel']):.5f}  "
          f"std_rel={float(bd['brightness_std_rel']):.5f}  "
          f"min_rel={float(bd['min_brightness_rel']):.5f}  "
          f"max_rel={float(bd['max_brightness_rel']):.5f}")


RUNS_ROOT = Path("examples/reports/optimize_pupil_rgb")


def main():
    run_dir = RUNS_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    initial_params = CombinerParams.initial(amplitude=0.10, width_nm=15)
    params = initial_params
    state = adam_init(params)

    optimized_label = ("spacings + Gaussian reflectance" if OPTIMIZE_SPACINGS
                       else "Gaussian reflectance (spacings frozen)")
    num_vars = params.curves.amplitude.size + params.curves.sigma.size
    if OPTIMIZE_SPACINGS:
        num_vars += params.spacings.size
    print(f"── Pupil optimization ({optimized_label}) ──")
    print(f"Variables: {num_vars}")
    window_mm_x = KERNEL_SIZE_CELLS * float(_CELL_PITCH_X) / mm
    window_mm_y = KERNEL_SIZE_CELLS * float(_CELL_PITCH_Y) / mm
    print(f"Pupil window: {KERNEL_SIZE_CELLS}×{KERNEL_SIZE_CELLS} cells "
          f"({window_mm_x:.2f}×{window_mm_y:.2f} mm) — "
          f"sample grid {SAMPLE_NX}×{SAMPLE_NY}, eyebox grid {EYEBOX_NX}×{EYEBOX_NY}")
    print(f"Eyebox:    {2*EYEBOX_HALF_X/mm:.1f}×{2*EYEBOX_HALF_Y/mm:.1f} mm, "
          f"{EYEBOX_NX}×{EYEBOX_NY} cells")
    print(f"FOV:       ±{X_FOV/deg:.1f}° × ±{Y_FOV/deg:.1f}°, "
          f"{FOV_GRID.num_x}×{FOV_GRID.num_y} samples")
    print(f"Spectrum:  {SPECTRAL_SAMPLES} uniform samples, "
          f"{float(_w_lo)/nm:.0f}–{float(_w_hi)/nm:.0f} nm "
          f"(W > {SPECTRAL_THRESHOLD:.0%} of peak)")
    print(f"Shape target: projector W spectrum (preserve panel's D65 white)")
    print(f"I_in:      {INPUT_FLUX:.1f}  "
          f"(target_relative={merit_cfg_phase1.target_relative})")

    # Ceiling diagnostic — what brightness can we hope for?
    ceiling_curves = SumOfGaussiansCurve(
        amplitude=jnp.full_like(initial_params.curves.amplitude,
                                 bounds.amplitude_max),
        sigma=jnp.full_like(initial_params.curves.sigma,
                             fwhm_to_sigma(bounds.fwhm_max_nm * nm)),
        centers=initial_params.curves.centers,
    )
    ceiling_params = CombinerParams(
        spacings=initial_params.spacings,
        curves=ceiling_curves,
    )
    ceiling_response = _compute_spectral_response(ceiling_params)
    ceiling_lum_per_angle = jnp.sum(
        ceiling_response * LUMINANCE_TRACE_WEIGHTS.reshape(1, 1, -1), axis=-1)
    ceiling_brightness = jnp.mean(ceiling_lum_per_angle, axis=-1) / INPUT_FLUX
    print(f"Ceiling per-cell brightness (amp=max, FWHM=max): "
          f"min={float(ceiling_brightness.min()):.5f}  "
          f"mean={float(ceiling_brightness.mean()):.5f}  "
          f"max={float(ceiling_brightness.max()):.5f}  "
          f"(target={PER_CELL_TARGET})")

    initial_breakdown = breakdown_fn(params, merit_cfg_phase1)
    _print_breakdown("Initial merit (phase 1 weights)", initial_breakdown)

    loss_history = []

    print("\n── Phase 1: target-focused (drive every cell to target) ──")
    for step in range(PHASE1_STEPS):
        loss, grad = value_and_grad_phase1(params)
        params, state = adam_step(params, _mask_frozen(grad), state,
                                  adam_cfg_phase1)
        params = bounds.clip(params)
        loss_history.append(float(loss))
        print(f"step {step+1:4d}/{PHASE1_STEPS}  loss={float(loss):.8f}")

    phase1_breakdown = breakdown_fn(params, merit_cfg_phase1)
    _print_breakdown("Phase 1 result", phase1_breakdown)

    print("\n── Phase 2: target + spectrum-preserving shape ──")
    state = adam_init(params)
    for step in range(PHASE2_STEPS):
        loss, grad = value_and_grad_phase2(params)
        params, state = adam_step(params, _mask_frozen(grad), state,
                                  adam_cfg_phase2)
        params = bounds.clip(params)
        loss_history.append(float(loss))
        print(f"step {step+1:4d}/{PHASE2_STEPS}  loss={float(loss):.8f}")

    final_breakdown = breakdown_fn(params, merit_cfg_phase2)
    _print_breakdown("Final merit", final_breakdown)

    print("\nFinal spacings (mm):",
          [f"{float(spacing)/mm:.3f}" for spacing in params.spacings])
    print("Final amplitudes per mirror (per basis):")
    for mirror_idx in range(NUM_MIRRORS):
        mirror_amplitude = params.curves.amplitude[mirror_idx]
        row = "  ".join(f"{float(a):.4f}" for a in mirror_amplitude)
        print(f"  m{mirror_idx}: {row}")
    print("Final widths per mirror (nm):")
    for mirror_idx in range(NUM_MIRRORS):
        mirror_width = params.curves.sigma[mirror_idx]
        row = "  ".join(f"{float(w) / nm:.1f}" for w in mirror_width)
        print(f"  m{mirror_idx}: {row}")

    # Bounds-pegged diagnostic — surfaces whether the optimizer is stalled
    # because a parameter is sitting at its upper bound (the bound is
    # binding, not a true flat region of the loss).
    sigma_max = fwhm_to_sigma(bounds.fwhm_max_nm * nm)
    amp_pegged = int(jnp.sum(
        params.curves.amplitude >= bounds.amplitude_max - 1e-4))
    sigma_pegged = int(jnp.sum(params.curves.sigma >= sigma_max - 1e-4))
    spacing_max_mm = bounds.spacing_max_mm * mm
    spacing_min_mm = bounds.spacing_min_mm * mm
    spacing_pegged_hi = int(jnp.sum(params.spacings >= spacing_max_mm - 1e-4))
    spacing_pegged_lo = int(jnp.sum(params.spacings <= spacing_min_mm + 1e-4))
    print(f"\nParameters at upper bound:")
    print(f"  amplitudes: {amp_pegged} / {params.curves.amplitude.size} "
          f"(at {bounds.amplitude_max})")
    print(f"  sigmas:     {sigma_pegged} / {params.curves.sigma.size} "
          f"(at FWHM {bounds.fwhm_max_nm:.1f} nm)")
    print(f"  spacings:   {spacing_pegged_lo} at lower / "
          f"{spacing_pegged_hi} at upper "
          f"({bounds.spacing_min_mm}–{bounds.spacing_max_mm} mm)")

    response = _compute_spectral_response(params)
    luminance_per_angle = jnp.sum(
        response * LUMINANCE_TRACE_WEIGHTS.reshape(1, 1, -1), axis=-1)  # (S, A)
    mean_luminance = jnp.mean(luminance_per_angle, axis=-1)             # (S,)
    relative_brightness = mean_luminance / INPUT_FLUX

    grid = relative_brightness.reshape(EYEBOX_NY, EYEBOX_NX)
    print(f"\nEyebox brightness map (relative to input flux, target={PER_CELL_TARGET}):")
    for row in grid:
        print("  " + "  ".join(f"{float(v):.4f}" for v in row))
    print(f"min={float(relative_brightness.min()):.5f}  "
          f"max={float(relative_brightness.max()):.5f}  "
          f"std={float(relative_brightness.std()):.5f}")

    final_system = build_parametrized_system(
        params, probe_wavelengths=TRACE_WAVELENGTHS)
    report_path = save_optimization_report(
        run_dir,
        system=final_system,
        projectors=[PROJECTOR],
        fov_grid=FOV_GRID,
        merit_config=merit_cfg_phase2,
        optimizer_config={
            "algorithm": "adam_two_phase",
            "phase1": {
                "steps": PHASE1_STEPS,
                "peak_lr": adam_cfg_phase1.peak_lr,
                "warmup_steps": adam_cfg_phase1.warmup_steps,
                "focus": "target",
            },
            "phase2": {
                "steps": PHASE2_STEPS,
                "peak_lr": adam_cfg_phase2.peak_lr,
                "warmup_steps": adam_cfg_phase2.warmup_steps,
                "focus": "target+shape (spectrum-preserving)",
            },
            "schedule": "warmup_cosine_decay",
            "total_steps": PHASE1_STEPS + PHASE2_STEPS,
            "spectral_threshold": SPECTRAL_THRESHOLD,
            "spectral_samples": SPECTRAL_SAMPLES,
            "optimize_spacings": OPTIMIZE_SPACINGS,
        },
        param_bounds=bounds,
        initial_params=initial_params,
        final_params=params,
        initial_breakdown={k: float(v) for k, v in initial_breakdown.items()},
        final_breakdown={k: float(v) for k, v in final_breakdown.items()},
        loss_history=loss_history,
        eyebox_config={
            "half_x": EYEBOX_HALF_X,
            "half_y": EYEBOX_HALF_Y,
            "nx": EYEBOX_NX,
            "ny": EYEBOX_NY,
        },
    )
    print(f"\nSaved optimization report: {report_path}")

    # Cell-centered axes — match the merit's grid convention so the
    # report's heatmap renders cells of (EYEBOX_NX × EYEBOX_NY) tiling
    # exactly the [-EYEBOX_HALF, +EYEBOX_HALF] eyebox.
    pupil_x_mm = jnp.linspace(
        -EYEBOX_HALF_X + _CELL_PITCH_X / 2,
        EYEBOX_HALF_X - _CELL_PITCH_X / 2,
        EYEBOX_NX,
    )
    pupil_y_mm = jnp.linspace(
        -EYEBOX_HALF_Y + _CELL_PITCH_Y / 2,
        EYEBOX_HALF_Y - _CELL_PITCH_Y / 2,
        EYEBOX_NY,
    )
    scan_cfg = ScanConfig(
        base_direction=DEFAULT_LIGHT_DIRECTION,
        x_fov=float(X_FOV), y_fov=float(Y_FOV),
        num_x=FOV_GRID.num_x, num_y=FOV_GRID.num_y,
    )
    save_run(
        run_dir,
        final_system, PROJECTOR, scan_cfg,
        response=response,
        pupil_x_mm=pupil_x_mm,
        pupil_y_mm=pupil_y_mm,
        scan_angles=FOV_GRID.angles_grid,
        wavelengths_nm=TRACE_WAVELENGTHS / nm,
    )
    print(f"Saved run inputs + response to: {run_dir}")

    pupil_report_path = render_pupil_report(run_dir)
    print(f"Saved pupil report: {pupil_report_path}")


if __name__ == "__main__":
    main()
