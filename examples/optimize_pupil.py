"""Pupil optimization driver — spacings + Gaussian reflectance.

Optimizes the Talos combiner's 41 design variables (5 spacings + 36
Gaussian amplitude/width params) against :func:`helios.pupil_merit` so
that every cell of a pre-defined target eyebox region delivers the
same brightness (``target_relative`` of input flux), D65 white-balanced
and FOV-flat.

Two phases:

1. **Target** — pull every cell's brightness to the target.
2. **Target + shape** — keep brightness at target while polishing D65
   white balance and FOV uniformity.

Run::

    python examples/optimize_pupil.py
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
from apollo14.projector import PlayNitrideLed, FovGrid
from apollo14.spectral import SumOfGaussiansCurve

from helios.combiner_params import (
    CombinerParams, ParamBounds, build_parametrized_system, NUM_MIRRORS,
    fwhm_to_sigma,
)
from apollo14.trace import prepare_route
from helios.merit import build_combiner_branch_routes, d65_weights_at
from helios.eyebox import (
    trace_branch_over_fov, eyebox_grid_points,
)
from helios.photometry import luminance_weights as photopic_luminance_weights
from helios.pupil_merit import (
    PupilMeritConfig, pupil_merit, merit_breakdown,
)
from helios.adam import AdamConfig, adam_init, adam_step
from helios.io import save_optimization_report, save_run, ScanConfig
from helios.reports.pupil_report import render_pupil_report


# ── Eyebox target region (pre-defined, fixed) ───────────────────────────────

EYEBOX_RADIUS = 5.0 * mm         # 10×10 mm centered eyebox
EYEBOX_NX, EYEBOX_NY = 9, 9      # 49 cells, ~1.4 mm each

X_FOV = 8.0 * deg
Y_FOV = 8.0 * deg

# ── Broadband projector (PlayNitride micro-LED, combined R+G+B) ───────────

PROJECTOR_NX, PROJECTOR_NY = 100, 20

PROJECTOR = PlayNitrideLed.create_broadband(
    position=DEFAULT_LIGHT_POSITION, direction=DEFAULT_LIGHT_DIRECTION,
    beam_width=10.0 * mm, beam_height=2.0 * mm,
    nx=PROJECTOR_NX, ny=PROJECTOR_NY,
)

# ── Wavelength sampling (spectral band above 5% of peak) ─────────────────

SPECTRAL_THRESHOLD = 0.03
SPECTRAL_SAMPLES = 100

_wl_min, _wl_max = PROJECTOR.spectral_band(threshold=SPECTRAL_THRESHOLD)
# _wl_min, _wl_max = 400*nm, 700*nm
TRACE_WAVELENGTHS = jnp.linspace(_wl_min, _wl_max, SPECTRAL_SAMPLES)

# ── Continuous D65 weights at the traced wavelengths ──────────────────────

D65_TRACE_WEIGHTS = d65_weights_at(TRACE_WAVELENGTHS)

# ── Photopic V(λ)·Δλ·K_m weights at the traced wavelengths ─────────────────
# Photometric "brightness" — blue at 446 nm counts ~3% of green at 545 nm
# per watt. The merit's per-cell target becomes perceptually meaningful
# instead of radiometric.

LUMINANCE_TRACE_WEIGHTS = photopic_luminance_weights(TRACE_WAVELENGTHS)

# ── Per-cell brightness target ─────────────────────────────────────────────
# Fraction of input flux each of the 49 eyebox cells should receive.
# 0.002 × 49 ≈ 9.8% of input flux reaching the eyebox uniformly.

NUM_EYEBOX_CELLS = EYEBOX_NX * EYEBOX_NY    # 49
PER_CELL_TARGET = 0.002


# ── Merit & tracer configuration ────────────────────────────────────────────

FOV_GRID = FovGrid(DEFAULT_LIGHT_DIRECTION, X_FOV, Y_FOV, num_x=32, num_y=32)

# Phase 1: drive every cell toward the target brightness; ignore color.
merit_cfg_phase1 = PupilMeritConfig(
    target_relative=PER_CELL_TARGET,
    d65_weights=D65_TRACE_WEIGHTS,
    luminance_weights=LUMINANCE_TRACE_WEIGHTS,
    weight_target=1.0,
    weight_shape=0.0,
)

# Phase 2: keep cells at target while polishing D65 + FOV uniformity.
merit_cfg_phase2 = PupilMeritConfig(
    target_relative=PER_CELL_TARGET,
    d65_weights=D65_TRACE_WEIGHTS,
    luminance_weights=LUMINANCE_TRACE_WEIGHTS,
    weight_target=1.0,
    weight_shape=1.0,
)

bounds = ParamBounds()


# ── Reference input flux (photometric, matches luminance_weights) ──────────
# One direction's emitted *luminous* flux summed across traced wavelengths:
# N_rays × Σ_λ spectrum(λ) × V(λ) × Δλ × K_m. Threshold/cap percentages
# are now fractions of perceived input brightness, not radiometric flux.

NUM_RAYS = PROJECTOR_NX * PROJECTOR_NY
_spec_wls, _spec_rad = PROJECTOR.spectrum
_spectral_weights = jnp.interp(TRACE_WAVELENGTHS, _spec_wls, _spec_rad)
INPUT_FLUX = float(NUM_RAYS * jnp.sum(_spectral_weights * LUMINANCE_TRACE_WEIGHTS))

# ── Build the pupil cell grid and mask for the target region ────────────────
# We build the grid from the pupil element in the reference system. The
# grid is fixed for the whole optimization (target eyebox is pre-defined).

_ref_system = build_parametrized_system(
    CombinerParams.initial(), probe_wavelengths=TRACE_WAVELENGTHS)
_pupil = next(e for e in _ref_system.elements if isinstance(e, RectangularPupil))
EYEBOX_POINTS = eyebox_grid_points(
    _pupil.position, _pupil.normal, EYEBOX_RADIUS, EYEBOX_NX, EYEBOX_NY,
)   # (S, 3)
CELL_MASK = jnp.ones(EYEBOX_POINTS.shape[0])   # all grid cells belong to the target

BINNING_SIGMA = 0.8 * mm   # ~half grid spacing for smooth spatial gradients


# ── Loss function ───────────────────────────────────────────────────────────


def _compute_spectral_response(params: CombinerParams):
    """Trace the broadband projector at each wavelength, vmapped over wavelengths."""
    system = build_parametrized_system(
        params, probe_wavelengths=TRACE_WAVELENGTHS)
    branch_routes = build_combiner_branch_routes(
        system, num_mirrors=NUM_MIRRORS,
    )
    directions = FOV_GRID.flat_directions  # (A, 3)

    def trace_one_wavelength(wavelength):
        binned = jnp.zeros((directions.shape[0], EYEBOX_POINTS.shape[0]))
        for route in branch_routes:
            prepared = prepare_route(route, wavelength)
            binned = binned + trace_branch_over_fov(
                prepared, PROJECTOR, EYEBOX_POINTS, wavelength,
                directions, sigma=BINNING_SIGMA)  # (A, S)
        return binned.T  # (S, A)

    all_responses = jax.vmap(trace_one_wavelength)(TRACE_WAVELENGTHS)  # (N, S, A)
    return jnp.transpose(all_responses, (1, 2, 0))  # (S, A, N)


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


# ── Adam optimizer ──────────────────────────────────────────────────────────

PHASE1_STEPS = 100
PHASE2_STEPS = 300

adam_cfg_phase1 = AdamConfig(peak_lr=3e-3, warmup_steps=20, num_steps=PHASE1_STEPS)
adam_cfg_phase2 = AdamConfig(peak_lr=2e-3, warmup_steps=10, num_steps=PHASE2_STEPS)


# ── Run ─────────────────────────────────────────────────────────────────────

def _print_breakdown(label, bd):
    print(f"\n{label}: {float(bd['total']):.5f}")
    print(f"  target={float(bd['target']):.5f}  "
          f"shape={float(bd['shape']):.5f}")
    print(f"  brightness mean_rel={float(bd['mean_brightness_rel']):.5f}  "
          f"std_rel={float(bd['brightness_std_rel']):.5f}  "
          f"min_rel={float(bd['min_brightness_rel']):.5f}  "
          f"max_rel={float(bd['max_brightness_rel']):.5f}")


RUNS_ROOT = Path("examples/reports/optimize_pupil")


def main():
    run_dir = RUNS_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # σ=15 nm sits comfortably above ParamBounds.fwhm_min_nm=20 ⇒ σ_min ≈ 8.5 nm,
    # so the first Adam step doesn't immediately get clipped.
    initial_params = CombinerParams.initial(amplitude=0.10, width_nm=15)
    params = initial_params
    state = adam_init(params)

    print("── Pupil optimization (spacings + Gaussian reflectance) ──")
    print(f"Variables: {params.spacings.size + params.curves.amplitude.size + params.curves.sigma.size}")
    print(f"Eyebox:    {2*EYEBOX_RADIUS/mm:.1f}×{2*EYEBOX_RADIUS/mm:.1f} mm, "
          f"{EYEBOX_NX}×{EYEBOX_NY} cells")
    print(f"FOV:       ±{X_FOV/deg:.1f}° × ±{Y_FOV/deg:.1f}°, "
          f"{FOV_GRID.num_x}×{FOV_GRID.num_y} samples")
    print(f"Spectrum:  {SPECTRAL_SAMPLES} samples/channel, "
          f">{SPECTRAL_THRESHOLD:.0%} of peak")
    print(f"I_in:      {INPUT_FLUX:.1f}  "
          f"(target_relative={merit_cfg_phase1.target_relative})")

    # Ceiling diagnostic — what brightness can we hope for?
    # Saturate amplitudes and widths at their upper bounds (spacings stay
    # at the initial value). This isn't a true ceiling — the optimizer
    # may actually find higher per-cell brightness with non-uniform
    # amplitudes — but it tells you whether ``PER_CELL_TARGET`` is
    # plausibly reachable at all.
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
        params, state = adam_step(params, grad, state, adam_cfg_phase1)
        params = bounds.clip(params)
        loss_history.append(float(loss))
        print(f"step {step+1:4d}/{PHASE1_STEPS}  loss={float(loss):.5f}")

    phase1_breakdown = breakdown_fn(params, merit_cfg_phase1)
    _print_breakdown("Phase 1 result", phase1_breakdown)

    print("\n── Phase 2: target + D65/FOV uniformity ──")
    state = adam_init(params)
    for step in range(PHASE2_STEPS):
        loss, grad = value_and_grad_phase2(params)
        params, state = adam_step(params, grad, state, adam_cfg_phase2)
        params = bounds.clip(params)
        loss_history.append(float(loss))
        print(f"step {step+1:4d}/{PHASE2_STEPS}  loss={float(loss):.5f}")

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

    response = _compute_spectral_response(params)
    # Match the merit's brightness definition: luminance (V-weighted sum).
    # ``INPUT_FLUX`` is in the same units, so the printed values are
    # directly comparable to ``PER_CELL_TARGET``.
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
                "focus": "target+shape",
            },
            "schedule": "warmup_cosine_decay",
            "total_steps": PHASE1_STEPS + PHASE2_STEPS,
        },
        param_bounds=bounds,
        initial_params=initial_params,
        final_params=params,
        initial_breakdown={k: float(v) for k, v in initial_breakdown.items()},
        final_breakdown={k: float(v) for k, v in final_breakdown.items()},
        loss_history=loss_history,
        eyebox_config={
            "radius": EYEBOX_RADIUS,
            "nx": EYEBOX_NX,
            "ny": EYEBOX_NY,
        },
    )
    print(f"\nSaved optimization report: {report_path}")

    pupil_x_mm = jnp.linspace(-EYEBOX_RADIUS, EYEBOX_RADIUS, EYEBOX_NX)
    pupil_y_mm = jnp.linspace(-EYEBOX_RADIUS, EYEBOX_RADIUS, EYEBOX_NY)
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
