"""Pupil optimization driver — spacings + Gaussian reflectance.

Optimizes the Talos combiner's 41 design variables (5 spacings + 36
Gaussian amplitude/width params) against :func:`helios.pupil_merit` so
that a pre-defined target eyebox region on the pupil is:

1. As **full** as possible — every cell above a relative brightness
   threshold.
2. **D65 white-balanced and FOV-flat** on the cells that make it into
   the eyebox.

Everything the merit reports is relative to an input-flux reference,
so tuning ``threshold_relative`` doesn't depend on ray count.

Run::

    python examples/optimize_pupil.py
"""

import jax
import jax.numpy as jnp

from apollo14.units import mm, nm, deg
from apollo14.combiner import (
    DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
)
from apollo14.elements.pupil import RectangularPupil
from apollo14.projector import PlayNitrideLed, FovGrid

from helios.combiner_params import (
    CombinerParams, ParamBounds, build_parametrized_system, NUM_MIRRORS,
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


# ── Eyebox target region (pre-defined, fixed) ───────────────────────────────

EYEBOX_RADIUS = 5.0 * mm         # 10×10 mm centered eyebox
EYEBOX_NX, EYEBOX_NY = 7, 7      # 49 cells, ~1.4 mm each

X_FOV = 8.0 * deg
Y_FOV = 8.0 * deg

# ── Broadband projector (PlayNitride micro-LED, combined R+G+B) ───────────

PROJECTOR_NX, PROJECTOR_NY = 14, 14

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
# per watt. The merit's coverage / warm-up / cap thresholds become
# perceptually meaningful instead of radiometric.

LUMINANCE_TRACE_WEIGHTS = photopic_luminance_weights(TRACE_WAVELENGTHS)

# ── Per-cell brightness targets ─────────────────────────────────────────────
# Fractions of input flux each of the 49 eyebox cells should receive.
# With NUM_EYEBOX_CELLS=49: threshold 0.002 ≈ 9.8% of input flux reaching
# the eyebox uniformly, cap 0.003 ≈ 14.7% — a ~50% headroom band lets the
# optimizer find a uniform-ish solution without being whipsawed by an
# equality-only constraint.

NUM_EYEBOX_CELLS = EYEBOX_NX * EYEBOX_NY    # 49
PER_CELL_THRESHOLD = 0.002
PER_CELL_CAP = 0.0022


# ── Merit & tracer configuration ────────────────────────────────────────────

FOV_GRID = FovGrid(DEFAULT_LIGHT_DIRECTION, X_FOV, Y_FOV, num_x=14, num_y=14)

merit_cfg_phase1 = PupilMeritConfig(
    threshold_relative=PER_CELL_THRESHOLD,
    cap_relative=PER_CELL_CAP,
    d65_weights=D65_TRACE_WEIGHTS,
    luminance_weights=LUMINANCE_TRACE_WEIGHTS,
    weight_shape=0.2,
    weight_coverage=5.0,
    weight_warmup=1.0,
    weight_cap=0.1,
    sigmoid_steepness=8.0,
    soft_min_temperature=15.0,
)

merit_cfg_phase2 = PupilMeritConfig(
    threshold_relative=PER_CELL_THRESHOLD,
    cap_relative=PER_CELL_CAP,
    d65_weights=D65_TRACE_WEIGHTS,
    luminance_weights=LUMINANCE_TRACE_WEIGHTS,
    weight_shape=1.0,
    weight_coverage=3.0,
    weight_warmup=0.5,
    weight_cap=0.2,
    sigmoid_steepness=8.0,
    soft_min_temperature=15.0,
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

PHASE1_STEPS = 50
PHASE2_STEPS = 50

adam_cfg_phase1 = AdamConfig(peak_lr=3e-3, warmup_steps=20, num_steps=PHASE1_STEPS)
adam_cfg_phase2 = AdamConfig(peak_lr=2e-3, warmup_steps=10, num_steps=PHASE2_STEPS)


# ── Run ─────────────────────────────────────────────────────────────────────

def _print_breakdown(label, bd):
    print(f"\n{label}: {float(bd['total']):.5f}")
    print(f"  shape={float(bd['shape']):.5f}  "
          f"coverage={float(bd['coverage']):.5f}  "
          f"warmup={float(bd['warmup']):.5f}  "
          f"cap={float(bd['cap']):.5f}")
    print(f"  active_fraction={float(bd['active_fraction']):.3f}  "
          f"min_rel={float(bd['min_brightness_rel']):.4f}")


def main():
    initial_params = CombinerParams.initial(amplitude=0.10)
    params = initial_params
    state = adam_init(params)

    print("── Pupil optimization (spacings + Gaussian reflectance) ──")
    print(f"Variables: {params.spacings.size + params.amplitudes.size + params.widths.size}")
    print(f"Eyebox:    {2*EYEBOX_RADIUS/mm:.1f}×{2*EYEBOX_RADIUS/mm:.1f} mm, "
          f"{EYEBOX_NX}×{EYEBOX_NY} cells")
    print(f"FOV:       ±{X_FOV/deg:.1f}° × ±{Y_FOV/deg:.1f}°, "
          f"{FOV_GRID.num_x}×{FOV_GRID.num_y} samples")
    print(f"Spectrum:  {SPECTRAL_SAMPLES} samples/channel, "
          f">{SPECTRAL_THRESHOLD:.0%} of peak")
    print(f"I_in:      {INPUT_FLUX:.1f}  "
          f"(threshold_relative={merit_cfg_phase1.threshold_relative})")

    initial_breakdown = breakdown_fn(params, merit_cfg_phase1)
    _print_breakdown("Initial merit (phase 1 weights)", initial_breakdown)

    loss_history = []

    print("\n── Phase 1: coverage-focused (fill the eyebox) ──")
    for step in range(PHASE1_STEPS):
        loss, grad = value_and_grad_phase1(params)
        params, state = adam_step(params, grad, state, adam_cfg_phase1)
        params = bounds.clip(params)
        loss_history.append(float(loss))
        print(f"step {step+1:4d}/{PHASE1_STEPS}  loss={float(loss):.5f}")

    phase1_breakdown = breakdown_fn(params, merit_cfg_phase1)
    _print_breakdown("Phase 1 result", phase1_breakdown)

    print("\n── Phase 2: shape-focused (D65 white balance + uniformity) ──")
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
    print("Final amplitudes per mirror (R, G, B):")
    for mirror_idx in range(NUM_MIRRORS):
        mirror_amplitude = params.amplitudes[mirror_idx]
        print(f"  m{mirror_idx}: {float(mirror_amplitude[0]):.4f}  "
              f"{float(mirror_amplitude[1]):.4f}  {float(mirror_amplitude[2]):.4f}")
    print("Final widths per mirror (nm):")
    for mirror_idx in range(NUM_MIRRORS):
        mirror_width = params.widths[mirror_idx]
        print(f"  m{mirror_idx}: {float(mirror_width[0])*1e6:.1f}  "
              f"{float(mirror_width[1])*1e6:.1f}  {float(mirror_width[2])*1e6:.1f}")

    response = _compute_spectral_response(params)
    # Match the merit's brightness definition: luminance (V-weighted sum),
    # not radiometric sum. ``INPUT_FLUX`` is in luminance units, so
    # mixing radiance numerator with luminance denominator was hiding
    # values ~2000× below ``threshold_relative`` in the printed grid.
    luminance_per_angle = jnp.sum(
        response * LUMINANCE_TRACE_WEIGHTS.reshape(1, 1, -1), axis=-1)  # (S, A)
    mean_luminance = jnp.mean(luminance_per_angle, axis=-1)             # (S,)
    relative_brightness = mean_luminance / INPUT_FLUX

    threshold = merit_cfg_phase1.threshold_relative
    grid = relative_brightness.reshape(EYEBOX_NY, EYEBOX_NX)
    print(f"\nEyebox brightness map (relative to input flux, threshold={threshold}):")
    for row in grid:
        print("  " + "  ".join(f"{float(v):.4f}" for v in row))
    active_cells = int((relative_brightness >= threshold).sum())
    print(f"Active cells: {active_cells}/{relative_brightness.shape[0]}  "
          f"min={float(relative_brightness.min()):.5f}  "
          f"max={float(relative_brightness.max()):.5f}")

    final_system = build_parametrized_system(
        params, probe_wavelengths=TRACE_WAVELENGTHS)
    report_path = save_optimization_report(
        "examples/reports/optimize_pupil",
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
                "focus": "coverage",
            },
            "phase2": {
                "steps": PHASE2_STEPS,
                "peak_lr": adam_cfg_phase2.peak_lr,
                "warmup_steps": adam_cfg_phase2.warmup_steps,
                "focus": "shape",
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
    run_dir = save_run(
        "examples/reports/optimize_pupil",
        final_system, PROJECTOR, scan_cfg,
        response=response,
        pupil_x_mm=pupil_x_mm,
        pupil_y_mm=pupil_y_mm,
        scan_angles=FOV_GRID.angles_grid,
        wavelengths_nm=TRACE_WAVELENGTHS / nm,
    )


if __name__ == "__main__":
    main()
