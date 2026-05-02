"""Pupil optimization driver — three independent R/G/B projectors.

Sibling to :mod:`examples.optimize_pupil`. The broadband driver feeds the
optimizer a single W-spectrum projector and integrates the response
across ~100 fine wavelength samples; this driver instead traces three
separate narrow-band projectors (one per LED primary) and integrates
each channel's per-wavelength response into a single luminance value
before merit evaluation.

Why three projectors instead of W-with-a-fine-wavelength-axis
-------------------------------------------------------------
The broadband driver's phase-2 shape term penalizes deviation from
``D65[c] · mean_radiance(s)`` at *every* wavelength sample. The
projector is a 3-band micro-LED — it has effectively zero radiance at
wavelengths between the R/G/B peaks. No mirror tuning can put light at
wavelengths the projector doesn't emit, so the shape term is
structurally unsatisfiable: phase 2 plateaus at ~0.5 and never moves.

Folding the response into three channel luminances *before* the merit
collapses the wavelength axis to a 3-element vector that the existing
``pupil_merit`` shape term can actually drive to zero — D65 ratio is a
3-vector, the response is a 3-vector, the comparison is well-posed.

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
from helios.merit import (
    build_combiner_branch_routes, D65_WEIGHTS, DEFAULT_WAVELENGTHS,
)
from helios.eyebox import trace_branch_over_fov
from helios.photometry import luminance_weights as photopic_luminance_weights
from helios.pupil_merit import (
    PupilMeritConfig, pupil_merit, merit_breakdown,
)
from helios.adam import AdamConfig, adam_init, adam_step
from helios.io import save_optimization_report, save_run, ScanConfig
from helios.reports.pupil_report import render_pupil_report

jax.config.update("jax_compilation_cache_dir", "/home/ubuntu/.cache/jax")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# ── Eyebox target region (pre-defined, fixed) ───────────────────────────────

EYEBOX_HALF_X = 4.0 * mm         # 8 mm full width on x
EYEBOX_HALF_Y = 5.0 * mm         # 10 mm full width on y
EYEBOX_NX, EYEBOX_NY = 8, 10     # 80 cells, ~1.0×1.1 mm each

X_FOV = 8.0 * deg
Y_FOV = 8.0 * deg

# ── Three independent narrow-band projectors (R, G, B) ────────────────────
# All three sit at the same module location with the same beam geometry —
# they only differ in spectral content. ``PlayNitrideLed.create`` reads the
# corresponding column from the panel CSV and peak-normalizes it.

PROJECTOR_NX, PROJECTOR_NY = 50, 10
BEAM_WIDTH, BEAM_HEIGHT = 10.0 * mm, 2.0 * mm

CHANNELS = ("R", "G", "B")

PROJECTORS = {
    color: PlayNitrideLed.create(
        position=DEFAULT_LIGHT_POSITION,
        direction=DEFAULT_LIGHT_DIRECTION,
        beam_width=BEAM_WIDTH,
        beam_height=BEAM_HEIGHT,
        nx=PROJECTOR_NX, ny=PROJECTOR_NY,
        color=color,
    )
    for color in CHANNELS
}

# ── Per-channel wavelength sampling ────────────────────────────────────────
# Each LED primary covers a narrow band; spending 30 samples inside each
# band integrates per-channel luminance accurately while keeping the
# wavelength axis short overall (90 samples vs the broadband 100).

SPECTRAL_THRESHOLD = 0.05
SAMPLES_PER_CHANNEL = 30

WAVELENGTHS_PER_CHANNEL = {
    color: jnp.linspace(
        *PROJECTORS[color].spectral_band(threshold=SPECTRAL_THRESHOLD),
        SAMPLES_PER_CHANNEL,
    )
    for color in CHANNELS
}

# Probe grid for evaluating mirror reflectance curves — must span every
# trace wavelength used by any channel. We linspace across the union of
# band extents at the same density as the per-channel grids.
_band_lo = min(float(WAVELENGTHS_PER_CHANNEL[c][0]) for c in CHANNELS)
_band_hi = max(float(WAVELENGTHS_PER_CHANNEL[c][-1]) for c in CHANNELS)
PROBE_WAVELENGTHS = jnp.linspace(_band_lo, _band_hi, 120)

# Photopic V·Δλ·K_m weights *per channel* — used to integrate each
# channel's per-wavelength response into a single luminance value.
LUMINANCE_WEIGHTS_PER_CHANNEL = {
    color: photopic_luminance_weights(WAVELENGTHS_PER_CHANNEL[color])
    for color in CHANNELS
}

# ── Per-cell brightness target ─────────────────────────────────────────────

NUM_EYEBOX_CELLS = EYEBOX_NX * EYEBOX_NY
EYEBOX_TARGET = 0.12
PER_CELL_TARGET = EYEBOX_TARGET / NUM_EYEBOX_CELLS


# ── Merit configuration (3-channel) ────────────────────────────────────────
# ``pupil_merit`` is reused as-is. The response we feed it has shape
# ``(S, A, 3)`` instead of ``(S, A, N_wavelengths)``, and the channel
# values are *already* luminances — so ``luminance_weights`` is just
# ones(3) and ``d65_weights`` is the 3-element D65 simplex.

D65_3CH = D65_WEIGHTS                # already shape (3,) at R/G/B peaks
ONES_3CH = jnp.ones(3)               # channel sum = total luminance

FOV_GRID = FovGrid(DEFAULT_LIGHT_DIRECTION, X_FOV, Y_FOV, num_x=16, num_y=16)

merit_cfg_phase1 = PupilMeritConfig(
    target_relative=PER_CELL_TARGET,
    d65_weights=D65_3CH,
    luminance_weights=ONES_3CH,
    weight_target=1.0,
    weight_shape=0.0,
)
merit_cfg_phase2 = PupilMeritConfig(
    target_relative=PER_CELL_TARGET,
    d65_weights=D65_3CH,
    luminance_weights=ONES_3CH,
    weight_target=1.0,
    weight_shape=1.0,
)

bounds = ParamBounds(amplitude_max=0.25, fwhm_max_nm=60, fwhm_min_nm=10)

# When False, mirror inter-spacing is held at its initial value and only
# the Gaussian reflectance curves are tuned. Hard nearest-neighbor binning
# is then sufficient — see optimize_pupil.py for the reasoning.
OPTIMIZE_SPACINGS = True


# ── Reference input flux (sum of per-channel luminances) ───────────────────
# Each projector contributes its own per-direction luminous flux:
# N_rays × Σ_λ spec_c(λ) × V(λ) × Δλ × K_m. Total is the sum across the
# three channels — running R, G, B simultaneously at their max-drive
# (peak-normalized) spectra. The merit's ``target_relative`` is then a
# fraction of this combined max-drive luminance.

NUM_RAYS = PROJECTOR_NX * PROJECTOR_NY


def _channel_input_luminance(color: str) -> float:
    spec_wls, spec_rad = PROJECTORS[color].spectrum
    wavelengths = WAVELENGTHS_PER_CHANNEL[color]
    spec_at_wls = jnp.interp(wavelengths, spec_wls, spec_rad)
    luminance_weights = LUMINANCE_WEIGHTS_PER_CHANNEL[color]
    return float(NUM_RAYS * jnp.sum(spec_at_wls * luminance_weights))


CHANNEL_INPUT_LUMINANCE = {c: _channel_input_luminance(c) for c in CHANNELS}
INPUT_FLUX = float(sum(CHANNEL_INPUT_LUMINANCE.values()))

# ── Build the pupil cell grid and mask for the target region ────────────────

_ref_system = build_parametrized_system(
    CombinerParams.initial(), probe_wavelengths=PROBE_WAVELENGTHS)
_pupil = next(e for e in _ref_system.elements if isinstance(e, RectangularPupil))
EYEBOX_POINTS = planar_grid_points(
    _pupil.position, _pupil.normal,
    EYEBOX_HALF_X, EYEBOX_HALF_Y, EYEBOX_NX, EYEBOX_NY,
)   # (S, 3)
CELL_MASK = jnp.ones(EYEBOX_POINTS.shape[0])

_CELL_PITCH_X = 2 * EYEBOX_HALF_X / (EYEBOX_NX - 1)
_CELL_PITCH_Y = 2 * EYEBOX_HALF_Y / (EYEBOX_NY - 1)
BINNING_SIGMA = 0.5 * min(_CELL_PITCH_X, _CELL_PITCH_Y)


# ── Loss function ───────────────────────────────────────────────────────────


def _trace_channel(
    branch_routes,
    projector,
    wavelengths,
    luminance_weights,
):
    """Trace one projector across its band, return (S, A) channel luminance.

    Inner ``lax.scan`` walks the channel's narrow wavelength grid; for each
    wavelength we accumulate over branches and bin onto the eyebox. After
    the scan we do a V-weighted sum over the channel's wavelength axis to
    collapse it into a single per-(cell, FOV) luminance value.
    """
    directions = FOV_GRID.flat_directions  # (A, 3)

    def trace_one_wavelength(_, wavelength):
        binned = jnp.zeros((directions.shape[0], EYEBOX_POINTS.shape[0]))
        for route in branch_routes:
            prepared = prepare_route(route, wavelength)
            binned = binned + trace_branch_over_fov(
                prepared, projector, EYEBOX_POINTS, wavelength,
                directions,
                sigma=BINNING_SIGMA if OPTIMIZE_SPACINGS else None)
        return None, binned.T  # (S, A)

    _, per_wavelength = jax.lax.scan(
        jax.checkpoint(trace_one_wavelength), None, wavelengths)  # (N, S, A)

    # V-weighted sum over the wavelength axis → (S, A) luminance.
    return jnp.tensordot(luminance_weights, per_wavelength, axes=(0, 0))


def _compute_3channel_response(params: CombinerParams) -> jnp.ndarray:
    """Trace all three projectors, return ``(S, A, 3)`` luminance per channel."""
    system = build_parametrized_system(
        params, probe_wavelengths=PROBE_WAVELENGTHS)
    branch_routes = build_combiner_branch_routes(
        system, num_mirrors=NUM_MIRRORS)

    channel_responses = [
        _trace_channel(
            branch_routes,
            PROJECTORS[color],
            WAVELENGTHS_PER_CHANNEL[color],
            LUMINANCE_WEIGHTS_PER_CHANNEL[color],
        )
        for color in CHANNELS
    ]
    return jnp.stack(channel_responses, axis=-1)  # (S, A, 3)


def loss_fn_phase1(params: CombinerParams) -> jnp.ndarray:
    response = _compute_3channel_response(params)
    return pupil_merit(response, INPUT_FLUX, merit_cfg_phase1, cell_mask=CELL_MASK)


def loss_fn_phase2(params: CombinerParams) -> jnp.ndarray:
    response = _compute_3channel_response(params)
    return pupil_merit(response, INPUT_FLUX, merit_cfg_phase2, cell_mask=CELL_MASK)


def breakdown_fn(params: CombinerParams, merit_cfg: PupilMeritConfig) -> dict:
    response = _compute_3channel_response(params)
    return merit_breakdown(response, INPUT_FLUX, merit_cfg, cell_mask=CELL_MASK)


value_and_grad_phase1 = jax.jit(jax.value_and_grad(loss_fn_phase1))
value_and_grad_phase2 = jax.jit(jax.value_and_grad(loss_fn_phase2))


def _mask_frozen(grad: CombinerParams) -> CombinerParams:
    if OPTIMIZE_SPACINGS:
        return grad
    return grad._replace(spacings=jnp.zeros_like(grad.spacings))


# ── Adam optimizer ──────────────────────────────────────────────────────────

PHASE1_STEPS = 100
PHASE2_STEPS = 200

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
    print(f"── RGB pupil optimization ({optimized_label}) ──")
    print(f"Variables: {num_vars}")
    print(f"Eyebox:    {2*EYEBOX_HALF_X/mm:.1f}×{2*EYEBOX_HALF_Y/mm:.1f} mm, "
          f"{EYEBOX_NX}×{EYEBOX_NY} cells")
    print(f"FOV:       ±{X_FOV/deg:.1f}° × ±{Y_FOV/deg:.1f}°, "
          f"{FOV_GRID.num_x}×{FOV_GRID.num_y} samples")
    for color in CHANNELS:
        wavelengths = WAVELENGTHS_PER_CHANNEL[color]
        print(f"Channel {color}: {SAMPLES_PER_CHANNEL} samples, "
              f"{float(wavelengths[0])/nm:.1f}–{float(wavelengths[-1])/nm:.1f} nm, "
              f"luminance contribution = {CHANNEL_INPUT_LUMINANCE[color]:.1f}")
    print(f"I_in:      {INPUT_FLUX:.1f}  "
          f"(target_relative={merit_cfg_phase1.target_relative})")

    # Ceiling diagnostic — saturate amplitudes/widths at their upper bounds.
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
    ceiling_response = _compute_3channel_response(ceiling_params)
    ceiling_brightness = (jnp.mean(jnp.sum(ceiling_response, axis=-1), axis=-1)
                          / INPUT_FLUX)
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
        print(f"step {step+1:4d}/{PHASE1_STEPS}  loss={float(loss):.5f}")

    phase1_breakdown = breakdown_fn(params, merit_cfg_phase1)
    _print_breakdown("Phase 1 result", phase1_breakdown)

    print("\n── Phase 2: target + D65/FOV uniformity (3-channel) ──")
    state = adam_init(params)
    for step in range(PHASE2_STEPS):
        loss, grad = value_and_grad_phase2(params)
        params, state = adam_step(params, _mask_frozen(grad), state,
                                  adam_cfg_phase2)
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

    response = _compute_3channel_response(params)
    # Each channel value is already a luminance, so the per-cell brightness
    # is the channel sum (luminance_weights = ones(3)).
    luminance_per_angle = jnp.sum(response, axis=-1)                # (S, A)
    mean_luminance = jnp.mean(luminance_per_angle, axis=-1)         # (S,)
    relative_brightness = mean_luminance / INPUT_FLUX

    grid = relative_brightness.reshape(EYEBOX_NY, EYEBOX_NX)
    print(f"\nEyebox brightness map (relative to input flux, target={PER_CELL_TARGET}):")
    for row in grid:
        print("  " + "  ".join(f"{float(v):.4f}" for v in row))
    print(f"min={float(relative_brightness.min()):.5f}  "
          f"max={float(relative_brightness.max()):.5f}  "
          f"std={float(relative_brightness.std()):.5f}")

    # Per-channel relative brightness (mean over cells × FOV).
    per_channel_rel = (jnp.mean(response, axis=(0, 1))
                       / (INPUT_FLUX / 3.0))
    print(f"Per-channel mean brightness (R, G, B) relative to I_in/3: "
          f"{float(per_channel_rel[0]):.4f}  "
          f"{float(per_channel_rel[1]):.4f}  "
          f"{float(per_channel_rel[2]):.4f}")
    print(f"D65 target ratios:                                     "
          f"{float(D65_3CH[0]):.4f}  "
          f"{float(D65_3CH[1]):.4f}  "
          f"{float(D65_3CH[2]):.4f}")

    final_system = build_parametrized_system(
        params, probe_wavelengths=PROBE_WAVELENGTHS)
    report_path = save_optimization_report(
        run_dir,
        system=final_system,
        projectors=[PROJECTORS[c] for c in CHANNELS],
        fov_grid=FOV_GRID,
        merit_config=merit_cfg_phase2,
        optimizer_config={
            "algorithm": "adam_two_phase_rgb",
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
                "focus": "target+shape (3-channel)",
            },
            "schedule": "warmup_cosine_decay",
            "total_steps": PHASE1_STEPS + PHASE2_STEPS,
            "channels": list(CHANNELS),
            "samples_per_channel": SAMPLES_PER_CHANNEL,
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

    pupil_x_mm = jnp.linspace(-EYEBOX_HALF_X, EYEBOX_HALF_X, EYEBOX_NX)
    pupil_y_mm = jnp.linspace(-EYEBOX_HALF_Y, EYEBOX_HALF_Y, EYEBOX_NY)
    scan_cfg = ScanConfig(
        base_direction=DEFAULT_LIGHT_DIRECTION,
        x_fov=float(X_FOV), y_fov=float(Y_FOV),
        num_x=FOV_GRID.num_x, num_y=FOV_GRID.num_y,
    )
    # ``save_run`` accepts a single representative projector; the three
    # channels share position/direction/beam geometry so any of them is
    # sufficient for the manifest. The 3-element ``wavelengths_nm`` axis
    # carries the actual channel labels (LED peak wavelengths).
    save_run(
        run_dir,
        final_system, PROJECTORS["G"], scan_cfg,
        response=response,
        pupil_x_mm=pupil_x_mm,
        pupil_y_mm=pupil_y_mm,
        scan_angles=FOV_GRID.angles_grid,
        wavelengths_nm=DEFAULT_WAVELENGTHS / nm,
    )
    print(f"Saved run inputs + response to: {run_dir}")

    pupil_report_path = render_pupil_report(run_dir)
    print(f"Saved pupil report: {pupil_report_path}")


if __name__ == "__main__":
    main()
