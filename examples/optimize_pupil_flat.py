"""Pupil optimization driver — 10 flat (wavelength-uniform) mirrors at
fixed spacing.

How this differs from :mod:`examples.optimize_pupil_rgb`
-------------------------------------------------------
The RGB driver gives every mirror a :class:`SumOfGaussiansCurve` —
per-color reflectance bumps (amplitude **and** width σ per basis) — and
also optimizes the inter-mirror spacings, so it can both white-balance
the pupil and reshape the geometry.

This driver is the deliberately simpler case:

- **10 mirrors** instead of 6.
- **Flat reflectance** — each mirror is a :class:`ConstantCurve`, a
  single wavelength-uniform scalar (think "5% across the whole
  spectrum"), not a Gaussian. The only design variables are the 10
  per-mirror reflectance levels.
- **Fixed geometry** — the inter-mirror spacing is frozen at
  ``FIXED_SPACING_MM``; it is never a design variable, so hard
  nearest-neighbor binning is sufficient (no soft Gaussian binning).

Because a flat mirror applies the same reflectance at every wavelength,
it preserves the projector's spectral shape up to a scalar — there is no
color shift to correct. So the merit's spectrum-preserving *shape* term
is irrelevant here and is switched off: the optimization is a single
phase that simply pulls every eyebox cell's brightness toward the target
and equalizes the cell-to-cell spread.

Run::

    python examples/optimize_pupil_flat.py
"""

from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp

from apollo14.units import mm, nm, deg
from apollo14.combiner import (
    DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
    compensated_reflectances,
)
from apollo14.elements.pupil import RectangularPupil
from apollo14.geometry import planar_grid_points
from apollo14.projector import PlayNitrideLed, FovGrid
from apollo14.spectral import ConstantCurve

from helios.combiner_params import (
    CombinerParams, ParamBounds, build_parametrized_system,
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

# ── Combiner geometry (this driver overrides the 6-mirror default) ──────────
# The spacing is fixed for the whole run — it is *not* a design variable.
# 10 mirrors × 9 gaps × 1.3 mm = 11.7 mm, comfortably inside the chassis's
# ~18 mm usable length.

NUM_MIRRORS = 8
FIXED_SPACING_MM = 1.3
BASE_RATIO = 0.05  # nominal per-mirror reflected fraction of the original beam

# ── Eyebox target region (pre-defined, fixed) ───────────────────────────────

EYEBOX_HALF_X = 4.0 * mm         # 8 mm full width on x
EYEBOX_HALF_Y = 5.0 * mm         # 10 mm full width on y
EYEBOX_NX, EYEBOX_NY = 8, 10     # 80 cells, exactly 1×1 mm each (cell-centered)

X_FOV = 8.0 * deg
Y_FOV = 8.0 * deg

# ── Single broadband projector (panel's calibrated white) ─────────────────

PROJECTOR_NX, PROJECTOR_NY = 25, 5
ANGULAR_STEPS_X, ANGULAR_STEPS_Y = 8, 8

PROJECTOR = PlayNitrideLed.create_broadband(
    position=DEFAULT_LIGHT_POSITION, direction=DEFAULT_LIGHT_DIRECTION,
    beam_width=10.0 * mm, beam_height=2.0 * mm,
    nx=PROJECTOR_NX, ny=PROJECTOR_NY,
)

# ── Wavelength sampling ─────────────────────────────────────────────────────
# Span W's above-threshold band uniformly. Even though the mirrors are
# wavelength-flat, the glass is dispersive, so different wavelengths land
# at slightly different pupil positions — tracing the full spectrum keeps
# that chromatic spread in the eyebox response, and the report's V(λ)
# luminance integration needs the per-sample Δλ.

SPECTRAL_THRESHOLD = 0.10
SPECTRAL_SAMPLES = 100

_w_lo, _w_hi = PROJECTOR.spectral_band(threshold=SPECTRAL_THRESHOLD)
TRACE_WAVELENGTHS = jnp.linspace(_w_lo, _w_hi, SPECTRAL_SAMPLES)

# Projector W spectrum at the trace wavelengths. Flat mirrors preserve
# this shape automatically, so it is only used as a (zero-weighted)
# placeholder for the merit's shape target — never actually penalized.
_spec_wls, _spec_rad = PROJECTOR.spectrum
_W_AT_TRACE = jnp.interp(TRACE_WAVELENGTHS, _spec_wls, _spec_rad)
SHAPE_TARGET = _W_AT_TRACE / _W_AT_TRACE.sum()

# Photopic V·Δλ·K_m weights at the trace wavelengths (uniform spacing,
# so ``mean(diff)`` inside ``photopic_luminance_weights`` is the actual
# per-sample Δλ — the report's report-time V-weighting matches).
LUMINANCE_TRACE_WEIGHTS = photopic_luminance_weights(TRACE_WAVELENGTHS)

# ── Per-cell brightness target ─────────────────────────────────────────────

NUM_EYEBOX_CELLS = EYEBOX_NX * EYEBOX_NY
# The merit excludes the 4 corner cells (see ``CELL_MASK`` below), so only
# ``NUM_ACTIVE_EYEBOX_CELLS`` are actually pulled to ``PER_CELL_TARGET``.
# Dividing the eyebox budget by the *active* count keeps the achieved eyebox
# total at ``EYEBOX_TARGET``; dividing by the full cell count makes every
# active cell aim ~5% low so the eyebox systematically undershoots.
NUM_EXCLUDED_CORNER_CELLS = 4
NUM_ACTIVE_EYEBOX_CELLS = NUM_EYEBOX_CELLS - NUM_EXCLUDED_CORNER_CELLS
EYEBOX_TARGET = 0.07
PER_CELL_TARGET = EYEBOX_TARGET / NUM_ACTIVE_EYEBOX_CELLS


# ── Merit & tracer configuration ────────────────────────────────────────────

FOV_GRID = FovGrid(DEFAULT_LIGHT_DIRECTION, X_FOV, Y_FOV,
                   num_x=ANGULAR_STEPS_X, num_y=ANGULAR_STEPS_Y)

# Single phase: pull every cell to the brightness target *exactly* —
# symmetric squared error, so over-target cells get pulled down and
# under-target cells get pulled up. The shape (color) term is off:
# wavelength-flat mirrors can't change color balance, so there is nothing
# for it to optimize.
merit_cfg = PupilMeritConfig(
    target_relative=PER_CELL_TARGET,
    d65_weights=SHAPE_TARGET,
    luminance_weights=LUMINANCE_TRACE_WEIGHTS,
    weight_target=1.0,
    weight_shape=0.0,
    asymmetric_target=False,
)

# Amplitude is the only design variable; the FWHM/spacing fields are
# carried only so ``save_optimization_report`` has a uniform bounds block
# to serialize — they do not apply to a flat curve at fixed geometry.
bounds = ParamBounds(amplitude_min=0.0, amplitude_max=0.20)


# ── Reference input flux (photometric, matches luminance_weights) ──────────

NUM_RAYS = PROJECTOR_NX * PROJECTOR_NY
INPUT_FLUX = float(NUM_RAYS * jnp.sum(_W_AT_TRACE * LUMINANCE_TRACE_WEIGHTS))

# ── Eye-pupil moving-window aggregation ───────────────────────────────────
# Each "merit cell" represents what a real eye-pupil (~3 mm diameter) would
# see if positioned at that point in the eyebox. Implemented as a 2-D mean
# convolution of the fine-grid binned response with a (K × K)-cell box
# kernel — physically meaningful, and provides spatial smoothness.
#
# The fine sample grid is enlarged by ``KERNEL_SIZE_CELLS // 2`` cells on
# every side so that ``mode="valid"`` convolution outputs *exactly* the
# original (EYEBOX_NY, EYEBOX_NX) shape — no boundary truncation.
#
# Setting ``KERNEL_SIZE_CELLS = 1`` falls back to point sampling.

KERNEL_SIZE_CELLS = 3
PADDING_CELLS = KERNEL_SIZE_CELLS // 2

_ref_system = build_parametrized_system(
    CombinerParams(
        spacings=jnp.full((NUM_MIRRORS - 1,), FIXED_SPACING_MM * mm),
        curves=ConstantCurve.uniform(amplitude=BASE_RATIO,
                                     num_mirrors=NUM_MIRRORS),
    ),
    probe_wavelengths=TRACE_WAVELENGTHS,
)
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
# consumes.
EYEBOX_POINTS = planar_grid_points(
    _pupil.position, _pupil.normal,
    SAMPLE_HALF_X, SAMPLE_HALF_Y, SAMPLE_NX, SAMPLE_NY,
    cell_centered=True,
)   # (SAMPLE_NX * SAMPLE_NY, 3)
# Exclude the 4 corner cells from the merit — geometric coverage at the
# extreme corners drops below what the optimizer can equalize, so weighting
# them in pulls the whole design down toward a worse compromise.
_cell_mask_2d = jnp.ones((EYEBOX_NY, EYEBOX_NX))
_cell_mask_2d = _cell_mask_2d.at[0, 0].set(0.0)
_cell_mask_2d = _cell_mask_2d.at[0, -1].set(0.0)
_cell_mask_2d = _cell_mask_2d.at[-1, 0].set(0.0)
_cell_mask_2d = _cell_mask_2d.at[-1, -1].set(0.0)
CELL_MASK = _cell_mask_2d.reshape(-1)


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
    and the report both expect. Spacings are fixed, so hard
    nearest-neighbor binning is used (no soft binning).
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
                sigma=None,  # spacings frozen → hard nearest-neighbor binning
                vmap_directions=True)  # (A, S_sample); A=64 fits comfortably
        # (S_sample, A) → moving-window mean → (S_eyebox, A). Windowing
        # *inside* the scan body keeps the per-iteration activation tape
        # at eyebox size rather than sample size.
        windowed = _window_mean(
            binned.T, SAMPLE_NY, SAMPLE_NX, KERNEL_SIZE_CELLS)
        return None, windowed  # (S_eyebox, A)

    # ``jax.checkpoint`` rematerializes each wavelength's forward during
    # backward instead of saving the full per-(wavelength, direction, ray,
    # cell) activation tape.
    _, all_responses = jax.lax.scan(
        jax.checkpoint(trace_one_wavelength), None, TRACE_WAVELENGTHS)  # (N, S_eyebox, A)
    return jnp.transpose(all_responses, (1, 2, 0))  # (S_eyebox, A, N)


def loss_fn(params: CombinerParams) -> jnp.ndarray:
    response = _compute_spectral_response(params)
    return pupil_merit(response, INPUT_FLUX, merit_cfg, cell_mask=CELL_MASK)


def breakdown_fn(params: CombinerParams) -> dict:
    response = _compute_spectral_response(params)
    return merit_breakdown(response, INPUT_FLUX, merit_cfg, cell_mask=CELL_MASK)


value_and_grad = jax.jit(jax.value_and_grad(loss_fn))


def _freeze_spacings(grad: CombinerParams) -> CombinerParams:
    """Zero the spacing gradient — the geometry is fixed by design."""
    return grad._replace(spacings=jnp.zeros_like(grad.spacings))


def _clip(params: CombinerParams) -> CombinerParams:
    """Clip the flat reflectance into ``[amplitude_min, amplitude_max]``.

    Spacings are frozen, so only the amplitude needs clamping; the curve
    has no σ/centers to clip.
    """
    clipped = jnp.clip(params.curves.amplitude,
                       bounds.amplitude_min, bounds.amplitude_max)
    return params._replace(curves=ConstantCurve(amplitude=clipped))


# ── Adam optimizer ──────────────────────────────────────────────────────────

NUM_STEPS = 300
adam_cfg = AdamConfig(peak_lr=3e-3, warmup_steps=20, num_steps=NUM_STEPS)


# ── Run ─────────────────────────────────────────────────────────────────────

def _print_breakdown(label, bd):
    print(f"\n{label}: {float(bd['total']):.5f}")
    print(f"  target={float(bd['target']):.5f}  "
          f"shape={float(bd['shape']):.5f}")
    print(f"  brightness mean_rel={float(bd['mean_brightness_rel']):.5f}  "
          f"std_rel={float(bd['brightness_std_rel']):.5f}  "
          f"min_rel={float(bd['min_brightness_rel']):.5f}  "
          f"max_rel={float(bd['max_brightness_rel']):.5f}")


RUNS_ROOT = Path("examples/reports/optimize_pupil_flat")


def main():
    run_dir = RUNS_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Warm-start with the Talos cascade compensation: each mirror's flat
    # reflectance is ``ratio / (1 − i·ratio)`` so the absolute reflected
    # fraction of the *original* beam is equal across mirrors. Without
    # this, the optimizer wastes iterations re-discovering that downstream
    # mirrors need higher local reflectance to compensate for upstream
    # transmission losses.
    compensated_amps = compensated_reflectances(
        BASE_RATIO, NUM_MIRRORS, num_samples=1).reshape(-1)  # (M,)
    initial_params = CombinerParams(
        spacings=jnp.full((NUM_MIRRORS - 1,), FIXED_SPACING_MM * mm),
        curves=ConstantCurve(amplitude=compensated_amps.astype(jnp.float32)),
    )
    params = initial_params
    state = adam_init(params)

    print("── Pupil optimization (flat reflectance, spacings frozen) ──")
    print(f"Variables: {params.curves.amplitude.size} "
          f"(one flat reflectance per mirror)")
    window_mm_x = KERNEL_SIZE_CELLS * float(_CELL_PITCH_X) / mm
    window_mm_y = KERNEL_SIZE_CELLS * float(_CELL_PITCH_Y) / mm
    print(f"Mirrors:   {NUM_MIRRORS}, fixed spacing {FIXED_SPACING_MM:.2f} mm")
    print(f"Pupil window: {KERNEL_SIZE_CELLS}×{KERNEL_SIZE_CELLS} cells "
          f"({window_mm_x:.2f}×{window_mm_y:.2f} mm) — "
          f"sample grid {SAMPLE_NX}×{SAMPLE_NY}, eyebox grid {EYEBOX_NX}×{EYEBOX_NY}")
    print(f"Eyebox:    {2*EYEBOX_HALF_X/mm:.1f}×{2*EYEBOX_HALF_Y/mm:.1f} mm, "
          f"{EYEBOX_NX}×{EYEBOX_NY} cells")
    print(f"FOV:       ±{X_FOV/deg/2:.1f}° × ±{Y_FOV/deg/2:.1f}°, "
          f"{FOV_GRID.num_x}×{FOV_GRID.num_y} samples")
    print(f"Spectrum:  {SPECTRAL_SAMPLES} uniform samples, "
          f"{float(_w_lo)/nm:.0f}–{float(_w_hi)/nm:.0f} nm "
          f"(W > {SPECTRAL_THRESHOLD:.0%} of peak)")
    print(f"I_in:      {INPUT_FLUX:.1f}  "
          f"(target_relative={merit_cfg.target_relative})")

    # Ceiling diagnostic — what brightness can we hope for if every mirror
    # is pegged at the maximum reflectance?
    ceiling_params = CombinerParams(
        spacings=initial_params.spacings,
        curves=ConstantCurve(
            amplitude=jnp.full_like(initial_params.curves.amplitude,
                                    bounds.amplitude_max)),
    )
    ceiling_response = _compute_spectral_response(ceiling_params)
    ceiling_lum_per_angle = jnp.sum(
        ceiling_response * LUMINANCE_TRACE_WEIGHTS.reshape(1, 1, -1), axis=-1)
    ceiling_brightness = jnp.mean(ceiling_lum_per_angle, axis=-1) / INPUT_FLUX
    print(f"Ceiling per-cell brightness (reflectance=max): "
          f"min={float(ceiling_brightness.min()):.5f}  "
          f"mean={float(ceiling_brightness.mean()):.5f}  "
          f"max={float(ceiling_brightness.max()):.5f}  "
          f"(target={PER_CELL_TARGET})")

    initial_breakdown = breakdown_fn(params)
    _print_breakdown("Initial merit", initial_breakdown)

    loss_history = []

    print("\n── Optimization: drive every cell to the brightness target ──")
    for step in range(NUM_STEPS):
        loss, grad = value_and_grad(params)
        params, state = adam_step(params, _freeze_spacings(grad), state,
                                  adam_cfg)
        params = _clip(params)
        loss_history.append(float(loss))
        print(f"step {step+1:4d}/{NUM_STEPS}  loss={float(loss):.8f}")

    final_breakdown = breakdown_fn(params)
    _print_breakdown("Final merit", final_breakdown)

    print("\nFixed spacings (mm):",
          [f"{float(spacing)/mm:.3f}" for spacing in params.spacings])
    print("Final flat reflectance per mirror:")
    for mirror_idx in range(NUM_MIRRORS):
        reflectance = float(params.curves.amplitude[mirror_idx])
        print(f"  m{mirror_idx}: {reflectance:.4f}")

    # Bounds-pegged diagnostic — surfaces whether the optimizer is stalled
    # because a reflectance is sitting at a bound (the bound is binding,
    # not a true flat region of the loss).
    amp_pegged_hi = int(jnp.sum(
        params.curves.amplitude >= bounds.amplitude_max - 1e-4))
    amp_pegged_lo = int(jnp.sum(
        params.curves.amplitude <= bounds.amplitude_min + 1e-4))
    print(f"\nReflectances at a bound: "
          f"{amp_pegged_lo} at lower / {amp_pegged_hi} at upper "
          f"({bounds.amplitude_min}–{bounds.amplitude_max})")

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
        merit_config=merit_cfg,
        optimizer_config={
            "algorithm": "adam",
            "steps": NUM_STEPS,
            "peak_lr": adam_cfg.peak_lr,
            "warmup_steps": adam_cfg.warmup_steps,
            "schedule": "warmup_cosine_decay",
            "focus": "target (flat reflectance, fixed spacing)",
            "spectral_threshold": SPECTRAL_THRESHOLD,
            "spectral_samples": SPECTRAL_SAMPLES,
            "num_mirrors": NUM_MIRRORS,
            "fixed_spacing_mm": FIXED_SPACING_MM,
            "optimize_spacings": False,
            "curve": "ConstantCurve (wavelength-flat)",
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
