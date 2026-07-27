"""Pupil optimization driver — Perseus combiner, per-color RGB reflectance.

The Perseus analogue of :mod:`examples.optimize_pupil_rgb`. It optimizes the
**library Perseus** system (:func:`helios.perseus_params.build_parametrized_perseus`
— the differentiable counterpart of :func:`apollo14.perseus.build_perseus_system`)
rather than the flat/Talos cascade:

- **Tilted 10-mirror stack** inside a 3.0 mm chassis, with the Perseus
  rigid-body tilts baked into the builder (pantoscopic tilt about the combiner
  center, projector leaning ``PERSEUS_PROJECTOR_TILT`` ≈ 13.4° into it). The
  library geometry now matches ``examples/visualize_perseus.py`` exactly.
- **Per-color RGB reflectance** — each mirror is a
  :class:`~apollo14.spectral.SumOfGaussiansCurve` on a 5-band basis, so the
  optimizer can white-balance the eyebox (amplitudes **and** widths per band).
- **Frozen spacings** — the inter-mirror gaps are fixed at the Perseus
  ``1.0 mm`` mirror-center y-spacing and never optimized, so hard
  nearest-neighbor binning suffices (no soft/bilinear binning, no gradients).

Two phases, exactly as in the RGB driver:

1. **Target** — pull every eyebox cell to the per-cell brightness target
   (symmetric squared error). Shape (color) term off.
2. **Target + shape** — hold brightness while polishing spectrum-preservation
   (``response(cell, fov, λ) ∝ W(λ)``), which is D65 white for this panel.

The one geometry knob that differs from the Talos drivers is the field of
view: Perseus is scanned over ``±5° × ±6°`` (10° about world-x, 12° about the
in-plane perpendicular) — the FOV established for this combiner.

Run::

    python examples/optimize_pupil_perseus.py
"""

from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp

from apollo14.units import mm, nm, deg
from apollo14.combiner import compensated_reflectances
from apollo14.elements.pupil import RectangularPupil
from apollo14.geometry import planar_grid_points
from apollo14.perseus import (
    PERSEUS_BEAM_HEIGHT,
    PERSEUS_BEAM_WIDTH,
    PERSEUS_CHASSIS_Z,
    PERSEUS_EYEBOX_SIZE,
    PERSEUS_FOV_AROUND_PROJECTOR_Y,
    PERSEUS_FOV_AROUND_X,
    PERSEUS_LIGHT_POSITION,
    PERSEUS_MIRROR_Y_SPACING,
    PERSEUS_NUM_MIRRORS,
    PERSEUS_PROJECTOR_DIRECTION,
)
from apollo14.projector import PlayNitrideLed, FovGrid
from apollo14.spectral import ConstantCurve, SumOfGaussiansCurve

from helios.combiner_params import CombinerParams, ParamBounds, fwhm_to_sigma
from helios.perseus_params import build_parametrized_perseus
from apollo14.trace import prepare_route
from helios.merit import build_combiner_branch_routes
from helios.eyebox import trace_branch_over_fov
from helios.photometry import luminance_weights as photopic_luminance_weights
from helios.pupil_merit import PupilMeritConfig, pupil_merit, merit_breakdown
from helios.adam import AdamConfig, adam_init, adam_step
from helios.io import save_optimization_report, save_run, ScanConfig
from helios.reports.pupil_report import render_pupil_report
from helios.jax_cache import enable_jax_compilation_cache

# Persistent JIT cache survives across runs so successive optimizations
# skip the multi-second compile. Cache dir auto-detects from
# ``$XDG_CACHE_HOME`` or ``~/.cache``, so the script is portable.
enable_jax_compilation_cache()

# ── Perseus geometry (fixed; the builder holds the tilts/chassis) ───────────
# num_mirrors, chassis depth and mirror spacing come from the Perseus library
# constants, so this driver optimizes exactly the system build_perseus_system
# builds — which now matches examples/visualize_perseus.py element-for-element.
NUM_MIRRORS = PERSEUS_NUM_MIRRORS          # 10
CHASSIS_Z = PERSEUS_CHASSIS_Z              # 3.0 mm full combiner depth
SEED_SPACING = PERSEUS_MIRROR_Y_SPACING    # 1.0 mm mirror-center y-gap, frozen
FROZEN_SPACINGS = jnp.full((NUM_MIRRORS - 1,), SEED_SPACING)

# Spacings are NOT a design variable here — only the reflectance curves are.
# Frozen spacings ⇒ hard nearest-neighbor binning is sufficient.
OPTIMIZE_SPACINGS = False

# Reflectance model:
#   "rgb"  — each mirror is a 5-band SumOfGaussiansCurve (amplitude + width per
#            band), optimized in two phases (target, then target + spectrum-
#            preserving shape) for white balance. Like optimize_pupil_rgb.
#   "flat" — each mirror is a single wavelength-uniform ConstantCurve scalar,
#            optimized in ONE target phase (a flat mirror preserves the panel's
#            spectrum, so there is no color term). Like optimize_pupil_flat.
CURVE_MODE = "flat"   # "rgb" | "flat"

# ── Eyebox target region (pre-defined, fixed) ───────────────────────────────
# The Perseus target eyebox: an 8×8 mm square on the pupil plane, 1×1 mm cells.

EYEBOX_HALF_X = PERSEUS_EYEBOX_SIZE / 2      # 4 mm → 8 mm full width on x
EYEBOX_HALF_Y = PERSEUS_EYEBOX_SIZE / 2      # 4 mm → 8 mm full width on y
EYEBOX_NX, EYEBOX_NY = 8, 8                  # 64 cells, exactly 1×1 mm each

# Perseus FOV: 10° about world-x, 12° about the in-plane perpendicular axis.
X_FOV = PERSEUS_FOV_AROUND_X
Y_FOV = PERSEUS_FOV_AROUND_PROJECTOR_Y

# ── Single broadband projector (panel's calibrated white) ─────────────────
# Aimed along the Perseus beam axis (tilted PERSEUS_PROJECTOR_TILT into the
# combiner), so its rays enter through the aperture the builder places on the
# same axis — the exact projector the visualization uses.

PROJECTOR_NX, PROJECTOR_NY = 25, 5
ANGULAR_STEPS_X, ANGULAR_STEPS_Y = 8, 8

PROJECTOR_DIRECTION = PERSEUS_PROJECTOR_DIRECTION

PROJECTOR = PlayNitrideLed.create_broadband(
    position=PERSEUS_LIGHT_POSITION, direction=PROJECTOR_DIRECTION,
    beam_width=PERSEUS_BEAM_WIDTH, beam_height=PERSEUS_BEAM_HEIGHT,
    nx=PROJECTOR_NX, ny=PROJECTOR_NY,
)

# ── Wavelength sampling ─────────────────────────────────────────────────────
# Span W's above-threshold band uniformly. Threshold 0.10 gives a clean
# three-peak envelope without splitting any LED's measured spectrum.

SPECTRAL_THRESHOLD = 0.10
SPECTRAL_SAMPLES = 100

_w_lo, _w_hi = PROJECTOR.spectral_band(threshold=SPECTRAL_THRESHOLD)
TRACE_WAVELENGTHS = jnp.linspace(_w_lo, _w_hi, SPECTRAL_SAMPLES)

# ── Spectrum-preserving shape target ───────────────────────────────────────
# Match the response to the projector's own (peak-normalized) emission shape —
# already calibrated to D65 — which is satisfiable for a 3-band panel, unlike
# chasing a continuous D65 SPD. At valley wavelengths both response and target
# are small, so the shape residual stays bounded.

_spec_wls, _spec_rad = PROJECTOR.spectrum
_W_AT_TRACE = jnp.interp(TRACE_WAVELENGTHS, _spec_wls, _spec_rad)
SHAPE_TARGET = _W_AT_TRACE / _W_AT_TRACE.sum()

# Photopic V·Δλ·K_m weights at the trace wavelengths (uniform spacing, so
# ``mean(diff)`` inside ``photopic_luminance_weights`` is the actual Δλ).
LUMINANCE_TRACE_WEIGHTS = photopic_luminance_weights(TRACE_WAVELENGTHS)

# ── Per-cell brightness target ─────────────────────────────────────────────

NUM_EYEBOX_CELLS = EYEBOX_NX * EYEBOX_NY
# The merit excludes the 4 corner cells (see ``CELL_MASK``); dividing the eyebox
# budget by the *active* count keeps the achieved eyebox total at EYEBOX_TARGET.
NUM_EXCLUDED_CORNER_CELLS = 4
NUM_ACTIVE_EYEBOX_CELLS = NUM_EYEBOX_CELLS - NUM_EXCLUDED_CORNER_CELLS
EYEBOX_TARGET = 0.07
PER_CELL_TARGET = EYEBOX_TARGET / NUM_ACTIVE_EYEBOX_CELLS


# ── Merit & tracer configuration ────────────────────────────────────────────

FOV_GRID = FovGrid(PROJECTOR_DIRECTION, X_FOV, Y_FOV,
                   num_x=ANGULAR_STEPS_X, num_y=ANGULAR_STEPS_Y)

# Phase 1: pull every cell to the brightness target (symmetric squared error).
# Shape term off — color balance is phase 2's job.
merit_cfg_phase1 = PupilMeritConfig(
    target_relative=PER_CELL_TARGET,
    d65_weights=SHAPE_TARGET,
    luminance_weights=LUMINANCE_TRACE_WEIGHTS,
    weight_target=1.0,
    weight_shape=0.0,
    asymmetric_target=False,
)

# Phase 2: hold brightness while polishing spectrum-preservation. The two terms
# have very different natural magnitudes (target ~1e-2, shape ~1e-5 once phase 1
# has produced an approximately spectrum-preserving design), so a large
# weight_shape rebalances the gradient so shape can actually drive deviation
# down. Tune downward (1e4, 1e3) if L_target rises during phase 2.
merit_cfg_phase2 = PupilMeritConfig(
    target_relative=PER_CELL_TARGET,
    d65_weights=SHAPE_TARGET,
    luminance_weights=LUMINANCE_TRACE_WEIGHTS,
    weight_target=1.0,
    weight_shape=100000.0,
    asymmetric_target=False,
)

if CURVE_MODE == "flat":
    # Flat (ConstantCurve): only the per-mirror amplitude is a design variable.
    bounds = ParamBounds(amplitude_min=0.0, amplitude_max=0.20)
else:
    # RGB (SumOfGaussiansCurve): amplitude + Gaussian width (FWHM) per band.
    bounds = ParamBounds(amplitude_max=0.25, fwhm_max_nm=15, fwhm_min_nm=5)


# ── Reference input flux (photometric, matches luminance_weights) ──────────

NUM_RAYS = PROJECTOR_NX * PROJECTOR_NY
INPUT_FLUX = float(NUM_RAYS * jnp.sum(_W_AT_TRACE * LUMINANCE_TRACE_WEIGHTS))

# ── Eye-pupil moving-window aggregation ───────────────────────────────────
# Each merit cell represents what a ~3 mm eye-pupil would see at that eyebox
# point: a (K × K)-cell mean convolution of the fine-binned response. The
# sample grid is padded by ``KERNEL_SIZE_CELLS // 2`` cells per side so a
# ``mode="valid"`` convolution outputs exactly the (EYEBOX_NY, EYEBOX_NX) grid.

KERNEL_SIZE_CELLS = 3
PADDING_CELLS = KERNEL_SIZE_CELLS // 2

_ref_system = build_parametrized_perseus(
    CombinerParams.initial(num_mirrors=NUM_MIRRORS),
    probe_wavelengths=TRACE_WAVELENGTHS, chassis_z=CHASSIS_Z)
_pupil = next(e for e in _ref_system.elements if isinstance(e, RectangularPupil))

# Cell-centered convention: NX × NY equal cells exactly tile the half-extent box.
_CELL_PITCH_X = 2 * EYEBOX_HALF_X / EYEBOX_NX
_CELL_PITCH_Y = 2 * EYEBOX_HALF_Y / EYEBOX_NY

SAMPLE_HALF_X = EYEBOX_HALF_X + PADDING_CELLS * _CELL_PITCH_X
SAMPLE_HALF_Y = EYEBOX_HALF_Y + PADDING_CELLS * _CELL_PITCH_Y
SAMPLE_NX = EYEBOX_NX + 2 * PADDING_CELLS
SAMPLE_NY = EYEBOX_NY + 2 * PADDING_CELLS

# Points the tracer bins onto: the padded sample grid, cell-centered. After
# binning we convolve with the (KERNEL × KERNEL) mean kernel and the result
# lives on the inner (EYEBOX_NY, EYEBOX_NX) grid — what the merit consumes.
EYEBOX_POINTS = planar_grid_points(
    _pupil.position, _pupil.normal,
    SAMPLE_HALF_X, SAMPLE_HALF_Y, SAMPLE_NX, SAMPLE_NY,
    cell_centered=True,
)   # (SAMPLE_NX * SAMPLE_NY, 3)
# Exclude the 4 corner cells — geometric coverage there drops below what the
# optimizer can equalize, so weighting them pulls the whole design down.
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

    Returns ``(S, A, N)`` per-wavelength radiance — the shape the merit and
    report both expect. Spacings are frozen, so hard nearest-neighbor binning
    (``sigma=None``, ``lattice=None``) is used.
    """
    system = build_parametrized_perseus(
        params, probe_wavelengths=TRACE_WAVELENGTHS,
        chassis_z=CHASSIS_Z)
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
        # (S_sample, A) → moving-window mean → (S_eyebox, A). Windowing inside
        # the scan body keeps the per-iteration activation tape at eyebox size.
        windowed = _window_mean(
            binned.T, SAMPLE_NY, SAMPLE_NX, KERNEL_SIZE_CELLS)
        return None, windowed  # (S_eyebox, A)

    # ``jax.checkpoint`` rematerializes each wavelength's forward during
    # backward instead of saving the full per-(wavelength, direction, ray,
    # cell) activation tape.
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


def _freeze_spacings(grad: CombinerParams) -> CombinerParams:
    """Zero the spacing gradient — the geometry is fixed by design."""
    return grad._replace(spacings=jnp.zeros_like(grad.spacings))


def _clip(params: CombinerParams) -> CombinerParams:
    """Clip curves into bounds and re-pin spacings to their frozen value.

    Flat curves clip the single amplitude; RGB curves use ``bounds.clip``
    (amplitude + σ). Either way ``FROZEN_SPACINGS`` is re-injected so the frozen
    geometry stays exact (``bounds.clip`` would otherwise rescale spacings).
    """
    if CURVE_MODE == "flat":
        clipped_amp = jnp.clip(params.curves.amplitude,
                               bounds.amplitude_min, bounds.amplitude_max)
        return params._replace(spacings=FROZEN_SPACINGS,
                               curves=ConstantCurve(amplitude=clipped_amp))
    clipped = bounds.clip(params)
    return clipped._replace(spacings=FROZEN_SPACINGS)


# ── Adam optimizer ──────────────────────────────────────────────────────────

PHASE1_STEPS = 100
PHASE2_STEPS = 100

adam_cfg_phase1 = AdamConfig(peak_lr=3e-3, warmup_steps=20, num_steps=PHASE1_STEPS)
# Phase 2 polishes the shape term in a flat region; drop the LR so Adam's
# 1/sqrt(variance) step actually descends instead of dithering.
adam_cfg_phase2 = AdamConfig(peak_lr=5e-4, warmup_steps=10, num_steps=PHASE2_STEPS)


# ── Curve-mode helpers (flat ConstantCurve vs RGB SumOfGaussiansCurve) ──────

# 5-band RGB reflectance basis (fixed centers): R/G/B primaries (446, 545,
# 627 nm) plus two intermediates (490, 590 nm) so a per-mirror curve can be
# near-wavelength-flat at a modest FWHM without one Gaussian bridging the R–B gap.
_RGB_CENTERS = jnp.array([446.0, 490.0, 545.0, 590.0, 627.0]) * nm


def _warm_start_amps() -> jnp.ndarray:
    """Talos cascade-compensation seed amplitudes ``(M,)``.

    Each mirror's absolute reflected fraction of the original beam is equal
    (``ratio / (1 - i·ratio)``), so downstream mirrors start higher to offset
    upstream transmission losses. Pick-off scaled to the mirror count.
    """
    base_ratio = 0.05 * 6 / NUM_MIRRORS
    return compensated_reflectances(
        base_ratio, NUM_MIRRORS, num_samples=1).reshape(-1).astype(jnp.float32)


def _build_initial_params() -> CombinerParams:
    """Warm-started initial design for the selected ``CURVE_MODE``."""
    amps = _warm_start_amps()   # (M,)
    if CURVE_MODE == "flat":
        return CombinerParams(spacings=FROZEN_SPACINGS,
                              curves=ConstantCurve(amplitude=amps))
    seed = CombinerParams.initial(
        num_mirrors=NUM_MIRRORS, amplitude=float(amps.mean()), width_nm=80,
        spacing_mm=float(SEED_SPACING / mm), centers=_RGB_CENTERS)
    return seed._replace(
        spacings=FROZEN_SPACINGS,
        curves=SumOfGaussiansCurve(
            amplitude=jnp.broadcast_to(amps[:, None],
                                       seed.curves.amplitude.shape),
            sigma=seed.curves.sigma,
            centers=seed.curves.centers,
        ),
    )


def _ceiling_params(template: CombinerParams) -> CombinerParams:
    """All-mirrors-at-max design, for the brightness-ceiling diagnostic."""
    if CURVE_MODE == "flat":
        curves = ConstantCurve(amplitude=jnp.full_like(
            template.curves.amplitude, bounds.amplitude_max))
    else:
        curves = SumOfGaussiansCurve(
            amplitude=jnp.full_like(template.curves.amplitude, bounds.amplitude_max),
            sigma=jnp.full_like(template.curves.sigma,
                                fwhm_to_sigma(bounds.fwhm_max_nm * nm)),
            centers=template.curves.centers)
    return CombinerParams(spacings=template.spacings, curves=curves)


def _num_design_vars(params: CombinerParams) -> int:
    if CURVE_MODE == "flat":
        return int(params.curves.amplitude.size)
    return int(params.curves.amplitude.size + params.curves.sigma.size)


def _print_final_curves(params: CombinerParams) -> None:
    """Print optimized reflectance per mirror + a bounds-pegged diagnostic."""
    if CURVE_MODE == "flat":
        print("Final flat reflectance per mirror:")
        for mirror_idx in range(NUM_MIRRORS):
            print(f"  m{mirror_idx}: {float(params.curves.amplitude[mirror_idx]):.4f}")
        pegged_lo = int(jnp.sum(params.curves.amplitude <= bounds.amplitude_min + 1e-4))
        pegged_hi = int(jnp.sum(params.curves.amplitude >= bounds.amplitude_max - 1e-4))
        print(f"\nReflectances at a bound: {pegged_lo} at lower / "
              f"{pegged_hi} at upper ({bounds.amplitude_min}–{bounds.amplitude_max})")
        return
    print("Final amplitudes per mirror (per basis):")
    for mirror_idx in range(NUM_MIRRORS):
        row = "  ".join(f"{float(a):.4f}" for a in params.curves.amplitude[mirror_idx])
        print(f"  m{mirror_idx}: {row}")
    print("Final widths per mirror (nm):")
    for mirror_idx in range(NUM_MIRRORS):
        row = "  ".join(f"{float(w) / nm:.1f}" for w in params.curves.sigma[mirror_idx])
        print(f"  m{mirror_idx}: {row}")
    sigma_max = fwhm_to_sigma(bounds.fwhm_max_nm * nm)
    amp_pegged = int(jnp.sum(params.curves.amplitude >= bounds.amplitude_max - 1e-4))
    sigma_pegged = int(jnp.sum(params.curves.sigma >= sigma_max - 1e-4))
    print(f"\nParameters at upper bound:")
    print(f"  amplitudes: {amp_pegged} / {params.curves.amplitude.size} "
          f"(at {bounds.amplitude_max})")
    print(f"  sigmas:     {sigma_pegged} / {params.curves.sigma.size} "
          f"(at FWHM {bounds.fwhm_max_nm:.1f} nm)")


# ── Run ─────────────────────────────────────────────────────────────────────

def _print_breakdown(label, bd):
    print(f"\n{label}: {float(bd['total']):.5f}")
    print(f"  target={float(bd['target']):.5f}  "
          f"shape={float(bd['shape']):.5f}")
    print(f"  brightness mean_rel={float(bd['mean_brightness_rel']):.5f}  "
          f"std_rel={float(bd['brightness_std_rel']):.5f}  "
          f"min_rel={float(bd['min_brightness_rel']):.5f}  "
          f"max_rel={float(bd['max_brightness_rel']):.5f}")


RUNS_ROOT = Path("examples/reports/optimize_pupil_perseus")


def main():
    run_dir = RUNS_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    initial_params = _build_initial_params()
    params = initial_params
    state = adam_init(params)
    is_flat = CURVE_MODE == "flat"

    model_label = "flat reflectance" if is_flat else "RGB reflectance"
    var_detail = ("one flat reflectance per mirror" if is_flat
                  else "per-mirror per-band amplitude + sigma")
    print(f"── Perseus pupil optimization ({model_label}, spacings frozen) ──")
    print(f"Variables: {_num_design_vars(params)} ({var_detail})")
    window_mm_x = KERNEL_SIZE_CELLS * float(_CELL_PITCH_X) / mm
    window_mm_y = KERNEL_SIZE_CELLS * float(_CELL_PITCH_Y) / mm
    print(f"Mirrors:   {NUM_MIRRORS}, frozen spacing {float(SEED_SPACING)/mm:.2f} mm, "
          f"chassis {float(CHASSIS_Z)/mm:.2f} mm")
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
    if not is_flat:
        print(f"Shape target: projector W spectrum (preserve panel's D65 white)")
    print(f"I_in:      {INPUT_FLUX:.1f}  "
          f"(target_relative={merit_cfg_phase1.target_relative})")

    # Ceiling diagnostic — brightest achievable if every mirror pegs at max.
    ceiling_response = _compute_spectral_response(_ceiling_params(initial_params))
    ceiling_lum_per_angle = jnp.sum(
        ceiling_response * LUMINANCE_TRACE_WEIGHTS.reshape(1, 1, -1), axis=-1)
    ceiling_brightness = jnp.mean(ceiling_lum_per_angle, axis=-1) / INPUT_FLUX
    print(f"Ceiling per-cell brightness (reflectance=max): "
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
        params, state = adam_step(params, _freeze_spacings(grad), state,
                                  adam_cfg_phase1)
        params = _clip(params)
        loss_history.append(float(loss))
        print(f"step {step+1:4d}/{PHASE1_STEPS}  loss={float(loss):.8f}")

    phase1_breakdown = breakdown_fn(params, merit_cfg_phase1)
    _print_breakdown("Phase 1 result", phase1_breakdown)

    # Phase 2 (RGB only): polish spectrum-preservation. A flat mirror preserves
    # the panel spectrum by construction, so a shape phase has nothing to do.
    if is_flat:
        final_merit_cfg = merit_cfg_phase1
        final_breakdown = phase1_breakdown
    else:
        print("\n── Phase 2: target + spectrum-preserving shape ──")
        state = adam_init(params)
        for step in range(PHASE2_STEPS):
            loss, grad = value_and_grad_phase2(params)
            params, state = adam_step(params, _freeze_spacings(grad), state,
                                      adam_cfg_phase2)
            params = _clip(params)
            loss_history.append(float(loss))
            print(f"step {step+1:4d}/{PHASE2_STEPS}  loss={float(loss):.8f}")
        final_merit_cfg = merit_cfg_phase2
        final_breakdown = breakdown_fn(params, final_merit_cfg)

    _print_breakdown("Final merit", final_breakdown)

    print("\nFrozen spacings (mm):",
          [f"{float(spacing)/mm:.3f}" for spacing in params.spacings])
    _print_final_curves(params)

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

    final_system = build_parametrized_perseus(
        params, probe_wavelengths=TRACE_WAVELENGTHS,
        chassis_z=CHASSIS_Z)

    if is_flat:
        optimizer_config = {
            "algorithm": "adam",
            "steps": PHASE1_STEPS,
            "peak_lr": adam_cfg_phase1.peak_lr,
            "warmup_steps": adam_cfg_phase1.warmup_steps,
            "focus": "target (flat reflectance, fixed spacing)",
            "curve": "ConstantCurve (wavelength-flat)",
        }
    else:
        optimizer_config = {
            "algorithm": "adam_two_phase",
            "phase1": {"steps": PHASE1_STEPS, "peak_lr": adam_cfg_phase1.peak_lr,
                       "warmup_steps": adam_cfg_phase1.warmup_steps, "focus": "target"},
            "phase2": {"steps": PHASE2_STEPS, "peak_lr": adam_cfg_phase2.peak_lr,
                       "warmup_steps": adam_cfg_phase2.warmup_steps,
                       "focus": "target+shape (spectrum-preserving)"},
            "schedule": "warmup_cosine_decay",
            "total_steps": PHASE1_STEPS + PHASE2_STEPS,
            "curve": "SumOfGaussiansCurve (5-band RGB)",
        }
    optimizer_config.update({
        "spectral_threshold": SPECTRAL_THRESHOLD,
        "spectral_samples": SPECTRAL_SAMPLES,
        "num_mirrors": NUM_MIRRORS,
        "optimize_spacings": OPTIMIZE_SPACINGS,
        "curve_mode": CURVE_MODE,
        "geometry": "perseus (tilted 10-mirror, build_parametrized_perseus)",
    })

    report_path = save_optimization_report(
        run_dir,
        system=final_system,
        projectors=[PROJECTOR],
        fov_grid=FOV_GRID,
        merit_config=final_merit_cfg,
        optimizer_config=optimizer_config,
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

    # Cell-centered axes matching the merit grid convention.
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
        base_direction=PROJECTOR_DIRECTION,
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
