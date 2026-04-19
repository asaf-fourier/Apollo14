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
from helios.merit import build_combiner_pupil_routes, DEFAULT_WAVELENGTHS
from helios.eyebox import (
    compute_eyebox_response, eyebox_grid_points,
)
from helios.pupil_merit import (
    PupilMeritConfig, pupil_merit, merit_breakdown,
)
from helios.adam import AdamConfig, adam_init, adam_step, lr_schedule
from helios.io import save_optimization_report, save_run, ScanConfig
from helios.reports.run_report import render_report


# ── Eyebox target region (pre-defined, fixed) ───────────────────────────────

EYEBOX_RADIUS = 5.0 * mm         # 10×10 mm centered eyebox
EYEBOX_NX, EYEBOX_NY = 7, 7      # 49 cells, ~1.4 mm each

X_FOV = 7.0 * deg
Y_FOV = 7.0 * deg

# ── RGB projectors (PlayNitride micro-LEDs, one per color) ─────────────────

PROJECTOR_NX, PROJECTOR_NY = 7, 7
BLUE_FALLOFF = 0.02 / (6.0 * deg)

PROJECTORS = [
    PlayNitrideLed.create(
        position=DEFAULT_LIGHT_POSITION, direction=DEFAULT_LIGHT_DIRECTION,
        beam_width=10.0 * mm, beam_height=2.0 * mm,
        nx=PROJECTOR_NX, ny=PROJECTOR_NY, color="R",
    ),
    PlayNitrideLed.create(
        position=DEFAULT_LIGHT_POSITION, direction=DEFAULT_LIGHT_DIRECTION,
        beam_width=10.0 * mm, beam_height=2.0 * mm,
        nx=PROJECTOR_NX, ny=PROJECTOR_NY, color="G",
    ),
    PlayNitrideLed.create(
        position=DEFAULT_LIGHT_POSITION, direction=DEFAULT_LIGHT_DIRECTION,
        beam_width=10.0 * mm, beam_height=2.0 * mm,
        nx=PROJECTOR_NX, ny=PROJECTOR_NY, color="B",
        falloff_x=BLUE_FALLOFF, falloff_y=BLUE_FALLOFF,
    ),
]

# ── Merit & tracer configuration ────────────────────────────────────────────

FOV_GRID = FovGrid(DEFAULT_LIGHT_DIRECTION, X_FOV, Y_FOV, num_x=7, num_y=7)

merit_cfg = PupilMeritConfig(
    threshold_relative=0.02,       # require ≥2% of input flux per eyebox cell
    cap_relative=0.08,             # and ≤8% — efficiency cap
    weight_shape=1.0,
    weight_coverage=3.0,
    weight_warmup=0.5,
    weight_cap=0.2,
    sigmoid_steepness=40.0,
    soft_min_temperature=15.0,
)

bounds = ParamBounds()


# ── Reference input flux ────────────────────────────────────────────────────
# True on-axis flux: sum of actual ray intensities across all three channels.

INPUT_FLUX = sum(
    float(jnp.sum(proj.generate_rays(
        direction=DEFAULT_LIGHT_DIRECTION, wavelength=wl).intensity))
    for proj, wl in zip(PROJECTORS, DEFAULT_WAVELENGTHS)
)


# ── Build the pupil cell grid and mask for the target region ────────────────
# We build the grid from the pupil element in the reference system. The
# grid is fixed for the whole optimization (target eyebox is pre-defined).

_ref_system = build_parametrized_system(CombinerParams.initial())
_pupil = next(e for e in _ref_system.elements if isinstance(e, RectangularPupil))
EYEBOX_POINTS = eyebox_grid_points(
    _pupil.position, _pupil.normal, EYEBOX_RADIUS, EYEBOX_NX, EYEBOX_NY,
)   # (S, 3)
CELL_MASK = jnp.ones(EYEBOX_POINTS.shape[0])   # all grid cells belong to the target


# ── Loss function ───────────────────────────────────────────────────────────


def _compute_rgb_response(params: CombinerParams):
    """Trace each wavelength with its own projector and stack responses."""
    system = build_parametrized_system(params)
    routes_per_wavelength = build_combiner_pupil_routes(
        system, DEFAULT_WAVELENGTHS, num_mirrors=NUM_MIRRORS,
    )
    channel_responses = []
    for wl_idx, projector in enumerate(PROJECTORS):
        single_response = compute_eyebox_response(
            [routes_per_wavelength[wl_idx]], projector,
            FOV_GRID, EYEBOX_POINTS,
            wavelengths=[DEFAULT_WAVELENGTHS[wl_idx]],
        )  # (S, A, 1)
        channel_responses.append(single_response[..., 0])  # (S, A)
    return jnp.stack(channel_responses, axis=-1)  # (S, A, 3)


def loss_fn(params: CombinerParams) -> jnp.ndarray:
    """Scalar pupil merit as a function of design variables."""
    response = _compute_rgb_response(params)
    return pupil_merit(response, INPUT_FLUX, merit_cfg, cell_mask=CELL_MASK)


def breakdown_fn(params: CombinerParams) -> dict:
    """Per-term merit breakdown for logging."""
    response = _compute_rgb_response(params)
    return merit_breakdown(response, INPUT_FLUX, merit_cfg, cell_mask=CELL_MASK)


value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))


# ── Adam optimizer ──────────────────────────────────────────────────────────

adam_cfg = AdamConfig(peak_lr=5e-4, warmup_steps=20, num_steps=400)


# ── Run ─────────────────────────────────────────────────────────────────────

def main():
    initial_params = CombinerParams.initial()
    params = initial_params
    state = adam_init(params)

    print("── Pupil optimization (spacings + Gaussian reflectance) ──")
    print(f"Variables: {params.spacings.size + params.amplitudes.size + params.widths.size}")
    print(f"Eyebox:    {2*EYEBOX_RADIUS/mm:.1f}×{2*EYEBOX_RADIUS/mm:.1f} mm, "
          f"{EYEBOX_NX}×{EYEBOX_NY} cells")
    print(f"FOV:       ±{X_FOV/deg:.1f}° × ±{Y_FOV/deg:.1f}°, "
          f"{FOV_GRID.num_x}×{FOV_GRID.num_y} samples")
    print(f"I_in:      {INPUT_FLUX:.1f}  (threshold_relative={merit_cfg.threshold_relative})")

    initial_breakdown = breakdown_fn(params)
    print(f"\nInitial merit: {float(initial_breakdown['total']):.5f}")
    print(f"  shape={float(initial_breakdown['shape']):.5f}  "
          f"coverage={float(initial_breakdown['coverage']):.5f}  "
          f"warmup={float(initial_breakdown['warmup']):.5f}  "
          f"cap={float(initial_breakdown['cap']):.5f}")
    print(f"  active_fraction={float(initial_breakdown['active_fraction']):.3f}  "
          f"min_rel={float(initial_breakdown['min_brightness_rel']):.4f}")

    loss_history = []
    for step in range(adam_cfg.num_steps):
        loss, grad = value_and_grad_fn(params)
        params, state = adam_step(params, grad, state, adam_cfg)
        params = bounds.clip(params)
        loss_history.append(float(loss))
        print(f"step {step+1:4d}  loss={float(loss):.5f}")

    final_breakdown = breakdown_fn(params)
    print(f"\nFinal merit:   {float(final_breakdown['total']):.5f}")
    print(f"  shape={float(final_breakdown['shape']):.5f}  "
          f"coverage={float(final_breakdown['coverage']):.5f}  "
          f"warmup={float(final_breakdown['warmup']):.5f}  "
          f"cap={float(final_breakdown['cap']):.5f}")
    print(f"  active_fraction={float(final_breakdown['active_fraction']):.3f}  "
          f"min_rel={float(final_breakdown['min_brightness_rel']):.4f}")

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

    final_system = build_parametrized_system(params)
    report_path = save_optimization_report(
        "examples/reports/optimize_pupil",
        system=final_system,
        projectors=PROJECTORS,
        fov_grid=FOV_GRID,
        merit_config=merit_cfg,
        optimizer_config={
            "algorithm": "adam",
            "peak_learning_rate": adam_cfg.peak_lr,
            "warmup_steps": adam_cfg.warmup_steps,
            "schedule": "warmup_cosine_decay",
            "beta1": adam_cfg.beta1,
            "beta2": adam_cfg.beta2,
            "epsilon": adam_cfg.epsilon,
            "num_steps": adam_cfg.num_steps,
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

    response = _compute_rgb_response(params)
    pupil_x_mm = jnp.linspace(-EYEBOX_RADIUS, EYEBOX_RADIUS, EYEBOX_NX)
    pupil_y_mm = jnp.linspace(-EYEBOX_RADIUS, EYEBOX_RADIUS, EYEBOX_NY)
    scan_cfg = ScanConfig(
        base_direction=DEFAULT_LIGHT_DIRECTION,
        x_fov=float(X_FOV), y_fov=float(Y_FOV),
        num_x=FOV_GRID.num_x, num_y=FOV_GRID.num_y,
    )
    run_dir = save_run(
        "examples/reports/optimize_pupil",
        final_system, PROJECTORS[0], scan_cfg,
        response=response,
        pupil_x_mm=pupil_x_mm,
        pupil_y_mm=pupil_y_mm,
        scan_angles=FOV_GRID.angles_grid,
        wavelengths_nm=DEFAULT_WAVELENGTHS / nm,
    )
    html_path = render_report(run_dir)
    print(f"Saved run report: {html_path}")


if __name__ == "__main__":
    main()
