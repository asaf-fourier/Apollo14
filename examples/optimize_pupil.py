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

from apollo14.units import mm, deg
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


# ── Eyebox target region (pre-defined, fixed) ───────────────────────────────

EYEBOX_RADIUS = 5.0 * mm         # 10×10 mm centered eyebox
EYEBOX_NX, EYEBOX_NY = 7, 7      # 49 cells, ~1.4 mm each

X_FOV = 10.0 * deg
Y_FOV = 10.0 * deg

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
# One FOV direction delivers one full beam — since every beam ray has
# intensity 1 in the tracer's accounting, input_flux equals the number of
# rays per beam. This keeps ``threshold_relative`` invariant to beam density.

INPUT_FLUX = float(PROJECTOR_NX * PROJECTOR_NY)


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

LEARNING_RATE = 5e-4
ADAM_BETA1, ADAM_BETA2, ADAM_EPS = 0.9, 0.999, 1e-8
NUM_STEPS = 200


def adam_init(params):
    zeros = jax.tree.map(jnp.zeros_like, params)
    return zeros, zeros, jnp.array(0.0)


def adam_step(params, state):
    moment, variance, step_count = state
    loss, grad = value_and_grad_fn(params)
    step_count = step_count + 1
    moment = jax.tree.map(
        lambda mom, grd: ADAM_BETA1 * mom + (1 - ADAM_BETA1) * grd,
        moment, grad,
    )
    variance = jax.tree.map(
        lambda var, grd: ADAM_BETA2 * var + (1 - ADAM_BETA2) * grd ** 2,
        variance, grad,
    )
    corrected_moment = jax.tree.map(
        lambda mom: mom / (1 - ADAM_BETA1 ** step_count), moment,
    )
    corrected_variance = jax.tree.map(
        lambda var: var / (1 - ADAM_BETA2 ** step_count), variance,
    )
    new_params = jax.tree.map(
        lambda param, mom, var: param - LEARNING_RATE * mom / (jnp.sqrt(var) + ADAM_EPS),
        params, corrected_moment, corrected_variance,
    )
    new_params = bounds.clip(new_params)
    return new_params, (moment, variance, step_count), loss


# ── Run ─────────────────────────────────────────────────────────────────────

def main():
    params = CombinerParams.initial()
    state = adam_init(params)

    print("── Pupil optimization (spacings + Gaussian reflectance) ──")
    print(f"Variables: {params.spacings.size + params.amplitudes.size + params.widths.size}")
    print(f"Eyebox:    {2*EYEBOX_RADIUS/mm:.1f}×{2*EYEBOX_RADIUS/mm:.1f} mm, "
          f"{EYEBOX_NX}×{EYEBOX_NY} cells")
    print(f"FOV:       ±{X_FOV/deg:.1f}° × ±{Y_FOV/deg:.1f}°, "
          f"{FOV_GRID.num_x}×{FOV_GRID.num_y} samples")
    print(f"I_in:      {INPUT_FLUX:.1f}  (threshold_relative={merit_cfg.threshold_relative})")

    init = breakdown_fn(params)
    print(f"\nInitial merit: {float(init['total']):.5f}")
    print(f"  shape={float(init['shape']):.5f}  "
          f"coverage={float(init['coverage']):.5f}  "
          f"warmup={float(init['warmup']):.5f}  "
          f"cap={float(init['cap']):.5f}")
    print(f"  active_fraction={float(init['active_fraction']):.3f}  "
          f"min_rel={float(init['min_brightness_rel']):.4f}")

    for step in range(NUM_STEPS):
        params, state, loss = adam_step(params, state)
        print(f"step {step+1:4d}  loss={float(loss):.5f}")

    final = breakdown_fn(params)
    print(f"\nFinal merit:   {float(final['total']):.5f}")
    print(f"  shape={float(final['shape']):.5f}  "
          f"coverage={float(final['coverage']):.5f}  "
          f"warmup={float(final['warmup']):.5f}  "
          f"cap={float(final['cap']):.5f}")
    print(f"  active_fraction={float(final['active_fraction']):.3f}  "
          f"min_rel={float(final['min_brightness_rel']):.4f}")

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


if __name__ == "__main__":
    main()
