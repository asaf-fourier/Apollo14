"""
Eyebox optimization — 23 variables for uniform D65-balanced FOV.

Optimizes per-mirror per-color reflectances (6x3 = 18 variables) and
inter-mirror intervals (5 variables) so the combiner delivers uniform,
D65-white-balanced intensity across the eyebox and FOV.

Uses Adam (manual, no extra dependencies) with the full optimization
loop JIT-compiled via jax.lax.scan.
"""

import json
import jax
import jax.numpy as jnp
from typing import NamedTuple

from apollo14.combiner import CombinerConfig
from apollo14.jax_tracer import params_from_config
from apollo14.units import mm

from helios.eyebox import (
    EyeboxConfig, eyebox_sample_points,
    compute_eyebox_response, eyebox_merit, visible_fov,
)


# ── Optimization variables ──────────────────────────────────────────────────

class OptVars(NamedTuple):
    """Optimization variables (JAX pytree — works with jax.grad)."""
    reflectances: jnp.ndarray   # (M, 3) per-mirror, per-color reflectance
    intervals: jnp.ndarray      # (M-1,) distance between consecutive mirrors


# ── Setup ───────────────────────────────────────────────────────────────────

config = CombinerConfig.default()
base_params = params_from_config(config)
n_glass = float(config.chassis.material.n(config.light.wavelength))
M = config.num_mirrors

eyebox_pts = eyebox_sample_points(
    config.pupil.center, config.pupil.normal, config.pupil.radius)

mc = EyeboxConfig(
    target_intensity=0.06,
    n_fov_x=5,
    n_fov_y=5,
    sigma=2.0,
    w_uniformity=1.0,
    w_intensity=10.0,
)

# Precompute the offset direction (mirrors step along -y, scaled by normal)
mirror_y_width = config.mirror.y_width
chassis_z = float(config.chassis.dimensions[2])
mirror_edge_to_center_y = 0.5 * jnp.sqrt(mirror_y_width ** 2 - chassis_z ** 2)
first_pos = (config.chassis.first_mirror_center -
             jnp.array([0.0, float(mirror_edge_to_center_y), 0.0]))
unit_offset = jnp.array([0.0, 1.0 / float(config.mirror.normal[1]), 0.0])

# Initial values
init_interval = config.chassis.distance_between_mirrors  # 1.47 mm
init_vars = OptVars(
    reflectances=base_params.mirror_reflectances,  # (M, 3)
    intervals=jnp.full(M - 1, init_interval),      # (M-1,)
)


# ── Build params from optimization variables ────────────────────────────────

def vars_to_params(v):
    """Convert OptVars → CombinerParams by updating reflectances + positions."""
    # Positions from intervals: cumulative offset from first mirror
    cumulative = jnp.concatenate([jnp.zeros(1), jnp.cumsum(v.intervals)])
    positions = first_pos[None, :] - cumulative[:, None] * unit_offset[None, :]

    return base_params._replace(
        mirror_reflectances=v.reflectances,
        mirror_positions=positions,
    )


# ── Loss function ───────────────────────────────────────────────────────────

def loss_fn(v):
    """Eyebox merit as a function of reflectances (M,3) + intervals (M-1,)."""
    params = vars_to_params(v)
    response, _ = compute_eyebox_response(
        params, n_glass,
        config.light.position, config.light.direction,
        config.light.x_fov, config.light.y_fov,
        eyebox_pts, mc,
    )
    return eyebox_merit(response, mc)


value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))


# ── Adam optimizer ──────────────────────────────────────────────────────────

lr = 5e-4
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
num_steps = 300

# Physical bounds
refl_min, refl_max = 0.01, 0.30
interval_min, interval_max = 0.5 * mm, 3.0 * mm


def clip_vars(v):
    """Clamp to physical bounds."""
    return OptVars(
        reflectances=jnp.clip(v.reflectances, refl_min, refl_max),
        intervals=jnp.clip(v.intervals, interval_min, interval_max),
    )


def adam_step(carry, _):
    """One Adam update step (pytree-aware)."""
    v, m, v_adam, t = carry

    loss, grad = value_and_grad_fn(v)
    t = t + 1

    m = jax.tree.map(lambda mi, gi: beta1 * mi + (1 - beta1) * gi, m, grad)
    v_adam = jax.tree.map(lambda vi, gi: beta2 * vi + (1 - beta2) * gi ** 2, v_adam, grad)

    m_hat = jax.tree.map(lambda mi: mi / (1 - beta1 ** t), m)
    v_hat = jax.tree.map(lambda vi: vi / (1 - beta2 ** t), v_adam)

    v_new = jax.tree.map(
        lambda xi, mh, vh: xi - lr * mh / (jnp.sqrt(vh) + eps),
        v, m_hat, v_hat,
    )
    v_new = clip_vars(v_new)

    return (v_new, m, v_adam, t), loss


# ── Run optimization ────────────────────────────────────────────────────────

print("── Eyebox Optimization: 23 Variables ──")
print(f"   18 reflectances (6 mirrors x 3 colors)")
print(f"    5 inter-mirror intervals\n")

init_loss = float(loss_fn(init_vars))
print(f"Initial intervals (mm):  {[f'{float(d):.2f}' for d in init_vars.intervals]}")
print(f"Initial merit:           {init_loss:.6f}")

# Initialize Adam state (zeros with same pytree structure)
zero_state = jax.tree.map(jnp.zeros_like, init_vars)
init_carry = (init_vars, zero_state, zero_state, jnp.array(0.0))

print(f"\nRunning {num_steps} Adam steps (JIT-compiled)...")
final_carry, loss_history = jax.lax.scan(
    adam_step, init_carry, None, length=num_steps)

final_vars = final_carry[0]
final_loss = float(loss_history[-1])
print("Done.\n")


# ── Results ─────────────────────────────────────────────────────────────────

print("── Results ──\n")
print(f"Final merit:       {final_loss:.6f}")
print(f"Improvement:       {init_loss / final_loss:.1f}x\n")

print("Intervals (mm):")
print(f"  Initial: {[f'{float(d):.2f}' for d in init_vars.intervals]}")
print(f"  Final:   {[f'{float(d):.2f}' for d in final_vars.intervals]}")

print(f"\nReflectances per mirror (initial → final):")
print(f"  {'Mirror':<8} {'R_init':>7}→{'R_fin':>6}  {'G_init':>7}→{'G_fin':>6}  {'B_init':>7}→{'B_fin':>6}")
for i in range(M):
    ri, gi, bi = [float(init_vars.reflectances[i, c]) for c in range(3)]
    rf, gf, bf = [float(final_vars.reflectances[i, c]) for c in range(3)]
    print(f"  {i:<8} {ri:>7.4f}→{rf:>6.4f}  {gi:>7.4f}→{gf:>6.4f}  {bi:>7.4f}→{bf:>6.4f}")

# Convergence
print(f"\nLoss at step   1: {float(loss_history[0]):.6f}")
print(f"Loss at step 100: {float(loss_history[99]):.6f}")
print(f"Loss at step 200: {float(loss_history[199]):.6f}")
print(f"Loss at step 300: {float(loss_history[-1]):.6f}")


# ── FOV diagnostic ──────────────────────────────────────────────────────────

print("\n── Visible FOV Diagnostic ──\n")

sample_labels = ["center", "corner_1", "corner_2", "corner_3", "corner_4"]


def compute_fov(v):
    params = vars_to_params(v)
    response, _ = compute_eyebox_response(
        params, n_glass,
        config.light.position, config.light.direction,
        config.light.x_fov, config.light.y_fov,
        eyebox_pts, mc,
    )
    return visible_fov(response, mc.visibility_threshold)


init_fov = compute_fov(init_vars)
final_fov = compute_fov(final_vars)

print(f"  {'Sample':<10} {'Initial':>10} {'Final':>10}")
for i, label in enumerate(sample_labels):
    print(f"  {label:<10} {float(init_fov[i]):>9.1%} {float(final_fov[i]):>9.1%}")


# ── Save results to JSON ──────────────────────────────────────────────────

results = {
    "num_steps": num_steps,
    "initial_merit": float(init_loss),
    "final_merit": float(final_loss),
    "improvement": float(init_loss / final_loss),
    "intervals_mm": {
        "initial": [float(d) for d in init_vars.intervals],
        "final": [float(d) for d in final_vars.intervals],
    },
    "reflectances": {
        "initial": [[float(init_vars.reflectances[i, c]) for c in range(3)] for i in range(M)],
        "final": [[float(final_vars.reflectances[i, c]) for c in range(3)] for i in range(M)],
    },
    "loss_history": [float(l) for l in loss_history],
    "visible_fov": {
        label: {"initial": float(init_fov[i]), "final": float(final_fov[i])}
        for i, label in enumerate(sample_labels)
    },
}

out_path = "examples/optimize_eyebox_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
print("Done.")
