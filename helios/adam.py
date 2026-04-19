"""Adam optimizer with warmup + cosine decay LR schedule.

Generic pytree-aware Adam that works with any JAX-compatible parameter
structure (NamedTuples, nested dicts, etc.).  The optimizer does not
own the loss function — the caller supplies gradients, keeping the
optimizer decoupled from the merit/tracer stack.

Usage::

    from helios.adam import AdamConfig, adam_init, adam_step

    cfg = AdamConfig(peak_lr=5e-4, num_steps=400)
    state = adam_init(params)
    for step in range(cfg.num_steps):
        loss, grad = value_and_grad_fn(params)
        params, state = adam_step(params, grad, state, cfg)
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class AdamConfig:
    peak_lr: float = 5e-4
    warmup_steps: int = 20
    num_steps: int = 400
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8


def lr_schedule(step_count, config: AdamConfig) -> jnp.ndarray:
    warmup = jnp.minimum(step_count / config.warmup_steps, 1.0)
    decay_fraction = ((step_count - config.warmup_steps)
                      / (config.num_steps - config.warmup_steps))
    cosine_decay = 0.5 * (1.0 + jnp.cos(
        jnp.pi * jnp.clip(decay_fraction, 0.0, 1.0)))
    return config.peak_lr * warmup * cosine_decay


def adam_init(params):
    """Initialize Adam state: (moment, variance, step_count)."""
    zeros = jax.tree.map(jnp.zeros_like, params)
    return zeros, zeros, jnp.array(0.0)


def adam_step(params, grad, state, config: AdamConfig):
    """One Adam update step. Returns (new_params, new_state)."""
    moment, variance, step_count = state
    step_count = step_count + 1
    learning_rate = lr_schedule(step_count, config)
    moment = jax.tree.map(
        lambda mom, grd: config.beta1 * mom + (1 - config.beta1) * grd,
        moment, grad,
    )
    variance = jax.tree.map(
        lambda var, grd: config.beta2 * var + (1 - config.beta2) * grd ** 2,
        variance, grad,
    )
    corrected_moment = jax.tree.map(
        lambda mom: mom / (1 - config.beta1 ** step_count), moment,
    )
    corrected_variance = jax.tree.map(
        lambda var: var / (1 - config.beta2 ** step_count), variance,
    )
    new_params = jax.tree.map(
        lambda param, mom, var: param - learning_rate * mom / (jnp.sqrt(var) + config.epsilon),
        params, corrected_moment, corrected_variance,
    )
    return new_params, (moment, variance, step_count)
