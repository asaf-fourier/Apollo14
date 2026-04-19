import jax
import jax.numpy as jnp

from helios.adam import AdamConfig, adam_init, adam_step, lr_schedule


# ── LR schedule ────────────────────────────────────────────────────────────


def test_lr_schedule_zero_at_start():
    cfg = AdamConfig(peak_lr=1e-3, warmup_steps=10, num_steps=100)
    lr = lr_schedule(jnp.array(0.0), cfg)
    assert float(lr) == 0.0


def test_lr_schedule_peak_at_warmup_end():
    cfg = AdamConfig(peak_lr=1e-3, warmup_steps=10, num_steps=100)
    lr = lr_schedule(jnp.array(10.0), cfg)
    assert jnp.isclose(lr, 1e-3, atol=1e-8)


def test_lr_schedule_linear_warmup():
    cfg = AdamConfig(peak_lr=1e-3, warmup_steps=20, num_steps=100)
    lr_mid = lr_schedule(jnp.array(10.0), cfg)
    assert jnp.isclose(lr_mid, 0.5e-3, atol=1e-8)


def test_lr_schedule_zero_at_end():
    cfg = AdamConfig(peak_lr=1e-3, warmup_steps=10, num_steps=100)
    lr = lr_schedule(jnp.array(100.0), cfg)
    assert jnp.isclose(lr, 0.0, atol=1e-8)


def test_lr_schedule_monotone_decay_after_warmup():
    cfg = AdamConfig(peak_lr=1e-3, warmup_steps=10, num_steps=100)
    lrs = [float(lr_schedule(jnp.array(float(s)), cfg))
           for s in range(10, 101)]
    for i in range(1, len(lrs)):
        assert lrs[i] <= lrs[i - 1] + 1e-10


# ── Adam init ──────────────────────────────────────────────────────────────


def test_adam_init_zeros():
    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    moment, variance, step_count = adam_init(params)
    assert jnp.all(moment["w"] == 0.0)
    assert jnp.all(variance["w"] == 0.0)
    assert float(step_count) == 0.0


def test_adam_init_preserves_structure():
    params = (jnp.ones(3), jnp.ones((2, 2)))
    moment, variance, step_count = adam_init(params)
    assert moment[0].shape == (3,)
    assert moment[1].shape == (2, 2)


# ── Adam step ──────────────────────────────────────────────────────────────


def test_adam_step_updates_params():
    params = {"x": jnp.array([5.0, 5.0])}
    grad = {"x": jnp.array([1.0, -1.0])}
    state = adam_init(params)
    cfg = AdamConfig(peak_lr=0.1, warmup_steps=1, num_steps=100)
    new_params, new_state = adam_step(params, grad, state, cfg)
    assert not jnp.allclose(new_params["x"], params["x"])


def test_adam_step_increments_step_count():
    params = {"x": jnp.array([1.0])}
    grad = {"x": jnp.array([0.1])}
    state = adam_init(params)
    cfg = AdamConfig()
    _, state = adam_step(params, grad, state, cfg)
    _, _, step_count = state
    assert float(step_count) == 1.0
    _, state = adam_step(params, grad, state, cfg)
    _, _, step_count = state
    assert float(step_count) == 2.0


def test_adam_minimizes_quadratic():
    """Adam should reduce f(x) = x² after several steps."""
    params = jnp.array([10.0])
    state = adam_init(params)
    cfg = AdamConfig(peak_lr=0.5, warmup_steps=1, num_steps=200)

    def loss_and_grad(p):
        return jnp.sum(p ** 2), 2.0 * p

    for _ in range(100):
        loss, grad = loss_and_grad(params)
        params, state = adam_step(params, grad, state, cfg)

    assert float(jnp.sum(params ** 2)) < 0.1


def test_adam_works_with_named_tuple():
    """Adam should handle NamedTuple pytrees (like CombinerParams)."""
    from helios.combiner_params import CombinerParams
    params = CombinerParams.initial()
    grad = jax.tree.map(jnp.ones_like, params)
    state = adam_init(params)
    cfg = AdamConfig(peak_lr=1e-3, warmup_steps=1, num_steps=100)
    new_params, _ = adam_step(params, grad, state, cfg)
    assert new_params.spacings.shape == params.spacings.shape
    assert not jnp.allclose(new_params.spacings, params.spacings)
