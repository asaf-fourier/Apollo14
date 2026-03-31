import jax.numpy as jnp

mm = 1.0
nm = mm / 1e6
deg = jnp.pi / 180.0

EPSILON = 1e-6  # geometric tolerance (ray misses, near-parallel)
FP_EPSILON = 1e-9  # floating point comparison tolerance
