"""Ray state pytree threaded through the segmented tracer."""

from typing import NamedTuple

import jax.numpy as jnp


class Ray(NamedTuple):
    """Single ray: position, direction, intensity.

    All element ``jax_interact`` functions take a ``Ray`` and return a new
    ``Ray`` plus ``(hit, valid)`` metadata for the trace trail. An invalid
    interaction zeros the ray's intensity and leaves its position frozen.
    """
    pos: jnp.ndarray         # (3,)
    dir: jnp.ndarray         # (3,)
    intensity: jnp.ndarray   # scalar
