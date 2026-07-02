"""Parametric spectral curves for partial-mirror reflectance.

A spectral curve carries its own design variables (Gaussian amplitudes,
B-spline coefficients, raw samples — whatever parametrization fits) and
exposes a single ``sample(wavelengths)`` method. A ``PartialMirror``
holds a curve and asks it for a sampled reflectance vector at
construction time; the tracer never sees the curve directly.

All curves are JAX pytrees so ``jax.grad`` flows through their
parameters and pytree-aware optimizers (Adam in :mod:`helios.adam`) can
treat them as ordinary leaves.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class SumOfGaussiansCurve(NamedTuple):
    """Sum of ``B`` Gaussian bumps::

        r(λ) = Σ_b amplitude[b] · exp( −(λ − center[b])² / (2·σ[b]²) )

    Optimization variables are ``amplitude`` and ``sigma``. ``centers``
    are fixed by design — they're wrapped in
    :func:`jax.lax.stop_gradient` inside :meth:`sample`, so a
    pytree-aware optimizer that updates every leaf still leaves them at
    zero gradient (and therefore unchanged).

    Shapes broadcast over leading batch dims: every field can be
    ``(B,)`` (one curve) or ``(M, B)`` (a stack of M curves), and
    :meth:`sample` returns ``(K,)`` or ``(M, K)`` accordingly.
    """
    amplitude: jnp.ndarray   # (..., B) — design variable
    sigma: jnp.ndarray       # (..., B) — design variable
    centers: jnp.ndarray     # (..., B) — fixed (stop_gradient at use)

    def sample(self, wavelengths: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the curve at the given ``(K,)`` wavelength grid.

        Returns shape ``(..., K)`` — the leading batch dims of the
        curve fields, with ``K`` appended.
        """
        centers = jax.lax.stop_gradient(self.centers)
        offset = wavelengths - centers[..., None]                  # (..., B, K)
        exponent = -(offset ** 2) / (
            2.0 * self.sigma[..., None] ** 2 + 1e-18)
        per_basis = self.amplitude[..., None] * jnp.exp(exponent)  # (..., B, K)
        return jnp.sum(per_basis, axis=-2)                         # (..., K)

    @classmethod
    def uniform(
        cls,
        centers: jnp.ndarray,
        amplitude: float = 0.05,
        sigma: float = 20.0,
        num_mirrors: int | None = None,
    ) -> "SumOfGaussiansCurve":
        """Construct a curve with uniform amplitude/sigma over all bases.

        If ``num_mirrors`` is given, the result is batched to shape
        ``(num_mirrors, B)`` — one curve per mirror, all initialized
        identically. ``centers`` are broadcast across the batch.
        """
        centers = jnp.asarray(centers, dtype=jnp.float32)
        num_basis = centers.shape[0]
        if num_mirrors is None:
            return cls(
                amplitude=jnp.full((num_basis,), amplitude,
                                    dtype=jnp.float32),
                sigma=jnp.full((num_basis,), sigma, dtype=jnp.float32),
                centers=centers,
            )
        return cls(
            amplitude=jnp.full((num_mirrors, num_basis), amplitude,
                                dtype=jnp.float32),
            sigma=jnp.full((num_mirrors, num_basis), sigma,
                            dtype=jnp.float32),
            centers=jnp.broadcast_to(centers,
                                      (num_mirrors, num_basis)).copy(),
        )

    def at(self, idx) -> "SumOfGaussiansCurve":
        """Index a single curve out of an ``(M, B)`` batched curve.

        Equivalent to ``jax.tree.map(lambda x: x[idx], curve)`` but
        explicit at the call site.
        """
        return SumOfGaussiansCurve(
            amplitude=self.amplitude[idx],
            sigma=self.sigma[idx],
            centers=self.centers[idx],
        )


class ConstantCurve(NamedTuple):
    """Wavelength-flat reflectance::

        r(λ) = amplitude        (constant across the whole spectrum)

    The only optimization variable is ``amplitude`` — a single scalar
    per mirror, with no spectral shape. Unlike :class:`SumOfGaussiansCurve`
    there is no width or center: a flat partial reflector applies the
    same reflectance at every wavelength, so it preserves the
    projector's spectral shape up to a scalar (no color shift) and there
    is nothing per-color to tune.

    Shapes broadcast over leading batch dims: ``amplitude`` can be a
    scalar ``()`` (one curve) or ``(M,)`` (a stack of M curves), and
    :meth:`sample` returns ``(K,)`` or ``(M, K)`` accordingly — matching
    the :class:`SumOfGaussiansCurve` protocol so it drops into
    ``build_parametrized_system`` and ``PartialMirror`` unchanged.
    """
    amplitude: jnp.ndarray   # (...,) — design variable, one scalar per curve

    def sample(self, wavelengths: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the flat curve at the given ``(K,)`` wavelength grid.

        Returns shape ``(..., K)`` — the leading batch dims of
        ``amplitude`` with ``K`` appended, every value equal to the
        corresponding ``amplitude``. Differentiable w.r.t. ``amplitude``.
        """
        wavelengths = jnp.asarray(wavelengths)
        amplitude = jnp.asarray(self.amplitude)
        # Broadcast amplitude (...,) across the K wavelengths → (..., K).
        return amplitude[..., None] * jnp.ones_like(wavelengths)

    @classmethod
    def uniform(
        cls,
        amplitude: float = 0.05,
        num_mirrors: int | None = None,
    ) -> "ConstantCurve":
        """Construct a flat curve with the same reflectance everywhere.

        If ``num_mirrors`` is given, the result is batched to shape
        ``(num_mirrors,)`` — one flat curve per mirror, all initialized
        to ``amplitude``.
        """
        if num_mirrors is None:
            return cls(amplitude=jnp.asarray(amplitude, dtype=jnp.float32))
        return cls(amplitude=jnp.full((num_mirrors,), amplitude,
                                       dtype=jnp.float32))

    def at(self, idx) -> "ConstantCurve":
        """Index a single flat curve out of an ``(M,)`` batched curve."""
        return ConstantCurve(amplitude=self.amplitude[idx])
