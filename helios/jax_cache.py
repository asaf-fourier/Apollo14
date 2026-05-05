"""JAX persistent compilation cache helper.

JAX's persistent compilation cache writes compiled kernels to disk so
subsequent runs of the same JIT-compiled function start instantly
instead of paying the (often multi-second) compile cost again. The
cache is process-global, opt-in, and configured via three
``jax.config`` keys.

This module wraps that configuration in a one-call helper that picks a
sensible per-user default (``~/.cache/jax``) so callers don't have to
hardcode machine-specific paths.

Typical use, at the top of an example or training script::

    from helios.jax_cache import enable_jax_compilation_cache
    enable_jax_compilation_cache()

Override the location for a shared/cluster cache::

    enable_jax_compilation_cache("/scratch/shared/jax-cache")
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

import jax


def _default_cache_dir() -> Path:
    """Per-host cache dir.

    Layout: ``<base>/jax/<system>-<machine>-<node>`` where ``base`` is
    ``$XDG_CACHE_HOME`` when set, else ``~/.cache``. Including the
    hostname and CPU architecture isolates AOT-compiled XLA:CPU
    kernels per machine — JAX's persistent cache stores binaries
    compiled with the *generating* machine's CPU features (``avx2``,
    ``wbnoinvd``, etc.) and loading them on a different host can
    SIGILL on missing instructions. The XLA:CPU loader prints a
    warning like "Target machine feature +wbnoinvd is not supported
    on the host machine" when this mismatch happens.
    """
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    host_tag = f"{platform.system()}-{platform.machine()}-{platform.node()}"
    return base / "jax" / host_tag


def enable_jax_compilation_cache(
    cache_dir: Path | str | None = None,
    *,
    min_compile_time_secs: float = 0.0,
    min_entry_size_bytes: int = -1,
) -> Path:
    """Turn on JAX's persistent compilation cache.

    Args:
        cache_dir: Where compiled kernels are stored. ``None`` (default)
            picks ``$XDG_CACHE_HOME/jax`` if set, otherwise
            ``~/.cache/jax`` — both auto-detected from the current
            user's environment, so the same script works across
            machines without per-host edits.
        min_compile_time_secs: Only cache kernels that took at least
            this long to compile. ``0.0`` caches every JIT'd kernel,
            which is what you usually want for a personal cache.
        min_entry_size_bytes: Only cache kernels whose serialized
            payload exceeds this size. ``-1`` disables the size filter.

    Returns:
        The resolved cache directory (created if it didn't exist).
    """
    resolved = Path(cache_dir).expanduser() if cache_dir is not None \
        else _default_cache_dir()
    resolved.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(resolved))
    jax.config.update("jax_persistent_cache_min_compile_time_secs",
                      float(min_compile_time_secs))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes",
                      int(min_entry_size_bytes))
    return resolved
