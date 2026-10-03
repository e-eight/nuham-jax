"""Runtime settings shared by drivers."""

import os
from pathlib import Path

import jax


def default_cache_dir() -> Path:
    """Return ``$XDG_CACHE_HOME/nuham/jax`` (``~/.cache/nuham/jax`` by default)."""
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "nuham" / "jax"


def enable_compilation_cache(directory: str | Path | None = None) -> Path:
    """Persist JAX's compiled kernels on disk so later runs skip compilation.

    The lab contraction compiles one kernel per padded shape (a few hundred at emax = 10), which
    costs several seconds per run without a cache. Call this before the first jitted call.

    Parameters
    ----------
    directory
        Cache directory; defaults to :func:`default_cache_dir`.

    Returns
    -------
    Path
        The directory in use.
    """
    path = Path(directory) if directory is not None else default_cache_dir()
    path.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(path))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return path
