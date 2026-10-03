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


def default_flow_sets() -> int | None:
    """Return how many parameter sets share one SRG flow batch by default.

    ``1`` on the CPU backend, where larger batches were slower per set (cache pressure, and a
    batch runs as long as its slowest entry); ``None`` (all sets of a scan batch together) on
    accelerators, where a batch of 6 sets halved the flow time on a GH200 (``docs/benchmarks.md``).
    """
    return 1 if jax.default_backend() == "cpu" else None
