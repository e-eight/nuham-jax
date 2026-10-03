"""Runtime settings shared by drivers."""

import argparse
import os
from dataclasses import dataclass, replace
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


_COMPRESSORS = ("auto", "gzip", "isal")
_ASSEMBLERS = ("numpy", "device")
_FORMATTERS = ("host", "device")


@dataclass(frozen=True)
class WriterOptions:
    """How the ``me2j`` values are assembled, formatted and compressed.

    A ``None`` field takes its default for the JAX backend (:meth:`resolve`): on the CPU the numpy
    assembler, the host formatter and one compression thread; on an accelerator the device
    assembler and formatter and four compression threads, the fastest combination measured on a
    GH200 (an emax 14 run took 8.1 s instead of 25 s; ``docs/benchmarks.md``).

    Attributes
    ----------
    compresslevel
        gzip level of ``.gz`` output; 1 is fast and gives files within 10% of level 4.
    compressor
        ``"auto"`` (ISA-L's igzip if the optional ``isal`` package is installed, else ``gzip``),
        ``"gzip"`` or ``"isal"``.
    compress_threads
        Threads of igzip's parallel compressor; ignored by ``gzip``.
    assembler
        ``"numpy"`` or ``"device"`` (a JAX kernel; the lab matrices then stay on the device).
    formatter
        ``"host"`` or ``"device"`` (needs ``assembler="device"``).
    """

    compresslevel: int = 1
    compressor: str = "auto"
    compress_threads: int | None = None
    assembler: str | None = None
    formatter: str | None = None

    def __post_init__(self) -> None:
        """Reject values outside the supported choices."""
        if self.compressor not in _COMPRESSORS:
            raise ValueError(f"unknown compressor {self.compressor!r}; use one of {_COMPRESSORS}")
        if self.assembler is not None and self.assembler not in _ASSEMBLERS:
            raise ValueError(f"unknown assembler {self.assembler!r}; use one of {_ASSEMBLERS}")
        if self.formatter is not None and self.formatter not in _FORMATTERS:
            raise ValueError(f"unknown formatter {self.formatter!r}; use one of {_FORMATTERS}")
        if self.compress_threads is not None and self.compress_threads < 1:
            raise ValueError("compress_threads must be at least 1")

    def resolve(self) -> "WriterOptions":
        """Return a copy with every ``None`` replaced by its default for the current backend."""
        accelerator = jax.default_backend() != "cpu"
        assembler = self.assembler or ("device" if accelerator else "numpy")
        formatter = self.formatter or (
            "device" if assembler == "device" and accelerator else "host"
        )
        if formatter == "device" and assembler != "device":
            raise ValueError("formatter='device' needs assembler='device'")
        threads = self.compress_threads or (min(4, os.cpu_count() or 1) if accelerator else 1)
        return replace(self, compress_threads=threads, assembler=assembler, formatter=formatter)

    @property
    def on_device(self) -> bool:
        """Whether the lab matrices stay on the device (``assembler="device"``); resolve first."""
        return self.assembler == "device"


def add_writer_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the writer command-line options (see :class:`WriterOptions`) to ``parser``."""
    parser.add_argument("--gzip-level", type=int, default=1, help="gzip level of the output (1-9)")
    parser.add_argument("--compressor", choices=_COMPRESSORS, default="auto")
    parser.add_argument("--compress-threads", type=int, default=None, help="default: by backend")
    parser.add_argument(
        "--assembler", choices=_ASSEMBLERS, default=None, help="default: by backend"
    )
    parser.add_argument(
        "--formatter", choices=_FORMATTERS, default=None, help="default: by backend"
    )


def writer_options_from_args(args: argparse.Namespace) -> WriterOptions:
    """Build the :class:`WriterOptions` of the arguments added by :func:`add_writer_arguments`."""
    return WriterOptions(
        args.gzip_level, args.compressor, args.compress_threads, args.assembler, args.formatter
    )
