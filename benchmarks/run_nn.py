"""Benchmark runner for NN ``me2j`` generation: stage times, memory and environment as JSON.

Lives outside the package on purpose: it imports ``nuham`` like any user. Run repeatedly in one
process, the first repeat includes building or loading the bracket tables and compiling kernels,
later repeats are warm. Peak host RSS is for the whole process so far, so it only ever grows.

    uv run python benchmarks/run_nn.py NN.bin --emax 8 --e2max 16 --hw 16 --lam 2.0 \
        --repeat 2 --json out.json [--compute-only]
"""

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import jax

import nuham  # noqa: F401  sets JAX allocation defaults before the first device use
from nuham.angmom.moshinsky_table import MoshinskyTable
from nuham.generate import generate_me2j
from nuham.interactions.nn_file import read_nn_mom
from nuham.runtime import enable_compilation_cache
from nuham.timing import StageTimer

STAGES = ("project", "srg", "lab_coeffs", "lab_contract", "assemble", "write")


def environment() -> dict:
    """Describe the machine, backend and software versions the run happened on."""
    dev = jax.devices()[0]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).parent, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "device": f"{dev.platform}:{dev.device_kind}",
        "x64": bool(jax.config.jax_enable_x64),
        "git_commit": commit,
    }


def peak_device_bytes() -> int | None:
    """Peak device memory in bytes, or ``None`` where the backend does not report it (CPU)."""
    stats = jax.devices()[0].memory_stats()
    return None if not stats else stats.get("peak_bytes_in_use")


def main() -> None:
    """Parse arguments, run the repeats and print and optionally save the JSON record."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("nn_file")
    p.add_argument("--emax", type=int, required=True)
    p.add_argument("--e2max", type=int, required=True)
    p.add_argument("--hw", type=float, required=True)
    p.add_argument("--lam", type=float, default=None)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--compute-only", action="store_true", help="skip assembly and file writing")
    p.add_argument("--gzip-level", type=int, default=1)
    p.add_argument("--bracket-cache", default=None)
    p.add_argument("--no-compile-cache", action="store_true")
    p.add_argument("--compressor", choices=["auto", "gzip", "isal"], default="auto")
    p.add_argument("--json", default=None, help="also write the record to this file")
    args = p.parse_args()
    if not args.no_compile_cache:
        enable_compilation_cache()
    nn = read_nn_mom(args.nn_file)
    mosh = MoshinskyTable(args.bracket_cache)
    runs = []
    with tempfile.TemporaryDirectory() as tmp:
        out = None if args.compute_only else Path(tmp) / "bench.me2j.gz"
        for i in range(args.repeat):
            timer = StageTimer()
            start = time.perf_counter()
            n = generate_me2j(
                nn, out, args.hw, args.lam, args.emax, args.e2max, mosh=mosh,
                compresslevel=args.gzip_level, timer=timer, compressor=args.compressor,
            )  # fmt: skip
            total = time.perf_counter() - start
            runs.append(
                {
                    "repeat": i,
                    "total_s": total,
                    "stages_s": {k: timer.seconds.get(k, 0.0) for k in STAGES},
                    "unaccounted_s": total - sum(timer.seconds.values()),
                    "values": n,
                    "file_bytes": out.stat().st_size if out is not None else None,
                    "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                    "peak_device_gb": (b / 1e9 if (b := peak_device_bytes()) else None),
                }
            )
    record = {
        "env": environment(),
        "case": {k: getattr(args, k) for k in ("nn_file", "emax", "e2max", "hw", "lam")}
        | {
            "compute_only": args.compute_only,
            "gzip_level": args.gzip_level,
            "compressor": args.compressor,
        },
        "runs": runs,
    }
    text = json.dumps(record, indent=2)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")


if __name__ == "__main__":
    main()
