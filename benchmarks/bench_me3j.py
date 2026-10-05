"""Benchmarks of the 3N lab stage and of the me3j writer (Phase 5.4), as JSON.

``lab``    one lab channel ``(2J, P, 2T)``: host T coefficients and device contraction of random
           symmetric Jacobi operators on the real antisymmetric bases (the largest production
           channel is ``--j2 9 --parity 1 --t2 1`` at ``--emax 14 --e3max 16``). Stage times,
           lab dimension, peak memory.
``write``  the file writer for a file of ``emax/e2max/e3max`` with random single-precision lab
           matrices: values per second of the tuple enumeration and lookup alone, then with the
           text formatting and compression. A file with a smaller ``emax`` and the same ``e2max``
           and ``e3max`` is a prefix of the production file (orbits come in file order), so a
           small ``--emax`` measures the production rate without its memory.

    uv run python benchmarks/bench_me3j.py lab --emax 14 --e3max 16 --j2 9 --json lab.json
    uv run python benchmarks/bench_me3j.py write --emax 10 --e2max 28 --e3max 16 --json write.json
"""

import argparse
import json
import resource
import tempfile
import time
from pathlib import Path

import numpy as np
from run_nn import environment, peak_device_bytes

import nuham  # noqa: F401  sets JAX allocation defaults before the first device use
from nuham.basis.lab import make_orbits
from nuham.nnn.jacobi import JacobiChannel, build_channel
from nuham.nnn.lab_transform import lab_scalar_device
from nuham.nnn.me3j import LabMatrix3, me3j_values, write_me3j, write_me3j_fields
from nuham.nnn.packed import pack_lower
from nuham.nnn.tcoef import lab_states, make_brackets
from nuham.runtime import default_workers
from nuham.timing import StageTimer


def lab_record(
    emax: int, e2max: int, e3max: int, j2: int, parity: int, t2: int, repeat: int = 1
) -> dict:
    """Time the lab transform of one channel with random Jacobi operators."""
    timer = StageTimer()
    orb = make_orbits(emax)
    rng = np.random.default_rng(0)
    with timer.stage("jacobi_bases"):
        chans, vjac = {}, {}
        for jr in range(1, 2 * e3max + 4, 2):
            for p in (1, -1):
                ch = build_channel(jr, p, t2, e3max)
                if ch.blocks:
                    chans[jr, p] = JacobiChannel(jr, p, t2, e3max, ch.blocks)
                    x = rng.normal(size=(ch.n_as, ch.n_as))
                    vjac[jr, p] = (x + x.T) / 2
    with timer.stage("brackets"):
        brackets = make_brackets()
    repeats = []
    for _ in range(repeat):  # the second run has every kernel compiled: it separates compilation
        run = StageTimer()
        t0 = time.perf_counter()
        lab, m, _ = lab_scalar_device(
            j2, parity, t2, orb, e2max, e3max, vjac, chans, brackets, np.float32, timer=run
        )
        repeats.append(
            {
                "seconds": time.perf_counter() - t0,
                **{k: round(x, 3) for k, x in run.seconds.items()},
            }
        )
    return {
        "emax": emax, "e2max": e2max, "e3max": e3max, "j2": j2, "parity": parity, "t2": t2,
        "lab_states": len(lab), "matrix_gb": m.nbytes / 1e9,
        "runs": repeats,
        "stages": {k: round(x, 3) for k, x in timer.seconds.items()},
        "peak_host_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        "peak_device_bytes": peak_device_bytes(),
    }  # fmt: skip


def write_record(
    emax: int, e2max: int, e3max: int, compresslevel: int, workers: int, threads: int, fields: bool
) -> dict:
    """Time the writer on random lab matrices."""
    orb = make_orbits(emax)
    rng = np.random.default_rng(0)
    labs = {}
    for t2 in (1, 3):
        for j2 in range(1, 2 * e3max + 4, 2):
            for p in (1, -1):
                lab = lab_states(orb, j2, p, t2, e2max, e3max)
                if len(lab):
                    m = rng.standard_normal((len(lab), len(lab)), dtype=np.float32)
                    labs[j2, p, t2] = LabMatrix3.make(lab, pack_lower(m), len(orb))
    t0 = time.perf_counter()
    n = sum(len(c) for c in me3j_values(orb, e2max, e3max, labs, workers=workers))
    enumerate_seconds = time.perf_counter() - t0
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bench.me3j.gz"
        t0 = time.perf_counter()
        if fields:
            write_me3j_fields(
                path,
                me3j_values(orb, e2max, e3max, labs, workers=workers, as_fields=True),
                compresslevel,
                compress_threads=threads,
            )
        else:
            write_me3j(
                path,
                me3j_values(orb, e2max, e3max, labs, workers=workers),
                compresslevel,
                compress_threads=threads,
            )
        write_seconds = time.perf_counter() - t0
        size = path.stat().st_size
    return {
        "emax": emax, "e2max": e2max, "e3max": e3max, "workers": workers,
        "compress_threads": threads, "worker_formatting": fields, "values": n,
        "lab_channels": len(labs), "lab_matrices_gb": sum(c.packed.nbytes for c in labs.values()) / 1e9,  # noqa: E501
        "enumerate_lookup_seconds": enumerate_seconds, "total_write_seconds": write_seconds,
        "values_per_second_lookup": n / enumerate_seconds,
        "values_per_second_total": n / write_seconds, "gz_bytes": size,
        "peak_host_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
    }  # fmt: skip


def main() -> None:
    """Run the chosen measurement and print and optionally save the record."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="what", required=True)
    for name in ("lab", "write"):
        sp = sub.add_parser(name)
        sp.add_argument("--emax", type=int, default=8)
        sp.add_argument("--e2max", type=int, default=None, help="default 2 emax")
        sp.add_argument("--e3max", type=int, default=8)
        sp.add_argument("--json", default=None)
        if name == "lab":
            sp.add_argument("--j2", type=int, default=3)
            sp.add_argument("--parity", type=int, default=1)
            sp.add_argument("--t2", type=int, default=1)
            sp.add_argument(
                "--repeat", type=int, default=1, help="runs in one process (warm after the first)"
            )
        else:
            sp.add_argument("--compresslevel", type=int, default=1)
            sp.add_argument("--workers", type=int, default=None, help="default: all CPUs")
            sp.add_argument("--compress-threads", type=int, default=1)
            sp.add_argument("--fields", action="store_true", help="workers format the values")
    args = p.parse_args()
    if getattr(args, "workers", 0) is None:
        args.workers = default_workers()  # a silent 1 once cost a GPU-hour of slow runs
    print("config:", json.dumps({k: v for k, v in vars(args).items() if k != "json"}), flush=True)
    e2max = 2 * args.emax if args.e2max is None else args.e2max
    if args.what == "lab":
        row = lab_record(args.emax, e2max, args.e3max, args.j2, args.parity, args.t2, args.repeat)
    else:
        row = write_record(
            args.emax,
            e2max,
            args.e3max,
            args.compresslevel,
            args.workers,
            args.compress_threads,
            args.fields,
        )
    print(json.dumps(row), flush=True)
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {"benchmark": args.what, "environment": environment(), "rows": [row]}, indent=1
            )
        )


if __name__ == "__main__":
    main()
