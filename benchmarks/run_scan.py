"""Benchmark runner for (hw, lambda) scans: time per set against batch size, as JSON.

For each batch size the whole grid is generated twice in one process, the first pass is a
warm-up (compilation) and the second is reported. A batch size of 0 means a loop of separate
single-set ``generate_me2j`` calls, the baseline a scan has to beat.

    uv run python benchmarks/run_scan.py NN.bin --emax 8 --e2max 16 --hw 12 16 20 --lam 1.8 2.0 \
        --batch-sizes 0 1 2 6 --json scan.json [--compute-only]
"""

import argparse
import itertools
import json
import resource
import tempfile
import time
from pathlib import Path

from run_nn import STAGES, environment, peak_device_bytes

import nuham  # noqa: F401  sets JAX allocation defaults before the first device use
from nuham.angmom.moshinsky_table import MoshinskyTable
from nuham.generate import generate_me2j
from nuham.interactions.nn_file import read_nn_mom
from nuham.runtime import enable_compilation_cache
from nuham.scan import ScanPoint, generate_scan
from nuham.timing import StageTimer


def main() -> None:
    """Parse arguments, run every batch size and print and optionally save the JSON record."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("nn_file")
    p.add_argument("--emax", type=int, required=True)
    p.add_argument("--e2max", type=int, required=True)
    p.add_argument("--hw", type=float, nargs="+", required=True)
    p.add_argument("--lam", type=float, nargs="+", required=True)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--flow-sets", type=int, default=1, help="points whose SRG flows share a batch")
    p.add_argument("--compute-only", action="store_true", help="skip assembly and file writing")
    p.add_argument("--bracket-cache", default=None)
    p.add_argument("--no-compile-cache", action="store_true")
    p.add_argument("--json", default=None, help="also write the record to this file")
    args = p.parse_args()
    if not args.no_compile_cache:
        enable_compilation_cache()
    nn = read_nn_mom(args.nn_file)
    mosh = MoshinskyTable(args.bracket_cache)
    grid = list(itertools.product(args.hw, args.lam))
    runs = []
    with tempfile.TemporaryDirectory() as tmp:

        def paths() -> list[Path | None]:
            return [
                None if args.compute_only else Path(tmp) / f"s{i}.me2j.gz" for i in range(len(grid))
            ]

        for batch in args.batch_sizes:
            for _pass in range(2):  # pass 0 warms up, pass 1 is reported
                timer = StageTimer()
                out = paths()
                start = time.perf_counter()
                if batch == 0:
                    for (hw, lam), path in zip(grid, out, strict=True):
                        generate_me2j(
                            nn, path, hw, lam, args.emax, args.e2max, mosh=mosh, timer=timer
                        )
                else:
                    points = [ScanPoint(hw, lam, o) for (hw, lam), o in zip(grid, out, strict=True)]
                    generate_scan(
                        nn,
                        points,
                        args.emax,
                        args.e2max,
                        mosh=mosh,
                        batch_size=batch,
                        timer=timer,
                        flow_sets=args.flow_sets,
                    )
                total = time.perf_counter() - start
            runs.append(
                {
                    "batch_size": batch,
                    "n_sets": len(grid),
                    "total_s": total,
                    "per_set_s": total / len(grid),
                    "stages_s": {k: timer.seconds.get(k, 0.0) for k in STAGES},
                    "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                    "peak_device_gb": (b / 1e9 if (b := peak_device_bytes()) else None),
                }
            )
    record = {
        "env": environment(),
        "case": {"nn_file": args.nn_file, "emax": args.emax, "e2max": args.e2max}
        | {"grid": grid, "compute_only": args.compute_only, "flow_sets": args.flow_sets},
        "runs": runs,
    }
    text = json.dumps(record, indent=2)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")


if __name__ == "__main__":
    main()
