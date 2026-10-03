"""Host memory held by the lab matrices of K scan points, against their real size.

Builds the relative potentials and then the lab matrices of every ``(J, parity)`` for K points, and
reports the resident-set growth per point next to the bytes of the matrices themselves. A large
ratio means memory is held by something other than the matrices (for instance pinned host buffers
on an accelerator; compare ``NUHAM_PINNED_COPY=0`` and ``=1``).

    NUHAM_PINNED_COPY=0 uv run python benchmarks/bench_memory.py NN.bin --emax 12 --points 3
"""

import argparse
import gc
import json

import jax

import nuham  # noqa: F401  sets JAX allocation defaults before the first device use
from nuham.angmom.moshinsky_table import MoshinskyTable
from nuham.basis.lab import lab_channels, make_orbits
from nuham.interactions.nn_file import read_nn_mom
from nuham.kernels.transfer import use_pinned_by_default
from nuham.lab_transform import lab_channel_matrices_many
from nuham.me2j import LabChannel
from nuham.relative_nn import evolve_relative_nn_many
from nuham.runtime import enable_compilation_cache


def rss_gb() -> float:
    """Return the current resident set size of this process in GB."""
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * 4096 / 1e9


def main() -> None:
    """Run the measurement for each requested number of points and print one JSON record."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("nn_file")
    p.add_argument("--emax", type=int, required=True)
    p.add_argument("--points", type=int, nargs="+", default=[1, 3])
    p.add_argument("--bracket-cache", default=None)
    args = p.parse_args()
    enable_compilation_cache()
    nn = read_nn_mom(args.nn_file)
    orb = make_orbits(args.emax)
    mosh = MoshinskyTable(args.bracket_cache)
    results = []
    for k in args.points:
        gc.collect()
        params = [(12.0 + 4.0 * i, 2.0) for i in range(k)]
        relatives = [
            {(c.j, c.parity, c.s, c.tz): c for c in channels}
            for channels in evolve_relative_nn_many(nn, params, flow_sets=1)
        ]
        gc.collect()
        before = rss_gb()
        channels: list[dict[tuple[int, int], LabChannel]] = [{} for _ in range(k)]
        peak = before
        for jp in lab_channels(orb, 2 * args.emax):
            matrices = lab_channel_matrices_many(orb, 2 * args.emax, *jp, relatives, mosh)
            for i, m in enumerate(matrices):
                channels[i][jp] = LabChannel(m, len(orb))
            peak = max(peak, rss_gb())
        gc.collect()
        after = rss_gb()
        nbytes = sum(m.v.nbytes for c in channels[0].values() for m in c.matrices.values())
        results.append(
            {
                "points": k,
                "rss_before_gb": before,
                "rss_after_gb": after,
                "rss_peak_gb": peak,
                "growth_per_point_gb": (after - before) / k,
                "matrix_gb_per_point": nbytes / 1e9,
            }
        )
        del channels, relatives
    print(
        json.dumps(
            {
                "emax": args.emax,
                "backend": jax.default_backend(),
                "pinned_copy": use_pinned_by_default(),
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
