"""Generation of ``me2j`` files for many ``(hw, lambda)`` sets that share work.

The Moshinsky brackets and the lab transformation coefficients do not depend on ``hw`` or the SRG
cutoff, and the SRG flows and the lab contractions of different sets batch well, so a scan costs
much less per set than separate runs.
"""

import argparse
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from nuham.angmom.moshinsky_table import MoshinskyTable
from nuham.basis.lab import lab_channels, make_orbits
from nuham.interactions.nn_file import MomentumNN, read_nn_mom
from nuham.lab_transform import lab_channel_matrices_many
from nuham.me2j import LabChannel, assemble, write_me2j, write_me2j_timed
from nuham.relative_nn import evolve_relative_nn_many
from nuham.runtime import default_flow_sets, enable_compilation_cache
from nuham.timing import StageTimer


@dataclass(frozen=True)
class ScanPoint:
    """One parameter set of a scan and the file it is written to.

    Attributes
    ----------
    hw
        Oscillator energy in MeV.
    lam
        SRG cutoff in fm^-1, or ``None`` for the bare interaction.
    path
        Output file, or ``None`` to compute without writing.
    """

    hw: float
    lam: float | None
    path: str | Path | None


def generate_scan(
    nn: MomentumNN,
    points: Sequence[ScanPoint],
    emax: int,
    e2max: int,
    lmax: int | None = None,
    coulomb: bool = True,
    mosh: MoshinskyTable | None = None,
    compresslevel: int = 1,
    n2max: int = 100,
    batch_size: int = 2,
    timer: StageTimer | None = None,
    flow_sets: int | None = None,
) -> list[int]:
    """Generate one NN ``me2j`` file per scan point, sharing work between points.

    Points are processed in batches of ``batch_size``. Within a batch the lab coefficients of each
    ``(J, parity)`` are built once and contracted for all points, and the SRG flows run in groups
    of ``flow_sets`` points. Every point of
    a batch keeps all its lab matrices in memory until its file is written (about 2 GB at
    emax 14), so ``batch_size`` trades speed against memory. Each file equals the one a separate
    :func:`nuham.generate.generate_me2j` call would write, up to the flow tolerances.

    Parameters
    ----------
    nn
        Momentum-space interaction.
    points
        The parameter sets and their output files.
    emax, e2max, lmax, coulomb, mosh, compresslevel, n2max
        As in :func:`nuham.generate.generate_me2j`.
    batch_size
        Number of points processed together.
    timer
        Optional stage timer with the stages of :func:`nuham.generate.generate_me2j`.
    flow_sets
        Points whose SRG flows share a batch; ``None`` takes :func:`nuham.runtime.default_flow_sets`
        (1 on a CPU, all points of a batch on an accelerator).

    Returns
    -------
    list[int]
        The number of values written for each point (0 where ``path`` is ``None``).
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    mosh = MoshinskyTable() if mosh is None else mosh
    flow_sets = default_flow_sets() if flow_sets is None else flow_sets
    orb = make_orbits(emax, lmax)
    counts: list[int] = []
    for first in range(0, len(points), batch_size):
        batch = points[first : first + batch_size]
        relatives = [
            {(c.j, c.parity, c.s, c.tz): c for c in channels}
            for channels in evolve_relative_nn_many(
                nn,
                [(p.hw, p.lam) for p in batch],
                n2max=n2max,
                coulomb=coulomb,
                timer=timer,
                flow_sets=flow_sets,
            )
        ]
        channels: list[dict[tuple[int, int], LabChannel]] = [{} for _ in batch]
        for jp in lab_channels(orb, e2max):
            matrices = lab_channel_matrices_many(orb, e2max, *jp, relatives, mosh, timer)
            for k, m in enumerate(matrices):
                channels[k][jp] = LabChannel(m, len(orb))
        for point, chans in zip(batch, channels, strict=True):
            if point.path is None:
                counts.append(0)
                continue
            chunks = assemble(orb, e2max, chans)
            if timer is None:
                counts.append(write_me2j(point.path, chunks, compresslevel=compresslevel))
            else:
                counts.append(write_me2j_timed(point.path, chunks, timer, compresslevel))
    return counts


def main() -> None:
    """Command-line entry point: ``python -m nuham.scan --help``."""
    p = argparse.ArgumentParser(description="Generate NN me2j files for a grid of hw and lambda.")
    p.add_argument("nn_file", help="NuHamil momentum-space *.bin file")
    p.add_argument(
        "output",
        help="output file pattern with {hw} and {lam}, e.g. NN_hw{hw:g}_lam{lam:g}_emax14.me2j.gz",
    )
    p.add_argument("--hw", type=float, nargs="+", required=True, help="oscillator energies in MeV")
    p.add_argument("--lam", type=float, nargs="+", required=True, help="SRG cutoffs in fm^-1")
    p.add_argument("--emax", type=int, required=True)
    p.add_argument("--e2max", type=int, required=True)
    p.add_argument("--lmax", type=int, default=None)
    p.add_argument("--no-coulomb", action="store_true")
    p.add_argument("--n2max", type=int, default=100, help="relative space of the SRG flow")
    p.add_argument("--batch-size", type=int, default=2, help="points processed together")
    p.add_argument(
        "--flow-sets",
        type=int,
        default=None,
        help="points per SRG flow batch (default: by backend)",
    )
    p.add_argument("--gzip-level", type=int, default=1)
    p.add_argument("--bracket-cache", default=None, help="directory for cached Moshinsky tables")
    p.add_argument("--no-compile-cache", action="store_true")
    args = p.parse_args()
    if not args.no_compile_cache:
        enable_compilation_cache()
    points = [
        ScanPoint(hw, lam, args.output.format(hw=hw, lam=lam))
        for hw, lam in itertools.product(args.hw, args.lam)
    ]
    counts = generate_scan(
        read_nn_mom(args.nn_file), points, args.emax, args.e2max, args.lmax, not args.no_coulomb,
        MoshinskyTable(args.bracket_cache), args.gzip_level, args.n2max, args.batch_size,
        flow_sets=args.flow_sets,
    )  # fmt: skip
    for point, n in zip(points, counts, strict=True):
        print(f"wrote {n} values to {point.path}")


if __name__ == "__main__":
    main()
