"""End-to-end generation of an NN ``me2j`` file from a momentum-space interaction."""

import argparse
from pathlib import Path

from nuham.angmom.moshinsky_table import MoshinskyTable
from nuham.basis.lab import lab_channels, make_orbits
from nuham.interactions.nn_file import MomentumNN, read_nn_mom
from nuham.lab_transform import lab_channel_matrices
from nuham.me2j import LabChannel, assemble, write_me2j
from nuham.relative_nn import evolve_relative_nn


def generate_me2j(
    nn: MomentumNN,
    path: str | Path,
    hw: float,
    lam: float | None,
    emax: int,
    e2max: int,
    lmax: int | None = None,
    coulomb: bool = True,
    mosh: MoshinskyTable | None = None,
) -> int:
    """Generate the NN interaction in the lab HO basis and write it as ``me2j``.

    Parameters
    ----------
    nn
        Momentum-space interaction.
    path
        Output file (gzip if it ends in ``.gz``).
    hw
        Oscillator energy in MeV.
    lam
        SRG cutoff in fm^-1, or ``None`` for the bare interaction.
    emax, e2max, lmax
        Single-particle and two-body truncations; ``lmax`` defaults to ``emax``.
    coulomb
        Whether to include the pp Coulomb interaction.
    mosh
        Optional bracket table to reuse between calls.

    Returns
    -------
    int
        The number of values written.
    """
    mosh = MoshinskyTable() if mosh is None else mosh
    relative = {
        (c.j, c.parity, c.s, c.tz): c
        for c in evolve_relative_nn(nn, hw, lam, n2max=100, coulomb=coulomb)
    }
    orb = make_orbits(emax, lmax)
    channels = {
        jp: LabChannel(lab_channel_matrices(orb, e2max, *jp, relative, mosh), len(orb))
        for jp in lab_channels(orb, e2max)
    }
    return write_me2j(path, assemble(orb, e2max, channels))


def main() -> None:
    """Command-line entry point: ``python -m nuham.generate --help``."""
    p = argparse.ArgumentParser(description="Generate an NN me2j file (EM500 etc.) for imsrg++.")
    p.add_argument("nn_file", help="NuHamil momentum-space *.bin file")
    p.add_argument("output", help="output file, e.g. NN_hw16_emax14_e2max28.me2j.gz")
    p.add_argument("--hw", type=float, required=True, help="oscillator energy in MeV")
    p.add_argument("--lam", type=float, default=None, help="SRG cutoff in fm^-1 (omit for bare)")
    p.add_argument("--emax", type=int, required=True)
    p.add_argument("--e2max", type=int, required=True)
    p.add_argument("--lmax", type=int, default=None)
    p.add_argument("--no-coulomb", action="store_true")
    p.add_argument("--bracket-cache", default=None, help="directory for cached Moshinsky tables")
    args = p.parse_args()
    mosh = MoshinskyTable(args.bracket_cache)
    n = generate_me2j(
        read_nn_mom(args.nn_file), args.output, args.hw, args.lam, args.emax, args.e2max,
        args.lmax, not args.no_coulomb, mosh,
    )  # fmt: skip
    print(f"wrote {n} values to {args.output}")


if __name__ == "__main__":
    main()
