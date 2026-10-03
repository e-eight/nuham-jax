"""End-to-end generation of an NN ``me2j`` file from a momentum-space interaction."""

import argparse
from pathlib import Path

from nuham.angmom.moshinsky_table import MoshinskyTable
from nuham.basis.lab import lab_channels, make_orbits
from nuham.interactions.nn_file import MomentumNN, read_nn_mom
from nuham.lab_transform import lab_channel_matrices
from nuham.me2j import LabChannel, assemble, assemble_device, write_me2j, write_me2j_timed
from nuham.relative_nn import evolve_relative_nn
from nuham.runtime import enable_compilation_cache
from nuham.timing import StageTimer


def generate_me2j(
    nn: MomentumNN,
    path: str | Path | None,
    hw: float,
    lam: float | None,
    emax: int,
    e2max: int,
    lmax: int | None = None,
    coulomb: bool = True,
    mosh: MoshinskyTable | None = None,
    compresslevel: int = 1,
    n2max: int = 100,
    timer: StageTimer | None = None,
    compressor: str = "auto",
    assembler: str = "numpy",
) -> int:
    """Generate the NN interaction in the lab HO basis and write it as ``me2j``.

    Parameters
    ----------
    nn
        Momentum-space interaction.
    path
        Output file (gzip if it ends in ``.gz``). ``None`` runs compute-only: the lab matrices
        are built but not assembled or written, and the return value is 0.
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
    compresslevel, compressor
        Level and compressor of the gzip output (see :func:`nuham.me2j.write_me2j`).
    assembler
        ``"numpy"`` (:func:`nuham.me2j.assemble`) or ``"device"``
        (:func:`nuham.me2j.assemble_device`, a JAX kernel on the default device).
    n2max
        Truncation ``2n + l <= n2max`` of the relative space in which the SRG flow runs.
    timer
        Optional stage timer. Records ``project``, ``srg``, ``lab_coeffs``, ``lab_contract``,
        ``assemble`` (building value chunks) and ``write`` (formatting and the final flush;
        compression overlaps in a thread and is only seen as back-pressure here).

    Returns
    -------
    int
        The number of values written.
    """
    mosh = MoshinskyTable() if mosh is None else mosh
    relative = {
        (c.j, c.parity, c.s, c.tz): c
        for c in evolve_relative_nn(nn, hw, lam, n2max=n2max, coulomb=coulomb, timer=timer)
    }
    orb = make_orbits(emax, lmax)
    channels = {
        jp: LabChannel(
            lab_channel_matrices(orb, e2max, *jp, relative, mosh, timer, assembler == "device"),
            len(orb),
        )
        for jp in lab_channels(orb, e2max)
    }
    if path is None:
        return 0
    chunks = (assemble_device if assembler == "device" else assemble)(orb, e2max, channels)
    if timer is None:
        return write_me2j(path, chunks, compresslevel=compresslevel, compressor=compressor)
    return write_me2j_timed(path, chunks, timer, compresslevel, compressor)


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
    p.add_argument("--n2max", type=int, default=100, help="relative space of the SRG flow")
    p.add_argument("--gzip-level", type=int, default=1, help="gzip level of the output (1-9)")
    p.add_argument("--assembler", choices=["numpy", "device"], default="numpy")
    p.add_argument(
        "--compressor", choices=["auto", "gzip", "isal"], default="auto",
        help="gzip implementation (auto: isal if installed, see --help of the isal extra)",
    )  # fmt: skip
    p.add_argument("--bracket-cache", default=None, help="directory for cached Moshinsky tables")
    p.add_argument(
        "--compile-cache",
        default=None,
        help="directory for JAX's compilation cache (default ~/.cache/nuham/jax)",
    )
    p.add_argument("--no-compile-cache", action="store_true", help="do not cache compiled kernels")
    args = p.parse_args()
    if not args.no_compile_cache:
        enable_compilation_cache(args.compile_cache)
    mosh = MoshinskyTable(args.bracket_cache)
    n = generate_me2j(
        read_nn_mom(args.nn_file), args.output, args.hw, args.lam, args.emax, args.e2max,
        args.lmax, not args.no_coulomb, mosh, args.gzip_level, args.n2max,
        compressor=args.compressor,
        assembler=args.assembler,
    )  # fmt: skip
    print(f"wrote {n} values to {args.output}")


if __name__ == "__main__":
    main()
