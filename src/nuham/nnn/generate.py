"""End-to-end generation of a 3N ``me3j`` file: Jacobi Hamiltonian, SRG, lab transform, writer."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nuham.angmom.moshinsky_table import MoshinskyTable
from nuham.basis.lab import make_orbits
from nuham.nn.input_file import MomentumNN
from nuham.nnn.hamiltonian import isospin_nn
from nuham.nnn.jacobi import JacobiChannel, build_channel, ramp_nmax
from nuham.nnn.lab_transform import lab_scalar_device
from nuham.nnn.local_force.assembly import local_3nf_matrix
from nuham.nnn.local_force.radial import RadialTables, nuhamil_meshes, radial_tables
from nuham.nnn.me3j import LabMatrix3, me3j_values, write_me3j
from nuham.nnn.srg3 import three_body_operator
from nuham.nnn.tcoef import make_brackets
from nuham.runtime import WriterOptions
from nuham.timing import StageTimer, stage


@dataclass(frozen=True)
class LocalForce:
    """Parameters of the local N2LO 3N force (NuHamil's ``Regulator="Local"``).

    The defaults are the 3NF400 set used for the NuHamil reference run.
    """

    lecs: dict[str, float] = field(
        default_factory=lambda: {"c1": -0.81, "c3": -3.2, "c4": 5.4, "cD": -0.2, "cE": 0.098}
    )
    cutoff: float = 400.0
    power: int = 2


def jacobi_operators(
    nn: MomentumNN,
    hw: float,
    lam: float,
    e3max: int,
    ramp: str,
    force: LocalForce | None,
    table: MoshinskyTable | None = None,
    timer: StageTimer | None = None,
    rtol: float = 1e-8,
    jmax3: int | None = None,
) -> tuple[dict[tuple[int, int, int], np.ndarray], dict[tuple[int, int, int], JacobiChannel]]:
    """Evolved three-body operators ``V3`` of all Jacobi channels, cut to ``N <= e3max``.

    Channels ``(2J, P, 2T)`` run over ``2J <= jmax3``, by default ``2 e3max + 3``, the range that
    can occur in the lab transform of ``e3max`` (a smaller ``jmax3`` drops the higher ones, as
    NuHamil's input of the same name does). Each flow runs in the full ramp space and is dropped
    after its leading block is kept, so memory stays at the size of the largest single channel.
    ``force=None`` omits the genuine force.

    Returns
    -------
    tuple
        ``V3`` and the matching truncated :class:`JacobiChannel` (their ``cfp`` fix the gauge)
        by ``(2J, P, 2T)``.
    """
    table = MoshinskyTable(d=1 / 3) if table is None else table
    with stage(timer, "nn_relative"):
        bare, evolved = isospin_nn(nn, hw, lam)
    meshes = nuhamil_meshes()
    tables: dict[int, RadialTables] = {}
    ops, chans = {}, {}
    for t2 in (1, 3):
        for j2 in range(1, (2 * e3max + 3 if jmax3 is None else jmax3) + 1, 2):
            for parity in (1, -1):
                nmax = ramp_nmax(j2, ramp)
                if nmax < e3max:
                    raise ValueError(f"ramp {ramp!r} is below e3max = {e3max} at 2J = {j2}")
                with stage(timer, "jacobi_basis"):
                    channel = build_channel(j2, parity, t2, nmax, table)
                if not channel.blocks:  # no antisymmetric state: nothing to evolve or transform
                    continue
                v3n = None
                if force:
                    with stage(timer, "force"):
                        if nmax not in tables:
                            tables[nmax] = radial_tables(
                                meshes, force.cutoff, force.power, nmax + 2
                            )
                        v3n = local_3nf_matrix(channel, hw, meshes, tables[nmax], force.lecs)
                with stage(timer, "flow"):
                    v3 = three_body_operator(
                        channel, hw, lam, bare, evolved, v3n, e3max, rtol=rtol, atol=rtol
                    )
                kept = tuple(b for b in channel.blocks if b.n_quanta <= e3max)
                chans[j2, parity, t2] = JacobiChannel(j2, parity, t2, e3max, kept)
                ops[j2, parity, t2] = v3
    return ops, chans


def lab_matrices(
    emax: int,
    e2max: int,
    e3max: int,
    ops: dict[tuple[int, int, int], np.ndarray],
    chans: dict[tuple[int, int, int], JacobiChannel],
    timer: StageTimer | None = None,
    dtype: type = np.float64,
) -> dict[tuple[int, int, int], LabMatrix3]:
    """Lab matrices ``6 sum T^T V3 T`` of every ``(2J, P, 2T)`` channel that has states.

    The contraction runs on the device in float64 and the matrices are stored as ``dtype`` on the
    host (``np.float32`` halves the memory; the file has eight decimals anyway).
    """
    orb = make_orbits(emax)
    brackets = make_brackets()
    out = {}
    with stage(timer, "lab"):
        for t2 in (1, 3):
            vjac = {(j, p): v for (j, p, t), v in ops.items() if t == t2}
            jac = {(j, p): c for (j, p, t), c in chans.items() if t == t2}
            for j2 in range(1, 2 * e3max + 4, 2):
                for parity in (1, -1):
                    lab, m, order = lab_scalar_device(
                        j2, parity, t2, orb, e2max, e3max, vjac, jac, brackets, dtype
                    )
                    if len(lab):
                        out[j2, parity, t2] = LabMatrix3.make(lab, m, len(orb), order)
    return out


def generate_me3j(
    nn: MomentumNN,
    path: str | Path | None,
    hw: float,
    lam: float,
    emax: int,
    e2max: int,
    e3max: int,
    ramp: str,
    force: LocalForce | None = None,
    timer: StageTimer | None = None,
    options: WriterOptions | None = None,
    rtol: float = 1e-8,
    jmax3: int | None = None,
    lab_dtype: type = np.float64,
) -> int:
    """Generate the 3N interaction in the lab HO basis and write it as ``me3j``.

    Parameters
    ----------
    nn
        Momentum-space NN interaction.
    path
        Output ``.me3j.gz`` file; ``None`` stops after the lab matrices (returns 0).
    hw, lam
        Oscillator energy (MeV) and SRG cutoff (fm^-1).
    emax, e2max, e3max
        Truncations; the file needs ``e3max <= e2max``.
    ramp
        Flow space per ``J``, NuHamil's syntax (``"flat8"``, ``"ramp40-5-36-..."``).
    force
        Local 3N force, or ``None`` for the induced force of the NN interaction alone.
    options
        Compression settings (only ``compresslevel``, ``compressor`` and ``compress_threads``
        are used so far; the assembly and formatting are the host paths).
    rtol
        Flow tolerance (rtol = atol in units of ``hw``).
    jmax3
        Largest doubled ``J`` of the Jacobi channels; ``None`` is ``2 e3max + 3``.
    lab_dtype
        Storage type of the lab matrices (``np.float32`` for production sizes).

    Returns
    -------
    int
        The number of values written.
    """
    if e3max > e2max:
        raise ValueError("e3max > e2max gives a file that imsrg++ cannot read (spec 5d 2.3)")
    ops, chans = jacobi_operators(
        nn, hw, lam, e3max, ramp, force, timer=timer, rtol=rtol, jmax3=jmax3
    )
    labs = lab_matrices(emax, e2max, e3max, ops, chans, timer, lab_dtype)
    if path is None:
        return 0
    opts = (WriterOptions() if options is None else options).resolve()
    assert opts.compress_threads is not None
    with stage(timer, "write"):
        return write_me3j(
            path,
            me3j_values(make_orbits(emax), e2max, e3max, labs),
            opts.compresslevel,
            opts.compressor,
            opts.compress_threads,
        )
