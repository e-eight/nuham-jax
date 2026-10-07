"""End-to-end generation of a 3N ``me3j`` file: Jacobi Hamiltonian, SRG, lab transform, writer."""

import argparse
import cProfile
import hashlib
import itertools
import json
import multiprocessing
import os
import sys
import warnings
from collections import deque
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from nuham.angmom.moshinsky_table import MoshinskyTable
from nuham.basis.lab import make_orbits
from nuham.nn.input_file import MomentumNN, read_nn_mom
from nuham.nnn.hamiltonian import JMAX_NN, N2MAX_FLOW, isospin_nn
from nuham.nnn.jacobi import JacobiChannel, basis_digest, load_or_build_channel, ramp_nmax
from nuham.nnn.lab_transform import contract_blocks
from nuham.nnn.local_force.assembly import antisymmetrize, local_3nf_matrix
from nuham.nnn.local_force.radial import RadialTables, nuhamil_meshes, radial_tables
from nuham.nnn.me3j import LabMatrix3, me3j_values, write_me3j, write_me3j_fields
from nuham.nnn.regulator import regulator_matrix
from nuham.nnn.srg3 import three_body_operator
from nuham.nnn.tcoef import LabBlock, lab_t_blocks, make_brackets, warm_brackets
from nuham.runtime import (
    WriterOptions,
    add_writer_arguments,
    default_workers,
    enable_compilation_cache,
    writer_options_from_args,
)
from nuham.timing import StageTimer, stage


@dataclass(frozen=True)
class LocalForce:
    """Parameters of the local N2LO 3N force (NuHamil's ``Regulator="Local"`` or ``"LNL"``).

    The defaults are the 3NF400 set used for the NuHamil reference run. A ``nonlocal_cutoff``
    makes it NuHamil's ``LNL`` force: ``cutoff`` is then the local cutoff and the non-local
    Gaussian regulator of ``docs/spec-5b2-lnl-regulator.md`` (same ``power``) multiplies the
    antisymmetrised force on both sides before the flow.
    """

    lecs: dict[str, float] = field(
        default_factory=lambda: {"c1": -0.81, "c3": -3.2, "c4": 5.4, "cD": -0.2, "cE": 0.098}
    )
    cutoff: float = 400.0
    power: int = 2
    jmax3: int | None = None
    """Largest doubled ``J`` of the Jacobi channels that get the genuine force; above it only the
    induced force of the NN interaction is kept, as for NuHamil's ``J3max_initial_3nf``
    (``None``: all channels, NuHamil's default). The angular coefficients of the high-``J``
    channels are most of the cost of a production run."""
    nonlocal_cutoff: float | None = field(default=None, repr=False)
    """Cutoff (MeV) of the non-local regulator, ``None`` for the purely local force. Left out of
    ``repr`` so that the cache keys of local runs stay as they were; :func:`_force_key` adds it."""


PRESETS: dict[str, LocalForce] = {
    "3NF400": LocalForce(),
    "3NF500": LocalForce(
        {"c1": -0.81, "c3": -3.2, "c4": 5.4, "cD": 0.83, "cE": -0.052}, cutoff=500.0
    ),
    "N3LOEMlnl": LocalForce(
        {"c1": -0.81, "c3": -3.2, "c4": 5.4, "cD": 0.7, "cE": -0.06},
        cutoff=650.0,
        nonlocal_cutoff=500.0,
    ),
}
"""NuHamil's named forces for the EM500 NN interaction (``NuHamil_3BME.py``): 3NF400, 3NF500
(``3NF500-BetaDecay-Local``) and the local plus non-local ``N3LOEMlnl``."""


def _force_key(force: LocalForce | None) -> object:
    """Cache-key part of ``force``: its ``repr``, plus the non-local cutoff when there is one."""
    if force is None or force.nonlocal_cutoff is None:
        return force
    return (force, force.nonlocal_cutoff)


CACHE_VERSION = 2
"""Bump when a change of the physics code makes cached operators stale.

``tests/test_v3_reference.py`` fails when the numbers of a small ``V3`` move: that is the signal
to bump this (and to update the reference if the change is intended)."""


def _nn_digest(nn: MomentumNN) -> str:
    """Short digest of the momentum-space interaction's numbers."""
    h = hashlib.sha1()
    for a in (nn.k, nn.w):
        h.update(np.ascontiguousarray(a).tobytes())
    for c in nn.channels:
        h.update(repr((c.j, c.parity, c.s, c.tz)).encode())
        h.update(np.ascontiguousarray(c.v).tobytes())
    return h.hexdigest()[:16]


def _v3_path(cache_dir: str | Path, tag: str, j2: int, parity: int, t2: int, nmax: int) -> Path:
    """File of one channel's cached ``V3``.

    The tag fixes the run and, with the channel's :func:`~nuham.nnn.jacobi.basis_digest`, the AS
    gauge the operator is expressed in: a rebuilt basis never meets an operator of another one.
    """
    return (
        Path(cache_dir) / "v3" / f"v3_{tag}_j{j2}_p{'+' if parity > 0 else '-'}_t{t2}_N{nmax}.npy"
    )


class StageLog:
    """Per-channel stage times as JSON lines, and the same times added to a run total.

    One record per Jacobi or lab channel, flushed as it is written, so a job that dies or is
    cancelled still leaves the profile of the channels it finished.
    """

    def __init__(self, total: StageTimer | None, path: str | Path | None) -> None:
        """Add every channel's times to ``total`` and write them to ``path`` (``None``: no file)."""
        self.total = total
        self._file = None if path is None else open(path, "a", buffering=1)

    def channel(self, kind: str, label: dict, timer: StageTimer) -> None:
        """Record the times of one channel; ``label`` describes it."""
        if self.total is not None:
            for name, seconds in timer.seconds.items():
                self.total.seconds[name] = self.total.seconds.get(name, 0.0) + seconds
                self.total.calls[name] = self.total.calls.get(name, 0) + timer.calls[name]
        if self._file is not None:
            record = {
                "kind": kind,
                **label,
                "seconds": {k: round(v, 4) for k, v in timer.seconds.items()},
            }
            self._file.write(json.dumps(record) + "\n")

    def close(self) -> None:
        """Close the file, if any."""
        if self._file is not None:
            self._file.close()


def _record(
    total: StageTimer | None, log: StageLog | None, kind: str, ct: StageTimer, **label: int
) -> None:
    """Hand one channel's times to the log, or to the run total when there is no log."""
    if log is not None:
        log.channel(kind, label, ct)
    elif total is not None:
        for name, seconds in ct.seconds.items():
            total.seconds[name] = total.seconds.get(name, 0.0) + seconds
            total.calls[name] = total.calls.get(name, 0) + ct.calls[name]


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
    cache_dir: str | Path | None = None,
    workers: int = 1,
    log: StageLog | None = None,
    js: Sequence[int] | None = None,
    n2max: int = N2MAX_FLOW,
    jmax_nn: int = JMAX_NN,
) -> tuple[dict[tuple[int, int, int], np.ndarray], dict[tuple[int, int, int], JacobiChannel]]:
    """Evolved three-body operators ``V3`` of all Jacobi channels, cut to ``N <= e3max``.

    Channels ``(2J, P, 2T)`` run over ``2J <= jmax3``, by default ``2 e3max + 3``, the range that
    can occur in the lab transform of ``e3max`` (a smaller ``jmax3`` drops the higher ones, as
    NuHamil's input of the same name does). Each flow runs in the full ramp space and is dropped
    after its leading block is kept, so memory stays at the size of the largest single channel.
    ``force=None`` omits the genuine force. ``workers`` processes compute the angular
    coefficients of the force (a single process took 650 s for the largest channel on the GH200,
    where 30 took 119 s). ``cache_dir`` keeps the Jacobi bases between runs
    (:func:`nuham.nnn.jacobi.load_or_build_channel`) and, in ``cache_dir/v3``, each channel's
    finished ``V3``: a run that stopped (time limit, crash) resumes after the last channel it
    completed, and a run with other ``hw``, ``lam``, interaction, force, ``e3max`` or ``rtol``
    never reads them. ``log`` receives the stage times of every channel. ``js`` restricts the
    channels to those doubled ``J`` (for timing a few channels of a large run). ``n2max`` and
    ``jmax_nn`` are the NN flow's relative space and largest relative ``J`` (NuHamil's values by
    default); they are part of the cache key like the other physics settings.

    Returns
    -------
    tuple
        ``V3`` and the matching truncated :class:`JacobiChannel` (their ``cfp`` fix the gauge)
        by ``(2J, P, 2T)``.
    """
    table = MoshinskyTable(d=1 / 3) if table is None else table
    with stage(timer, "nn_relative"):
        bare, evolved = isospin_nn(nn, hw, lam, n2max, jmax_nn)
    meshes = nuhamil_meshes()
    tables: dict[int, RadialTables] = {}
    ops, chans = {}, {}
    tag = hashlib.sha1(
        repr(
            (CACHE_VERSION, _nn_digest(nn), hw, lam, e3max, rtol, _force_key(force), n2max, jmax_nn)
        ).encode()
    ).hexdigest()[:16]
    for t2 in (1, 3):
        for j2 in range(1, (2 * e3max + 3 if jmax3 is None else jmax3) + 1, 2):
            if js is not None and j2 not in js:
                continue
            for parity in (1, -1):
                nmax = ramp_nmax(j2, ramp)
                if nmax < e3max:
                    raise ValueError(f"ramp {ramp!r} is below e3max = {e3max} at 2J = {j2}")
                ct = StageTimer()
                with ct.stage("jacobi_basis"):
                    channel = load_or_build_channel(j2, parity, t2, nmax, table, cache_dir)
                if not channel.blocks:  # no antisymmetric state: nothing to evolve or transform
                    continue
                v3_file = (
                    None
                    if cache_dir is None
                    else _v3_path(cache_dir, tag + basis_digest(channel), j2, parity, t2, nmax)
                )
                kept = tuple(b for b in channel.blocks if b.n_quanta <= e3max)
                if v3_file is not None and v3_file.exists():
                    chans[j2, parity, t2] = JacobiChannel(j2, parity, t2, e3max, kept)
                    ops[j2, parity, t2] = np.load(v3_file)
                    _record(timer, log, "jacobi", ct, j2=j2, parity=parity, t2=t2, nmax=nmax,
                            n_as=channel.n_as, n_nas=channel.n_nas, cached=1)  # fmt: skip
                    continue
                v3n = None
                if force and (force.jmax3 is None or j2 <= force.jmax3):
                    with ct.stage("force"):
                        if nmax not in tables:
                            tables[nmax] = radial_tables(
                                meshes, force.cutoff, force.power, nmax + 2
                            )
                        v3n = local_3nf_matrix(
                            channel, hw, meshes, tables[nmax], force.lecs, workers=workers, timer=ct
                        )
                    if force.nonlocal_cutoff is not None:
                        with ct.stage("regulator"):
                            nas = np.concatenate([b.nas for b in channel.blocks])
                            r = regulator_matrix(nas, hw, force.nonlocal_cutoff, force.power)
                            fa = antisymmetrize(r, channel)
                            v3n = fa @ v3n @ fa
                            v3n.block_until_ready()
                v3 = three_body_operator(
                    channel, hw, lam, bare, evolved, v3n, e3max, rtol=rtol, atol=rtol, timer=ct
                )
                chans[j2, parity, t2] = JacobiChannel(j2, parity, t2, e3max, kept)
                ops[j2, parity, t2] = v3
                if v3_file is not None:
                    v3_file.parent.mkdir(parents=True, exist_ok=True)
                    np.save(v3_file.with_suffix(".tmp.npy"), v3)
                    v3_file.with_suffix(".tmp.npy").rename(v3_file)
                _record(timer, log, "jacobi", ct, j2=j2, parity=parity, t2=t2, nmax=nmax,
                        n_as=channel.n_as, n_nas=channel.n_nas)  # fmt: skip
    return ops, chans


_LAB_WORKER: dict = {}


def _lab_task(key: tuple[int, int, int]) -> tuple:
    """Host coefficients of one lab channel in a worker process (inputs inherited by ``fork``)."""
    w = _LAB_WORKER
    j2, parity, t2 = key
    return lab_t_blocks(
        j2, parity, t2, w["orb"], w["e2max"], w["e3max"], w["jac"][t2], w["brackets"]
    )


def lab_matrices(
    emax: int,
    e2max: int,
    e3max: int,
    ops: dict[tuple[int, int, int], np.ndarray],
    chans: dict[tuple[int, int, int], JacobiChannel],
    timer: StageTimer | None = None,
    dtype: type = np.float64,
    log: StageLog | None = None,
    workers: int = 1,
) -> dict[tuple[int, int, int], LabMatrix3]:
    """Lab matrices ``6 sum T^T V3 T`` of every ``(2J, P, 2T)`` channel that has states.

    The contraction runs on the device in float64 and the matrices are stored as ``dtype`` on the
    host (``np.float32`` halves the memory; the file has eight decimals anyway). ``log``
    receives the stage times of every channel. The coefficients are built on the host, one
    channel at a time (about 44 s for the largest at e3max 16 on the GH200, 17 minutes for all);
    with ``workers > 1`` forked processes build them for several channels at once, at most
    ``min(workers, 8)`` finished ones waiting for the device, which bounds the host memory.
    """
    orb = make_orbits(emax)
    brackets = make_brackets()
    keys = [(j2, p, t2) for t2 in (1, 3) for j2 in range(1, 2 * e3max + 4, 2) for p in (1, -1)]
    vjac = {t2: {(j, p): v for (j, p, t), v in ops.items() if t == t2} for t2 in (1, 3)}
    jac = {t2: {(j, p): c for (j, p, t), c in chans.items() if t == t2} for t2 in (1, 3)}
    out = {}

    def finish(key: tuple[int, int, int], lab: LabBlock, blocks: dict, ct: StageTimer) -> None:
        j2, parity, t2 = key
        lab, m, order = contract_blocks(lab, blocks, vjac[t2], jac[t2], dtype, timer=ct)
        if len(lab):
            out[key] = LabMatrix3.make(lab, m, len(orb), order)
        _record(timer, log, "lab", ct, j2=j2, parity=parity, t2=t2, n_lab=len(lab))

    with stage(timer, "lab"):
        if workers <= 1:
            for key in keys:
                ct = StageTimer()
                with ct.stage("tcoef"):
                    lab, blocks = lab_t_blocks(*key, orb, e2max, e3max, jac[key[2]], brackets)
                finish(key, lab, blocks, ct)
            return out
        with stage(timer, "brackets"):
            warm_brackets(brackets, e3max)  # JAX builds them; never inside a forked worker
        _LAB_WORKER.update(orb=orb, e2max=e2max, e3max=e3max, jac=jac, brackets=brackets)
        ctx = multiprocessing.get_context("fork")
        window = max(1, min(workers, 8))
        pending = iter(keys)
        with ProcessPoolExecutor(workers, mp_context=ctx) as pool:
            inflight: deque = deque()
            for key in itertools.islice(pending, window):
                inflight.append((key, pool.submit(_lab_task, key)))
            while inflight:
                key, future = inflight.popleft()
                ct = StageTimer()
                with ct.stage("tcoef_wait"):  # time the device loop waited for this channel
                    lab, blocks = future.result()
                if (nxt := next(pending, None)) is not None:
                    inflight.append((nxt, pool.submit(_lab_task, nxt)))
                finish(key, lab, blocks, ct)
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
    workers: int | None = None,
    cache_dir: str | Path | None = None,
    stop_after: str = "write",
    stage_log: str | Path | None = None,
    js: Sequence[int] | None = None,
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
        are used so far; the assembly and formatting are the host paths). Unset compression
        threads become ``min(8, workers)``: eight threads wrote 71 M values/s on the GH200, four
        48 M.
    rtol
        Flow tolerance (rtol = atol in units of ``hw``).
    jmax3
        Largest doubled ``J`` of the Jacobi channels; ``None`` is ``2 e3max + 3``.
    lab_dtype
        Storage type of the lab matrices (``np.float32`` for production sizes).
    workers
        Processes for the angular coefficients of the 3N force and for the lookup, ordering and
        formatting of the file's values (:func:`nuham.nnn.me3j.me3j_values`); ``None`` is every
        available CPU (:func:`nuham.runtime.default_workers`).
    cache_dir
        Directory that keeps the Jacobi bases between runs (they do not depend on ``hw``, ``lam``).

    stop_after
        ``"jacobi"`` (the operators of all Jacobi channels), ``"lab"`` (also the lab matrices) or
        ``"write"`` (the whole file); an earlier stop returns 0. For timing one part of a large run.
    stage_log
        File that receives the stage times of every channel as JSON lines (appended).
    js
        Compute only the Jacobi channels with these doubled ``J`` (needs ``stop_after="jacobi"``:
        the lab transform needs all channels). For timing a few channels of a large run.

    Returns
    -------
    int
        The number of values written.
    """
    if stop_after not in ("jacobi", "lab", "write"):
        raise ValueError(f"stop_after must be 'jacobi', 'lab' or 'write', not {stop_after!r}")
    if js is not None and stop_after != "jacobi":
        raise ValueError("js restricts the Jacobi channels, so the run must stop after 'jacobi'")
    workers = default_workers() if workers is None else workers
    if workers == 1 and default_workers() > 4:
        warnings.warn(
            f"workers=1 with {default_workers()} CPUs available: the 3N force and the file "
            "lookup will run on one process",
            stacklevel=2,
        )
    if e3max > e2max:
        raise ValueError("e3max > e2max gives a file that imsrg++ cannot read (spec 5d 2.3)")
    log = StageLog(timer, stage_log)
    try:
        ops, chans = jacobi_operators(
            nn, hw, lam, e3max, ramp, force, timer=timer, rtol=rtol, jmax3=jmax3,
            cache_dir=cache_dir, workers=workers, log=log, js=js,
        )  # fmt: skip
        if stop_after == "jacobi":
            return 0
        labs = lab_matrices(emax, e2max, e3max, ops, chans, timer, lab_dtype, log, workers)
    finally:
        log.close()
    if path is None or stop_after == "lab":
        return 0
    opts = options if options is not None else WriterOptions()
    threads = opts.compress_threads or min(8, workers)
    opts = replace(opts, compress_threads=threads).resolve()
    assert opts.compress_threads is not None
    orb = make_orbits(emax)
    with stage(timer, "write"):
        if workers > 1:
            fields = me3j_values(orb, e2max, e3max, labs, workers=workers, as_fields=True)
            return write_me3j_fields(
                path, fields, opts.compresslevel, opts.compressor, opts.compress_threads
            )
        return write_me3j(
            path,
            me3j_values(orb, e2max, e3max, labs),
            opts.compresslevel,
            opts.compressor,
            opts.compress_threads,
        )


def run_config(args: argparse.Namespace, workers: int, writer: WriterOptions) -> dict:
    """Describe what the run will use, for the first line of the log.

    Printed before any work starts, so a wrong worker count, compression thread count or stop
    point is visible in the first seconds of a job instead of in its timings.
    """
    import jax

    return {
        "workers": workers,
        "cpus_available": default_workers(),
        "compress_threads": writer.compress_threads or min(8, workers),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "emax": args.emax, "e2max": args.e2max, "e3max": args.e3max, "ramp": args.ramp,
        "hw": args.hw, "lam": args.lam, "genuine_3nf": not args.no_3nf,
        "lab_dtype": args.lab_dtype, "rtol": args.rtol, "stop_after": args.stop_after,
        "jacobi_cache": args.jacobi_cache, "stage_log": args.stage_log, "js": args.js,
        "j3max_3nf": args.j3max_3nf, "force": None if args.no_3nf else repr(force_from_args(args)),
        "cutoff_nl": None if args.no_3nf else force_from_args(args).nonlocal_cutoff,
    }  # fmt: skip


def force_from_args(args: argparse.Namespace) -> LocalForce:
    """Return the preset ``args.preset`` with every option given on the command line applied."""
    base = PRESETS[args.preset]
    given = {k: getattr(args, k) for k in base.lecs}
    lecs = {k: base.lecs[k] if v is None else v for k, v in given.items()}
    return LocalForce(
        lecs,
        base.cutoff if args.cutoff is None else args.cutoff,
        base.power if args.power is None else args.power,
        args.j3max_3nf,
        base.nonlocal_cutoff if args.cutoff_nl is None else args.cutoff_nl,
    )


def main() -> None:
    """Command-line entry point: ``python -m nuham.nnn.generate --help``."""
    p = argparse.ArgumentParser(
        description="Generate a 3N me3j file (local N2LO force) for imsrg++."
    )
    p.add_argument("nn_file", help="NuHamil momentum-space *.bin file")
    p.add_argument("output", help="output file, e.g. N3LO_3NF400_hw16_e14_e2_28_e3_16.me3j.gz")
    p.add_argument("--hw", type=float, required=True, help="oscillator energy in MeV")
    p.add_argument("--lam", type=float, required=True, help="SRG cutoff in fm^-1")
    p.add_argument("--emax", type=int, required=True)
    p.add_argument("--e2max", type=int, required=True)
    p.add_argument("--e3max", type=int, required=True)
    p.add_argument(
        "--ramp", required=True, help="flow space per J: flat8 or ramp40-5-36-7-32-9-28-11-24"
    )
    p.add_argument(
        "--jmax3", type=int, default=None, help="largest doubled J (default 2 e3max + 3)"
    )
    p.add_argument("--no-3nf", action="store_true", help="induced 3N force of the NN force only")
    p.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="3NF400",
        help="named force of NuHamil_3BME.py (default 3NF400); the options below override it",
    )
    p.add_argument("--cutoff", type=float, default=None, help="local 3NF regulator cutoff (MeV)")
    p.add_argument(
        "--cutoff-nl",
        type=float,
        default=None,
        help="cutoff (MeV) of the non-local regulator (LNL force); the local force alone if unset",
    )
    p.add_argument(
        "--power", type=int, default=None, help="regulator power, local and non-local (default 2)"
    )
    p.add_argument(
        "--j3max-3nf",
        type=int,
        default=None,
        help="largest doubled J of the Jacobi channels with the genuine 3NF (default: all; "
        "above it only the induced force, like NuHamil's J3max_initial_3nf)",
    )
    p.add_argument(
        "--js", default=None, help="comma-separated doubled J: only these Jacobi channels (timing)"
    )
    for name in ("c1", "c3", "c4", "cD", "cE"):
        p.add_argument(f"--{name}", type=float, default=None)
    p.add_argument("--rtol", type=float, default=1e-8, help="flow tolerance (rtol = atol / hw)")
    p.add_argument("--lab-dtype", choices=("float32", "float64"), default="float32")
    p.add_argument(
        "--workers", type=int, default=None, help="worker processes (default: every available CPU)"
    )
    p.add_argument("--jacobi-cache", default=None, help="directory for cached Jacobi bases")
    p.add_argument(
        "--stop-after",
        choices=("jacobi", "lab", "write"),
        default="write",
        help="stop after the Jacobi operators or the lab matrices (timing one part of a big run)",
    )
    p.add_argument("--stage-log", default=None, help="append per-channel stage times (JSON lines)")
    p.add_argument(
        "--cprofile", default=None, help="write cProfile statistics of the main process here"
    )
    add_writer_arguments(p)
    p.add_argument("--compile-cache", default=None, help="directory for JAX's compilation cache")
    p.add_argument("--no-compile-cache", action="store_true")
    args = p.parse_args()
    workers = default_workers() if args.workers is None else args.workers
    writer = writer_options_from_args(args)
    print("config:", json.dumps(run_config(args, workers, writer)), flush=True)
    if workers > default_workers():
        print(
            f"WARNING: {workers} workers on {default_workers()} available CPUs",
            file=sys.stderr, flush=True,
        )  # fmt: skip
    if not args.no_compile_cache:
        enable_compilation_cache(args.compile_cache)
    force = None
    if not args.no_3nf:
        force = force_from_args(args)
    timer = StageTimer()
    profiler = cProfile.Profile() if args.cprofile else None
    if profiler is not None:
        profiler.enable()
    n = generate_me3j(
        read_nn_mom(args.nn_file), args.output, args.hw, args.lam, args.emax, args.e2max,
        args.e3max, args.ramp, force, timer, writer, args.rtol, args.jmax3,
        getattr(np, args.lab_dtype), workers, args.jacobi_cache, args.stop_after, args.stage_log,
        None if args.js is None else [int(x) for x in args.js.split(",")],
    )  # fmt: skip
    if profiler is not None:
        profiler.disable()
        profiler.dump_stats(args.cprofile)
    print(f"wrote {n} values to {args.output}")
    print("stages:", {k: round(v, 1) for k, v in timer.seconds.items()})


if __name__ == "__main__":
    main()
