"""Driver for the relative-space NN pipeline: projection, Coulomb, SRG flow, ``V = H - T``."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from nuham.basis.relative import relative_states
from nuham.constants import oscillator_length_sq
from nuham.kernels.nn.coulomb import coulomb_matrix
from nuham.kernels.nn.projection import project_channel
from nuham.kernels.nn.relative import kinetic_matrix
from nuham.kernels.srg import srg_alpha, srg_evolve_batch
from nuham.nn.input_file import MomentumNN
from nuham.timing import StageTimer, stage


@dataclass(frozen=True)
class RelativeChannel:
    """Evolved relative-space potential of one (J, parity, S, Tz) channel.

    Attributes
    ----------
    j, parity, s, tz
        Channel quantum numbers; ``tz`` is -1 for pp, 0 for pn, +1 for nn.
    n, l
        Quantum numbers of the basis states, l-major.
    v
        Potential ``V = H - T_rel`` in MeV, shape ``(len(n), len(n))``.
    """

    j: int
    parity: int
    s: int
    tz: int
    n: np.ndarray
    l: np.ndarray
    v: np.ndarray


def _bucket_size(dim: int, granularity: int) -> int:
    """Round ``dim`` up to a multiple of ``granularity``."""
    return -(-dim // granularity) * granularity


def _tridiagonal_parts(t: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Zero-padded diagonal and first off-diagonal of a tridiagonal matrix."""
    dim = t.shape[0]
    diag = np.zeros(size)
    off = np.zeros(size - 1)
    diag[:dim] = np.diag(t)
    off[: dim - 1] = np.diag(t, 1)
    return diag, off


@dataclass(frozen=True)
class _Prepared:
    """A channel projected to the HO basis, before the flow."""

    channel_key: tuple[int, int, int, int]
    n: np.ndarray
    l: np.ndarray
    v: np.ndarray
    t: np.ndarray


def evolve_relative_nn(
    nn: MomentumNN,
    hw: float,
    lam: float | None,
    n2max: int = 100,
    jmax: int = 8,
    coulomb: bool = True,
    rtol: float = 1e-10,
    atol: float = 1e-10,
    granularity: int = 16,
    timer: StageTimer | None = None,
) -> tuple[RelativeChannel, ...]:
    """Project the NN potential to the HO relative basis and SRG-evolve every channel.

    Coulomb is added to the pp channels before the flow and is evolved with them, as NuHamil does
    by default. The flow runs in the full ``2n + l <= n2max`` space of each channel. Channels
    are padded to a multiple of ``granularity`` and flowed in one batch per padded size. This is
    :func:`evolve_relative_nn_many` for a single parameter set.

    Parameters
    ----------
    nn
        Momentum-space interaction.
    hw
        Oscillator energy in MeV.
    lam
        SRG cutoff in fm^-1; ``None`` or ``0`` skips the flow (bare interaction).
    n2max
        Truncation on ``2n + l``.
    jmax
        Largest total angular momentum kept.
    coulomb
        Whether to add the pp Coulomb interaction.
    rtol, atol
        Flow tolerances; ``atol`` in MeV.
    granularity
        Padding multiple for batching channels of different dimension.
    timer
        Optional stage timer; records ``project`` and ``srg``.

    Returns
    -------
    tuple[RelativeChannel, ...]
        Evolved channels in the order of the input file.
    """
    return evolve_relative_nn_many(
        nn, [(hw, lam)], n2max, jmax, coulomb, rtol, atol, granularity, timer
    )[0]


def evolve_relative_nn_many(
    nn: MomentumNN,
    params: Sequence[tuple[float, float | None]],
    n2max: int = 100,
    jmax: int = 8,
    coulomb: bool = True,
    rtol: float = 1e-10,
    atol: float = 1e-10,
    granularity: int = 16,
    timer: StageTimer | None = None,
    flow_sets: int | None = None,
) -> list[tuple[RelativeChannel, ...]]:
    """Evolve the interaction for several ``(hw, lam)`` sets, flowing all sets in shared batches.

    Projection is per set (it depends on ``hw``). The flows of ``flow_sets`` sets at a time are
    grouped by padded size into one batch each, with per-entry ``hw`` and flow parameter. Each
    result equals the one of a single-set call. Larger flow batches suit devices with spare
    parallelism; on a CPU they were slower per set (cache pressure, and a batch runs as long as
    its slowest entry), see ``docs/benchmarks.md``.

    Parameters
    ----------
    nn
        Momentum-space interaction.
    params
        ``(hw, lam)`` pairs; ``lam`` is ``None`` or ``0`` for the bare interaction.
    n2max, jmax, coulomb, rtol, atol, granularity, timer
        As in :func:`evolve_relative_nn`.
    flow_sets
        Number of sets whose flows run in one batch; ``None`` runs all sets together.

    Returns
    -------
    list[tuple[RelativeChannel, ...]]
        One tuple of channels per set, in the order of ``params``.
    """
    prepared: list[list[_Prepared]] = []
    with stage(timer, "project"):
        for hw, _lam in params:
            b = math.sqrt(oscillator_length_sq(hw))
            chans = []
            for ch in nn.channels:
                if ch.j > jmax:
                    continue
                n, l = relative_states(ch.ls, n2max)
                v = project_channel(nn.k, nn.w, ch.v, ch.ls, n2max, b)
                if coulomb and ch.tz == -1:
                    v = v + coulomb_matrix(ch.ls, n2max, hw)
                t = kinetic_matrix(n, l, hw)
                chans.append(
                    _Prepared((ch.j, ch.parity, ch.s, ch.tz), n, l, np.asarray(v), np.asarray(t))
                )
            prepared.append(chans)
    with stage(timer, "srg"):
        step = len(params) if flow_sets is None else max(1, flow_sets)
        potentials = [
            vs
            for first in range(0, len(params), step)
            for vs in _flow(
                prepared[first : first + step],
                params[first : first + step],
                rtol,
                atol,
                granularity,
            )
        ]
    return [
        tuple(RelativeChannel(*p.channel_key, p.n, p.l, v) for p, v in zip(chans, vs, strict=True))
        for chans, vs in zip(prepared, potentials, strict=True)
    ]


def _flow(
    prepared: list[list[_Prepared]],
    params: Sequence[tuple[float, float | None]],
    rtol: float,
    atol: float,
    granularity: int,
) -> list[list[np.ndarray]]:
    """Flow all channels of all sets in padded batches and return ``V = H - T`` for each.

    Sets with ``lam`` of ``None`` or 0 skip the flow and keep their projected potential.
    """
    results = [[p.v for p in chans] for chans in prepared]
    groups: dict[int, list[tuple[int, int]]] = {}
    alpha_of: dict[int, float] = {}
    for i, ((_hw, lam), chans) in enumerate(zip(params, prepared, strict=True)):
        if not lam:
            continue
        alpha_of[i] = srg_alpha(lam)
        for c, p in enumerate(chans):
            groups.setdefault(_bucket_size(len(p.n), granularity), []).append((i, c))
    for size, members in groups.items():
        h0 = np.zeros((len(members), size, size))
        t_diag = np.zeros((len(members), size))
        t_off = np.zeros((len(members), size - 1))
        hws = np.empty(len(members))
        alphas = np.empty(len(members))
        for row, (i, c) in enumerate(members):
            p = prepared[i][c]
            dim = len(p.n)
            h0[row, :dim, :dim] = p.v + p.t
            t_diag[row], t_off[row] = _tridiagonal_parts(p.t, size)
            hws[row] = params[i][0]
            alphas[row] = alpha_of[i]
        evolved = np.asarray(
            srg_evolve_batch(
                jnp.asarray(h0),
                jnp.asarray(t_diag),
                jnp.asarray(t_off),
                jnp.asarray(hws),
                jnp.asarray(alphas),
                rtol,
                atol,
            )
        )
        for row, (i, c) in enumerate(members):
            p = prepared[i][c]
            dim = len(p.n)
            results[i][c] = evolved[row, :dim, :dim] - p.t
    return results
