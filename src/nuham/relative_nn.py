"""Driver for the relative-space NN pipeline: projection, Coulomb, SRG flow, ``V = H - T``."""

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from nuham.basis.relative import relative_states
from nuham.constants import oscillator_length_sq
from nuham.interactions.nn_file import MomentumNN
from nuham.kernels.coulomb import coulomb_matrix
from nuham.kernels.projection import project_channel
from nuham.kernels.relative import kinetic_matrix
from nuham.kernels.srg import srg_alpha, srg_evolve_batch
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
    are padded to a multiple of ``granularity`` and flowed in one batch per padded size.

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
    b = math.sqrt(oscillator_length_sq(hw))
    prepared = []
    with stage(timer, "project"):
        for ch in nn.channels:
            if ch.j > jmax:
                continue
            n, l = relative_states(ch.ls, n2max)
            v = project_channel(nn.k, nn.w, ch.v, ch.ls, n2max, b)
            if coulomb and ch.tz == -1:
                v = v + coulomb_matrix(ch.ls, n2max, hw)
            t = kinetic_matrix(n, l, hw)
            prepared.append(
                _Prepared((ch.j, ch.parity, ch.s, ch.tz), n, l, np.asarray(v), np.asarray(t))
            )
    with stage(timer, "srg"):
        potentials = (
            [p.v for p in prepared]
            if not lam
            else _flow(prepared, hw, lam, rtol, atol, granularity)
        )
    return tuple(
        RelativeChannel(*p.channel_key, p.n, p.l, v)
        for p, v in zip(prepared, potentials, strict=True)
    )


def _flow(
    prepared: list[_Prepared], hw: float, lam: float, rtol: float, atol: float, granularity: int
) -> list[np.ndarray]:
    """Flow channels in padded batches of equal size and return ``V = H - T`` for each."""
    alpha = srg_alpha(lam)
    groups: dict[int, list[int]] = {}
    for i, p in enumerate(prepared):
        groups.setdefault(_bucket_size(len(p.n), granularity), []).append(i)
    results: list[np.ndarray] = [np.empty(0)] * len(prepared)
    for size, members in groups.items():
        h0 = np.zeros((len(members), size, size))
        t_diag = np.zeros((len(members), size))
        t_off = np.zeros((len(members), size - 1))
        for row, i in enumerate(members):
            dim = len(prepared[i].n)
            h0[row, :dim, :dim] = prepared[i].v + prepared[i].t
            t_diag[row], t_off[row] = _tridiagonal_parts(prepared[i].t, size)
        evolved = np.asarray(
            srg_evolve_batch(
                jnp.asarray(h0), jnp.asarray(t_diag), jnp.asarray(t_off), hw, alpha, rtol, atol
            )
        )
        for row, i in enumerate(members):
            dim = len(prepared[i].n)
            results[i] = evolved[row, :dim, :dim] - prepared[i].t
    return results
