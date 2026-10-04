"""Laboratory matrices of a 3N Jacobi operator on the device (Phase 5.4).

Same result as :func:`nuham.nnn.tcoef.lab_scalar` (which stays as the slow host reference), built
block by block in ``N = e1 + e2 + e3`` with :func:`nuham.kernels.nnn.lab.congruence_block`. The
coefficients of a channel are built on the host (:func:`nuham.nnn.tcoef.lab_t_blocks`) and
uploaded once; the matrix is kept in ``N``-major state order, which the writer's lookup undoes
(:meth:`nuham.nnn.me3j.LabMatrix3.make`).
"""

import jax.numpy as jnp
import numpy as np

from nuham.basis.lab import Orbits
from nuham.kernels.nnn.lab import congruence_block, place_block
from nuham.kernels.transfer import to_host
from nuham.nnn.jacobi import JacobiChannel
from nuham.nnn.tcoef import Brackets, LabBlock, lab_t_blocks

Run = tuple[int, int, np.ndarray, tuple[int, int]]
"""``(start, stop, positions in the channel's AS order, (jrel, parity))`` of one group's rows."""


def _bucket(n: int, granularity: int) -> int:
    """Round ``n`` up to a multiple of ``granularity``."""
    return -(-n // granularity) * granularity


def _runs(
    labels: list[tuple[int, int, int, int, int]], offs: dict[tuple[int, int], dict[int, int]]
) -> dict[tuple[int, int, int], Run]:
    """Rows of each ``g = (jrel, ncm, lcm)`` in one block: a contiguous run and its AS positions."""
    out: dict[tuple[int, int, int], Run] = {}
    start = 0
    while start < len(labels):
        e, _, jrel, ncm, lcm = labels[start]
        stop = start
        while stop < len(labels) and labels[stop][2:] == (jrel, ncm, lcm) and labels[stop][0] == e:
            stop += 1
        key = (jrel, (-1) ** e)
        base = offs.get(key, {}).get(e)
        if base is not None:
            pos = base + np.array([lab[1] for lab in labels[start:stop]])
            out[jrel, ncm, lcm] = (start, stop, pos, key)
        start = stop
    return out


def _pair_entries(
    bra: dict[tuple[int, int, int], Run],
    ket: dict[tuple[int, int, int], Run],
    base: dict[tuple[int, int], tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Source index in the flat operators and block position of every entry of ``V_bd``."""
    src, rows, cols = [], [], []
    for g, (b0, b1, pos_b, key) in bra.items():
        run = ket.get(g)
        if run is None or key not in base:
            continue
        k0, k1, pos_k, _ = run
        first, size = base[key]
        src.append((first + pos_b[:, None] * size + pos_k[None, :]).ravel())
        rows.append(np.repeat(np.arange(b0, b1), k1 - k0))
        cols.append(np.tile(np.arange(k0, k1), b1 - b0))
    if not src:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    return np.concatenate(src), np.concatenate(rows), np.concatenate(cols)


def lab_scalar_device(
    j2: int,
    parity: int,
    t2: int,
    orb: Orbits,
    e2max: int,
    e3max: int,
    vjac: dict[tuple[int, int], np.ndarray],
    channels: dict[tuple[int, int], JacobiChannel],
    brackets: Brackets,
    dtype: type = np.float64,
    granularity: int = 64,
) -> tuple[LabBlock, np.ndarray, np.ndarray]:
    """Laboratory matrix ``6 sum T^T V_jac T`` of one channel, contracted on the device.

    Arguments as in :func:`nuham.nnn.tcoef.lab_scalar`, plus the storage type of the result and
    the granularity to which block sizes are rounded up (zero padding; it limits the number of
    distinct compiled shapes). The arithmetic is float64 whatever ``dtype`` is.

    Returns
    -------
    tuple
        The lab states, the matrix in ``N``-major order, and ``order``: the lab-state index of
        each row (and column) of that matrix.
    """
    lab, blocks = lab_t_blocks(j2, parity, t2, orb, e2max, e3max, channels, brackets)
    offs = {}
    for key, ch in channels.items():
        sizes = [len(b.cfp) for b in ch.blocks]
        offs[key] = {
            b.n_quanta: int(o) for b, o in zip(ch.blocks, np.cumsum([0, *sizes]), strict=False)
        }
    keys = [k for k in vjac if vjac[k].size]
    base, flat, pos = {}, [], 0
    for key in keys:
        base[key] = (pos, len(vjac[key]))
        flat.append(vjac[key].ravel())
        pos += vjac[key].size
    v_flat = jnp.asarray(np.concatenate([*flat, np.zeros(1)]))
    ns = sorted(blocks)
    order = np.concatenate([blocks[n][0] for n in ns]) if ns else np.zeros(0, dtype=np.int64)
    first_col = dict(zip(ns, np.cumsum([0] + [len(blocks[n][0]) for n in ns])[:-1], strict=True))
    t_dev, runs = {}, {}
    for n in ns:
        t_as, labels = blocks[n][1], blocks[n][2]
        rows_pad, cols_pad = (
            _bucket(t_as.shape[0], granularity),
            _bucket(t_as.shape[1], granularity),
        )
        padded = np.zeros((rows_pad, cols_pad))
        padded[: t_as.shape[0], : t_as.shape[1]] = t_as
        t_dev[n] = jnp.asarray(padded)
        runs[n] = _runs(labels, offs)
    out = jnp.zeros((len(lab), len(lab)))
    for ib, nb in enumerate(ns):
        for nk in ns[: ib + 1]:
            src, rows, cols = _pair_entries(runs[nb], runs[nk], base)
            if not len(src):
                continue
            size = _bucket(len(src), 1 << max(0, len(src).bit_length() - 3))
            fill = size - len(src)
            block = congruence_block(
                t_dev[nb],
                t_dev[nk],
                v_flat,
                jnp.asarray(np.pad(src, (0, fill), constant_values=len(v_flat) - 1)),
                jnp.asarray(np.pad(rows, (0, fill))),
                jnp.asarray(np.pad(cols, (0, fill))),
            )
            nb_cols, nk_cols = len(blocks[nb][0]), len(blocks[nk][0])
            out = place_block(out, 6.0 * block[:nb_cols, :nk_cols], first_col[nb], first_col[nk])
    return lab, to_host(out.astype(dtype)), order
