"""Laboratory matrices of a 3N Jacobi operator on the device (Phase 5.4).

Same result as :func:`nuham.nnn.tcoef.lab_scalar` (which stays as the slow host reference), built
block by block in ``N = e1 + e2 + e3`` with :func:`nuham.kernels.nnn.lab.congruence_block`. The
coefficients of a channel are built on the host (:func:`nuham.nnn.tcoef.lab_t_blocks`) and
uploaded once; the matrix is kept in ``N``-major state order, which the writer's lookup undoes
(:meth:`nuham.nnn.me3j.LabMatrix3.make`).
"""

import jax
import jax.numpy as jnp
import numpy as np

from nuham.basis.lab import Orbits
from nuham.kernels.nnn.lab import block_diagonal_operator, congruence_block
from nuham.kernels.transfer import to_host
from nuham.nnn.jacobi import JacobiChannel
from nuham.nnn.tcoef import Brackets, LabBlock, lab_t_blocks
from nuham.timing import StageTimer, stage

Run = tuple[int, int, np.ndarray, tuple[int, int]]
"""``(start, stop, positions in the channel's AS order, (jrel, parity))`` of one group's rows."""


def _rung(n: int, base: int) -> int:
    """Round ``n`` up to the ladder ``base, 1.5 base, 2 base, 3 base, 4 base, ...``.

    Every compiled kernel is specific to its shapes, and the blocks of a channel have dozens of
    different sizes: on the GH200 compiling one kernel per block pair took longer than the
    products themselves (41 s for the largest channel). A coarse ladder makes pairs share shapes;
    the padding is zeros that cost up to 1.5 times the size in each dimension.
    """
    size = base
    while size < n:
        size = size * 3 // 2 if size & (size - 1) == 0 else size * 4 // 3
    return size


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
    granularity: int = 256,
    timer: StageTimer | None = None,
) -> tuple[LabBlock, np.ndarray, np.ndarray]:
    """Laboratory matrix ``6 sum T^T V_jac T`` of one channel, contracted on the device.

    Arguments as in :func:`nuham.nnn.tcoef.lab_scalar`, plus the storage type of the result and
    the smallest size of the ladder ``granularity * 2^k`` to which block sizes are rounded up
    (zero padding; it limits the number of distinct compiled shapes). The arithmetic is float64
    whatever ``dtype`` is. ``timer`` records ``tcoef`` (host coefficients), ``upload``
    (coefficients and operators to the device) and ``gemm`` (index arrays, compilation, the
    device products and the copy of every block to the host).

    Returns
    -------
    tuple
        The lab states, the matrix in ``N``-major order, and ``order``: the lab-state index of
        each row (and column) of that matrix.
    """
    with stage(timer, "tcoef"):
        lab, blocks = lab_t_blocks(j2, parity, t2, orb, e2max, e3max, channels, brackets)
    return contract_blocks(lab, blocks, vjac, channels, dtype, granularity, timer)


def contract_blocks(
    lab: LabBlock,
    blocks: dict[int, tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int, int]]]],
    vjac: dict[tuple[int, int], np.ndarray],
    channels: dict[tuple[int, int], JacobiChannel],
    dtype: type = np.float64,
    granularity: int = 256,
    timer: StageTimer | None = None,
) -> tuple[LabBlock, np.ndarray, np.ndarray]:
    """Contract coefficients built elsewhere with ``vjac``: the device part of the transform.

    ``blocks`` are the coefficients of :func:`nuham.nnn.tcoef.lab_t_blocks`, say from a worker
    process; the other arguments are those of :func:`lab_scalar_device`.

    Returns
    -------
    tuple
        As :func:`lab_scalar_device`.
    """
    offs = {}
    for key, ch in channels.items():
        sizes = [len(b.cfp) for b in ch.blocks]
        offs[key] = {
            b.n_quanta: int(o) for b, o in zip(ch.blocks, np.cumsum([0, *sizes]), strict=False)
        }
    with stage(timer, "upload"):
        keys = [k for k in vjac if vjac[k].size]
        base, flat, pos = {}, [], 0
        for key in keys:
            base[key] = (pos, len(vjac[key]))
            flat.append(vjac[key].ravel())
            pos += vjac[key].size
        v_flat = jnp.asarray(np.concatenate([*flat, np.zeros(1)]))
        ns = sorted(blocks)
        order = np.concatenate([blocks[n][0] for n in ns]) if ns else np.zeros(0, dtype=np.int64)
        first_col = dict(
            zip(ns, np.cumsum([0] + [len(blocks[n][0]) for n in ns])[:-1], strict=True)
        )
        t_dev, runs = {}, {}
        for n in ns:
            t_as, labels = blocks[n][1], blocks[n][2]
            rows_pad, cols_pad = (
                _rung(t_as.shape[0], granularity),
                _rung(t_as.shape[1], granularity),
            )
            padded = np.zeros((rows_pad, cols_pad))
            padded[: t_as.shape[0], : t_as.shape[1]] = t_as
            t_dev[n] = jnp.asarray(padded)
            runs[n] = _runs(labels, offs)
        jax.block_until_ready(list(t_dev.values()))
    out = np.zeros((len(lab), len(lab)), dtype=dtype)
    with stage(timer, "gemm"):
        for ib, nb in enumerate(ns):
            for nk in ns[: ib + 1]:
                src, rows, cols = _pair_entries(runs[nb], runs[nk], base)
                if not len(src):
                    continue
                size = _rung(len(src), 1 << 16)
                fill = size - len(src)
                v_bd = block_diagonal_operator(
                    v_flat,
                    jnp.asarray(np.pad(src, (0, fill), constant_values=len(v_flat) - 1)),
                    jnp.asarray(np.pad(rows, (0, fill))),
                    jnp.asarray(np.pad(cols, (0, fill))),
                    (t_dev[nb].shape[0], t_dev[nk].shape[0]),
                )
                block = to_host(congruence_block(t_dev[nb], t_dev[nk], v_bd, 6.0, dtype))
                b0, k0 = first_col[nb], first_col[nk]
                nb_cols, nk_cols = len(blocks[nb][0]), len(blocks[nk][0])
                part = block[:nb_cols, :nk_cols]
                out[b0 : b0 + nb_cols, k0 : k0 + nk_cols] = part
                out[k0 : k0 + nk_cols, b0 : b0 + nb_cols] = part.T
    return lab, out, order
