"""Assembly of the local N2LO 3N force in the antisymmetrised Jacobi basis (Phase 5.2.4).

For one ``(J, P, T)`` the non-antisymmetrised (NAS) matrix is
``W = 3 sum_t LEC_t V_t`` with ``V_t`` the representative-term matrices of
:mod:`nuham.nnn.local_force.angular` projected on HO states, and the antisymmetrised matrix is
``cfp W cfp^T`` block by block in ``N`` (the factor 3 replaces the sum over the three permutations,
checked in ``docs/study-local-3nf.md``). The LECs are in GeV^-1 for c1, c3, c4.
"""

import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from nuham.constants import HBARC, M_REDUCED_PN
from nuham.kernels.nnn.antisym import antisymmetrize as antisymmetrize_dense
from nuham.kernels.nnn.antisym import dense_from_coo
from nuham.kernels.nnn.projection import (
    orbital_table,
    project_weighted_pairs,
    scatter_pair_blocks,
)
from nuham.nnn.jacobi import JacobiChannel
from nuham.nnn.local_force.angular import (
    TERMS,
    Channel,
    channel_array,
    coefficient_table,
    radial_keys,
)
from nuham.nnn.local_force.radial import LocalMeshes, RadialTables, radial_function
from nuham.timing import StageTimer, stage

_WORKER: dict = {}


def _init_worker(channels: np.ndarray, j2: int, t2: int, x_max: int, factors: dict) -> None:
    """Store the read-only inputs of :func:`_pair_coefficients` in the worker process."""
    _WORKER.update(channels=channels, j2=j2, t2=t2, x_max=x_max, factors=factors)


def _pair_coefficients(pairs: list[tuple[int, int]]) -> np.ndarray:
    """Return ``3 LEC C_a`` of every channel pair: shape ``(len(pairs), n_radial)``.

    The columns are the radial functions of every term in the order of ``factors`` and
    :func:`radial_keys`, which is the order of the radial table ``nas_matrix`` multiplies with.
    """
    w = _WORKER
    ib, ik = np.array(pairs, dtype=np.int64).T
    bra, ket = w["channels"][ib], w["channels"][ik]
    return np.hstack(
        [
            factor * coefficient_table(term, bra, ket, w["j2"], w["t2"], w["x_max"])
            for term, factor in w["factors"].items()
        ]
    )


def nas_matrix(
    nas: np.ndarray,
    j2: int,
    t2: int,
    hw: float,
    meshes: LocalMeshes,
    tables: RadialTables,
    lecs: dict[str, float],
    *,
    workers: int = 1,
    chunk: int = 128,
    timer: StageTimer | None = None,
) -> jax.Array:
    """Return ``W = 3 sum_t LEC_t V_t`` on the NAS states ``nas``, as a dense device array.

    Rows of ``nas`` are ``n12 l12 s12 j12 t12 n3 l3 j3`` (``nnn.jacobi.NAS_FIELDS``). The radial
    tables must have been built for ``x_max >= max(N) + 2``.

    The work is split by cost: the Racah coefficients of the channel pairs are host NumPy code
    (:func:`~nuham.nnn.local_force.angular.coefficient_table`, spread over ``workers``
    processes), the weighted sum over radial functions is one matrix product per chunk of
    ``chunk`` pairs, and the HO projection and the scatter of its blocks into the dense matrix
    run batched on the JAX device (the matrix never visits the host). Stage times
    (``coefficients`` wait, ``projection``, ``scatter`` and ``device_wait``, the work still
    queued on the device after the last chunk) go to ``timer``.
    """
    x_max = tables.zx.shape[0] - 1
    nu = 2.0 * M_REDUCED_PN * hw / HBARC**2
    l_max = int(nas[:, [1, 6]].max())
    n_max = int(nas[:, [0, 5]].max())
    orbitals = orbital_table(nu, meshes.xi, meshes.xi_w, l_max, n_max)
    by_channel: dict[Channel, np.ndarray] = {}
    grouped: dict[Channel, list[int]] = defaultdict(list)
    for i, row in enumerate(nas):
        grouped[Channel(row[1], row[2], row[3], row[4], row[6], row[7])].append(i)
    channels = sorted(grouped)
    for c in channels:
        by_channel[c] = np.array(grouped[c])
    factors = {t: 3.0 * lecs[t] for t in TERMS if lecs.get(t, 0.0) != 0.0}
    radial = []
    for term in factors:
        for key in radial_keys(term, x_max):
            radial.append(radial_function(term, key, tables, meshes).ravel())
    radial_flat = np.array(radial)
    radial_dev = jnp.asarray(radial_flat)
    pairs = [(ib, ik) for ib in range(len(channels)) for ik in range(ib + 1)]
    chunks = [pairs[i : i + chunk] for i in range(0, len(pairs), chunk)]
    init = (channel_array(channels), j2, t2, x_max, factors)
    n_nas = len(nas)
    width = max(len(v) for v in by_channel.values())
    out = jnp.zeros((n_nas, n_nas))
    n12 = jnp.asarray(np.append(nas[:, 0], 0), dtype=jnp.int32)
    n3 = jnp.asarray(np.append(nas[:, 5], 0), dtype=jnp.int32)
    if workers > 1:
        ctx = multiprocessing.get_context("fork")
        pool = ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker, initargs=init)
        results = pool.map(_pair_coefficients, chunks)
    else:
        _init_worker(*init)
        pool = None
        results = (_pair_coefficients(c) for c in chunks)
    if timer is not None:
        results = timer.wrap("coefficients", results)
    for pair_chunk, coefs in zip(chunks, results, strict=True):
        weights = np.zeros((chunk, len(radial_flat)))
        weights[: len(coefs)] = coefs
        live = np.flatnonzero(np.abs(weights).sum(axis=1) > 0.0)
        if len(live) == 0:
            continue
        lb = np.array([channels[ib].l12 for ib, _ in pair_chunk] + [0] * (chunk - len(pair_chunk)))
        l3b = np.array([channels[ib].l3 for ib, _ in pair_chunk] + [0] * (chunk - len(pair_chunk)))
        lk = np.array([channels[ik].l12 for _, ik in pair_chunk] + [0] * (chunk - len(pair_chunk)))
        l3k = np.array([channels[ik].l3 for _, ik in pair_chunk] + [0] * (chunk - len(pair_chunk)))
        with stage(timer, "projection"):
            m = project_weighted_pairs(
                orbitals,
                jnp.asarray(lb),
                jnp.asarray(l3b),
                jnp.asarray(lk),
                jnp.asarray(l3k),
                jnp.asarray(weights),
                radial_dev,
            )
        with stage(timer, "scatter"):
            rows, cols, mirror = _pair_indices(
                pair_chunk, live, by_channel, channels, chunk, n_nas, width
            )
            out = scatter_pair_blocks(out, m, rows, cols, mirror, n12, n3)
    if pool is not None:
        pool.shutdown()
    with stage(timer, "device_wait"):
        out.block_until_ready()
    return out


def _pair_indices(
    pair_chunk: list[tuple[int, int]],
    live: np.ndarray,
    by_channel: dict[Channel, np.ndarray],
    channels: list[Channel],
    chunk: int,
    n_nas: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NAS row and column indices of the live pairs of a chunk, padded for the device scatter.

    Pairs that are not live (no coefficient) and the padding of a short chunk get only padding
    entries (``n_nas``). All lists have the same ``width`` (the largest channel), so one compiled
    scatter serves every chunk of a matrix.
    """
    rows = np.full((chunk, width), n_nas, dtype=np.int32)
    cols = np.full((chunk, width), n_nas, dtype=np.int32)
    mirror = np.zeros(chunk, dtype=bool)
    for p in live:
        ib, ik = pair_chunk[p]
        bra, ket = by_channel[channels[ib]], by_channel[channels[ik]]
        rows[p, : len(bra)] = bra
        cols[p, : len(ket)] = ket
        mirror[p] = ib != ik
    return rows, cols, mirror


def _to_device_dense(w: np.ndarray | jax.Array | sp.sparray | sp.spmatrix, n: int) -> jax.Array:
    """``w`` as a dense device array; sparse input goes up as entries, padded to few shapes."""
    if not isinstance(w, sp.sparray | sp.spmatrix):
        return jnp.asarray(w)
    coo = sp.coo_array(w)
    return dense_from_coo(*_padded_entries(coo.row, coo.col, coo.data, n), (n, n))


def _padded_entries(
    rows: np.ndarray, cols: np.ndarray, values: np.ndarray, n: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Entry lists as device arrays, padded to a multiple of 2**20 by entries dropped on scatter."""
    length = max(1, -(-len(values) // 2**20)) * 2**20
    pad = length - len(values)
    return (
        jnp.asarray(np.pad(rows, (0, pad), constant_values=n).astype(np.int32)),
        jnp.asarray(np.pad(cols, (0, pad), constant_values=n).astype(np.int32)),
        jnp.asarray(np.pad(values, (0, pad))),
    )


def antisymmetrize(
    w: np.ndarray | jax.Array | sp.csr_array,
    channel: JacobiChannel,
    blocks: list | None = None,
    timer: StageTimer | None = None,
) -> jax.Array:
    """Return ``cfp W cfp^T`` (block-diagonal ``cfp`` in ``N``) as a dense device array.

    ``blocks`` default to those of ``channel``. ``w`` may be a SciPy sparse array (the pair
    operators of Phase 5.3 are); it and the block-diagonal ``cfp`` go to the device as entry lists
    and are made dense there. With a ``timer`` each step waits for the device and is charged to
    ``as_cfp`` (``C`` on the device), ``as_w`` (``W`` on the device) or ``as_gemm`` (the products),
    so the stages are real times and not dispatch times.
    """
    blocks = list(channel.blocks) if blocks is None else blocks
    n_nas = sum(len(b.nas) for b in blocks)
    n_as = sum(len(b.cfp) for b in blocks)
    with stage(timer, "as_cfp"):
        c = sp.coo_array(sp.block_diag([b.cfp for b in blocks]))
        c_dev = dense_from_coo(*_padded_entries(c.row, c.col, c.data, n_as), (n_as, n_nas))
        if timer is not None:
            c_dev.block_until_ready()
    with stage(timer, "as_w"):
        w_dev = _to_device_dense(w, n_nas)
        if timer is not None:
            w_dev.block_until_ready()
    with stage(timer, "as_gemm"):
        out = antisymmetrize_dense(w_dev, c_dev)
        if timer is not None:
            out.block_until_ready()
    return out


def local_3nf_matrix(
    channel: JacobiChannel,
    hw: float,
    meshes: LocalMeshes,
    tables: RadialTables,
    lecs: dict[str, float],
    *,
    workers: int = 1,
    timer: StageTimer | None = None,
) -> jax.Array:
    """Bare local N2LO 3N force in the AS Jacobi basis of ``channel`` (MeV), on the device."""
    nas = np.concatenate([b.nas for b in channel.blocks])
    w = nas_matrix(
        nas, channel.j2, channel.t2, hw, meshes, tables, lecs, workers=workers, timer=timer
    )
    with stage(timer, "antisymmetrize"):
        v = antisymmetrize(w, channel, timer=timer)
        v.block_until_ready()
    return v
