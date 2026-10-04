"""The Darmstadt ``me3j`` format: tuple enumeration and value lookup in file order (Phase 5.4).

A value is ``V[(J, P, T)][(abc, J12, T12), (def, J45, T45)]``, the fully antisymmetrised,
un-normalised lab matrix element (:func:`nuham.nnn.tcoef.lab_scalar`). The file lists, for every
bra triple ``a >= b >= c`` of orbits and every ket triple ``(d, e, f)`` that does not come after it,
all ``(J12, J45, J)`` and the five isospin combinations ``(T12, T45, 2T) = (0,0,1), (0,1,1),
(1,0,1), (1,1,1), (1,1,3)``. The ket condition of NuHamil's nested loops is the same as
"``(d, e, f)`` is not after ``(a, b, c)`` in the triple order", so a bra triple pairs with a prefix
of the triple list of equal parity. The text layout is that of ``me2j`` (:func:`write_me3j`).
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nuham.basis.lab import Orbits
from nuham.nn.me2j import write_me2j
from nuham.nnn.tcoef import LabBlock

HEADER = "NNN int. calculated by nuham (JAX)"
ISOSPIN = np.array([(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1), (1, 1, 3)])
"""``(T12, T45, 2T)`` in the order of the file."""
_CHUNK = 2_000_000
_JSTRIDE = 64


def _key(
    i1: np.ndarray, i2: np.ndarray, i3: np.ndarray, j12: np.ndarray, t12: np.ndarray, n: int
) -> np.ndarray:
    """Sortable integer label of a lab state; its order is the order of states in a channel."""
    return (((i1 * n + i2) * n + i3) * _JSTRIDE + j12) * 2 + t12


@dataclass(frozen=True)
class LabMatrix3:
    """The lab matrix of one ``(J, P, T)`` channel with the labels of its rows.

    Attributes
    ----------
    keys
        Sorted state labels (:func:`_key`).
    rows
        Row of ``m`` of each sorted label.
    m
        The matrix, shape ``(n, n)``, in the row order given to :meth:`make`.
    """

    keys: np.ndarray
    rows: np.ndarray
    m: np.ndarray

    @staticmethod
    def make(
        lab: LabBlock, m: np.ndarray, n_orbits: int, order: np.ndarray | None = None
    ) -> "LabMatrix3":
        """Pair the matrix ``m`` with the labels of the states ``lab``.

        ``order`` gives the state of each row of ``m`` (``None``: the rows are in channel order),
        as returned by :func:`nuham.nnn.lab_transform.lab_scalar_device`.
        """
        keys = _key(lab.i1, lab.i2, lab.i3, lab.j12, lab.t12, n_orbits)
        keys = keys if order is None else keys[order]
        rows = np.argsort(keys)
        return LabMatrix3(keys[rows], rows, m)


def triples(
    orb: Orbits, e2max: int, e3max: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Orbit triples ``a >= b >= c`` of the file in its order, and their parity (``+-1``)."""
    n = len(orb)
    i1, i2, i3 = (x.ravel() for x in np.meshgrid(*[np.arange(n)] * 3, indexing="ij"))
    e1, e2, e3 = orb.e[i1], orb.e[i2], orb.e[i3]
    keep = (
        (i2 <= i1)
        & (i3 <= i2)
        & (e1 + e2 <= e2max)
        & (e1 + e3 <= e2max)
        & (e2 + e3 <= e2max)
        & (e1 + e2 + e3 <= e3max)
    )
    i1, i2, i3 = i1[keep], i2[keep], i3[keep]
    return i1, i2, i3, (-1) ** (orb.l[i1] + orb.l[i2] + orb.l[i3])


def _expand(count: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Parent index and running offset of ``count[i]`` children per parent, parents in order."""
    parent = np.repeat(np.arange(len(count)), count)
    first = np.cumsum(count) - count
    return parent, np.arange(parent.size) - np.repeat(first, count)


def bra_tuples(
    orb: Orbits, tri: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], bra: int
) -> tuple[np.ndarray, ...]:
    """Tuples of one bra triple, in file order.

    Returns
    -------
    tuple[np.ndarray, ...]
        ``ket`` (triple index), ``j12``, ``j45``, ``J2`` (doubled), ``T12``, ``T45``, ``2T``.
    """
    i1, i2, i3, par = tri
    a, b, c = int(i1[bra]), int(i2[bra]), int(i3[bra])
    kets = np.flatnonzero(par[: bra + 1] == par[bra])
    jb = orb.j2
    lo12, hi12 = abs(jb[a] - jb[b]) // 2, (jb[a] + jb[b]) // 2
    n12 = hi12 - lo12 + 1
    ka, kb = jb[i1[kets]], jb[i2[kets]]
    lo45, n45 = np.abs(ka - kb) // 2, (ka + kb) // 2 - np.abs(ka - kb) // 2 + 1
    # (ket, j12) rows, then j45 for each, then the total J for each.
    r1, off1 = _expand(np.full(len(kets), n12))
    r2, off2 = _expand(n45[r1])
    ket = r1[r2]
    j12 = lo12 + off1[r2]
    j45 = lo45[ket] + off2
    jf = jb[i3[kets]][ket]
    lo = np.maximum(np.abs(2 * j12 - jb[c]), np.abs(2 * j45 - jf))
    hi = np.minimum(2 * j12 + jb[c], 2 * j45 + jf)
    r3, off3 = _expand(np.maximum((hi - lo) // 2 + 1, 0))
    ket, j12, j45, jt = kets[ket[r3]], j12[r3], j45[r3], lo[r3] + 2 * off3
    iso = np.tile(ISOSPIN, (len(ket), 1))
    rep = np.repeat(np.arange(len(ket)), len(ISOSPIN))
    return ket[rep], j12[rep], j45[rep], jt[rep], iso[:, 0], iso[:, 1], iso[:, 2]


def count_values(orb: Orbits, e2max: int, e3max: int) -> int:
    """Return the number of values of the file (spec 5d: 24 890 at 4/8/4)."""
    tri = triples(orb, e2max, e3max)
    return sum(len(bra_tuples(orb, tri, b)[0]) for b in range(len(tri[0])))


def me3j_values(
    orb: Orbits,
    e2max: int,
    e3max: int,
    labs: dict[tuple[int, int, int], LabMatrix3],
    chunk: int = _CHUNK,
) -> Iterator[np.ndarray]:
    """Yield the values of the file in order, about ``chunk`` at a time.

    Parameters
    ----------
    labs
        Lab matrices by ``(2J, P, 2T)``. A tuple whose channel or states are absent, like the
        Pauli-forbidden ones, gets 0.0, as in NuHamil's ``GetThBME``.
    """
    n = len(orb)
    tri = triples(orb, e2max, e3max)
    i1, i2, i3, par = tri
    pending: list[tuple[np.ndarray, ...]] = []
    size = 0

    def flush() -> np.ndarray:
        cols = [np.concatenate(x) for x in zip(*pending, strict=True)]
        bra, ket, j12, j45, jt, t12, t45, t2, parity = cols
        out = np.zeros(len(bra))
        chan = (jt * 2 + (t2 > 1)) * 2 + (parity > 0)
        for cid in np.unique(chan):
            sel = np.flatnonzero(chan == cid)
            key = (int(jt[sel[0]]), int(parity[sel[0]]), int(t2[sel[0]]))
            lab = labs.get(key)
            if lab is None:
                continue
            kb = _key(i1[bra[sel]], i2[bra[sel]], i3[bra[sel]], j12[sel], t12[sel], n)
            kk = _key(i1[ket[sel]], i2[ket[sel]], i3[ket[sel]], j45[sel], t45[sel], n)
            rb, rk = np.searchsorted(lab.keys, kb), np.searchsorted(lab.keys, kk)
            ok = (rb < len(lab.keys)) & (rk < len(lab.keys))
            ok[ok] = (lab.keys[rb[ok]] == kb[ok]) & (lab.keys[rk[ok]] == kk[ok])
            out[sel[ok]] = lab.m[lab.rows[rb[ok]], lab.rows[rk[ok]]]
        return out

    for bra in range(len(i1)):
        ket, j12, j45, jt, t12, t45, t2 = bra_tuples(orb, tri, bra)
        pending.append(
            (np.full(len(ket), bra), ket, j12, j45, jt, t12, t45, t2, par[np.full(len(ket), bra)])
        )
        size += len(ket)
        if size >= chunk:
            yield flush()
            pending, size = [], 0
    if pending:
        yield flush()


def write_me3j(
    path: str | Path,
    chunks: Iterator[np.ndarray],
    compresslevel: int = 1,
    compressor: str = "auto",
    compress_threads: int = 1,
) -> int:
    """Write values as ``10f16.8`` text (gzip for ``.gz``) under the ``me3j`` header.

    This is :func:`nuham.nn.me2j.write_me2j`, whose layout is the same, with its own header line.
    """
    return write_me2j(path, chunks, HEADER, compresslevel, compressor, compress_threads)
