"""The Darmstadt ``me2j`` format: tuple enumeration, assembly of the four isospin values, I/O."""

import gzip
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from nuham.basis.lab import Orbits
from nuham.lab_transform import LabMatrix

HEADER = "NN int. calculated by nuham (JAX)"


def unordered_pairs(orb: Orbits, e2max: int) -> np.ndarray:
    """Pairs ``(a, b)`` with ``b <= a`` and ``e_a + e_b <= e2max``, in file order, shape (n, 2)."""
    out = [(a, b) for a in range(len(orb)) for b in range(a + 1) if orb.e[a] + orb.e[b] <= e2max]
    return np.array(out, dtype=np.int64)


def tuple_chunks(
    orb: Orbits, e2max: int
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]]:
    """Enumerate the ``(a, b, c, d, J)`` tuples of the file, one chunk per ``(a, b)``.

    The tuples satisfy ``(c, d) <= (a, b)`` in file order, equal parity of ``l_a + l_b`` and
    ``l_c + l_d``, and ``J`` runs over all couplings of both pairs.

    Yields
    ------
    tuple[int, int, np.ndarray, np.ndarray, np.ndarray]
        ``a``, ``b`` and arrays ``c``, ``d``, ``J`` of equal length, in file order.
    """
    pairs = unordered_pairs(orb, e2max)
    lsum = orb.l[pairs[:, 0]] + orb.l[pairs[:, 1]]
    jlo = np.abs(orb.j2[pairs[:, 0]] - orb.j2[pairs[:, 1]])
    jhi = orb.j2[pairs[:, 0]] + orb.j2[pairs[:, 1]]
    for u, (a, b) in enumerate(pairs):
        v = np.flatnonzero((lsum[: u + 1] - lsum[u]) % 2 == 0)
        j_min = np.maximum(jlo[v], jlo[u]) // 2
        j_max = np.minimum(jhi[v], jhi[u]) // 2
        count = np.maximum(j_max - j_min + 1, 0)
        which = np.repeat(np.arange(len(v)), count)
        offset = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
        yield int(a), int(b), pairs[v][which, 0], pairs[v][which, 1], j_min[which] + offset


class LabChannel:
    """Lab matrices of one ``(J, parity)`` with lookup tables from an orbit pair to its index."""

    def __init__(self, matrices: dict[int, LabMatrix], n_orbits: int) -> None:
        """Index the pair bases of the pp (``-1``), pn (``0``) and nn (``+1``) matrices."""
        self.matrices = matrices
        self.position: dict[int, np.ndarray] = {}
        for tz, m in matrices.items():
            pos = np.full((n_orbits, n_orbits), -1, dtype=np.int64)
            pos[m.pair_a, m.pair_b] = np.arange(len(m.pair_a))
            self.position[tz] = pos

    def element(self, tz: int, a: int, b: int, c: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Matrix elements ``<ab|V|cd>`` for arrays ``c, d``; zero outside the pair basis."""
        pos = self.position[tz]
        i = pos[a, b]
        k = pos[c, d]
        out = np.zeros(len(k))
        if i < 0:
            return out
        ok = k >= 0
        out[ok] = self.matrices[tz].v[i, k[ok]]
        return out


def _phase(orb: Orbits, x: int | np.ndarray, y: int | np.ndarray, j: int) -> np.ndarray | float:
    """Exchange phase ``-(-1)^(j_x + j_y - J)`` of the pn lab states."""
    return -((-1.0) ** ((orb.j2[x] + orb.j2[y]) // 2 - j))


def assemble(
    orb: Orbits,
    e2max: int,
    channels: dict[tuple[int, int], LabChannel],
) -> Iterator[np.ndarray]:
    """Yield the four ``me2j`` values per tuple, chunk by chunk, shape ``(n, 4)``.

    The columns are ``T = 0``, nn, pn ``T = 1`` and pp, in the un-normalised Darmstadt convention.
    """
    for a, b, c, d, jj in tuple_chunks(orb, e2max):
        vals = np.zeros((len(c), 4))
        par = (-1) ** (orb.l[a] + orb.l[b])
        for j in np.unique(jj):
            sel = np.flatnonzero(jj == j)
            ch = channels[(int(j), par)]
            cs, ds = c[sel], d[sel]
            nf = (math.sqrt(2.0) if a == b else 1.0) * np.where(cs == ds, math.sqrt(2.0), 1.0)
            ph_ab = _phase(orb, a, b, int(j))
            ph_cd = _phase(orb, cs, ds, int(j))
            x1 = ch.element(0, a, b, cs, ds)
            x2 = ch.element(0, b, a, ds, cs) * ph_ab * ph_cd
            x3 = ch.element(0, a, b, ds, cs) * ph_cd
            x4 = ch.element(0, b, a, cs, ds) * ph_ab
            vals[sel, 0] = 0.5 * (x1 + x2 - x3 - x4)
            vals[sel, 1] = ch.element(1, a, b, cs, ds) * nf
            vals[sel, 2] = 0.5 * (x1 + x2 + x3 + x4)
            vals[sel, 3] = ch.element(-1, a, b, cs, ds) * nf
        yield vals


def write_me2j(path: str | Path, chunks: Iterator[np.ndarray], header: str = HEADER) -> int:
    """Write values as gzip text: one header line, then Fortran ``10f16.8`` (10 values per line).

    Parameters
    ----------
    path
        Output file; a ``.gz`` suffix gives gzip, anything else plain text.
    chunks
        Arrays of values, flattened in order.
    header
        Free-text first line.

    Returns
    -------
    int
        The number of values written.
    """
    total = 0
    pending = np.empty(0)
    fh = gzip.open(path, "wt", compresslevel=4) if str(path).endswith(".gz") else open(path, "w")
    with fh:
        fh.write(header + "\n")
        for chunk in chunks:
            pending = np.concatenate([pending, np.asarray(chunk).ravel()])
            full = len(pending) // 10 * 10
            for line in pending[:full].reshape(-1, 10):
                fh.write("".join(f"{x:16.8f}" for x in line) + "\n")
            total += full
            pending = pending[full:]
        if len(pending):
            fh.write("".join(f"{x:16.8f}" for x in pending) + "\n")
            total += len(pending)
    return total


def read_me2j(path: str | Path) -> np.ndarray:
    """Read all values of an ``me2j`` file (gzip or plain), skipping the header line."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        fh.readline()
        return np.array(fh.read().split(), dtype=np.float64)
