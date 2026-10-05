"""Cached Moshinsky bracket matrices for fixed total quanta ``N`` and coupled momentum ``lam``.

The transformation is a rotation of the two-oscillator Fock space by ``theta = atan(sqrt(d))``
(45 degrees for equal masses). At fixed ``N`` and ``lam`` the bracket matrix is
``(-1)^(N + lam) exp(theta G) P`` with a sparse antisymmetric generator ``G`` built from
oscillator ladder-operator matrix elements and ``P`` the permutation exchanging the
centre-of-mass and relative labels. This agrees with the explicit Kamuntavicius sum, kept as a
slow reference in ``refs/moshinsky_reference.py``, to about 1e-13 and is far faster.
"""

import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from nuham.angmom.racah import sixj
from nuham.kernels.rotation import expm_antisymmetric


def cm_rel_states(total: int) -> list[tuple[int, int, int, int]]:
    """States ``(n_cm, l_cm, n_rel, l_rel)`` with ``2 n_cm + l_cm + 2 n_rel + l_rel = total``."""
    out = []
    for e_cm in range(total + 1):
        e_rel = total - e_cm
        for n_cm in range(e_cm // 2 + 1):
            for n_rel in range(e_rel // 2 + 1):
                out.append((n_cm, e_cm - 2 * n_cm, n_rel, e_rel - 2 * n_rel))
    return out


def particle_states(total: int) -> list[tuple[int, int, int, int]]:
    """Orbital pairs ``(n1, l1, n2, l2)`` with ``2 n1 + l1 + 2 n2 + l2 = total``."""
    out = []
    for e1 in range(total + 1):
        e2 = total - e1
        for n1 in range(e1 // 2 + 1):
            for n2 in range(e2 // 2 + 1):
                out.append((n1, e1 - 2 * n1, n2, e2 - 2 * n2))
    return out


_PAD = 32
"""Generators are zero-padded to a multiple of this size so ``jit`` compiles few shapes."""


def _generator(states: list[tuple[int, int, int, int]], lam: int) -> np.ndarray:
    """Rotation generator ``G = -2 (S - S^T)`` with ``S = a1^dagger . a2`` in the coupled basis.

    Parameters
    ----------
    states
        Orbital pairs ``(n1, l1, n2, l2)`` of one total ``N`` that couple to ``lam``.
    lam
        Total orbital angular momentum.

    Returns
    -------
    np.ndarray
        Real antisymmetric matrix over ``states``.
    """
    index = {s: i for i, s in enumerate(states)}
    s_op = np.zeros((len(states), len(states)))
    for n1, l1, n2, l2 in states:
        col = index[(n1, l1, n2, l2)]
        raise_1 = [((n1, l1 + 1), math.sqrt((l1 + 1) * (n1 + l1 + 1.5)))]
        if l1 > 0:
            raise_1.append(((n1 + 1, l1 - 1), math.sqrt(l1 * (n1 + 1))))
        lower_2 = []
        if n2 > 0:
            lower_2.append(((n2 - 1, l2 + 1), math.sqrt((l2 + 1) * n2)))
        if l2 > 0:
            lower_2.append(((n2, l2 - 1), math.sqrt(l2 * (n2 + l2 + 0.5))))
        for (n1p, l1p), m1 in raise_1:
            for (n2p, l2p), m2 in lower_2:
                row = index.get((n1p, l1p, n2p, l2p))
                if row is not None:
                    phase = (-1.0) ** (l1 + l2p + lam)
                    six = sixj(2 * l1p, 2 * l2p, 2 * lam, 2 * l2, 2 * l1, 2)
                    s_op[row, col] += phase * six * m1 * m2
    return -2.0 * (s_op - s_op.T)


class MoshinskyTable:
    """Lazily built, cached bracket matrices (equal masses unless ``d`` is given).

    For fixed ``total`` and ``lam`` the bracket matrix over the states that satisfy both triangle
    conditions is square and orthogonal; every other row and column is exactly zero. Only that
    compact block is kept, in memory and on disk (about a tenth of the zero-padded size);
    :meth:`gather` and :meth:`matrix` rebuild the zero-padded view on demand.
    """

    def __init__(self, cache_dir: str | Path | None = None, d: float = 1.0) -> None:
        """Create an empty table.

        Parameters
        ----------
        cache_dir
            Optional directory where finished blocks are stored as ``N<total>_lam<lam>_c.npy``
            (with a ``_d<d>`` suffix for ``d != 1``) and reused by later runs. The brackets do
            not depend on ħω, the SRG cutoff or the interaction, so one cache serves every run.
        d
            Mass ratio of the two particles; ``1`` for two nucleons.
        """
        self._d = d
        self._cache_dir = None if cache_dir is None else Path(cache_dir)
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._blocks: dict[tuple[int, int], np.ndarray] = {}
        self._positions: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._rows: dict[int, dict[tuple[int, int, int, int], int]] = {}
        self._cols: dict[int, dict[tuple[int, int, int, int], int]] = {}

    def warm(self, max_total: int) -> None:
        """Build (or load) every block with ``total <= max_total``.

        Blocks are computed with JAX, which must not run in a forked worker process, so a driver
        that forks workers that read the table calls this first.
        """
        for total in range(max_total + 1):
            for lam in range(total + 1):
                self._block(total, lam)

    def row_index(self, total: int) -> dict[tuple[int, int, int, int], int]:
        """Map from ``(n_cm, l_cm, n_rel, l_rel)`` to its row for the given total."""
        if total not in self._rows:
            self._rows[total] = {s: i for i, s in enumerate(cm_rel_states(total))}
        return self._rows[total]

    def col_index(self, total: int) -> dict[tuple[int, int, int, int], int]:
        """Map from ``(n1, l1, n2, l2)`` to its column for the given total."""
        if total not in self._cols:
            self._cols[total] = {s: i for i, s in enumerate(particle_states(total))}
        return self._cols[total]

    def _valid_states(self, total: int, lam: int) -> list[tuple[int, int, int, int]]:
        """States of this total whose two orbital momenta couple to ``lam``."""
        return [s for s in self.col_index(total) if abs(s[1] - s[3]) <= lam <= s[1] + s[3]]

    def _block(self, total: int, lam: int) -> np.ndarray:
        """Compact square block over :meth:`_valid_states`, from memory, disk or computation."""
        key = (total, lam)
        if key in self._blocks:
            return self._blocks[key]
        suffix = "" if self._d == 1.0 else f"_d{self._d:g}"
        path = (
            None
            if self._cache_dir is None
            else self._cache_dir / f"N{total}_lam{lam}_c{suffix}.npy"
        )
        if path is not None and path.exists():
            block = np.load(path)
        else:
            block = self._rotation_block(self._valid_states(total, lam), total, lam)
            if path is not None:
                np.save(path, block)
        self._blocks[key] = block
        return block

    def _compact_positions(self, total: int, lam: int) -> tuple[np.ndarray, np.ndarray]:
        """Compact index of every row and column (``-1`` where the entries are zero)."""
        key = (total, lam)
        if key not in self._positions:
            local = {s: i for i, s in enumerate(self._valid_states(total, lam))}
            rows = np.array([local.get(s, -1) for s in self.row_index(total)], dtype=np.int64)
            cols = np.array([local.get(s, -1) for s in self.col_index(total)], dtype=np.int64)
            self._positions[key] = (rows, cols)
        return self._positions[key]

    def gather(self, total: int, lam: int, row_idx: np.ndarray, col_idx: np.ndarray) -> np.ndarray:
        """Bracket entries ``[row_idx, col_idx]`` of the zero-padded matrix, shape ``(r, c)``.

        Parameters
        ----------
        total, lam
            Total quanta and coupled orbital momentum.
        row_idx, col_idx
            Row and column positions in the orders of :meth:`row_index` and :meth:`col_index`.

        Returns
        -------
        np.ndarray
            ``matrix(total, lam)[np.ix_(row_idx, col_idx)]`` without building the full matrix.
        """
        block = self._block(total, lam)
        rows_all, cols_all = self._compact_positions(total, lam)
        r = rows_all[row_idx]
        c = cols_all[col_idx]
        out = np.zeros((len(r), len(c)))
        keep_r, keep_c = r >= 0, c >= 0
        if keep_r.any() and keep_c.any():
            out[np.ix_(keep_r, keep_c)] = block[np.ix_(r[keep_r], c[keep_c])]
        return out

    def matrix(self, total: int, lam: int) -> np.ndarray:
        """Zero-padded bracket matrix ``<cm rel; lam | 1 2; lam>``, shape ``(n_rows, n_cols)``.

        Built on demand and not cached; use :meth:`gather` in hot paths.
        """
        rows_all, cols_all = self._compact_positions(total, lam)
        return self.gather(total, lam, np.arange(len(rows_all)), np.arange(len(cols_all)))

    def _rotation_block(
        self, states: list[tuple[int, int, int, int]], total: int, lam: int
    ) -> np.ndarray:
        """Compute ``(-1)^(N + lam) exp(theta G) P`` over ``states`` (rows and columns alike)."""
        if not states:
            return np.zeros((0, 0))
        theta = math.atan(math.sqrt(self._d))
        gen = theta * _generator(states, lam)
        size = -(-len(states) // _PAD) * _PAD
        padded = np.zeros((size, size))
        padded[: len(states), : len(states)] = gen
        rot = np.asarray(expm_antisymmetric(jnp.asarray(padded)))[: len(states), : len(states)]
        local = {s: i for i, s in enumerate(states)}
        swap = np.array([local[(s[2], s[3], s[0], s[1])] for s in states])
        return (-1.0) ** (total + lam) * rot[:, np.argsort(swap)]
