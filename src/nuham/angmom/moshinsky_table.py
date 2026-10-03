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
    """Lazily built, cached bracket matrices (equal masses).

    For fixed ``total`` and ``lam`` the matrix over ``cm_rel_states`` x ``particle_states`` is
    orthogonal once the entries that violate a triangle condition (which are zero) are included.
    """

    def __init__(self, cache_dir: str | Path | None = None, d: float = 1.0) -> None:
        """Create an empty table.

        Parameters
        ----------
        cache_dir
            Optional directory where finished matrices are stored as ``N<total>_lam<lam>.npy``
            (with a ``_d<d>`` suffix for ``d != 1``) and reused by later runs. The brackets do
            not depend on ħω, the SRG cutoff or the interaction, so one cache serves every run.
        d
            Mass ratio of the two particles; ``1`` for two nucleons.
        """
        self._d = d
        self._cache_dir = None if cache_dir is None else Path(cache_dir)
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._matrices: dict[tuple[int, int], np.ndarray] = {}
        self._rows: dict[int, dict[tuple[int, int, int, int], int]] = {}
        self._cols: dict[int, dict[tuple[int, int, int, int], int]] = {}

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

    def matrix(self, total: int, lam: int) -> np.ndarray:
        """Bracket matrix ``<cm rel; lam | 1 2; lam>`` of shape ``(n_rows, n_cols)``."""
        key = (total, lam)
        suffix = "" if self._d == 1.0 else f"_d{self._d:g}"
        path = (
            None if self._cache_dir is None else self._cache_dir / f"N{total}_lam{lam}{suffix}.npy"
        )
        if key not in self._matrices and path is not None and path.exists():
            self._matrices[key] = np.load(path)
        if key not in self._matrices:
            self._matrices[key] = self._rotation_matrix(total, lam)
            if path is not None:
                np.save(path, self._matrices[key])
        return self._matrices[key]

    def _rotation_matrix(self, total: int, lam: int) -> np.ndarray:
        """Build the matrix as ``(-1)^(N + lam) exp(theta G) P`` and scatter it to full size."""
        rows = self.row_index(total)
        cols = self.col_index(total)
        mat = np.zeros((len(rows), len(cols)))
        states = [s for s in cols if abs(s[1] - s[3]) <= lam <= s[1] + s[3]]
        if not states:
            return mat
        theta = math.atan(math.sqrt(self._d))
        gen = theta * _generator(states, lam)
        size = -(-len(states) // _PAD) * _PAD
        padded = np.zeros((size, size))
        padded[: len(states), : len(states)] = gen
        rot = np.asarray(expm_antisymmetric(jnp.asarray(padded)))[: len(states), : len(states)]
        local = {s: i for i, s in enumerate(states)}
        swap = np.array([local[(s[2], s[3], s[0], s[1])] for s in states])
        sub = (-1.0) ** (total + lam) * rot[:, np.argsort(swap)]
        r_idx = np.array([rows[s] for s in states])
        c_idx = np.array([cols[s] for s in states])
        mat[np.ix_(r_idx, c_idx)] = sub
        return mat
