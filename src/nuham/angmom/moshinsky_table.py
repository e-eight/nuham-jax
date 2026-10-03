"""Cached Moshinsky bracket matrices for fixed total quanta ``N`` and coupled momentum ``lam``."""

from pathlib import Path

import numpy as np

from nuham.angmom.moshinsky import bracket


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


class MoshinskyTable:
    """Lazily built, cached bracket matrices (equal masses).

    For fixed ``total`` and ``lam`` the matrix over ``cm_rel_states`` x ``particle_states`` is
    orthogonal once the entries that violate a triangle condition (which are zero) are included.
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """Create an empty table.

        Parameters
        ----------
        cache_dir
            Optional directory where finished matrices are stored as ``N<total>_lam<lam>.npy``
            and reused by later runs. The brackets do not depend on ħω, the SRG cutoff or the
            interaction, so one cache serves every run.
        """
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
        path = None if self._cache_dir is None else self._cache_dir / f"N{total}_lam{lam}.npy"
        if key not in self._matrices and path is not None and path.exists():
            self._matrices[key] = np.load(path)
        if key not in self._matrices:
            rows = list(self.row_index(total))
            cols = list(self.col_index(total))
            mat = np.zeros((len(rows), len(cols)))
            for i, (ncm, lcm, nr, lr) in enumerate(rows):
                if abs(lcm - lr) > lam or lam > lcm + lr:
                    continue
                for j, (n1, l1, n2, l2) in enumerate(cols):
                    if abs(l1 - l2) <= lam <= l1 + l2:
                        mat[i, j] = bracket(ncm, lcm, nr, lr, n1, l1, n2, l2, lam)
            self._matrices[key] = mat
            if path is not None:
                np.save(path, mat)
        return self._matrices[key]
