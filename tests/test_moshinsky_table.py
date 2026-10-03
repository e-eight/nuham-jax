from pathlib import Path

import numpy as np
import pytest

from nuham.angmom.moshinsky_table import MoshinskyTable


@pytest.mark.parametrize("total", [3, 6, 8])
def test_matrices_are_orthogonal(total: int) -> None:
    table = MoshinskyTable()
    for lam in range(total + 1):
        m = table.matrix(total, lam)
        assert m.shape[0] == m.shape[1]
        keep = np.flatnonzero(np.abs(m).sum(axis=1) > 0)
        sub = m[np.ix_(keep, np.flatnonzero(np.abs(m).sum(axis=0) > 0))]
        np.testing.assert_allclose(sub @ sub.T, np.eye(len(keep)), atol=1e-11)


def test_disk_cache_round_trip(tmp_path: Path) -> None:
    first = MoshinskyTable(tmp_path).matrix(5, 2)
    assert (tmp_path / "N5_lam2.npy").exists()
    second = MoshinskyTable(tmp_path).matrix(5, 2)
    np.testing.assert_array_equal(first, second)
