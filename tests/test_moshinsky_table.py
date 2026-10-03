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


@pytest.mark.parametrize("d", [1.0, 2.0])
def test_rotation_backend_matches_bracket_formula(d: float) -> None:
    fast = MoshinskyTable(d=d)
    slow = MoshinskyTable(d=d, method="bracket")
    for total in range(0, 10):
        for lam in range(total + 1):
            np.testing.assert_allclose(fast.matrix(total, lam), slow.matrix(total, lam), atol=1e-12)


def test_rotation_backend_is_orthogonal_at_high_n() -> None:
    table = MoshinskyTable()
    for total, lam in [(20, 0), (24, 7), (28, 14), (28, 28)]:
        m = table.matrix(total, lam)
        keep = np.flatnonzero(np.abs(m).sum(axis=1) > 0)
        sub = m[np.ix_(keep, np.flatnonzero(np.abs(m).sum(axis=0) > 0))]
        np.testing.assert_allclose(sub @ sub.T, np.eye(len(keep)), atol=1e-12)


def test_stretched_state_has_exact_parity_zero() -> None:
    # (x^2 - y^2)^6 has no odd powers, so this bracket vanishes exactly.
    table = MoshinskyTable()
    m = table.matrix(12, 12)
    assert abs(m[table.row_index(12)[(0, 6, 0, 6)], table.col_index(12)[(0, 7, 0, 5)]]) < 1e-13


def test_cache_name_includes_mass_ratio(tmp_path: Path) -> None:
    MoshinskyTable(tmp_path, d=2.0).matrix(4, 1)
    assert (tmp_path / "N4_lam1_d2.npy").exists()


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError):
        MoshinskyTable(method="nope")
