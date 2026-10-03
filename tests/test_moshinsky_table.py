from pathlib import Path

import numpy as np
import pytest

from nuham.angmom.moshinsky_table import MoshinskyTable


@pytest.mark.parametrize("total", [3, 6, 8])
@pytest.mark.parametrize("d", [1.0, 2.0])
def test_matrices_are_orthogonal(total: int, d: float) -> None:
    table = MoshinskyTable(d=d)
    for lam in range(total + 1):
        m = table.matrix(total, lam)
        assert m.shape[0] == m.shape[1]
        keep = np.flatnonzero(np.abs(m).sum(axis=1) > 0)
        sub = m[np.ix_(keep, np.flatnonzero(np.abs(m).sum(axis=0) > 0))]
        np.testing.assert_allclose(sub @ sub.T, np.eye(len(keep)), atol=1e-11)


def test_disk_cache_round_trip(tmp_path: Path) -> None:
    first = MoshinskyTable(tmp_path).matrix(5, 2)
    assert (tmp_path / "N5_lam2_c.npy").exists()
    second = MoshinskyTable(tmp_path).matrix(5, 2)
    np.testing.assert_array_equal(first, second)


def test_tables_are_orthogonal_at_high_n() -> None:
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
    assert (tmp_path / "N4_lam1_c_d2.npy").exists()


def test_gather_matches_the_zero_padded_matrix() -> None:
    table = MoshinskyTable()
    rng = np.random.default_rng(0)
    for total, lam in [(0, 0), (5, 2), (8, 3), (10, 10), (12, 1)]:
        full = table.matrix(total, lam)
        rows = rng.integers(0, full.shape[0], size=7)
        cols = rng.integers(0, full.shape[1], size=9)
        np.testing.assert_array_equal(
            table.gather(total, lam, rows, cols), full[np.ix_(rows, cols)]
        )


def test_invalid_rows_and_columns_are_exactly_zero() -> None:
    # (n_cm, l_cm, n_rel, l_rel) = (0, 0, 0, 3) cannot couple to lam = 0.
    table = MoshinskyTable()
    row = table.row_index(3)[(0, 0, 0, 3)]
    assert not table.matrix(3, 0)[row].any()
    assert table.gather(3, 0, np.array([row]), np.arange(table.matrix(3, 0).shape[1])).sum() == 0.0


def test_cache_stores_only_the_valid_block(tmp_path: Path) -> None:
    table = MoshinskyTable(tmp_path)
    full = table.matrix(12, 5)
    stored = np.load(tmp_path / "N12_lam5_c.npy")
    assert stored.shape[0] == stored.shape[1] < min(full.shape)
    np.testing.assert_allclose(stored @ stored.T, np.eye(len(stored)), atol=1e-12)
    np.testing.assert_array_equal(MoshinskyTable(tmp_path).matrix(12, 5), full)


def test_empty_block_when_no_state_couples() -> None:
    table = MoshinskyTable()
    assert table.matrix(2, 9).shape[0] > 0 and not table.matrix(2, 9).any()
