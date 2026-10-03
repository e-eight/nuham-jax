"""Optional checks against the slow explicit-sum reference in ``refs/`` (skipped if absent)."""

import numpy as np
import pytest
from sympy.physics.wigner import clebsch_gordan

from nuham.angmom.moshinsky_table import MoshinskyTable

ref = pytest.importorskip("moshinsky_reference")


@pytest.mark.parametrize("d", [1.0, 2.0])
def test_tables_match_explicit_sum(d: float) -> None:
    table = MoshinskyTable(d=d)
    for total in range(0, 10):
        rows = list(table.row_index(total))
        cols = list(table.col_index(total))
        for lam in range(total + 1):
            expected = np.zeros((len(rows), len(cols)))
            for i, (ncm, lcm, nr, lr) in enumerate(rows):
                for j, (n1, l1, n2, l2) in enumerate(cols):
                    if abs(lcm - lr) <= lam <= lcm + lr and abs(l1 - l2) <= lam <= l1 + l2:
                        expected[i, j] = ref.bracket(ncm, lcm, nr, lr, n1, l1, n2, l2, lam, d)
            np.testing.assert_allclose(table.matrix(total, lam), expected, atol=1e-12)


def test_cg_zero_matches_sympy() -> None:
    for la in range(7):
        for lb in range(7):
            for l in range(abs(la - lb), la + lb + 1):
                want = float(clebsch_gordan(la, lb, l, 0, 0, 0))
                assert ref.cg_zero(la, lb, l) == pytest.approx(want, abs=1e-14)
    assert ref.cg_zero(1, 1, 1) == 0.0
