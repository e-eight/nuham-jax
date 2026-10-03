import itertools
import json
from pathlib import Path

import pytest

from nuham.angmom.moshinsky import bracket

GOLDEN = json.loads((Path(__file__).parent / "data" / "moshinsky_golden.json").read_text())


@pytest.mark.parametrize("case", GOLDEN)
def test_matches_oracle(case: dict) -> None:
    assert bracket(*case["args"], case["d"]) == pytest.approx(case["value"], abs=1e-12)


def _pairs(e_total: int, lam: int) -> list[tuple[int, int, int, int]]:
    """All ``(n1, l1, n2, l2)`` with ``2n1+l1 + 2n2+l2 = e_total`` coupled to ``lam``."""
    out = []
    for e1 in range(e_total + 1):
        for n1, n2 in itertools.product(range(e1 // 2 + 1), range(((e_total - e1) // 2) + 1)):
            l1, l2 = e1 - 2 * n1, e_total - e1 - 2 * n2
            if abs(l1 - l2) <= lam <= l1 + l2:
                out.append((n1, l1, n2, l2))
    return out


@pytest.mark.parametrize(("e_total", "lam"), [(4, 0), (6, 2), (7, 3), (8, 1)])
@pytest.mark.parametrize("d", [1.0, 2.0])
def test_bracket_matrix_is_orthogonal(e_total: int, lam: int, d: float) -> None:
    cm_rel = []
    for e_cm in range(e_total + 1):
        for n_cm in range(e_cm // 2 + 1):
            l_cm = e_cm - 2 * n_cm
            for n_r in range(((e_total - e_cm) // 2) + 1):
                l_r = e_total - e_cm - 2 * n_r
                if abs(l_cm - l_r) <= lam <= l_cm + l_r:
                    cm_rel.append((n_cm, l_cm, n_r, l_r))
    parts = _pairs(e_total, lam)
    assert len(cm_rel) == len(parts)
    rows = [[bracket(*cr, *p, lam, d) for p in parts] for cr in cm_rel]
    for i, ri in enumerate(rows):
        for j, rj in enumerate(rows):
            dot = sum(x * y for x, y in zip(ri, rj, strict=True))
            assert dot == pytest.approx(1.0 if i == j else 0.0, abs=1e-11)


def test_trivial_bracket() -> None:
    assert bracket(0, 0, 0, 0, 0, 0, 0, 0, 0) == pytest.approx(1.0)
