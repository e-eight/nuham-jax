import json
from pathlib import Path

import numpy as np
import pytest

from nuham.angmom.moshinsky_table import MoshinskyTable

# Brackets generated once with the explicit Kamuntavicius sum (refs/oracle_numpy.py); they fix
# the phase convention that the rotation tables must reproduce.
GOLDEN = json.loads((Path(__file__).parent / "data" / "moshinsky_golden.json").read_text())
TABLES = {1.0: MoshinskyTable(), 2.0: MoshinskyTable(d=2.0)}


@pytest.mark.parametrize("case", GOLDEN)
def test_table_matches_golden_bracket(case: dict) -> None:
    n_cm, l_cm, n_rel, l_rel, n1, l1, n2, l2, lam = case["args"]
    total = 2 * n_cm + l_cm + 2 * n_rel + l_rel
    table = TABLES[case["d"]]
    value = table.matrix(total, lam)[
        table.row_index(total)[(n_cm, l_cm, n_rel, l_rel)], table.col_index(total)[(n1, l1, n2, l2)]
    ]
    assert value == pytest.approx(case["value"], abs=1e-12)


def test_vacuum_bracket_is_one() -> None:
    np.testing.assert_allclose(MoshinskyTable().matrix(0, 0), [[1.0]])
