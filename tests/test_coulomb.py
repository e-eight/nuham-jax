import math

import numpy as np
import pytest

import nuham  # noqa: F401
from nuham.constants import ALPHA_EM, HBARC, M_PROTON
from nuham.kernels.nn.coulomb import coulomb_matrix


def test_ground_state_expectation_is_analytic() -> None:
    hw = 16.0
    nu = math.sqrt(0.5 * M_PROTON * hw) / HBARC
    c = np.asarray(coulomb_matrix((0,), 10, hw))
    assert c[0, 0] == pytest.approx(HBARC / ALPHA_EM * 2.0 * nu / math.sqrt(math.pi), rel=1e-12)


def test_block_diagonal_in_l_and_symmetric() -> None:
    c = np.asarray(coulomb_matrix((0, 2), 12, 20.0))
    n_s = 7  # (12 - 0) // 2 + 1 states with l = 0
    np.testing.assert_allclose(c, c.T, atol=1e-12)
    assert np.all(c[:n_s, n_s:] == 0.0)
    assert c[0, 0] > 0.0
