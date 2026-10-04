import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from reference import read_rel_cache

import nuham  # noqa: F401
from nuham.basis.relative import relative_states
from nuham.constants import oscillator_length_sq
from nuham.kernels.nn.projection import project_channel
from nuham.kernels.nn.relative import kinetic_matrix
from nuham.nn.input_file import read_nn_mom

HW = 16.0
N2MAX = 100


@pytest.fixture(scope="module")
def nn(nn_bin: Path):
    return read_nn_mom(nn_bin)


def test_file_layout(nn) -> None:
    assert len(nn.channels) == 54
    assert nn.k.shape == (100,)
    assert nn.w.sum() == pytest.approx(8.0, abs=1e-12)
    for ch in nn.channels:
        np.testing.assert_allclose(ch.v, ch.v.T, atol=1e-9 * np.abs(ch.v).max())


def test_matches_nuhamil_cache(nn, rel_cache_hw16: Path) -> None:
    cache = read_rel_cache(rel_cache_hw16)
    b = math.sqrt(oscillator_length_sq(HW))
    worst = 0.0
    for ch in nn.channels:
        key = (ch.j, ch.parity, ch.s, ch.tz)
        n, l = relative_states(ch.ls, N2MAX)
        v = np.asarray(project_channel(nn.k, nn.w, ch.v, ch.ls, N2MAX, b))
        index = {(int(a), int(c)): i for i, (a, c) in enumerate(zip(n, l, strict=True))}
        for (n1, l1, n2, l2), val in cache[key].items():
            worst = max(worst, abs(v[index[(n1, l1)], index[(n2, l2)]] - val))
        assert len(cache[key]) == len(n) * (len(n) + 1) // 2
    assert worst < 1e-5


def test_deuteron_energy(nn) -> None:
    ch = next(c for c in nn.channels if (c.j, c.parity, c.s, c.tz) == (1, 1, 1, 0))
    assert ch.ls == (0, 2)
    b = math.sqrt(oscillator_length_sq(HW))
    n, l = relative_states(ch.ls, N2MAX)
    v = project_channel(nn.k, nn.w, ch.v, ch.ls, N2MAX, b)
    h = kinetic_matrix(n, l, HW) + v
    assert float(jnp.linalg.eigvalsh(h)[0]) == pytest.approx(-2.2245754, abs=2e-4)


def test_kinetic_is_symmetric_tridiagonal() -> None:
    n, l = relative_states((0, 2), 20)
    t = np.asarray(kinetic_matrix(n, l, HW))
    np.testing.assert_array_equal(t, t.T)
    assert t[0, 0] == pytest.approx(HW * 0.5 * 1.5)
    assert t[0, 1] == pytest.approx(HW * 0.5 * math.sqrt(1 * 1.5))
    assert t[0, 2] == 0.0
