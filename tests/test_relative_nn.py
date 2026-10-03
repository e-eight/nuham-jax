from pathlib import Path

import numpy as np
import pytest

import nuham  # noqa: F401
from nuham.interactions.nn_file import read_nn_mom
from nuham.kernels.relative import kinetic_matrix
from nuham.relative_nn import RelativeChannel, evolve_relative_nn

GOLDEN = np.load(Path(__file__).parent / "data" / "srg_golden.npz")
CASES = [(16.0, 2.0), (24.0, 1.8)]


@pytest.fixture(scope="module")
def nn(nn_bin: Path):
    return read_nn_mom(nn_bin)


@pytest.fixture(scope="module")
def evolved(nn) -> dict[tuple[float, float], tuple[RelativeChannel, ...]]:
    return {c: evolve_relative_nn(nn, *c) for c in CASES}


def key(hw: float, lam: float, ch: RelativeChannel) -> str:
    return f"hw{int(hw)}_lam{lam:.1f}_{ch.j}_{ch.parity}_{ch.s}_{ch.tz}"


@pytest.mark.parametrize(("hw", "lam"), CASES)
def test_matches_oracle(evolved: dict, hw: float, lam: float) -> None:
    seen = 0
    for ch in evolved[(hw, lam)]:
        k = key(hw, lam, ch)
        if k in GOLDEN:
            assert np.abs(ch.v - GOLDEN[k]).max() < 5e-8, k
            seen += 1
    assert seen == 8


def test_channel_set(evolved: dict) -> None:
    chans = evolved[CASES[0]]
    assert all(ch.j <= 8 for ch in chans)
    assert len(chans) == 54
    for ch in chans:
        np.testing.assert_allclose(ch.v, ch.v.T, atol=1e-8)


def test_spectrum_is_invariant_in_every_channel(nn, evolved: dict) -> None:
    hw, lam = CASES[0]
    bare = evolve_relative_nn(nn, hw, None)
    worst = 0.0
    for ev, b in zip(evolved[CASES[0]], bare, strict=True):
        t = np.asarray(kinetic_matrix(ev.n, ev.l, hw))
        lo = np.linalg.eigvalsh(t + b.v)
        hi = np.linalg.eigvalsh(t + ev.v)
        worst = max(worst, np.abs(lo - hi).max())
    assert worst < 1e-6


def test_deuteron_binding_energy_is_unchanged(nn, evolved: dict) -> None:
    hw, lam = CASES[0]
    ev = next(c for c in evolved[CASES[0]] if (c.j, c.parity, c.s, c.tz) == (1, 1, 1, 0))
    t = np.asarray(kinetic_matrix(ev.n, ev.l, hw))
    assert np.linalg.eigvalsh(t + ev.v)[0] == pytest.approx(-2.2245754, abs=1e-5)


def test_bare_pp_channel_includes_coulomb(nn) -> None:
    with_c = evolve_relative_nn(nn, 16.0, None, n2max=20, coulomb=True)
    without = evolve_relative_nn(nn, 16.0, None, n2max=20, coulomb=False)
    for a, b in zip(with_c, without, strict=True):
        if a.tz == -1:
            assert np.abs(a.v - b.v).max() > 0.1
        else:
            np.testing.assert_array_equal(a.v, b.v)
