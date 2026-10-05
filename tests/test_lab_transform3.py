"""Device contraction of the 3N lab matrix against the host reference (Phase 5.4)."""

import numpy as np
import pytest

import nuham  # noqa: F401  (enables float64)
from nuham.basis.lab import make_orbits
from nuham.nnn.jacobi import JacobiChannel, build_channel
from nuham.nnn.lab_transform import lab_scalar_device
from nuham.nnn.packed import unpack_lower
from nuham.nnn.tcoef import lab_scalar, make_brackets

EMAX = 6


@pytest.fixture(scope="module")
def setup():
    """Random symmetric Jacobi operators on the real AS bases of ``T = 1/2`` and ``3/2``."""
    rng = np.random.default_rng(7)
    chans, vjac = {}, {}
    for t2 in (1, 3):
        for j2 in range(1, 2 * EMAX + 4, 2):
            for p in (1, -1):
                ch = build_channel(j2, p, t2, EMAX)
                if not ch.blocks:
                    continue
                kept = JacobiChannel(j2, p, t2, EMAX, ch.blocks)
                v = rng.normal(size=(kept.n_as, kept.n_as))
                chans[t2, j2, p] = kept
                vjac[t2, j2, p] = (v + v.T) / 2
    return make_orbits(EMAX), make_brackets(), chans, vjac


@pytest.mark.parametrize(
    ("j2", "parity", "t2"), [(1, 1, 1), (3, -1, 1), (5, 1, 3), (7, -1, 1), (3, 1, 3)]
)
def test_device_equals_host(setup, j2: int, parity: int, t2: int) -> None:
    orb, brackets, chans, vjac = setup
    v = {(j, p): m for (t, j, p), m in vjac.items() if t == t2}
    c = {(j, p): m for (t, j, p), m in chans.items() if t == t2}
    lab, want = lab_scalar(j2, parity, t2, orb, 2 * EMAX, EMAX, v, c, brackets)
    lab_d, got, order = lab_scalar_device(
        j2, parity, t2, orb, 2 * EMAX, EMAX, v, c, brackets, granularity=32
    )
    assert len(lab_d) == len(lab) == len(order)
    got = unpack_lower(got, len(order))
    full = np.zeros_like(want)
    full[np.ix_(order, order)] = got
    assert np.abs(full - want).max() < 1e-9 * max(1.0, np.abs(want).max())
    np.testing.assert_allclose(full, full.T, atol=1e-12)
    single = lab_scalar_device(j2, parity, t2, orb, 2 * EMAX, EMAX, v, c, brackets, np.float32)[1]
    assert single.dtype == np.float32
    assert len(single) == len(order) * (len(order) + 1) // 2
    assert np.abs(unpack_lower(single, len(order)) - got).max() < 1e-5 * max(1.0, np.abs(got).max())


def test_packed_storage_round_trip_and_lookup() -> None:
    from nuham.nnn.packed import pack_lower, packed_index, packed_size

    rng = np.random.default_rng(0)
    m = rng.normal(size=(9, 9))
    m = m + m.T
    packed = pack_lower(m)
    assert len(packed) == packed_size(9)
    np.testing.assert_array_equal(unpack_lower(packed, 9), m)
    i, j = np.meshgrid(np.arange(9), np.arange(9), indexing="ij")
    np.testing.assert_array_equal(packed[packed_index(i, j)], m)
