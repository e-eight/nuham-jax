"""Tests of the 3N Jacobi basis against the NuHamil reference run (emax 4, Nmax 8)."""

import numpy as np
import pytest
from cfp_reader import read_cfp
from conftest import REF_WORK

from nuham.angmom.moshinsky_table import MoshinskyTable
from nuham.nnn.jacobi import (
    antisymmetrizer,
    build_channel,
    jacobi_channels,
    nas_states,
    ramp_nmax,
)

CFP_DIR = REF_WORK / "nnn_e4" / "cfp"

# NAS (Pauli-forbidden-only blocks dropped) / AS at Nmax = 8 and at Nmax = 4, from spec-5a 2.4.
# Rows: (T, P); columns: 2J = 1, 3, ..., 11.
EXPECTED = {
    (1, 1): [
        (190, 64, 44, 15), (304, 101, 60, 20), (340, 113, 54, 18),
        (320, 107, 36, 12), (268, 89, 18, 6), (201, 67, 5, 2),
    ],
    (1, -1): [
        (140, 47, 26, 9), (220, 73, 34, 11), (236, 79, 26, 9),
        (212, 71, 14, 5), (166, 55, 4, 1), (114, 38, 0, 0),
    ],
    (3, 1): [
        (94, 31, 21, 7), (150, 49, 29, 9), (170, 57, 27, 9),
        (157, 51, 16, 5), (134, 45, 9, 3), (97, 32, 0, 0),
    ],
    (3, -1): [
        (70, 23, 13, 4), (110, 37, 17, 6), (117, 39, 12, 4),
        (106, 35, 7, 2), (83, 28, 2, 1), (57, 19, 0, 0),
    ],
}  # fmt: skip


@pytest.fixture(scope="module")
def table() -> MoshinskyTable:
    """Shared in-memory ``d = 1/3`` bracket table."""
    return MoshinskyTable(d=1 / 3)


def test_ramp_and_channel_order() -> None:
    """Flat and ramp truncations, and T-outermost, J, then P = +1 before -1."""
    assert ramp_nmax(5, "flat8") == 8
    assert [ramp_nmax(j, "ramp4-3-6-7-8") for j in (1, 3, 5, 7, 9)] == [4, 4, 6, 6, 8]
    channels = jacobi_channels(11, "flat8")
    assert len(channels) == 24
    assert channels[:3] == [(1, 1, 1, 8), (1, -1, 1, 8), (3, 1, 1, 8)]
    assert channels[12] == (1, 1, 3, 8)


def test_nas_order_first_states() -> None:
    """The first NAS states of J=1/2+ T=1/2 follow NuHamil's loop order."""
    n0 = nas_states(1, 1, 1, 0)
    assert n0.tolist() == [[0, 0, 0, 0, 1, 0, 0, 1], [0, 0, 1, 1, 0, 0, 0, 1]]
    assert len(nas_states(1, 1, 1, 1)) == 0  # wrong parity


@pytest.mark.parametrize(("t2", "parity"), sorted(EXPECTED))
def test_state_counts(t2: int, parity: int, table: MoshinskyTable) -> None:
    """NAS and AS counts of all 24 channels at Nmax = 8 and Nmax = 4 equal the NuHamil log."""
    for k, (nas8, as8, nas4, as4) in enumerate(EXPECTED[(t2, parity)]):
        j2 = 2 * k + 1
        full = build_channel(j2, parity, t2, 8, table)
        assert (full.n_nas, full.n_as) == (nas8, as8)
        low = build_channel(j2, parity, t2, 4, table)
        assert (low.n_nas, low.n_as) == (nas4, as4)
        assert [b.n_quanta for b in low.blocks] == [b.n_quanta for b in full.blocks][
            : len(low.blocks)
        ]


def test_projector_and_multiplicities(table: MoshinskyTable) -> None:
    """A is a symmetric projector, X = (1 - 3A)/2 has spectrum {-1, 1/2}, zeros are mixed."""
    for n_quanta, nas_dim, as_dim in [(2, 12, 4), (4, 30, 10), (8, 90, 30)]:
        nas = nas_states(1, 1, 1, n_quanta)
        a = antisymmetrizer(nas, 1, 1, n_quanta, table)
        assert len(nas) == nas_dim
        assert abs(a - a.T).max() < 1e-13
        assert abs(a @ a - a).max() < 1e-13
        assert abs(np.trace(a) - as_dim) < 1e-12
        values = np.linalg.eigvalsh(a)
        assert np.minimum(abs(values), abs(1 - values)).max() < 1e-12
        x = (np.eye(nas_dim) - 3 * a) / 2
        assert abs((x + np.eye(nas_dim)) @ (x - 0.5 * np.eye(nas_dim))).max() < 1e-12


def test_basis_is_deterministic_isometry(table: MoshinskyTable) -> None:
    """Rows are orthonormal, C^T C = A, and rebuilding gives the identical matrix."""
    ch = build_channel(3, 1, 1, 6, table)
    again = build_channel(3, 1, 1, 6, table)
    for b, b2 in zip(ch.blocks, again.blocks, strict=True):
        a = antisymmetrizer(b.nas, 3, 1, b.n_quanta, table)
        assert abs(b.cfp @ b.cfp.T - np.eye(len(b.cfp))).max() < 1e-13
        assert abs(b.cfp.T @ b.cfp - a).max() < 1e-13
        assert np.array_equal(b.cfp, b2.cfp)


def test_pauli_forbidden_block_dropped(table: MoshinskyTable) -> None:
    """J=3/2+ T=1/2: the N=0 block has one NAS state, no AS state, and is dropped (305 -> 304)."""
    assert len(nas_states(3, 1, 1, 0)) == 1
    ch = build_channel(3, 1, 1, 8, table)
    assert ch.blocks[0].n_quanta == 2
    assert ch.n_nas == 304


@pytest.mark.parametrize("name", ["j1p+t1", "j3p+t1", "j5p-t1", "j7p+t3", "j11p-t1"])
def test_matches_nuhamil_cfp(name: str, table: MoshinskyTable) -> None:
    """Our NAS lists equal NuHamil's, and our projector equals NuHamil's C^T C to 1e-13."""
    path = CFP_DIR / f"cfp_A3_{name}_Nmax8.bin"
    if not path.exists():
        pytest.skip("refs/work/nnn_e4 not present")
    (j2, parity, t2, _), blocks = read_cfp(path)
    ours = build_channel(j2, parity, t2, 8, table)
    assert [b.n_quanta for b in ours.blocks] == [n for n, _, _ in blocks]
    for mine, (n_quanta, nas, cfp) in zip(ours.blocks, blocks, strict=True):
        assert np.array_equal(mine.nas, nas)
        a = antisymmetrizer(nas, j2, t2, n_quanta, table)
        assert abs(cfp.T @ cfp - a).max() < 1e-13
        assert abs(mine.cfp.T @ mine.cfp - cfp.T @ cfp).max() < 1e-13


def test_cached_channel_equals_built_channel(tmp_path) -> None:
    """A basis read back from the cache is the one that was built, bit for bit."""
    from nuham.nnn.jacobi import load_or_build_channel

    built = load_or_build_channel(3, 1, 1, 6, cache_dir=tmp_path)
    assert any(tmp_path.glob("jacobi_j3_p+_t1_N6.npz"))
    read = load_or_build_channel(3, 1, 1, 6, cache_dir=tmp_path)
    assert [b.n_quanta for b in read.blocks] == [b.n_quanta for b in built.blocks]
    for a, b in zip(read.blocks, built.blocks, strict=True):
        np.testing.assert_array_equal(a.nas, b.nas)
        np.testing.assert_array_equal(a.cfp, b.cfp)
