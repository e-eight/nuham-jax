import numpy as np

from nuham.basis.lab import lab_channels, make_orbits, ordered_pairs
from nuham.nn.me2j import tuple_chunks


def test_orbit_order_emax4() -> None:
    orb = make_orbits(4)
    names = [(int(e), int(l), int(j)) for e, l, j in zip(orb.e, orb.l, orb.j2, strict=True)]
    assert len(orb) == 15
    assert names[:8] == [(0, 0, 1), (1, 1, 1), (1, 1, 3), (2, 0, 1), (2, 2, 3), (2, 2, 5),
                         (3, 1, 1), (3, 1, 3)]  # fmt: skip
    assert list(orb.n[:4]) == [0, 0, 0, 1]


def test_lmax_restricts_orbits() -> None:
    orb = make_orbits(4, lmax=1)
    assert orb.l.max() == 1


def test_tuples_match_explicit_loops() -> None:
    orb = make_orbits(4)
    e2max = 8
    expected = []
    for a in range(len(orb)):
        for b in range(a + 1):
            if orb.e[a] + orb.e[b] > e2max:
                continue
            for c in range(a + 1):
                for d in range((b if c == a else c) + 1):
                    if orb.e[c] + orb.e[d] > e2max:
                        continue
                    if (orb.l[a] + orb.l[b] + orb.l[c] + orb.l[d]) % 2:
                        continue
                    lo = max(abs(orb.j2[a] - orb.j2[b]), abs(orb.j2[c] - orb.j2[d])) // 2
                    hi = min(orb.j2[a] + orb.j2[b], orb.j2[c] + orb.j2[d]) // 2
                    expected += [(a, b, c, d, jj) for jj in range(lo, hi + 1)]
    got = [
        (a, b, int(c), int(d), int(j))
        for a, b, cs, ds, js in tuple_chunks(orb, e2max)
        for c, d, j in zip(cs, ds, js, strict=True)
    ]
    assert got == expected
    assert len(got) == 6834


def test_pair_basis_respects_couplings() -> None:
    orb = make_orbits(4)
    a, b = ordered_pairs(orb, 8, 2, -1)
    assert np.all((-1) ** (orb.l[a] + orb.l[b]) == -1)
    assert np.all(np.abs(orb.j2[a] - orb.j2[b]) <= 4)
    assert (2, -1) in lab_channels(orb, 8)
