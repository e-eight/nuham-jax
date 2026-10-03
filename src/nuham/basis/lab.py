"""Laboratory-frame single-particle orbits and two-body pair bases."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Orbits:
    """HO single-particle orbits ``(n, l, j)`` ordered as in the me2j format.

    Orbits run by ``e = 2n + l``, then ``l``, then ``j = l - 1/2`` before ``j = l + 1/2``.

    Attributes
    ----------
    e, n, l, j2
        Integer arrays; ``j2`` is twice ``j``.
    """

    e: np.ndarray
    n: np.ndarray
    l: np.ndarray
    j2: np.ndarray

    def __len__(self) -> int:
        """Return the number of orbits."""
        return len(self.e)


def make_orbits(emax: int, lmax: int | None = None) -> Orbits:
    """Build the orbit list for ``e <= emax`` and ``l <= lmax`` (default ``emax``)."""
    lmax = emax if lmax is None else lmax
    rows = [
        (e, (e - l) // 2, l, tj)
        for e in range(emax + 1)
        for l in range(min(e, lmax) + 1)
        if (e - l) % 2 == 0
        for tj in (2 * l - 1, 2 * l + 1)
        if tj > 0
    ]
    e, n, l, j2 = (np.array(col, dtype=np.int64) for col in zip(*rows, strict=True))
    return Orbits(e, n, l, j2)


def ordered_pairs(orb: Orbits, e2max: int, j: int, parity: int) -> tuple[np.ndarray, np.ndarray]:
    """All ordered orbit pairs ``(a, b)`` that can couple to ``J`` with the given parity.

    Parameters
    ----------
    orb
        Orbits.
    e2max
        Bound on ``e_a + e_b``.
    j
        Total angular momentum ``J``.
    parity
        ``+1`` or ``-1``, the parity ``(-1)^(l_a + l_b)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Orbit indices ``a`` and ``b``, ordered with ``a`` major.
    """
    a, b = np.meshgrid(np.arange(len(orb)), np.arange(len(orb)), indexing="ij")
    a, b = a.ravel(), b.ravel()
    ok = (
        (orb.e[a] + orb.e[b] <= e2max)
        & (np.abs(orb.j2[a] - orb.j2[b]) <= 2 * j)
        & (2 * j <= orb.j2[a] + orb.j2[b])
        & ((-1) ** (orb.l[a] + orb.l[b]) == parity)
    )
    return a[ok], b[ok]


def lab_channels(orb: Orbits, e2max: int) -> list[tuple[int, int]]:
    """All ``(J, parity)`` with at least one ordered pair, in ascending order."""
    found = set()
    for a in range(len(orb)):
        for b in range(len(orb)):
            if orb.e[a] + orb.e[b] > e2max:
                continue
            par = (-1) ** (orb.l[a] + orb.l[b])
            lo = abs(orb.j2[a] - orb.j2[b]) // 2
            hi = (orb.j2[a] + orb.j2[b]) // 2
            found.update((jj, par) for jj in range(lo, hi + 1))
    return sorted(found)
