"""Slow, element-wise reference for the 3N Jacobi -> lab coefficients (throwaway oracle).

A direct transcription of the nested sums of NuHamil's ``set_nas_tcoefs`` (spec 5d section 4.2),
using the explicit Kamuntavicius brackets of ``refs/moshinsky_reference.py``. It is deliberately
kept out of ``src/`` (provenance rule) and is used by ``tests/test_tcoef3.py`` when ``refs/`` exists.
Doubled arguments follow the Fortran code.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moshinsky_reference import bracket  # noqa: E402

from nuham.angmom.racah import ninej, sixj  # noqa: E402


def tnas_element(row, col, j2, orb):
    """One ``T_NAS`` entry.

    Parameters
    ----------
    row
        ``(n12, l12, s12, j12, t12, n3, l3, j3, jrel, ncm, lcm)``.
    col
        ``(i1, i2, i3, jab, tab)`` with orbit indices into ``orb``.
    j2
        Doubled total ``J`` of the channel.
    orb
        :class:`nuham.basis.lab.Orbits`.
    """
    n12, l12, s12, j12, t12, n3, l3, j3, jrel, ncm, lcm = row
    i1, i2, i3, jab, tab = col
    if t12 != tab:
        return 0.0
    na, la, ja = int(orb.n[i1]), int(orb.l[i1]), int(orb.j2[i1])
    nb, lb, jb = int(orb.n[i2]), int(orb.l[i2]), int(orb.j2[i2])
    nc, lc, jc = int(orb.n[i3]), int(orb.l[i3]), int(orb.j2[i3])
    tc = 0.0
    for lab in range(max(abs(jab - s12), abs(la - lb)), min(jab + s12, la + lb) + 1):
        flab = (2 * lab + 1) * ninej(2 * la, 1, ja, 2 * lb, 1, jb, 2 * lab, 2 * s12, 2 * jab)
        if flab == 0.0:
            continue
        lit_max = min(
            lab + l12,
            jab + j12,
            2 * na + la + 2 * nb + lb - 2 * n12 - l12,
            2 * ncm + lcm + 2 * n3 + l3 - 2 * nc - lc,
        )
        for lit in range(max(abs(lab - l12), abs(jab - j12)), lit_max + 1):
            nit = (2 * na + la + 2 * nb + lb - 2 * n12 - l12 - lit) // 2
            flit = bracket(nit, lit, n12, l12, na, la, nb, lb, lab, 1.0) * sixj(
                2 * lit, 2 * l12, 2 * lab, 2 * s12, 2 * jab, 2 * j12
            )
            if flit == 0.0:
                continue
            lmin = max(abs(2 * j12 - j2), abs(2 * lit - jc), abs(2 * lcm - j3))
            lmax = min(2 * j12 + j2, 2 * lit + jc, 2 * lcm + j3)
            for big in range(lmin, lmax + 1, 2):
                fbig = (
                    (-1.0) ** ((big - jrel) // 2)
                    * (big + 1)
                    * sixj(2 * j12, 2 * lit, 2 * jab, jc, j2, big)
                    * sixj(2 * lcm, j3, big, 2 * j12, j2, jrel)
                )
                if fbig == 0.0:
                    continue
                for lam in range(
                    max(abs(lit - lc), abs(1 - big) // 2, abs(lcm - l3)),
                    min(lit + lc, (1 + big) // 2, lcm + l3) + 1,
                ):
                    flam = (
                        (2 * lam + 1)
                        * sixj(2 * lit, 2 * lc, 2 * lam, 1, big, jc)
                        * sixj(2 * lcm, 2 * l3, 2 * lam, 1, big, j3)
                        * bracket(ncm, lcm, n3, l3, nit, lit, nc, lc, lam, 2.0)
                    )
                    tc += flam * fbig * flab * flit
    return (
        tc
        * math.sqrt((ja + 1) * (jb + 1) * (jc + 1) * (2 * jab + 1) * (2 * s12 + 1) * (2 * j12 + 1))
        * math.sqrt((j3 + 1) * (jrel + 1))
        * (-1.0) ** (s12 + l12 + 1 + j12 + lc + l3 + (jc + j2) // 2)
    )
