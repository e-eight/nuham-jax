"""Independent references for the 3N Jacobi-to-lab coefficients.

Everything here is test infrastructure: an m-scheme evaluation of the antisymmetriser's lab matrix
(explicit Clebsch-Gordan coefficients and tensor permutations, no recoupling formulas), and readers
for NuHamil's reference outputs in ``refs/work/nnn_e4`` (Jacobi ``cfp`` and operator files, the
``me3j`` text file, and its tuple order).
"""

import gzip
import itertools
from pathlib import Path

import numpy as np

REF_DIR = Path(__file__).resolve().parents[1] / "refs" / "work" / "nnn_e4"


def _cg(j1, m1, j2, m2, j, m):
    """Clebsch-Gordan coefficient with doubled arguments (sympy, cached)."""
    from sympy import Rational
    from sympy.physics.wigner import clebsch_gordan

    return float(clebsch_gordan(*(Rational(x, 2) for x in (j1, j2, j, m1, m2, m))))


_CG_CACHE: dict = {}


def cg(j1, m1, j2, m2, j, m):
    """Cached :func:`_cg`."""
    key = (j1, m1, j2, m2, j, m)
    if key not in _CG_CACHE:
        _CG_CACHE[key] = _cg(*key)
    return _CG_CACHE[key]


def mscheme_state(orb, a, b, c, jab, tab, j2, t2, sp_index, dim):
    """Coupled lab state ``|(a b) Jab Tab, c; J T>`` (M = J, M_T = T) as a ``dim^3`` tensor."""
    psi = np.zeros((dim, dim, dim))
    ja, jb, jc = (int(orb.j2[x]) for x in (a, b, c))
    for ma in range(-ja, ja + 1, 2):
        for mb in range(-jb, jb + 1, 2):
            m12 = ma + mb
            if abs(m12) > 2 * jab:
                continue
            g1 = cg(ja, ma, jb, mb, 2 * jab, m12)
            if g1 == 0.0:
                continue
            for mc in range(-jc, jc + 1, 2):
                if m12 + mc != j2:
                    continue
                g2 = cg(2 * jab, m12, jc, mc, j2, j2)
                for ta in (-1, 1):
                    for tb in (-1, 1):
                        g3 = cg(1, ta, 1, tb, 2 * tab, ta + tb)
                        if g3 == 0.0 or ta + tb + 0 > 2 * tab:
                            continue
                        for tc in (-1, 1):
                            if ta + tb + tc != t2:
                                continue
                            g4 = cg(2 * tab, ta + tb, 1, tc, t2, t2)
                            psi[
                                sp_index[(a, ma, ta)], sp_index[(b, mb, tb)], sp_index[(c, mc, tc)]
                            ] += g1 * g2 * g3 * g4
    return psi


def antisym_matrix(orb, cols, j2, t2):
    """``sum_P sgn(P) <abc|P|def>`` between the lab states ``cols`` ``(i1, i2, i3, jab, tab)``.

    Computed in the m-scheme with explicit Clebsch-Gordan coefficients and tensor permutations,
    so it is independent of every recoupling formula used by the coefficients.
    """
    orbs = sorted({int(x) for col in cols for x in col[:3]})
    sp = [
        (o, m, t)
        for o in orbs
        for m in range(-int(orb.j2[o]), int(orb.j2[o]) + 1, 2)
        for t in (-1, 1)
    ]
    sp_index = {s: i for i, s in enumerate(sp)}
    dim = len(sp)
    psis = [mscheme_state(orb, c[0], c[1], c[2], c[3], c[4], j2, t2, sp_index, dim) for c in cols]
    perms = [
        (p, (-1) ** sum(p[i] > p[k] for i in range(3) for k in range(i + 1, 3)))
        for p in itertools.permutations(range(3))
    ]
    n = len(cols)
    out = np.zeros((n, n))
    for j in range(n):
        asym = sum(sg * np.transpose(psis[j], p) for p, sg in perms)
        for i in range(n):
            out[i, j] = float(np.sum(psis[i] * asym))
    return out


def read_cfp_bin(path):
    """Read NuHamil ``cfp_A3_*.bin``: ``(J2, P, T2, [(N, nas, cfp)])`` for blocks with n_as >= 1."""
    raw = Path(path).read_bytes()
    j2, par, t2, nmax = np.frombuffer(raw[:16], dtype="<i4")
    off, blocks = 16, []
    for _ in range(nmax + 1):
        n_nas, n_as, n = (int(x) for x in np.frombuffer(raw[off : off + 12], dtype="<i4"))
        off += 12
        nas = np.zeros((0, 8), dtype=np.int64)
        if n_nas >= 1:
            nas = np.frombuffer(raw[off : off + 32 * n_nas], dtype="<i4").reshape(n_nas, 8)
            off += 32 * n_nas
        if n_as >= 1:
            cfp = np.frombuffer(raw[off : off + 8 * n_as * n_nas], dtype="<f8")
            off += 8 * n_as * n_nas
            blocks.append((n, nas.astype(np.int64), cfp.reshape(n_as, n_nas).copy()))
    assert off == len(raw)
    return int(j2), int(par), int(t2), blocks


def read_op_bin(path):
    """NuHamil Jacobi operator file ``NNN-full_*.bin``: one square double matrix."""
    raw = Path(path).read_bytes()
    n = int(np.frombuffer(raw[:4], dtype="<i4")[0])
    return np.frombuffer(raw[4:], dtype="<f8").reshape(n, n).T.copy()


def file_tuples(orb, e2max, e3max):
    """Index tuples of the me3j file in NuHamil's writer order (spec 5d 6.2).

    Yields ``(i1, i2, i3, i4, i5, i6, j12, j45, j2, t12, t45, t2)`` with 0-based orbit indices.
    """
    no = len(orb)
    e, l, j2o = orb.e, orb.l, orb.j2
    for i1 in range(no):
        for i2 in range(i1 + 1):
            if e[i1] + e[i2] > e2max:
                continue
            for i3 in range(i2 + 1):
                if e[i1] + e[i3] > e2max or e[i2] + e[i3] > e2max or e[i1] + e[i2] + e[i3] > e3max:
                    continue
                p123 = (-1) ** (l[i1] + l[i2] + l[i3])
                for i4 in range(i1 + 1):
                    for i5 in range((i2 if i4 == i1 else i4) + 1):
                        if e[i4] + e[i5] > e2max:
                            continue
                        for i6 in range((i3 if (i4 == i1 and i5 == i2) else i5) + 1):
                            if (
                                e[i4] + e[i6] > e2max
                                or e[i5] + e[i6] > e2max
                                or e[i4] + e[i5] + e[i6] > e3max
                                or p123 != (-1) ** (l[i4] + l[i5] + l[i6])
                            ):
                                continue
                            for j12 in range(
                                abs(j2o[i1] - j2o[i2]) // 2, (j2o[i1] + j2o[i2]) // 2 + 1
                            ):
                                for j45 in range(
                                    abs(j2o[i4] - j2o[i5]) // 2, (j2o[i4] + j2o[i5]) // 2 + 1
                                ):
                                    lo = max(abs(2 * j12 - j2o[i3]), abs(2 * j45 - j2o[i6]))
                                    hi = min(2 * j12 + j2o[i3], 2 * j45 + j2o[i6])
                                    for jt in range(lo, hi + 1, 2):
                                        for t12 in (0, 1):
                                            for t45 in (0, 1):
                                                for tt in range(
                                                    1, min(2 * t12 + 1, 2 * t45 + 1) + 1, 2
                                                ):
                                                    yield (
                                                        i1,
                                                        i2,
                                                        i3,
                                                        i4,
                                                        i5,
                                                        i6,
                                                        j12,
                                                        j45,
                                                        jt,
                                                        t12,
                                                        t45,
                                                        tt,
                                                    )


def read_me3j(path):
    """Values of a (gzip) text me3j file as a float array, skipping the header line."""
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        f.readline()
        return np.array(f.read().split(), dtype=np.float64)
