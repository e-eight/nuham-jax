"""Jacobi-to-laboratory transformation coefficients of three nucleons (prototype of step 5.1b).

For one laboratory channel ``(J, P, T)`` and one oscillator block ``N = e1 + e2 + e3`` the
coefficient matrix ``T[(alpha, jrel, Ncm, Lcm), (a b c, J12, T12)]`` is the overlap of a
non-antisymmetrised (NAS) Jacobi state times a centre-of-mass state with the pair-coupled
laboratory state. It is built from two Moshinsky brackets (mass ratio 1 for the pair ``(a b)``,
mass ratio 2 for the pair's centre of mass and ``c``) and recoupling symbols.

The sums factor into a left stage, which depends on the pair ``(a, b, J12)`` and the Jacobi pair
state, and a right stage, which depends on ``c`` and the Jacobi particle-3 and centre-of-mass
state. The stages are joined by two summed angular momenta: ``lit``, the orbital momentum of the
pair's centre of mass, and ``Lam``. That is what makes the construction vectorisable.
Multiplying with the antisymmetriser's ``cfp`` rows gives the antisymmetric coefficients
:func:`tcoef_as`.

Angular momenta named ``j2``, ``t2``, ``j3``, ``jc``, ``jrel`` and the orbit's ``j2`` are doubled;
``l``, ``s12``, ``j12``, ``jab`` are true values, as in :mod:`nuham.nnn.jacobi`.
"""

from dataclasses import dataclass
from functools import cache

import numpy as np

from nuham.angmom.moshinsky_table import MoshinskyTable
from nuham.angmom.racah_array import ninej_array, sixj_array
from nuham.basis.lab import Orbits
from nuham.nnn.jacobi import JacobiChannel, build_channel, nas_states

_CUT = 1e-16
_CHUNK = 4_000_000
"""Largest number of (left, right) term pairs combined in one array operation."""


@dataclass(frozen=True)
class Brackets:
    """The two Moshinsky bracket tables the transformation needs (hw independent).

    Attributes
    ----------
    pair
        Mass ratio 1: lab pair ``(a, b)`` to pair relative and pair centre of mass.
    third
        Mass ratio 2: pair centre of mass and ``c`` to Jacobi coordinate 3 and total centre of mass.
    """

    pair: MoshinskyTable
    third: MoshinskyTable


def warm_brackets(brackets: Brackets, e3max: int) -> None:
    """Build every bracket block a lab transform of ``e3max`` can read, before forking workers."""
    brackets.pair.warm(e3max)
    brackets.third.warm(e3max)


def make_brackets(cache_dir: str | None = None) -> Brackets:
    """Bracket tables with ``d = 1`` and ``d = 2``, optionally with an on-disk cache."""
    return Brackets(MoshinskyTable(cache_dir), MoshinskyTable(cache_dir, d=2.0))


@cache
def _nas(j2: int, parity: int, t2: int, n: int) -> np.ndarray:
    """Return :func:`nuham.nnn.jacobi.nas_states`, cached."""
    return nas_states(j2, parity, t2, n)


def _tri(a: int, b: int, c: int) -> bool:
    """Whether doubled ``a, b, c`` satisfy the triangle rule with an even sum."""
    return abs(a - b) <= c <= a + b and (a + b + c) % 2 == 0


@dataclass(frozen=True)
class LabBlock:
    """Laboratory states of one ``(J, P, T)`` channel in NuHamil's order.

    Attributes
    ----------
    i1, i2, i3
        Orbit indices with ``i1 >= i2 >= i3``.
    j12, t12
        Pair angular momentum and isospin.
    n
        ``e1 + e2 + e3`` of every state.
    """

    i1: np.ndarray
    i2: np.ndarray
    i3: np.ndarray
    j12: np.ndarray
    t12: np.ndarray
    n: np.ndarray

    def __len__(self) -> int:
        """Return the number of states."""
        return len(self.i1)

    def select(self, n: int) -> "LabBlock":
        """Return the states with ``e1 + e2 + e3 = n`` in their original relative order."""
        k = self.n == n
        return LabBlock(*(x[k] for x in (self.i1, self.i2, self.i3, self.j12, self.t12, self.n)))


def lab_states(orb: Orbits, j2: int, parity: int, t2: int, e2max: int, e3max: int) -> LabBlock:
    """All laboratory states ``|a b c; J12 T12; J T>`` of a channel, in the order of spec 5d 2.3.

    Triples run over ``a >= b >= c`` (orbit index), then ``J12``, then ``T12``; pairs of identical
    orbits keep only ``J12 + T12`` odd. Pauli-forbidden three-body combinations stay in the basis.
    """
    no = len(orb)
    i1, i2, i3 = np.meshgrid(np.arange(no), np.arange(no), np.arange(no), indexing="ij")
    i1, i2, i3 = i1.ravel(), i2.ravel(), i3.ravel()
    e1, e2, e3 = orb.e[i1], orb.e[i2], orb.e[i3]
    keep = (
        (i2 <= i1)
        & (i3 <= i2)
        & (e1 + e2 <= e2max)
        & (e1 + e3 <= e2max)
        & (e2 + e3 <= e2max)
        & (e1 + e2 + e3 <= e3max)
        & ((-1) ** (orb.l[i1] + orb.l[i2] + orb.l[i3]) == parity)
    )
    i1, i2, i3 = i1[keep], i2[keep], i3[keep]
    out: list[tuple[np.ndarray, ...]] = []
    jlo = np.abs(orb.j2[i1] - orb.j2[i2]) // 2
    jhi = (orb.j2[i1] + orb.j2[i2]) // 2
    for j12 in range(int(jhi.max(initial=0)) + 1):
        ok = (jlo <= j12) & (j12 <= jhi) & (np.abs(2 * j12 - orb.j2[i3]) <= j2)
        ok &= j2 <= 2 * j12 + orb.j2[i3]
        for t12 in (0, 1):
            if not _tri(2 * t12, 1, t2):
                continue
            sel = ok & ~((i1 == i2) & ((j12 + t12) % 2 == 0))
            out.append(
                (i1[sel], i2[sel], i3[sel], np.full(sel.sum(), j12), np.full(sel.sum(), t12))
            )
    if not out:
        z = np.zeros(0, dtype=np.int64)
        return LabBlock(z, z, z, z, z, z)
    cols = [np.concatenate(c) for c in zip(*out, strict=True)]
    order = np.lexsort((cols[4], cols[3], cols[2], cols[1], cols[0]))
    a, b, c, j12, t12 = (x[order] for x in cols)
    return LabBlock(a, b, c, j12, t12, orb.e[a] + orb.e[b] + orb.e[c])


@dataclass(frozen=True)
class JacobiCmRows:
    """NAS Jacobi times centre-of-mass states of one ``(J, P, T, N)`` block.

    Attributes
    ----------
    qn
        Integer array ``(rows, 11)``: ``n12, l12, s12, j12, t12, n3, l3, j3, jrel, ncm, lcm``.
    groups
        One entry ``(jrel, ncm, lcm, e, start, stop)`` per contiguous run of rows that shares the
        Jacobi block ``(jrel, e)``; the rows inside a run are the NAS states of that block in order.
    """

    qn: np.ndarray
    groups: tuple[tuple[int, int, int, int, int, int], ...]

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self.qn)


def jacobi_cm_rows(j2: int, parity: int, t2: int, n: int) -> JacobiCmRows:
    """NAS Jacobi + centre-of-mass rows of channel ``(J, P, T)`` and block ``N``.

    ``E = N - 2 Ncm - Lcm`` is the Jacobi quanta; ``Jrel`` couples with ``Lcm`` to ``J``. States of
    Jacobi blocks that hold no antisymmetric state are kept here (they have zero ``cfp`` rows).
    """
    chunks, groups, start = [], [], 0
    if (-1) ** n == parity:
        for ncm in range(n // 2 + 1):
            for lcm in range(n - 2 * ncm + 1):
                e = n - 2 * ncm - lcm
                for jrel in range(1, 2 * n + 4, 2):
                    if not _tri(jrel, 2 * lcm, j2):
                        continue
                    nas = _nas(jrel, (-1) ** e, t2, e)
                    if len(nas) == 0:
                        continue
                    k = len(nas)
                    extra = np.empty((k, 3), dtype=np.int64)
                    extra[:] = (jrel, ncm, lcm)
                    chunks.append(np.hstack([nas, extra]))
                    groups.append((jrel, ncm, lcm, e, start, start + k))
                    start += k
    qn = np.vstack(chunks) if chunks else np.zeros((0, 11), dtype=np.int64)
    return JacobiCmRows(qn, tuple(groups))


def _unique_rows(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the unique rows of an integer array and the id of each input row."""
    u, inv = np.unique(a, axis=0, return_inverse=True)
    return u, inv.reshape(-1)


def _left_dense(
    et: int, lit: int, pairs: np.ndarray, pstates: np.ndarray, br: Brackets, orb: Orbits
) -> np.ndarray:
    """Left-stage factor of one pair energy ``et`` and one ``lit``, dense.

    Parameters
    ----------
    et
        Energy ``e_a + e_b`` of every row of ``pairs``.
    pairs
        ``(a, b, jab)`` rows, all with energy ``et``.
    pstates
        ``(n12, l12, s12, j12, t12)`` rows with ``e12 <= et - lit`` and ``et - lit - e12`` even
        (``nit`` follows from energy conservation).

    Returns
    -------
    np.ndarray
        Array ``(len(pairs), len(pstates))``.
    """
    na, la, ja = orb.n[pairs[:, 0]], orb.l[pairs[:, 0]], orb.j2[pairs[:, 0]]
    nb, lb, jb = orb.n[pairs[:, 1]], orb.l[pairs[:, 1]], orb.j2[pairs[:, 1]]
    jab = pairs[:, 2]
    n12, l12, s12, j12 = (pstates[:, k] for k in range(4))
    nit = (et - lit - (2 * n12 + l12)) // 2
    rix = br.pair.row_index(et)
    cix = br.pair.col_index(et)
    ridx = np.array([rix[(a, lit, b, c)] for a, b, c in zip(nit, n12, l12, strict=True)])
    cidx = np.array([cix[(a, b, c, d)] for a, b, c, d in zip(na, la, nb, lb, strict=True)])
    val = np.zeros((len(pairs), len(pstates)))
    # The symbols depend on few labels each, so evaluate them on the unique combinations only.
    u_pair, inv_pair = _unique_rows(np.stack([la, ja, lb, jb, jab], axis=1))
    u_q, inv_q = _unique_rows(np.stack([l12, s12, j12], axis=1))
    u_jab, inv_jab = np.unique(jab, return_inverse=True)
    inv_jab = inv_jab.reshape(-1)
    for lab in range(int((la + lb).max()) + 1):
        brk = br.pair.gather(et, lab, ridx, cidx).T
        if not brk.any():
            continue
        nj_u = np.stack(
            [
                ninej_array(
                    2 * u_pair[:, 0],
                    1,
                    u_pair[:, 1],
                    2 * u_pair[:, 2],
                    1,
                    u_pair[:, 3],
                    2 * lab,
                    2 * s,
                    2 * u_pair[:, 4],
                )  # fmt: skip
                for s in (0, 1)
            ],
            axis=1,
        )
        sj_u = sixj_array(
            2 * lit, 2 * u_q[None, :, 0], 2 * lab, 2 * u_q[None, :, 1], 2 * u_jab[:, None],
            2 * u_q[None, :, 2],
        )  # fmt: skip
        nj = nj_u[inv_pair[:, None], s12[None]]
        sj = sj_u[inv_jab[:, None], inv_q[None, :]]
        val += (2 * lab + 1) * nj * sj * brk
    pre_p = np.sqrt((2 * s12 + 1) * (2 * j12 + 1)) * (-1.0) ** (s12 + l12 + 1 + j12)
    pre_a = np.sqrt((ja + 1) * (jb + 1) * (2 * jab + 1))
    return val * pre_a[:, None] * pre_p[None]


def _left_terms(
    lit: int,
    pairs: np.ndarray,
    pstates: np.ndarray,
    br: Brackets,
    orb: Orbits,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Left-stage factor for one ``lit`` as sparse terms.

    Parameters
    ----------
    pairs
        Unique ``(a, b, jab)`` rows.
    pstates
        Unique ``(n12, l12, s12, j12, t12)`` rows.

    Returns
    -------
    tuple
        ``pair id, p id, nit, value`` of every non-zero term.
    """
    e_ab = orb.e[pairs[:, 0]] + orb.e[pairs[:, 1]]
    e12 = 2 * pstates[:, 0] + pstates[:, 1]
    res: list[tuple[np.ndarray, ...]] = []
    for et in np.unique(e_ab):
        ps = np.nonzero(e_ab == et)[0]
        qs = np.nonzero((e12 <= et - lit) & ((et - lit - e12) % 2 == 0))[0]
        if len(ps) == 0 or len(qs) == 0:
            continue
        val = _left_dense(int(et), lit, pairs[ps], pstates[qs], br, orb)
        ii, jj = np.nonzero(np.abs(val) > _CUT)
        res.append((ps[ii], qs[jj], (et - lit - e12[qs][jj]) // 2, val[ii, jj]))
    if not res:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z, np.zeros(0)
    a, b, c, d = (np.concatenate(x) for x in zip(*res, strict=True))
    return a, b, c, d


def _right_dense(
    et: int,
    lit: int,
    lam_big: int,
    j2: int,
    corbs: np.ndarray,
    qstates: np.ndarray,
    br: Brackets,
    orb: Orbits,
) -> np.ndarray:
    """Right-stage factor of one energy ``et`` and one ``(lit, Lam)``, dense.

    Parameters
    ----------
    et
        Energy ``2 ncm + lcm + 2 n3 + l3`` of every row of ``qstates``.
    lam_big
        Doubled ``Lam``.
    corbs
        Orbit indices of ``c`` with ``e_c <= et - lit`` and ``et - lit - e_c`` even.
    qstates
        ``(n3, l3, j3, ncm, lcm, jrel)`` rows, all with energy ``et``.

    Returns
    -------
    np.ndarray
        Array ``(len(corbs), len(qstates))``.
    """
    nc, lc, jc = orb.n[corbs], orb.l[corbs], orb.j2[corbs]
    n3, l3, j3, ncm, lcm = (qstates[:, k] for k in range(5))
    nit = (et - lit - (2 * nc + lc)) // 2
    rix = br.third.row_index(et)
    cix = br.third.col_index(et)
    ridx = np.array([rix[(a, b, c, d)] for a, b, c, d in zip(ncm, lcm, n3, l3, strict=True)])
    cidx = np.array([cix[(a, lit, b, c)] for a, b, c in zip(nit, nc, lc, strict=True)])
    val = np.zeros((len(corbs), len(qstates)))
    u_c, inv_c = _unique_rows(np.stack([lc, jc], axis=1))
    u_q, inv_q = _unique_rows(np.stack([lcm, l3, j3], axis=1))
    for lam in range(lit + int(lc.max()) + 1):
        brk = br.third.gather(et, lam, ridx, cidx).T
        if not brk.any():
            continue
        s1 = sixj_array(2 * lit, 2 * u_c[:, 0], 2 * lam, 1, lam_big, u_c[:, 1])[inv_c][:, None]
        s2 = sixj_array(2 * u_q[:, 0], 2 * u_q[:, 1], 2 * lam, 1, lam_big, u_q[:, 2])[inv_q][None]
        val += (2 * lam + 1) * s1 * s2 * brk
    pre_c = np.sqrt(jc + 1.0) * (-1.0) ** (lc + (jc + j2) // 2)
    pre_q = np.sqrt(j3 + 1.0) * (-1.0) ** l3
    return val * pre_c[:, None] * pre_q[None]


def _right_terms(
    lit: int,
    lam_big: int,
    j2: int,
    corbs: np.ndarray,
    qstates: np.ndarray,
    br: Brackets,
    orb: Orbits,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Right-stage factor for one ``(lit, Lam)`` as sparse terms.

    Parameters
    ----------
    lam_big
        Doubled ``Lam``.
    corbs
        Unique orbit indices of ``c``.
    qstates
        Unique ``(n3, l3, j3, ncm, lcm, jrel)`` rows.

    Returns
    -------
    tuple
        ``c id, q id, nit, value`` of every non-zero term.
    """
    ec = orb.e[corbs]
    eq = 2 * qstates[:, 3] + qstates[:, 4] + 2 * qstates[:, 0] + qstates[:, 1]
    res: list[tuple[np.ndarray, ...]] = []
    for et in np.unique(eq):
        qs = np.nonzero(eq == et)[0]
        cs = np.nonzero((ec <= et - lit) & ((et - lit - ec) % 2 == 0))[0]
        if len(cs) == 0:
            continue
        val = _right_dense(int(et), lit, lam_big, j2, corbs[cs], qstates[qs], br, orb)
        ii, jj = np.nonzero(np.abs(val) > _CUT)
        res.append((cs[ii], qs[jj], (et - lit - ec[cs][ii]) // 2, val[ii, jj]))
    if not res:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z, np.zeros(0)
    a, b, c, d = (np.concatenate(x) for x in zip(*res, strict=True))
    return a, b, c, d


def _index_map(keys: np.ndarray, ids: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Dense lookup table: ``out[keys[k]] = ids[k]``, ``-1`` elsewhere."""
    out = np.full(shape, -1, dtype=np.int64)
    out[tuple(keys.T)] = ids
    return out


def tcoef_nas(
    rows: JacobiCmRows,
    cols: LabBlock,
    j2: int,
    orb: Orbits,
    brackets: Brackets,
) -> np.ndarray:
    """NAS coefficient matrix ``T_NAS[rows, cols]`` of one ``(channel, N)`` block.

    ``rows`` and ``cols`` must belong to the same ``N``. The result is dense, shape
    ``(len(rows), len(cols))``, in double precision.
    """
    n_r, n_c = len(rows), len(cols)
    out = np.zeros((n_r, n_c))
    if n_r == 0 or n_c == 0:
        return out
    qn = rows.qn
    # unique left / right keys and their ids
    pst, p_of_row = _unique_rows(qn[:, :5])
    qst, q_of_row = _unique_rows(qn[:, [5, 6, 7, 9, 10, 8]])
    pair_key = np.stack([cols.i1, cols.i2, cols.j12], axis=1)
    pairs, pair_of_col = _unique_rows(pair_key)
    corbs, c_of_col = np.unique(cols.i3, return_inverse=True)
    c_of_col = c_of_col.reshape(-1)
    rowmap = _index_map(np.stack([p_of_row, q_of_row], 1), np.arange(n_r), (len(pst), len(qst)))
    colmap = _index_map(
        np.stack([pair_of_col, c_of_col, cols.t12], 1), np.arange(n_c), (len(pairs), len(corbs), 2)
    )
    flat = out.reshape(-1)

    j12_p, t12_p = pst[:, 3], pst[:, 4]
    jab_pair = pairs[:, 2]
    jc_c = orb.j2[corbs]
    lcm_q, j3_q, jrel_q = qst[:, 4], qst[:, 2], qst[:, 5]
    n = int(cols.n[0])
    j12_max = int(j12_p.max())
    jab_max = int(jab_pair.max())
    jc_max = int(jc_c.max())
    lcm_max, j3_max, jrel_max = int(lcm_q.max()), int(j3_q.max()), int(jrel_q.max())

    jab_g = np.arange(jab_max + 1)[:, None, None]
    jc_g = np.arange(jc_max + 1)[None, :, None]
    j12_g = np.arange(j12_max + 1)[None, None, :]
    lcm_g = np.arange(lcm_max + 1)[:, None, None, None]
    j3_g = np.arange(j3_max + 1)[None, :, None, None]
    j12_h = np.arange(j12_max + 1)[None, None, :, None]
    jrel_g = np.arange(jrel_max + 1)[None, None, None, :]

    g_cache: dict[int, np.ndarray] = {}
    lit_max = n
    for lit in range(lit_max + 1):
        lp, lq, lnit, lval = _left_terms(lit, pairs, pst, brackets, orb)
        if len(lval) == 0:
            continue
        for lam_big in range(1, 2 * lit + jc_max + 2, 2):
            a1 = sixj_array(2 * j12_g, 2 * lit, 2 * jab_g, jc_g, j2, lam_big)
            if not a1.any():
                continue
            rc, rq, rnit, rval = _right_terms(lit, lam_big, j2, corbs, qst, brackets, orb)
            if len(rval) == 0:
                continue
            if lam_big not in g_cache:
                g_cache[lam_big] = (
                    sixj_array(2 * lcm_g, j3_g, lam_big, 2 * j12_h, j2, jrel_g)
                    * (lam_big + 1.0)
                    * (-1.0) ** ((lam_big - jrel_g) // 2)
                    * np.sqrt(jrel_g + 1.0)
                )
            g = g_cache[lam_big]
            for nit in np.unique(np.intersect1d(lnit, rnit)):
                ls, rs = lnit == nit, rnit == nit
                lp_n, lq_n, lv_n = lp[ls], lq[ls], lval[ls]
                rc_n, rq_n, rv_n = rc[rs], rq[rs], rval[rs]
                step = max(1, _CHUNK // len(rv_n))
                for s in range(0, len(lv_n), step):
                    sl = slice(s, s + step)
                    pp, pq, pv = lp_n[sl], lq_n[sl], lv_n[sl]
                    ridx = rowmap[pq[:, None], rq_n[None, :]]
                    cidx = colmap[pp[:, None], rc_n[None, :], t12_p[pq][:, None]]
                    ok = (ridx >= 0) & (cidx >= 0)
                    if not ok.any():
                        continue
                    v = (
                        pv[:, None]
                        * rv_n[None, :]
                        * a1[jab_pair[pp][:, None], jc_c[rc_n][None, :], j12_p[pq][:, None]]
                        * g[
                            lcm_q[rq_n][None, :],
                            j3_q[rq_n][None, :],
                            j12_p[pq][:, None],
                            jrel_q[rq_n][None, :],
                        ]
                    )
                    # distinct (left, right) term pairs map to distinct matrix entries
                    flat[(ridx * n_c + cidx)[ok]] += v[ok]
    return out


def tcoef_nas_blocked(
    rows: JacobiCmRows,
    cols: LabBlock,
    j2: int,
    orb: Orbits,
    brackets: Brackets,
    stats: dict[str, int] | None = None,
) -> np.ndarray:
    """Build the matrix of :func:`tcoef_nas` from dense products over energy blocks.

    Energy conservation ties the labels together: a lab pair of energy ``e_ab`` and a Jacobi pair
    of energy ``e12`` fix ``2 nit + lit = e_ab - e12``, the energy ``N - e_ab`` of ``c`` and the
    energy ``N - e12`` of ``(n3, l3, Ncm, Lcm)``. So for each ``(e_ab, e12)`` and each
    ``(j12, jc)`` the sum over ``k = (lit, Lam)`` is one product ``A[(pair, p), k] @ B[k, (c, q)]``
    in which every element is a valid pair of a left and a right state. The term-wise
    :func:`tcoef_nas` is the reference this is tested against.

    Parameters
    ----------
    stats
        If given, receives ``products`` (number of dense products), ``elements`` (their total
        size) and ``valid`` (elements that land in the matrix).
    """
    n_r, n_c = len(rows), len(cols)
    out = np.zeros((n_r, n_c))
    if n_r == 0 or n_c == 0:
        return out
    n = int(cols.n[0])
    qn = rows.qn
    pst, p_of_row = _unique_rows(qn[:, :5])
    qst, q_of_row = _unique_rows(qn[:, [5, 6, 7, 9, 10, 8]])
    pairs, pair_of_col = _unique_rows(np.stack([cols.i1, cols.i2, cols.j12], axis=1))
    corbs, c_of_col = np.unique(cols.i3, return_inverse=True)
    c_of_col = c_of_col.reshape(-1)
    rowmap = _index_map(np.stack([p_of_row, q_of_row], 1), np.arange(n_r), (len(pst), len(qst)))
    colmap = _index_map(
        np.stack([pair_of_col, c_of_col, cols.t12], 1), np.arange(n_c), (len(pairs), len(corbs), 2)
    )
    flat = out.reshape(-1)

    e_pair = orb.e[pairs[:, 0]] + orb.e[pairs[:, 1]]
    e_c = orb.e[corbs]
    e12_p = 2 * pst[:, 0] + pst[:, 1]
    e_q = 2 * qst[:, 3] + qst[:, 4] + 2 * qst[:, 0] + qst[:, 1]
    j12_p, t12_p = pst[:, 3], pst[:, 4]
    jab_pair, jc_c = pairs[:, 2], orb.j2[corbs]
    lcm_q, j3_q, jrel_q = qst[:, 4], qst[:, 2], qst[:, 5]
    count = {"products": 0, "elements": 0, "valid": 0}
    u_jab, inv_jab = np.unique(jab_pair, return_inverse=True)
    u_q, inv_q = _unique_rows(np.stack([lcm_q, j3_q, jrel_q], axis=1))
    inv_jab, a1_cache, g_cache = inv_jab.reshape(-1), {}, {}

    def a1_table(j12: int, lit: int, jc: int, lam_big: int) -> np.ndarray:
        """6j of the left/right coupling for every lab pair, evaluated once per label set."""
        key = (j12, lit, jc, lam_big)
        if key not in a1_cache:
            a1_cache[key] = sixj_array(2 * j12, 2 * lit, 2 * u_jab, jc, j2, lam_big)[inv_jab]
        return a1_cache[key]

    def g_table(j12: int, lam_big: int) -> np.ndarray:
        """Centre-of-mass recoupling factor for every ``q``, evaluated once per label set."""
        key = (j12, lam_big)
        if key not in g_cache:
            lcm_u, j3_u, jrel_u = u_q[:, 0], u_q[:, 1], u_q[:, 2]
            g_cache[key] = (
                sixj_array(2 * lcm_u, j3_u, lam_big, 2 * j12, j2, jrel_u)
                * (lam_big + 1.0)
                * (-1.0) ** ((lam_big - jrel_u) // 2)
                * np.sqrt(jrel_u + 1.0)
            )[inv_q]
        return g_cache[key]

    for e_ab in np.unique(e_pair):
        ps_all = np.nonzero(e_pair == e_ab)[0]
        cs_all = np.nonzero(e_c == n - e_ab)[0]
        if len(cs_all) == 0:
            continue
        for e12 in np.unique(e12_p[e12_p <= e_ab]):
            pp_all = np.nonzero(e12_p == e12)[0]
            qs_all = np.nonzero(e_q == n - e12)[0]
            if len(qs_all) == 0:
                continue
            eps = int(e_ab - e12)
            left = {
                lit: _left_dense(int(e_ab), lit, pairs[ps_all], pst[pp_all], brackets, orb)
                for lit in range(eps % 2, eps + 1, 2)
            }
            right: dict[tuple[int, int], np.ndarray] = {}
            for j12 in np.unique(j12_p[pp_all]):
                pp = pp_all[j12_p[pp_all] == j12]
                pp_loc = np.searchsorted(pp_all, pp)
                for jc in np.unique(jc_c[cs_all]):
                    cs = cs_all[jc_c[cs_all] == jc]
                    cs_loc = np.searchsorted(cs_all, cs)
                    a_parts, b_parts = [], []
                    for lit, lval in left.items():
                        lo = max(abs(2 * lit - jc), abs(2 * int(j12) - j2))
                        hi = min(2 * lit + jc, 2 * int(j12) + j2)
                        for lam_big in range(lo, hi + 1, 2):
                            a1 = a1_table(int(j12), lit, int(jc), lam_big)[ps_all]
                            if not a1.any():
                                continue
                            if (lit, lam_big) not in right:
                                right[lit, lam_big] = _right_dense(
                                    int(n - e12), lit, lam_big, j2, corbs[cs_all],
                                    qst[qs_all], brackets, orb,
                                )  # fmt: skip
                            rval = right[lit, lam_big]
                            if not rval.any():
                                continue
                            g = g_table(int(j12), lam_big)[qs_all]
                            a_parts.append((lval[:, pp_loc] * a1[:, None]).T)
                            b_parts.append(rval[np.ix_(cs_loc, np.arange(len(qs_all)))] * g[None])
                    if not a_parts:
                        continue
                    a_mat = np.stack(a_parts, 2).reshape(len(pp) * len(ps_all), -1)
                    b_mat = np.stack(b_parts, 0).reshape(len(a_parts), -1)
                    blk = a_mat @ b_mat  # rows (p, pair), columns (c, q)
                    p_i = np.repeat(pp, len(ps_all))
                    pair_i = np.tile(ps_all, len(pp))
                    c_i = np.repeat(cs, len(qs_all))
                    q_i = np.tile(qs_all, len(cs))
                    ridx = rowmap[p_i[:, None], q_i[None, :]]
                    cidx = colmap[pair_i[:, None], c_i[None, :], t12_p[p_i][:, None]]
                    ok = (ridx >= 0) & (cidx >= 0)
                    count["products"] += 1
                    count["elements"] += ok.size
                    count["valid"] += int(ok.sum())
                    flat[(ridx * n_c + cidx)[ok]] += blk[ok]
    if stats is not None:
        stats.update(count)
    return out


def tcoef_as(
    t_nas: np.ndarray,
    rows: JacobiCmRows,
    channels: dict[tuple[int, int], JacobiChannel],
    t2: int,
) -> tuple[np.ndarray, list[tuple[int, int, int, int, int]]]:
    """Antisymmetric coefficients: ``cfp`` rows times the NAS coefficients.

    Parameters
    ----------
    t_nas
        Result of :func:`tcoef_nas`.
    rows
        Its row description.
    channels
        Jacobi channels ``{(jrel, parity): JacobiChannel}`` at ``T = t2`` holding ``cfp``.
    t2
        Doubled isospin (only used to index ``channels`` consistently).

    Returns
    -------
    tuple
        The matrix ``(n_as_cm, n_lab)`` and the labels ``(e, i, jrel, ncm, lcm)`` of its rows.
    """
    parts, labels = [], []
    for jrel, ncm, lcm, e, start, stop in rows.groups:
        ch = channels.get((jrel, (-1) ** e))
        if ch is None:
            continue
        for blk in ch.blocks:
            if blk.n_quanta != e:
                continue
            parts.append(blk.cfp @ t_nas[start:stop])
            labels += [(e, i, jrel, ncm, lcm) for i in range(len(blk.cfp))]
    if not parts:
        return np.zeros((0, t_nas.shape[1])), labels
    return np.vstack(parts), labels


def jacobi_channels_for(
    j2: int, t2: int, nmax: int, table: MoshinskyTable | None = None
) -> dict[tuple[int, int], JacobiChannel]:
    """Jacobi ``cfp`` channels ``(jrel, parity)`` needed by a lab channel with total ``J = j2``.

    ``Jrel`` ranges over every value that can couple with an ``Lcm <= nmax`` to ``j2``.
    """
    table = MoshinskyTable(d=1 / 3) if table is None else table
    out = {}
    for jrel in range(1, 2 * nmax + 4, 2):
        if jrel > j2 + 2 * nmax:
            break
        for parity in (1, -1):
            out[(jrel, parity)] = build_channel(jrel, parity, t2, nmax, table)
    return out


def lab_t_blocks(
    j2: int,
    parity: int,
    t2: int,
    orb: Orbits,
    e2max: int,
    e3max: int,
    channels: dict[tuple[int, int], JacobiChannel],
    brackets: Brackets,
) -> tuple[
    LabBlock, dict[int, tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int, int]]]]
]:
    """Antisymmetric coefficients of every ``N`` block of one lab channel.

    Returns
    -------
    tuple
        The lab states, and for each ``N = e1 + e2 + e3`` the lab-state indices of the block, the
        coefficient matrix ``T_as`` (rows ``(e, i, jrel, ncm, lcm)``, columns those states) and
        the row labels.
    """
    lab = lab_states(orb, j2, parity, t2, e2max, e3max)
    blocks: dict[int, tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int, int]]]] = {}
    for n in range(e3max + 1):
        if (-1) ** n != parity:
            continue
        cols = lab.select(n)
        rows = jacobi_cm_rows(j2, parity, t2, n)
        t_as, labels = tcoef_as(
            tcoef_nas_blocked(rows, cols, j2, orb, brackets), rows, channels, t2
        )
        blocks[n] = (np.nonzero(lab.n == n)[0], t_as, labels)
    return lab, blocks


def lab_scalar(
    j2: int,
    parity: int,
    t2: int,
    orb: Orbits,
    e2max: int,
    e3max: int,
    vjac: dict[tuple[int, int], np.ndarray],
    channels: dict[tuple[int, int], JacobiChannel],
    brackets: Brackets,
) -> tuple[LabBlock, np.ndarray]:
    """Laboratory matrix ``6 sum T^T V_jac T`` of a scalar Jacobi operator in one lab channel.

    Parameters
    ----------
    vjac
        ``{(jrel, parity): V}``, each ``V`` square over the antisymmetric states of that Jacobi
        channel up to ``E <= e3max`` in the order of ``channels[...].blocks`` (``N`` ascending).
    channels
        Matching Jacobi channels (their ``cfp`` fix the gauge of ``vjac``).

    Returns
    -------
    tuple
        The lab states and the dense symmetric matrix over them (a block diagonal in nothing:
        different ``N`` blocks couple through ``V``).
    """
    lab, blocks = lab_t_blocks(j2, parity, t2, orb, e2max, e3max, channels, brackets)
    offs = {}
    for key, ch in channels.items():
        o, off = {}, 0
        for blk in ch.blocks:
            o[blk.n_quanta] = (off, off + len(blk.cfp))
            off += len(blk.cfp)
        offs[key] = o
    out = np.zeros((len(lab), len(lab)))
    for n_bra, (c_bra, t_bra, lb_bra) in blocks.items():
        for c_ket, t_ket, lb_ket in blocks.values():
            by_bra: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
            for r, (e, i, jrel, ncm, lcm) in enumerate(lb_bra):
                by_bra.setdefault((jrel, ncm, lcm), []).append((r, e, i))
            by_ket: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
            for r, (e, i, jrel, ncm, lcm) in enumerate(lb_ket):
                by_ket.setdefault((jrel, ncm, lcm), []).append((r, e, i))
            for (jrel, ncm, lcm), lst_b in by_bra.items():
                lst_k = by_ket.get((jrel, ncm, lcm))
                if not lst_k:
                    continue
                key = (jrel, (-1) ** (n_bra - 2 * ncm - lcm))
                v = vjac.get(key)
                if v is None:
                    continue
                off = offs[key]
                ib = np.array([off[e][0] + i for _, e, i in lst_b])
                ik = np.array([off[e][0] + i for _, e, i in lst_k])
                rb = np.array([r for r, _, _ in lst_b])
                rk = np.array([r for r, _, _ in lst_k])
                out[np.ix_(c_bra, c_ket)] += 6.0 * t_bra[rb].T @ v[np.ix_(ib, ik)] @ t_ket[rk]
    return lab, out
