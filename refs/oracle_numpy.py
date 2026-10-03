#!/usr/bin/env python3
"""
SLOW THROWAWAY REFERENCE ("oracle"): a slow, throwaway numpy reference, NOT the package.

Plain numpy/scipy end-to-end re-implementation of NuHamil's NN pipeline
(bare HO relative ME -> +Coulomb(pp) -> SRG in full Nmax=100 space -> V=H-T ->
truncate e2max -> rel->lab transform -> me2j 4-column convention) used to check
specs 1-3 against the reference me2j files in refs/work.  See
docs/spec-4-end-to-end-check.md.

Env: ORACLE_SOLVER=vode (emulates NuHamil's DVODE-Adams 1e-8; default dop853 1e-10), ORACLE_SOURCE=bin|gz, ORACLE_CACHE=dir.
Usage:  python oracle_numpy.py [--n 200] [--seed 1] [--cases a,b,c] [--nproc 12]
        cases: srg20_hw16, srg20_hw20, srg18_hw16_e6
Slow pure-python Talmi-Moshinsky; compares a random sample of tuples
(or all with --n 0) and prints max |diff| per file and per column.
"""

import gzip, math, os, sys, time, random, pickle, argparse
from functools import lru_cache
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import eval_genlaguerre, gammaln
from sympy.physics.wigner import clebsch_gordan

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work") + "/"
CACHE = os.environ.get("ORACLE_CACHE", "/tmp/oracle_cache") + "/"
hc = 197.32705
mp = 938.27231
mn = 939.56563
alpha_em = 137.035999
mred = mp * mn / (mp + mn)

# ------------------------------------------------------------------ 6j / 9j (doubled-integer args)
_lf = [math.lgamma(i + 1) for i in range(400)]


def _tri(a, b, c):  # doubled args; returns log of triangle coeff or None
    if (a + b + c) % 2 or c < abs(a - b) or c > a + b:
        return None
    return 0.5 * (
        _lf[(a + b - c) // 2]
        + _lf[(a - b + c) // 2]
        + _lf[(-a + b + c) // 2]
        - _lf[(a + b + c) // 2 + 1]
    )


@lru_cache(None)
def sixj(j1, j2, j3, j4, j5, j6):
    t = [_tri(j1, j2, j3), _tri(j1, j5, j6), _tri(j4, j2, j6), _tri(j4, j5, j3)]
    if None in t:
        return 0.0
    pre = sum(t)
    a = (j1 + j2 + j3) // 2
    b = (j1 + j5 + j6) // 2
    c = (j4 + j2 + j6) // 2
    d = (j4 + j5 + j3) // 2
    e = (j1 + j2 + j4 + j5) // 2
    f = (j2 + j3 + j5 + j6) // 2
    g = (j3 + j1 + j6 + j4) // 2
    s = 0.0
    for k in range(max(a, b, c, d), min(e, f, g) + 1):
        s += (-1) ** k * math.exp(
            pre
            + _lf[k + 1]
            - _lf[k - a]
            - _lf[k - b]
            - _lf[k - c]
            - _lf[k - d]
            - _lf[e - k]
            - _lf[f - k]
            - _lf[g - k]
        )
    return s


@lru_cache(None)
def ninej(a, b, c, d, e, f, g, h, i):
    lo = max(abs(a - i), abs(d - h), abs(b - f))
    hi = min(a + i, d + h, b + f)
    s = 0.0
    for x in range(lo, hi + 1, 2):
        s += (
            (-1) ** x
            * (x + 1)
            * sixj(a, b, c, f, i, x)
            * sixj(d, e, f, b, x, h)
            * sixj(g, h, i, x, a, d)
        )
    return s


# ------------------------------------------------------------------ Moshinsky bracket (transcribed from MyLibrary gmosh, d=1)
def df(n):
    r = 1
    while n > 1:
        r *= n
        n -= 2
    return r


def tri(i, j, k):
    return df(i) / (df(j) * df(k))


@lru_cache(None)
def cg0(la, lb, l):
    return float(clebsch_gordan(la, lb, l, 0, 0, 0))


def g(e1, l1, ea, la, eb, lb):
    if l1 < abs(la - lb) or l1 > la + lb:
        return 0.0
    return cg0(la, lb, l1) * math.sqrt(
        (2 * la + 1)
        * (2 * lb + 1)
        * tri(e1 - l1, ea - la, eb - lb)
        * tri(e1 + l1 + 1, ea + la + 1, eb + lb + 1)
    )


def triag(a, b, c):
    return c < abs(a - b) or c > a + b


@lru_cache(None)
def gmosh(nl, ll, nr, lr, n1, l1, n2, l2, lm, d=1.0):
    ee = 2 * nl + ll
    er = 2 * nr + lr
    e1 = 2 * n1 + l1
    e2 = 2 * n2 + l2
    if ee + er != e1 + e2 or triag(ll, lr, lm) or triag(l1, l2, lm):
        return 0.0
    t = math.sqrt(d ** (e1 - er) / (1 + d) ** (e1 + e2))
    m = min(er, e2)
    s = 1.0
    r = 0.0
    for ed in range(m + 1):
        eb = er - ed
        ec = e2 - ed
        ea = e1 - er + ed
        if ea < 0:
            s *= -d
            continue
        for ld in range(ed, -1, -2):
            for lb in range(eb, -1, -2):
                if triag(ld, lb, lr):
                    continue
                for lc in range(ec, -1, -2):
                    if triag(ld, lc, l2):
                        continue
                    for la in range(ea, -1, -2):
                        if triag(la, lb, l1) or triag(la, ll, lc):
                            continue
                        r += (
                            s
                            * t
                            * ninej(
                                2 * la,
                                2 * lb,
                                2 * l1,
                                2 * lc,
                                2 * ld,
                                2 * l2,
                                2 * ll,
                                2 * lr,
                                2 * lm,
                            )
                            * g(e1, l1, ea, la, eb, lb)
                            * g(e2, l2, ec, lc, ed, ld)
                            * g(ee, ll, ea, la, ec, lc)
                            * g(er, lr, eb, lb, ed, ld)
                        )
        s *= -d
    return r * (-1) ** (n1 + n2 + nr + nl)


# ------------------------------------------------------------------ stage 1: bare relative file
def read_rel_file(hw):
    """-> dict (J,P,S,Z) -> (states list [(n,l)], symmetric V matrix) from NNint_A2_rel_*_bare_Nmax100_hw{hw}.gz"""
    fn = W + "NNint_A2_rel_N3LO_EM500_bare_Nmax100_hw%d.gz" % hw
    with gzip.open(fn, "rt") as f:
        f.readline()
        a = np.array(f.read().split(), dtype=float).reshape(-1, 9)
    ich = a[:, :8].astype(int)
    val = a[:, 8]
    out = {}
    keys = ich[:, :4]
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    for ci, key in enumerate(uniq):
        sel = np.where(inv == ci)[0]
        rows = ich[sel]
        st = sorted(
            {(r[4], r[5]) for r in rows} | {(r[6], r[7]) for r in rows}, key=lambda x: (x[1], x[0])
        )
        idx = {s: i for i, s in enumerate(st)}
        i1 = np.array([idx[(r[4], r[5])] for r in rows])
        i2 = np.array([idx[(r[6], r[7])] for r in rows])
        V = np.zeros((len(st), len(st)))
        V[i1, i2] = val[sel]
        V[i2, i1] = val[sel]
        out[tuple(int(x) for x in key)] = (st, V)
    return out


BIN = "/home/soham/Projects/nuclear-physics/NuHamil-public/input_nn_files/N3LO_EM500_kmax8_N100_Jmax8.bin"


def read_rel_bin(hw, N2max=100):
    """Spec 1: project the momentum-space .bin onto HO relative states in FULL precision (what NuHamil does when the
    NNint_A2_rel cache file does not exist; the cache file itself is rounded to 7 digits)."""
    import struct

    b = open(BIN, "rb").read()
    N, Jm, NC = struct.unpack("<3i", b[:12])
    k = np.frombuffer(b, "<f8", N, 12)
    w = np.frombuffer(b, "<f8", N, 12 + 8 * N)
    off = 12 + 16 * N
    b2 = hc**2 / (mred * hw)
    out = {}
    while off < len(b):
        j, p, s, z, nd = struct.unpack("<5i", b[off : off + 20])
        off += 20
        m = np.frombuffer(b, "<f8", nd * nd, off).reshape(nd, nd)
        off += 8 * nd * nd
        ls = [
            l
            for l in range(abs(j - s), j + s + 1)
            if (-1) ** l == p and not (abs(z) == 1 and (-1) ** (l + s) == -1)
        ]
        st = [(n, l) for l in ls for n in range(N2max // 2 + 1) if 2 * n + l <= N2max]
        O = np.zeros((len(ls) * N, len(st)))
        for i, (n, l) in enumerate(st):
            bi = ls.index(l)
            O[bi * N : (bi + 1) * N, i] = k * w * ho_radial(n, l, b2, k) * (-1) ** n
        out[(j, p, s, z)] = (st, O.T @ m @ O)
    return out


def ho_radial(n, l, anu, r):
    nu = math.sqrt(anu)
    x = nu * r
    return (
        math.sqrt(2 * nu)
        * np.exp(
            0.5 * gammaln(n + 1) - 0.5 * gammaln(n + l + 1.5) - 0.5 * x * x + (l + 1) * np.log(x)
        )
        * eval_genlaguerre(n, l + 0.5, x * x)
    )


def coulomb_matrix(st, hw):
    """add_coulomb_force: 500-pt GL on [0,40], a = 0.5 m_p hw / hc^2 ; diagonal in l"""
    xg, wg = np.polynomial.legendre.leggauss(500)
    r = 20 * (xg + 1)
    w = 20 * wg
    a = 0.5 * mp * hw / hc**2
    cache = {s: ho_radial(s[0], s[1], a, r) for s in st}
    C = np.zeros((len(st), len(st)))
    for i, s1 in enumerate(st):
        for j, s2 in enumerate(st):
            if s1[1] == s2[1]:
                C[i, j] = np.sum(cache[s1] * cache[s2] * w * (hc / alpha_em) / r)
    return C


# ------------------------------------------------------------------ stage 2: SRG
def kinetic(st, hw):
    n = len(st)
    T = np.zeros((n, n))
    for i, (n1, l1) in enumerate(st):
        for k, (n2, l2) in enumerate(st):
            if l1 != l2 or abs(n1 - n2) > 1:
                continue
            if n1 == n2:
                x = 2 * n1 + l1 + 1.5
            else:
                m = min(n1, n2)
                x = math.sqrt((m + 1) * (m + l1 + 1.5))
            T[i, k] = 0.5 * hw * x
    return T


SOURCE = os.environ.get(
    "ORACLE_SOURCE", ""
)  # '' = per-case default (see CASES); or force 'bin' / 'gz'
SOLVER = os.environ.get(
    "ORACLE_SOLVER", "dop853"
)  # 'vode' emulates NuHamil's integrator (rtol=atol=1e-8)


def srg_flow(H0, T, s_final, rtol=1e-10, atol=1e-10):
    n = len(H0)
    iu = np.tril_indices(n)

    def unpack(v):
        H = np.zeros((n, n))
        H[iu] = v
        return H + H.T - np.diag(np.diag(H))

    def rhs(s, v):
        H = unpack(v)
        eta = T @ H - H @ T
        return (eta @ H - H @ eta)[iu]

    if (
        SOLVER == "vode"
    ):  # emulate NuHamil: DVODE Adams (MF=10), rtol=atol=1e-8, packed lower triangle (same ordering as Fortran)
        from scipy.integrate import ode

        r = ode(lambda t, v: rhs(t, v)).set_integrator(
            "vode", method="adams", rtol=rtol, atol=atol, nsteps=50000, order=12
        )
        r.set_initial_value(H0[iu], 0.0)
        y = r.integrate(s_final)
        assert r.successful()
        return unpack(y)
    sol = solve_ivp(rhs, [0, s_final], H0[iu], method="DOP853", rtol=rtol, atol=atol)
    assert sol.success
    return unpack(sol.y[:, -1])


def evolved_rel(hw, lam, coulomb=True, evolve_coulomb=True, nmaxJ=8, tol=None, source=None):
    source = source or SOURCE
    tol = tol or (1e-8 if SOLVER == "vode" else 1e-10)
    """-> dict (J,P,S,Z) -> (idx dict, V_evolved).  Full N=100 space flow per channel, V = H - T."""
    os.makedirs(CACHE, exist_ok=True)
    fn = CACHE + "rel_hw%d_lam%.2f_c%d%d_tol%g_%s_%s.pkl" % (
        hw,
        lam,
        coulomb,
        evolve_coulomb,
        tol,
        SOLVER,
        source,
    )
    if os.path.exists(fn):
        return pickle.load(open(fn, "rb"))
    bare = read_rel_bin(hw) if source == "bin" else read_rel_file(hw)
    lc = lam * hc / math.sqrt(2 * mred)
    s_final = lc**-4
    out = {}
    for key, (st, V) in bare.items():
        J, P, S, Z = key
        T = kinetic(st, hw)
        C = coulomb_matrix(st, hw) if (coulomb and Z == -1) else 0.0
        if evolve_coulomb:
            V = V + C
        Vev = srg_flow(T + V, T, s_final, tol, tol) - T if lam > 0 else V
        if not evolve_coulomb:
            Vev = Vev + C  # ablation only (NuHamil default evolves Coulomb)
        out[key] = ({s: i for i, s in enumerate(st)}, Vev)
    pickle.dump(out, open(fn, "wb"))
    return out


# ------------------------------------------------------------------ stage 3: rel -> lab
def orbits(emax):
    o = []
    for e in range(emax + 1):
        for l in range(e + 1):
            if (e - l) % 2:
                continue
            for tj in (2 * l - 1, 2 * l + 1):
                if tj < 1:
                    continue
                o.append((e, (e - l) // 2, l, tj))
    return o


def tuples(emax, e2max):
    o = orbits(emax)
    out = []
    for a in range(len(o)):
        for b in range(a + 1):
            if o[a][0] + o[b][0] > e2max:
                continue
            for c in range(a + 1):
                for d in range((b if c == a else c) + 1):
                    if o[c][0] + o[d][0] > e2max:
                        continue
                    if (o[a][2] + o[b][2] + o[c][2] + o[d][2]) % 2:
                        continue
                    for J in range(
                        max(abs(o[a][3] - o[b][3]), abs(o[c][3] - o[d][3])) // 2,
                        min(o[a][3] + o[b][3], o[c][3] + o[d][3]) // 2 + 1,
                    ):
                        out.append((a, b, c, d, J))
    return o, out


REL = None  # global set per case (inherited by fork)
JMAX_NN = 8


def relME(j, s, z, n1, l1, n2, l2):
    if j > JMAX_NN:
        return 0.0  # J2max_NNint=8: relative J=9 channels are exactly zero
    ent = REL.get((j, (-1) ** l1, s, z))
    if ent is None:
        return 0.0
    idx, V = ent
    return V[idx[(n1, l1)], idx[(n2, l2)]]


@lru_cache(None)
def tcoef(a, b, Z, J, nrel, lrel, s, jrel, ncm, lcm):
    (ea, na, la, ja), (eb, nb, lb, jb) = a, b
    delab = 1 / math.sqrt(2) if (a == b and Z != 0) else 1.0
    ft = (1 + (-1) ** (la + lb + s - lcm)) / math.sqrt(2) if abs(Z) == 1 else 1.0
    if ft == 0:
        return 0.0
    tc = 0.0
    for lam in range(
        max(abs(lcm - lrel), abs(J - s), abs(la - lb)), min(lcm + lrel, J + s, la + lb) + 1
    ):
        tc += (
            (-1) ** (lcm + lrel + s + J)
            * math.sqrt((2 * lam + 1) * (2 * s + 1) * (ja + 1) * (jb + 1))
            * ninej(2 * la, 2 * lb, 2 * lam, 1, 1, 2 * s, ja, jb, 2 * J)
            * math.sqrt((2 * lam + 1) * (2 * jrel + 1))
            * sixj(2 * lcm, 2 * lrel, 2 * lam, 2 * s, 2 * J, 2 * jrel)
            * gmosh(ncm, lcm, nrel, lrel, na, la, nb, lb, lam)
            * ft
            * delab
        )
    return tc


def labME(a, b, c, d, Z, J):
    Nab = a[0] + b[0]
    Ncd = c[0] + d[0]
    P = (-1) ** (a[2] + b[2])
    if P != (-1) ** (c[2] + d[2]):
        return 0.0
    tot = 0.0
    for lcm in range(0, max(Nab, Ncd) + 1):
        for ncm in range(0, max(Nab, Ncd) // 2 + 1):
            Eb = Nab - 2 * ncm - lcm
            Ek = Ncd - 2 * ncm - lcm
            if Eb < 0 or Ek < 0:
                continue
            for s in (0, 1):
                for jrel in range(0, min(J + lcm, 13) + 1):
                    if triag(lcm, jrel, J):
                        continue
                    for nb_ in range(Eb // 2 + 1):
                        lb_ = Eb - 2 * nb_
                        if (-1) ** (lb_ + lcm) != P or triag(lb_, s, jrel):
                            continue
                        if abs(Z) == 1 and (lb_ + s) % 2:
                            continue
                        t1 = tcoef(a, b, Z, J, nb_, lb_, s, jrel, ncm, lcm)
                        if t1 == 0:
                            continue
                        for nk_ in range(Ek // 2 + 1):
                            lk_ = Ek - 2 * nk_
                            if (-1) ** (lk_ + lcm) != P or triag(lk_, s, jrel):
                                continue
                            if abs(Z) == 1 and (lk_ + s) % 2:
                                continue
                            t2 = tcoef(c, d, Z, J, nk_, lk_, s, jrel, ncm, lcm)
                            if t2 == 0:
                                continue
                            tot += t1 * t2 * relME(jrel, s, Z, nb_, lb_, nk_, lk_)
    return tot


ORB = None
TUP = None


def entry(i):
    """four me2j values for tuple i: [T=0, nn, pn T=1, pp]"""
    a, b, c, d, J = TUP[i]
    A, B, C, D = ORB[a], ORB[b], ORB[c], ORB[d]
    pp = labME(A, B, C, D, -1, J)
    nn = labME(A, B, C, D, 1, J)
    ph = lambda X, Y: -((-1) ** ((X[3] + Y[3]) // 2 - J))
    pn = lambda X, Y, Xp, Yp: labME(X, Y, Xp, Yp, 0, J)
    x1 = pn(A, B, C, D)
    x2 = pn(B, A, D, C) * ph(A, B) * ph(C, D)
    x3 = pn(A, B, D, C) * ph(C, D)
    x4 = pn(B, A, C, D) * ph(A, B)
    nf = (math.sqrt(2) if a == b else 1.0) * (math.sqrt(2) if c == d else 1.0)
    return np.array([0.5 * (x1 + x2 - x3 - x4), nn * nf, 0.5 * (x1 + x2 + x3 + x4), pp * nf])


def read_me2j(fn):
    with gzip.open(fn, "rt") as f:
        f.readline()
        return np.array(f.read().split(), dtype=float).reshape(-1, 4)


# last field: where the REFERENCE run got its bare relative V: 'bin' = projected in memory from the momentum .bin at full
# precision (cache file did not exist yet), 'gz' = read back the 7-digit-rounded NNint_A2_rel cache (see logs: "Read from file").
CASES = {  # name: (file, hw, lambda, emax, e2max, source)
    "srg20_hw16": (
        "TwBME-HO_NN-only_N3LO_EM500_srg2.0_hw16_emax4_e2max8.me2j.gz",
        16,
        2.0,
        4,
        8,
        "bin",
    ),
    "srg20_hw20": (
        "TwBME-HO_NN-only_N3LO_EM500_srg2.0_hw20_emax4_e2max8.me2j.gz",
        20,
        2.0,
        4,
        8,
        "bin",
    ),
    "srg18_hw16_e6": (
        "TwBME-HO_NN-only_N3LO_EM500_srg1.8_hw16_emax6_e2max12.me2j.gz",
        16,
        1.8,
        6,
        12,
        "gz",
    ),
}
COLS = ["T=0", "nn", "pn T=1", "pp"]


def run_case(name, n, seed, nproc, report_top=5, **kw):
    global REL, ORB, TUP
    fn, hw, lam, emax, e2max, src = CASES[name]
    t0 = time.time()
    kw.setdefault("source", SOURCE or src)
    REL = evolved_rel(hw, lam, **kw)
    print("[%s] SRG done %.1fs, %d channels" % (name, time.time() - t0, len(REL)), flush=True)
    ref = read_me2j(W + fn)
    ORB, TUP = orbits(emax), None
    ORB, TUP = tuples(emax, e2max)
    assert len(TUP) == len(ref), (len(TUP), len(ref))
    idx = list(range(len(TUP)))
    if n and n < len(idx):
        random.seed(seed)
        idx = sorted(random.sample(idx, n))
    import multiprocessing as mp_

    with mp_.get_context("fork").Pool(nproc) as pool:
        res = pool.map(entry, idx, chunksize=4)
    res = np.array(res)
    d = np.abs(res - ref[idx])
    print("[%s] compared %d/%d tuples in %.0fs" % (name, len(idx), len(TUP), time.time() - t0))
    print(
        "  max|diff| per column",
        dict(zip(COLS, ["%.2e" % x for x in d.max(0)])),
        " overall %.2e" % d.max(),
    )
    print("  max|ref|  per column", ["%.2f" % x for x in np.abs(ref[idx]).max(0)])
    flat = np.argsort(d.ravel())[::-1][:report_top]
    for f in flat:
        i, c = divmod(f, 4)
        a, b, cc, dd, J = TUP[idx[i]]
        print(
            "   worst: tuple",
            (ORB[a][1:], ORB[b][1:], ORB[cc][1:], ORB[dd][1:], J),
            COLS[c],
            "oracle %.8f ref %.8f" % (res[i, c], ref[idx[i], c]),
        )
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--nproc", type=int, default=12)
    a = ap.parse_args()
    for c in a.cases.split(","):
        run_case(c, a.n, a.seed, a.nproc)
