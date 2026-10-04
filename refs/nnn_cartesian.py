"""Brute-force Cartesian check of NuHamil's local N2LO 3N terms (Phase 5.2.1; throwaway oracle).

Builds the *representative* term of the cE, cD and c1 operators directly from the physics formulas
(spin and isospin as explicit 8x8 matrices, Cartesian gradients of the regulated Yukawa, regulated
deltas, a product quadrature over both Jacobi vectors) between NAS Jacobi HO states, and compares with
the per-term NAS elements NuHamil stores with ``save_3nf_before_lec``.

Nothing here uses NuHamil's partial-wave formulas: the only shared input is the definition of the
regulator exp(-(q/Lambda)^(2n)) and the constants.  The expected ratio (NuHamil V_term) / (one ordered
term) is 1 for cE and 2 for cD and c1 if NuHamil's factor 3 replaces the sum over permutations.

Usage: python refs/nnn_cartesian.py DIR TERM [--nmax N] [--lam 400]
where DIR holds ops/N2LO_3NF_HOPW_J1P+T1_Nmax*.bin.
"""

from __future__ import annotations

import argparse
import glob
from functools import lru_cache

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import eval_genlaguerre, gammaln, sph_harm_y, spherical_jn
from sympy import Rational
from sympy.physics.wigner import clebsch_gordan

HC = 197.32705
M_P, M_N = 938.27231, 939.56563
F_PI, G_A, M_PI, LAM_CHI = 92.2, 1.29, 138.038, 700.0


@lru_cache(maxsize=None)
def cg(j1: float, m1: float, j2: float, m2: float, j: float, m: float) -> float:
    """Clebsch-Gordan coefficient <j1 m1 j2 m2 | j m> (floats, half-integers allowed)."""
    if abs(m1 + m2 - m) > 1e-9:
        return 0.0
    r = Rational
    return float(clebsch_gordan(r(j1 * 2, 2), r(j2 * 2, 2), r(j * 2, 2), r(m1 * 2, 2), r(m2 * 2, 2), r(m * 2, 2)))


# --- radial functions -----------------------------------------------------------------------------


def ho_radial(n: int, l: int, nu: float, r: np.ndarray) -> np.ndarray:
    """HO radial function with int R^2 r^2 dr = 1 and positive slope at the origin."""
    norm = np.exp(0.5 * (np.log(2.0) + (l + 1.5) * np.log(nu) + gammaln(n + 1) - gammaln(n + l + 1.5)))
    x = nu * r**2
    return norm * r**l * np.exp(-0.5 * x) * eval_genlaguerre(n, l + 0.5, x)


def regulator(q: np.ndarray, lam: float, npow: int) -> np.ndarray:
    """exp(-(q hc / Lambda)^(2n)) with q in fm^-1."""
    return np.exp(-((q * HC / lam) ** (2 * npow)))


class RadialTables:
    """Splines of the regulated delta and Yukawa derivatives on a fine r grid (fm units)."""

    def __init__(self, lam: float, npow: int, rmax: float = 40.0, nr: int = 4000, nq: int = 4000):
        mu = M_PI / HC
        r = np.linspace(1e-3, rmax, nr)
        x, w = np.polynomial.legendre.leggauss(nq)
        qmax = 10.0
        q = 0.5 * qmax * (x + 1.0)
        w = 0.5 * qmax * w
        reg = regulator(q, lam, npow)
        qr = np.outer(r, q)
        c = 1.0 / (2 * np.pi**2)
        delta = c * (spherical_jn(0, qr) * (w * q**2 * reg)).sum(1)
        g = reg / (q**2 + mu**2)
        yp = -c * (spherical_jn(1, qr) * (w * q**3 * g)).sum(1)
        # j1'(x) = j0(x) - 2 j1(x)/x
        j1p = spherical_jn(0, qr) - 2 * spherical_jn(1, qr) / qr
        ypp = -c * (j1p * (w * q**4 * g)).sum(1)
        self.delta = CubicSpline(r, delta)
        self.yp = CubicSpline(r, yp)
        self.ypp = CubicSpline(r, ypp)
        self.rmax = rmax


# --- spin, isospin ----------------------------------------------------------------------------------

PAULI = [
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
]


def kron3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Kronecker product of three single-particle operators."""
    return np.kron(np.kron(a, b), c)


def pair_state(s: int, ms: int) -> np.ndarray:
    """Two spin-1/2 coupled to ``s`` with projection ``ms``, in the basis index 0 = up, 1 = down."""
    v = np.zeros(4, dtype=complex)
    for a in (0.5, -0.5):
        for b in (0.5, -0.5):
            v[(0 if a > 0 else 1) * 2 + (0 if b > 0 else 1)] = cg(0.5, a, 0.5, b, s, ms)
    return v


def three_spin_state(s12: int, ms12: int, ms3: float) -> np.ndarray:
    """(s1 s2)s12 ms12 times spin 3 with projection ms3, an 8-vector."""
    e = np.zeros(2, dtype=complex)
    e[0 if ms3 > 0 else 1] = 1.0
    return np.kron(pair_state(s12, ms12), e)


def isospin_vector(t12: int, t2: int, mt: float = 0.5) -> np.ndarray:
    """|(t12 1/2) T MT> as an 8-vector (T = t2/2)."""
    T = t2 / 2
    v = np.zeros(8, dtype=complex)
    for ms3 in (0.5, -0.5):
        m12 = mt - ms3
        if abs(m12) > t12:
            continue
        v += cg(t12, m12, 0.5, ms3, T, mt) * three_spin_state(t12, int(round(m12)), ms3)
    return v


def tau_dot_tau(i: int, j: int) -> np.ndarray:
    """tau_i . tau_j on the 8-dimensional isospin space (i, j in 0..2)."""
    out = np.zeros((8, 8), dtype=complex)
    for a in range(3):
        ops = [np.eye(2, dtype=complex)] * 3
        ops = list(ops)
        ops[i] = PAULI[a]
        ops[j] = PAULI[a]
        out += kron3(*ops)
    return out


def sigma_ab(i: int, j: int) -> np.ndarray:
    """Array (3, 3, 8, 8) of sigma_i^a sigma_j^b."""
    out = np.zeros((3, 3, 8, 8), dtype=complex)
    for a in range(3):
        for b in range(3):
            ops = [np.eye(2, dtype=complex)] * 3
            ops = list(ops)
            ops[i] = PAULI[a]
            ops[j] = PAULI[b]
            out[a, b] = kron3(*ops)
    return out


# --- angular functions of a NAS state ---------------------------------------------------------------


def angular_spin(l12: int, s12: int, j12: int, l3: int, j3: int, j2: int, th1, th2, dphi) -> np.ndarray:
    """Spin-angle function ((l12 s12)j12 (l3 1/2)j3) J M=J on a (th1, th2, dphi) grid.

    Returns an array (n1, n2, nphi, 8).  The first azimuth is fixed to zero (invariance of the
    integrand under a joint rotation about z, valid for diagonal M).
    """
    J = j2 / 2
    M = J
    J3 = j3 / 2
    t1, t2, ph = np.meshgrid(th1, th2, dphi, indexing="ij")
    out = np.zeros(t1.shape + (8,), dtype=complex)
    for ml12 in range(-l12, l12 + 1):
        y1 = sph_harm_y(l12, ml12, t1, 0.0 * t1)
        for ms12 in range(-s12, s12 + 1):
            mj12 = ml12 + ms12
            if abs(mj12) > j12:
                continue
            c1 = cg(l12, ml12, s12, ms12, j12, mj12)
            if c1 == 0.0:
                continue
            for ml3 in range(-l3, l3 + 1):
                y2 = sph_harm_y(l3, ml3, t2, ph)
                for ms3 in (0.5, -0.5):
                    mj3 = ml3 + ms3
                    if abs(mj3) > J3 + 1e-9:
                        continue
                    c2 = cg(l3, ml3, 0.5, ms3, J3, mj3)
                    c3 = cg(j12, mj12, J3, mj3, J, M)
                    coef = c1 * c2 * c3
                    if coef == 0.0:
                        continue
                    out += (coef * y1 * y2)[..., None] * three_spin_state(s12, ms12, ms3)
    return out


# --- the matrix elements ----------------------------------------------------------------------------


def matrix(
    states: np.ndarray,
    term: str,
    j2: int,
    t2: int,
    lam: float,
    npow: int = 2,
    hw: float = 16.0,
    nr: int = 36,
    rmax: float = 10.0,
    nth: int = 24,
    nphi: int = 24,
    tables: RadialTables | None = None,
) -> np.ndarray:
    """NAS matrix of one ordered/representative term (no LEC-independent factors dropped).

    ``states`` rows are (n12, l12, s12, j12, t12, n3, l3, j3).  Returns the real symmetric matrix in MeV.
    """
    m_red = M_P * M_N / (M_P + M_N)
    nu = 2 * m_red * hw / HC**2
    tab = tables or RadialTables(lam, npow)
    xr, wr = np.polynomial.legendre.leggauss(nr)
    xi = 0.5 * rmax * (xr + 1.0)
    wr = 0.5 * rmax * wr
    xc, wc = np.polynomial.legendre.leggauss(nth)
    th = np.arccos(xc)
    dphi = 2 * np.pi * np.arange(nphi) / nphi
    wphi = 2 * np.pi / nphi
    n = len(states)
    uniq = {}
    for k, (n12, l12, s12, j12, t12, n3, l3, j3) in enumerate(states):
        uniq[k] = angular_spin(l12, s12, j12, l3, j3, j2, th, th, dphi)
    rad = np.array([ho_radial(s[0], s[1], nu, xi) for s in states])  # ξ1 functions
    rad3 = np.array([ho_radial(s[5], s[6], nu, xi) for s in states])  # ξ2 functions
    tau = tau_dot_tau(1, 2)
    iso = [isospin_vector(s[4], t2) for s in states]
    iso_me = np.array([[np.vdot(iso[a], tau @ iso[b]) for b in range(n)] for a in range(n)])
    sig = sigma_ab(1, 2)
    # geometry on the grid: indices (i1, i2, c1, c2, phi)
    ct1 = np.cos(th)[:, None, None]
    st1 = np.sin(th)[:, None, None]
    ct2 = np.cos(th)[None, :, None]
    st2 = np.sin(th)[None, :, None]
    cph = np.cos(dphi)[None, None, :]
    sph = np.sin(dphi)[None, None, :]
    n1 = np.stack(np.broadcast_arrays(st1 * 1.0, 0 * st1, ct1 * 1.0 + 0 * st1), -1)  # phi1 = 0
    n2 = np.stack(
        np.broadcast_arrays(st2 * cph, st2 * sph, ct2 + 0 * cph),
        -1,
    )
    n1 = np.broadcast_to(n1, (nth, nth, nphi, 3))
    n2 = np.broadcast_to(n2, (nth, nth, nphi, 3))
    ang_w = (wc[:, None, None] * wc[None, :, None]) * wphi * 2 * np.pi  # phi1 integral gives 2 pi
    out = np.zeros((n, n), dtype=complex)
    pref = {
        "cE": LAM_CHI**-1 * HC**6 / F_PI**4,
        "cD": -(G_A / (8 * F_PI**2)) / (F_PI**2 * LAM_CHI) * HC**6,
        "c1": 0.5 * (G_A / (2 * F_PI)) ** 2 * (-4 * M_PI**2 / F_PI**2) * HC**4 * 1e-3,
    }[term]
    # bilinears of the spin-angle functions
    if term == "cE":
        bil = np.zeros((n, n, nth, nth, nphi), dtype=complex)
        for a in range(n):
            for b in range(n):
                bil[a, b] = np.einsum("ijks,ijks->ijk", uniq[a].conj(), uniq[b])
    else:
        bil = np.zeros((n, n, 3, 3, nth, nth, nphi), dtype=complex)
        for a in range(n):
            for b in range(n):
                for x in range(3):
                    for y in range(3):
                        bil[a, b, x, y] = np.einsum("ijks,st,ijkt->ijk", uniq[a].conj(), sig[x, y], uniq[b])
    for i1 in range(nr):
        for i2 in range(nr):
            x1, x2 = xi[i1], xi[i2]
            v21 = -np.sqrt(2.0) * x1 * n1  # r2 - r1
            v31 = np.sqrt(1.5) * x2 * n2 - x1 / np.sqrt(2.0) * n1  # r3 - r1
            r21 = np.linalg.norm(v21, axis=-1)
            r31 = np.maximum(np.linalg.norm(v31, axis=-1), 1e-3)
            w = wr[i1] * wr[i2] * x1**2 * x2**2 * ang_w
            if term == "cE":
                f = tab.delta(r21) * tab.delta(r31)
                rr = np.einsum("ijk,abijk->ab", w * f, bil)
            else:
                if term == "cD":
                    u = v21 / r21[..., None]
                    yp, ypp = tab.yp(r21), tab.ypp(r21)
                    g = -(
                        u[..., :, None] * u[..., None, :] * (ypp - yp / r21)[..., None, None]
                        + np.eye(3) * (yp / r21)[..., None, None]
                    )
                    g = g * tab.delta(r31)[..., None, None]
                else:
                    u = v21 / r21[..., None]
                    u3 = v31 / r31[..., None]
                    g = -(u[..., :, None] * u3[..., None, :]) * (tab.yp(r21) * tab.yp(r31))[..., None, None]
                rr = np.einsum("ijk,ijkxy,abxyijk->ab", w, g, bil)
            ra = rad[:, i1] * rad3[:, i2]
            out += rr * np.outer(ra, ra)
    return pref * (out * iso_me).real


def read_terms(path: str) -> tuple[tuple[int, ...], np.ndarray]:
    """Read one ``N2LO_3NF_HOPW`` file: header and the five NAS lower triangles (c1 c3 c4 cD cE)."""
    raw = open(path, "rb").read()
    j, p, t, nmax = np.frombuffer(raw[:16], dtype=np.int32)
    for nd in (np.int64, np.int32):
        sz = np.dtype(nd).itemsize
        n = int(np.frombuffer(raw[16 : 16 + sz], dtype=nd)[0])
        if len(raw) == 16 + sz + 5 * 8 * n:
            data = np.frombuffer(raw[16 + sz :], dtype=np.float64).reshape(5, n)
            return (int(j), int(p), int(t), int(nmax), n), data
    raise ValueError("unrecognised file layout")


def lower_to_full(v: np.ndarray, dim: int) -> np.ndarray:
    """Unpack a lower triangle stored by rows (ket <= bra order) into a symmetric matrix."""
    m = np.zeros((dim, dim))
    k = 0
    for i in range(dim):
        for j in range(i + 1):
            m[i, j] = m[j, i] = v[k]
            k += 1
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("term", choices=["cE", "cD", "c1"])
    ap.add_argument("--n", type=int, default=2, help="NAS states of lowest N blocks to use")
    ap.add_argument("--lam", type=float, default=400.0)
    ap.add_argument("--j2", type=int, default=1)
    ap.add_argument("--t2", type=int, default=1)
    ap.add_argument("--skip", type=int, default=0, help="leading NAS states absent from NuHamil's list")
    ap.add_argument("--par", type=int, default=1)
    ap.add_argument("--nr", type=int, default=30)
    ap.add_argument("--nth", type=int, default=20)
    a = ap.parse_args()
    from nuham.nnn.jacobi import nas_states

    f = sorted(glob.glob(f"{a.dir}/ops/N2LO_3NF_HOPW_J{a.j2}P{'+' if a.par > 0 else '-'}T{a.t2}_*.bin"))[0]
    hdr, data = read_terms(f)
    nmax = hdr[3]
    blocks = [nas_states(a.j2, a.par, a.t2, nq) for nq in range(0 if a.par > 0 else 1, nmax + 1, 2)]
    allst = np.concatenate(blocks)
    k = {"c1": 0, "c3": 1, "c4": 2, "cD": 3, "cE": 4}[a.term]
    dim = int(round((np.sqrt(8 * hdr[4] + 1) - 1) / 2))  # NuHamil's list can omit states of the top N block
    print('file dim', dim, 'ours', len(allst))
    full = lower_to_full(data[k], dim)
    sel = np.arange(min(a.n, len(allst)))
    print('block sizes', [len(b) for b in blocks])
    st = allst[a.skip + sel]
    xb = matrix(st, a.term, a.j2, a.t2, a.lam, nr=a.nr, nth=a.nth, nphi=a.nth)
    nh = full[np.ix_(sel, sel)]  # file index i is our index i + skip
    np.set_printoptions(precision=6, suppress=True, linewidth=160)
    print(st)
    print("brute\n", xb)
    print("NuHamil\n", nh)
    with np.errstate(all="ignore"):
        print("ratio\n", nh / xb)
    for name, sg in {
        "(-1)^l12": (-1.0) ** st[:, 1],
        "(-1)^l3": (-1.0) ** st[:, 6],
        "(-1)^(l12+n12)": (-1.0) ** (st[:, 1] + st[:, 0]),
        "(-1)^(l12+n3)": (-1.0) ** (st[:, 1] + st[:, 5]),
        "(-1)^(l12+n12+n3)": (-1.0) ** (st[:, 1] + st[:, 0] + st[:, 5]),
        "(-1)^(n12+n3)": (-1.0) ** (st[:, 0] + st[:, 5]),
    }.items():
        d = nh - np.outer(sg, sg) * xb * (2.0 if a.term != "cE" else 1.0)
        print(f"gauge {name}: max |NuHamil - s s r brute| = {np.abs(d).max():.3e} (scale {np.abs(nh).max():.3e})")
    big = np.abs(nh) > 1e-3 * np.abs(nh).max()
    print("max |NuHamil - r*brute| / max|NuHamil| for r in 1,2:", [np.abs(nh - r * xb).max() / np.abs(nh).max() for r in (1, 2)])
    print("median ratio on large elements:", np.median((nh / xb)[big & (np.abs(xb) > 0)]))
