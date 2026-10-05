"""Angular coefficients of the local chiral N2LO three-nucleon force (Phase 5.2.3).

For two pair-antisymmetric channels ``alpha = (l12, s12, j12, t12; l3, j3)`` of one ``(J, P, T)``
the non-antisymmetrised Jacobi matrix element of the representative term ``W_1`` is

    V(alpha, alpha'; n12 n3, n45 n6) = sum_a C_a(alpha, alpha') <n12 l12 n3 l3| g_a |n45 l45 n6 l6>

where ``g_a(xi1, xi2)`` are the radial functions of :mod:`nuham.nnn.local_force.radial` and the
coefficients ``C_a`` below hold all Racah algebra, spin-isospin factors and constants (LECs = 1;
multiply c1, c3, c4 by their value in GeV^-1). The formulas are Eqs. (15), (38), (42), (47) and (49)
of P. Navratil, Few-Body Syst. 41, 117 (2007), in that paper's notation: pair quantum numbers
``l s j t``, third nucleon ``L = l3`` and ``J3 = j3``, total ``J`` and ``T``. The paper's
``X``, ``V``, ``R``, ``Y``, ``Z``, ``K1..K4`` are the intermediate ranks.

All half-integers are passed as doubled integers (``j3``, ``j2``, ``t2``); the rest are integers.
The columns of the coefficient tables label the radial functions (see :func:`radial_keys`).

The sums of the paper are evaluated for many channel pairs at once: the nested ranges (``x``, ``r``,
``v``, ``y``, ``z``, ``K5``) are expanded into flat arrays of index tuples with NumPy, every factor
is computed on those arrays, and the products are accumulated per pair and radial function. The 6j
and 9j symbols come from their cached scalar functions, one call per distinct argument tuple. This
replaced nested Python loops (kept, slow, in ``refs/local3n_angular_scalar.py`` as the reference of
``tests/test_local3n_angular.py``) that took 95% of the force time of a production channel.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from math import comb, factorial, lgamma, sqrt
from typing import NamedTuple

import numpy as np

from nuham.angmom.racah import ninej, sixj
from nuham.constants import F_PI, G_A, HBARC, LAMBDA_CHI, M_PI

TERMS = ("c1", "c3", "c4", "cD", "cE")


class Channel(NamedTuple):
    """Quantum numbers of a pair-antisymmetric Jacobi channel (``j3`` doubled)."""

    l12: int
    s12: int
    j12: int
    t12: int
    l3: int
    j3: int


@cache
def six(a: int, b: int, c: int, d: int, e: int, f: int) -> float:
    """Return the 6j symbol of doubled arguments (cached)."""
    return sixj(a, b, c, d, e, f)


@cache
def nine(a: int, b: int, c: int, d: int, e: int, f: int, g: int, h: int, i: int) -> float:
    """Return the 9j symbol of doubled arguments (cached)."""
    return ninej(a, b, c, d, e, f, g, h, i)


@cache
def cg0(a: int, b: int, c: int) -> float:
    """Clebsch-Gordan coefficient ``<a 0 b 0 | c 0>`` for integer ``a, b, c``."""
    g = a + b + c
    if g % 2 or c < abs(a - b) or c > a + b:
        return 0.0
    h = g // 2
    delta = factorial(g - 2 * a) * factorial(g - 2 * b) * factorial(g - 2 * c) / factorial(g + 1)
    three_j = (
        (-1.0) ** h
        * sqrt(delta)
        * factorial(h)
        / (factorial(h - a) * factorial(h - b) * factorial(h - c))
    )
    return (-1.0) ** (a - b) * sqrt(2 * c + 1) * three_j


def hat(x: int) -> float:
    """``sqrt(2 x + 1)`` for an integer ``x``."""
    return sqrt(2 * x + 1)


def tau2_dot_tau3(t: int, tp: int, t2: int) -> float:
    """Isospin factor ``hat(t) hat(t') (-1)^(t+t'+T+1/2) {t t' 1; 1/2 1/2 1/2} {..; 1/2 1/2 T}``.

    Six times this is the reduced matrix element of ``tau_2 . tau_3`` between ``(t, 1/2) T`` states.
    """
    sign = (-1.0) ** (t + tp + (t2 + 1) // 2)
    return (
        hat(t) * hat(tp) * sign * six(2 * t, 2 * tp, 2, 1, 1, 1) * six(2 * t, 2 * tp, 2, 1, 1, t2)
    )


def tau_cross(t: int, tp: int, t2: int) -> float:
    """Isospin factor of ``tau_1 . (tau_2 x tau_3)`` (zero for ``T = 3/2``)."""
    sign = (-1.0) ** (tp + (t2 + 1) // 2)
    return (
        hat(t)
        * hat(tp)
        * sign
        * six(2 * t, 2 * tp, 2, 1, 1, t2)
        * nine(1, 1, 2 * tp, 1, 1, 2 * t, 2, 2, 2)
    )


def spin_factor(s: int, sp: int) -> float:
    """Spin factor ``hat(s) hat(s') {s s' 1; 1/2 1/2 1/2}`` of the sigma_2, sigma_3 terms."""
    return hat(s) * hat(sp) * six(2 * s, 2 * sp, 2, 1, 1, 1)


def _lec_free_constants() -> dict[str, float]:
    """Overall constants with the LEC set to 1 (MeV and fm units; c1, c3, c4 for 1 GeV^-1)."""
    return {
        "c1": -36.0 * M_PI**2 * G_A**2 * HBARC**4 * 1e-3 / F_PI**4,
        "c3": 18.0 * G_A**2 * HBARC**6 * 1e-3 / F_PI**4,
        "c4": -324.0 * G_A**2 * HBARC**6 * 1e-3 / F_PI**4,
        "cD": -9.0 * G_A * HBARC**6 / (F_PI**4 * LAMBDA_CHI),
        "cE": 6.0 * HBARC**6 / (F_PI**4 * LAMBDA_CHI),
    }


def radial_keys(term: str, x_max: int) -> list[tuple[int, ...]]:
    """Labels of the radial functions ``g_a`` of ``term`` (see ``radial.radial_function``)."""
    xs = range(x_max + 1)
    if term == "cE":
        return [(x,) for x in xs]
    if term == "cD":
        return [(k, x) for k in (0, 2) for x in xs]
    if term == "c1":
        return [(k3, x) for k3 in (0, 1) for x in xs]
    return [(k1, k2, k3, x) for k1 in (0, 2) for k2 in (0, 2) for k3 in range(k2 + 1) for x in xs]


# --- vectorised evaluation ----------------------------------------------------------------------

_LOG_FACT = np.array([lgamma(i + 1) for i in range(512)])
COMBOS = tuple((k1, k2, k3) for k1 in (0, 2) for k2 in (0, 2) for k3 in range(k2 + 1))
"""``(K1, K2, K3)`` of the c3 and c4 radial functions, in the order of :func:`radial_keys`."""


def _tpe_ranks() -> list[tuple[int, int, float]]:
    """Pairs ``(K1, K2)`` with weight ``(-1)^((K1+K2)/2) hat(K1) hat(K2) <1010|K10><1010|K20>``."""
    return [
        (k1, k2, (-1.0) ** ((k1 + k2) // 2) * hat(k1) * hat(k2) * cg0(1, 1, k1) * cg0(1, 1, k2))
        for k1 in (0, 2)
        for k2 in (0, 2)
    ]


_TPE_WEIGHT = {(k1, k2): w for k1, k2, w in _tpe_ranks()}


def n_keys(term: str, x_max: int) -> int:
    """Return the number of radial functions (columns of the coefficient table) of ``term``."""
    return len(radial_keys(term, x_max))


def _expand(count: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Parent index and running offset of ``count[i]`` children per parent, parents in order."""
    parent = np.repeat(np.arange(len(count)), count)
    first = np.cumsum(count) - count
    return parent, np.arange(parent.size) - np.repeat(first, count)


def _span(lo: np.ndarray, hi: np.ndarray, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Parent index and value of ``lo[i], lo[i] + step, ... <= hi[i]`` for every parent."""
    parent, off = _expand(np.maximum((hi - lo) // step + 1, 0))
    return parent, lo[parent] + step * off


def _span_parity(
    lo: np.ndarray, hi: np.ndarray, par: np.ndarray, par2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Values ``lo <= n <= hi`` with ``n % 2 == par % 2`` and ``n % 2 == par2 % 2`` (step 2).

    Parents whose two parities differ get no children.
    """
    par = np.asarray(par) % 2
    lo = lo + ((par - lo) % 2)
    hi = np.where(par == np.asarray(par2) % 2, hi, lo - 2)
    return _span(lo, hi, 2)


def _sign(n: np.ndarray) -> np.ndarray:
    """Return ``(-1)^n`` for an integer array."""
    return 1.0 - 2.0 * (np.asarray(n) % 2)


def _cg0_v(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Vector form of :func:`cg0` (zero where the coupling is not allowed)."""
    g = a + b + c
    ok = (g % 2 == 0) & (c >= np.abs(a - b)) & (c <= a + b)
    g, a, b, c = (np.where(ok, x, 0) for x in (g, a, b, c))
    h = g // 2
    log_three = 0.5 * (
        _LOG_FACT[g - 2 * a] + _LOG_FACT[g - 2 * b] + _LOG_FACT[g - 2 * c] - _LOG_FACT[g + 1]
    ) + (_LOG_FACT[h] - _LOG_FACT[h - a] - _LOG_FACT[h - b] - _LOG_FACT[h - c])
    value = _sign(h + a - b) * np.sqrt(2.0 * c + 1.0) * np.exp(log_three)
    return np.where(ok, value, 0.0)


@cache
def _fields(bits: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Return ``(shift, width)`` of each packed argument."""
    shifts = np.cumsum((0, *bits[:-1]))
    return tuple((int(s), w) for s, w in zip(shifts, bits, strict=True))


def _lookup(fn: Callable[..., float], bits: tuple[int, ...], *args: np.ndarray | int) -> np.ndarray:
    """Evaluate the cached scalar ``fn`` on arrays, once per distinct argument tuple.

    The tuples are packed into one integer per entry (``bits`` bits per argument) so that
    ``np.unique`` finds the distinct ones; the arguments must be non-negative and fit.
    """
    shape = np.broadcast(*args).shape
    key = np.zeros(shape, dtype=np.int64)
    shift = 0
    for arg, width in zip(args, bits, strict=True):
        arg = np.broadcast_to(np.asarray(arg, dtype=np.int64), shape)
        if arg.size and (arg.min() < 0 or arg.max() >= 1 << width):
            raise ValueError(f"argument outside 0..{(1 << width) - 1} in a Racah lookup")
        key |= arg << shift
        shift += width
    uniq, inverse = np.unique(key.ravel(), return_inverse=True)
    columns = [((uniq >> s) & ((1 << w) - 1)).tolist() for s, w in _fields(bits)]
    vals = np.array([fn(*row) for row in zip(*columns, strict=True)])
    return vals[inverse].reshape(shape)


def _six_v(*args: np.ndarray | int) -> np.ndarray:
    """6j symbols of doubled-integer arrays."""
    return _lookup(six, (8,) * 6, *args)


def _nine_orbit_v(r: np.ndarray, z: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``9j{r 1 z; l3 1/2 j3; l3' 1/2 j3'}`` of the third nucleon, for pair arrays ``a, b``."""
    one = np.ones_like(r)
    return _lookup(
        nine, (8, 2, 8, 8, 1, 8, 8, 1, 8),
        2 * r, 2 * one, 2 * z, 2 * a[:, 4], one, a[:, 5], 2 * b[:, 4], one, b[:, 5],
    )  # fmt: skip


def _nine_pair_v(
    r: np.ndarray, v: np.ndarray, z: np.ndarray, a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """9j symbols recoupling ``(l s) j`` and ``(L 1/2) J3`` to ranks ``(v,1,z)``, ``(r,1,z)``."""
    one = np.ones_like(v)
    pair = _lookup(
        nine, (8, 2, 8, 8, 2, 8, 8, 2, 8),
        2 * v, 2 * one, 2 * z, 2 * a[:, 0], 2 * a[:, 1], 2 * a[:, 2],
        2 * b[:, 0], 2 * b[:, 1], 2 * b[:, 2],
    )  # fmt: skip
    return pair * _nine_orbit_v(r, z, a, b)


def _prefactor(term: str, a: np.ndarray, b: np.ndarray, j2: int, t2: int) -> np.ndarray:
    """Overall factor of ``term`` of every pair (LECs = 1): constants, isospin, spin, dimensions."""
    la, sa, ja, ta, l3a, j3a = (a[:, i] for i in range(6))
    lb, sb, jb, tb, l3b, j3b = (b[:, i] for i in range(6))
    const = _lec_free_constants()[term]
    cross = term == "c4"
    iso = _lookup(
        lambda t, tp: tau_cross(t, tp, t2) if cross else tau2_dot_tau3(t, tp, t2), (2, 2), ta, tb
    )
    dims = np.sqrt(
        (2.0 * ja + 1)
        * (2.0 * jb + 1)
        * (j3a + 1.0)
        * (j3b + 1.0)
        * (2.0 * lb + 1)
        * (2.0 * l3b + 1)
    )
    if term == "cE":
        return const * iso * dims * _sign((j2 - 1 + j3b - j3a) // 2 + la + l3a + sa) * (sa == sb)
    if term == "c4":
        phase = _sign((j2 + j3a) // 2 + jb)
        return const * iso * np.sqrt((2.0 * sa + 1) * (2.0 * sb + 1)) * dims * phase
    spin = _lookup(spin_factor, (2, 2), sa, sb)
    return const * iso * spin * dims * _sign((j2 - j3a) // 2 + sa + jb)


def _ce_block(a: np.ndarray, b: np.ndarray, j2: int, x_max: int) -> np.ndarray:
    """Contact term, Eq. (15), for pairs with non-zero prefactor: shape ``(n, x_max + 1)``."""
    la, sa, ja, _, l3a, j3a = (a[:, i] for i in range(6))
    lb, sb, jb, _, l3b, j3b = (b[:, i] for i in range(6))
    lo = np.maximum(np.maximum(np.abs(la - lb), np.abs(l3a - l3b)), np.abs(ja - jb))
    p, x = _span_parity(lo, np.full(len(a), x_max), la + lb, l3a + l3b)
    w = (
        _sign(x)
        * (2 * x + 1)
        * _six_v(2 * x, 2 * lb[p], 2 * la[p], 2 * sa[p], 2 * ja[p], 2 * jb[p])
        * _six_v(2 * x, 2 * ja[p], 2 * jb[p], j2, j3b[p], j3a[p])
        * _six_v(2 * x, 2 * l3b[p], 2 * l3a[p], 1, j3a[p], j3b[p])
        * _cg0_v(lb[p], x, la[p])
        * _cg0_v(l3b[p], x, l3a[p])
    )
    out = np.bincount(p * (x_max + 1) + x, weights=w, minlength=len(a) * (x_max + 1))
    return out.reshape(len(a), x_max + 1)


def _cd_block(a: np.ndarray, b: np.ndarray, j2: int, x_max: int) -> np.ndarray:
    """One-pion-exchange contact term, Eq. (38): shape ``(n, 2, x_max + 1)`` for ``K = 0, 2``."""
    la, sa, ja, _, l3a, j3a = (a[:, i] for i in range(6))
    lb, sb, jb, _, l3b, j3b = (b[:, i] for i in range(6))
    n, width = len(a), x_max + 1
    total = np.zeros(n * 2 * width)
    for ik, k in enumerate((0, 2)):
        fk = sqrt(2 * k + 1) * (-1.0) ** (k // 2) * cg0(1, 1, k)
        pv, v = _span(np.abs(la - lb), la + lb, 2)
        lo = np.maximum(np.abs(l3a - l3b)[pv], np.abs(v - k))
        hi = np.minimum(np.minimum((l3a + l3b)[pv], v + k), x_max)
        i, x = _span_parity(lo, hi, (l3a + l3b)[pv], v + k)
        pv, v = pv[i], v[i]
        wv = (
            _sign(v)
            * np.sqrt(2.0 * v + 1)
            * _cg0_v(v, lb[pv], la[pv])
            * (2 * x + 1)
            * _cg0_v(x, np.full_like(x, k), v)
            * _cg0_v(x, l3b[pv], l3a[pv])
        )
        keep = wv != 0.0
        pv, v, x, wv = pv[keep], v[keep], x[keep], wv[keep]
        zlo = np.maximum(
            np.maximum(np.abs(v - 1), np.abs(ja - jb)[pv]),
            np.maximum(np.abs(x - 1), (np.abs(j3a - j3b) // 2)[pv]),
        )
        zhi = np.minimum(
            np.minimum(v + 1, (ja + jb)[pv]), np.minimum(x + 1, ((j3a + j3b) // 2)[pv])
        )
        i, z = _span(zlo, zhi)
        pv, v, x, wv = pv[i], v[i], x[i], wv[i]
        wz = (
            (2 * z + 1)
            * _nine_pair_v(x, v, z, a[pv], b[pv])
            * _six_v(2 * z, 2 * ja[pv], 2 * jb[pv], j2, j3b[pv], j3a[pv])
            * _six_v(2 * v, 2, 2 * z, 2, 2 * x, 2 * k)
        )
        idx = (pv * 2 + ik) * width + x
        total += np.bincount(idx, weights=fk * wv * wz, minlength=n * 2 * width)
    return total.reshape(n, 2, width)


def _c1_block(a: np.ndarray, b: np.ndarray, j2: int, x_max: int) -> np.ndarray:
    """Two-pion exchange c1, Eq. (42): shape ``(n, 2, x_max + 1)`` for ``K3 = 0, 1``."""
    la, sa, ja, _, l3a, j3a = (a[:, i] for i in range(6))
    lb, sb, jb, _, l3b, j3b = (b[:, i] for i in range(6))
    n, width = len(a), x_max + 1
    out = np.zeros((n, 2, width))
    for k3 in (0, 1):
        k4 = 1 - k3
        p = np.repeat(np.arange(n), width)
        x = np.tile(np.arange(width), n)
        lo = np.maximum(np.abs(l3a - l3b)[p], np.abs(x - k4))
        hi = np.minimum((l3a + l3b)[p], x + k4)
        i, r = _span_parity(lo, hi, (l3a + l3b)[p], x + k4)
        p, x = p[i], x[i]
        i, v = _span(np.abs(la - lb)[p], (la + lb)[p], 2)
        p, x, r = p[i], x[i], r[i]
        fvr = (
            np.sqrt(2.0 * v + 1)
            * np.sqrt(2.0 * r + 1)
            * _cg0_v(v, lb[p], la[p])
            * _cg0_v(r, l3b[p], l3a[p])
            * _cg0_v(x, np.full_like(x, k4), r)
        )
        keep = fvr != 0.0
        p, x, r, v, fvr = p[keep], x[keep], r[keep], v[keep], fvr[keep]
        lo = np.maximum.reduce(
            [
                np.abs(v - 1),
                np.abs(ja - jb)[p],
                np.abs(r - 1),
                (np.abs(j3a - j3b) // 2)[p],
                np.abs(x - k3),
            ]
        )
        hi = np.minimum.reduce([v + 1, (ja + jb)[p], r + 1, ((j3a + j3b) // 2)[p], x + k3])
        i, y = _span_parity(lo, hi, x + k3, v + 1)
        p, x, r, v, fvr = p[i], x[i], r[i], v[i], fvr[i]
        fy = (
            _sign(y)
            * np.sqrt(2.0 * y + 1)
            * _cg0_v(y, np.ones_like(y), v)
            * _nine_pair_v(r, v, y, a[p], b[p])
            * _six_v(2 * y, 2 * ja[p], 2 * jb[p], j2, j3b[p], j3a[p])
            * _cg0_v(x, np.full_like(x, k3), y)
            * _six_v(np.full_like(y, 2 * k4), 2, np.full_like(y, 2 * k3), 2 * y, 2 * x, 2 * r)
        )
        total = np.bincount(p * width + x, weights=fvr * fy, minlength=n * width)
        fk34 = sqrt(comb(3, 2 * k3) * (2 * k4 + 1))
        out[:, k3, :] = fk34 * (2 * np.arange(width) + 1) * total.reshape(n, width)
    return out


def _tpe_block(term: str, a: np.ndarray, b: np.ndarray, j2: int, x_max: int) -> np.ndarray:
    """c3 (Eq. 47) and c4 (Eq. 49) without the prefactor: shape ``(n, 8, x_max + 1)``."""
    la, sa, ja, _, l3a, j3a = (a[:, i] for i in range(6))
    lb, sb, jb, _, l3b, j3b = (b[:, i] for i in range(6))
    n, width, nc = len(a), x_max + 1, len(COMBOS)
    k1s, k2s, k3s = (np.array([cb[i] for cb in COMBOS]) for i in range(3))
    p = np.repeat(np.arange(n), nc * width)
    c = np.tile(np.repeat(np.arange(nc), width), n)
    x = np.tile(np.arange(width), n * nc)
    k1, k2, k3 = k1s[c], k2s[c], k3s[c]
    k4 = k2 - k3
    lo = np.maximum(np.abs(l3a - l3b)[p], np.abs(x - k4))
    hi = np.minimum((l3a + l3b)[p], x + k4)
    i, r = _span_parity(lo, hi, (l3a + l3b)[p], x + k4)
    p, c, x, k1, k2, k3, k4 = (arr[i] for arr in (p, c, x, k1, k2, k3, k4))
    i, v = _span(np.abs(la - lb)[p], (la + lb)[p], 2)
    p, c, x, k1, k2, k3, k4, r = (arr[i] for arr in (p, c, x, k1, k2, k3, k4, r))
    fvr = (
        _sign(v + r)
        * np.sqrt(2.0 * v + 1)
        * np.sqrt(2.0 * r + 1)
        * _cg0_v(v, lb[p], la[p])
        * _cg0_v(r, l3b[p], l3a[p])
        * _cg0_v(x, k4, r)
    )
    keep = fvr != 0.0
    p, c, x, k1, k2, k3, k4, r, v, fvr = (arr[keep] for arr in (p, c, x, k1, k2, k3, k4, r, v, fvr))
    lo = np.maximum.reduce([np.abs(v - k1), np.abs(r - k2), np.abs(x - k3)])
    hi = np.minimum.reduce([v + k1, r + k2, x + k3])
    i, y = _span_parity(lo, hi, x + k3, v + k1)
    p, c, x, k1, k2, k3, k4, r, v, fvr = (arr[i] for arr in (p, c, x, k1, k2, k3, k4, r, v, fvr))
    fy = (
        np.sqrt(2.0 * y + 1)
        * _cg0_v(y, k1, v)
        * _cg0_v(x, k3, y)
        * _six_v(2 * k3, 2 * k4, 2 * k2, 2 * r, 2 * y, 2 * x)
    )
    keep = fy != 0.0
    p, c, x, k1, k2, r, v, y, fvr, fy = (arr[keep] for arr in (p, c, x, k1, k2, r, v, y, fvr, fy))
    zlo = np.maximum.reduce(
        [np.abs(ja - jb)[p], np.abs(r - 1), (np.abs(j3a - j3b) // 2)[p], np.abs(y - 1)]
        + ([np.abs(v - 1)] if term == "c3" else [])
    )
    zhi = np.minimum.reduce(
        [(ja + jb)[p], r + 1, ((j3a + j3b) // 2)[p], y + 1] + ([v + 1] if term == "c3" else [])
    )
    i, z = _span(zlo, zhi)
    p, c, x, k1, k2, r, v, y, fvr, fy = (arr[i] for arr in (p, c, x, k1, k2, r, v, y, fvr, fy))
    fz = (
        _sign(z)
        * (2 * z + 1)
        * _six_v(2 * z, 2 * ja[p], 2 * jb[p], j2, j3b[p], j3a[p])
        * _six_v(2, 2, 2 * k2, 2 * r, 2 * y, 2 * z)
    )
    keep = fz != 0.0
    p, c, x, k1, k2, r, v, y, z, fvr, fy, fz = (
        arr[keep] for arr in (p, c, x, k1, k2, r, v, y, z, fvr, fy, fz)
    )
    if term == "c3":
        inner = fz * _nine_pair_v(r, v, z, a[p], b[p]) * _six_v(2, 2, 2 * k1, 2 * v, 2 * y, 2 * z)
    else:
        orbit = _nine_orbit_v(r, z, a[p], b[p])
        lo5 = np.maximum.reduce([np.abs(sa - sb)[p], np.abs(v - z), np.abs(k1 - 1)])
        hi5 = np.minimum.reduce([(sa + sb)[p], v + z, k1 + 1, np.full_like(v, 2)])
        i, k5 = _span(lo5, hi5)
        pi, vi, zi, yi, k1i = p[i], v[i], z[i], y[i], k1[i]
        spin_terms = (
            (2 * k5 + 1)
            * _lookup(
                nine,
                (8, 3, 8, 8, 2, 8, 8, 2, 8),
                2 * vi,
                2 * k5,
                2 * zi,
                2 * la[pi],
                2 * sa[pi],
                2 * ja[pi],
                2 * lb[pi],
                2 * sb[pi],
                2 * jb[pi],
            )  # fmt: skip
            * _lookup(
                nine,
                (2,) * 8 + (3,),
                np.ones_like(k5),
                np.ones_like(k5),
                2 * sa[pi],
                np.ones_like(k5),
                np.ones_like(k5),
                2 * sb[pi],
                2 * np.ones_like(k5),
                2 * np.ones_like(k5),
                2 * k5,
            )  # fmt: skip
            * _six_v(2 * k5, 2, 2 * k1i, 2 * yi, 2 * vi, 2 * zi)
            * _six_v(2, 2, 2, 2, 2 * k5, 2 * k1i)
        )
        spin = np.bincount(i, weights=spin_terms, minlength=len(p))
        inner = fz * orbit * spin
    total = np.bincount(
        (p * nc + c) * width + x, weights=fvr * fy * inner, minlength=n * nc * width
    )
    total = total.reshape(n, nc, width)
    w12 = np.array([_TPE_WEIGHT[k1, k2] for k1, k2, _ in COMBOS])
    fk34 = np.array([sqrt(comb(2 * k2 + 1, 2 * k3) * (2 * (k2 - k3) + 1)) for _, k2, k3 in COMBOS])
    return total * (w12 * fk34)[None, :, None] * (2 * np.arange(width) + 1)[None, None, :]


def coefficient_table(
    term: str, a: np.ndarray, b: np.ndarray, j2: int, t2: int, x_max: int, block: int = 128
) -> np.ndarray:
    """Coefficients ``C_a`` of ``term`` for many channel pairs, one row per pair.

    Parameters
    ----------
    term
        One of :data:`TERMS`.
    a, b
        Integer arrays ``(n, 6)`` of the bra and ket channels, columns
        ``l12, s12, j12, t12, l3, j3`` (``j3`` doubled).
    j2, t2
        Doubled total ``J`` and ``T``.
    x_max
        Largest rank ``X`` of the radial functions.
    block
        Pairs evaluated together; it bounds the size of the temporary index arrays.

    Returns
    -------
    np.ndarray
        Shape ``(n, n_keys(term, x_max))``; the columns follow :func:`radial_keys`.
    """
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    out = np.zeros((len(a), n_keys(term, x_max)))
    if not len(a):
        return out
    pref = _prefactor(term, a, b, j2, t2)
    active = np.flatnonzero(pref != 0.0)
    for start in range(0, len(active), block):
        sel = active[start : start + block]
        if term == "cE":
            part = _ce_block(a[sel], b[sel], j2, x_max)
        elif term == "cD":
            part = _cd_block(a[sel], b[sel], j2, x_max)
        elif term == "c1":
            part = _c1_block(a[sel], b[sel], j2, x_max)
        else:
            part = _tpe_block(term, a[sel], b[sel], j2, x_max)
        out[sel] = pref[sel, None] * part.reshape(len(sel), -1)
    return out


def channel_array(channels: list[Channel]) -> np.ndarray:
    """Stack channels into the integer array ``(n, 6)`` that :func:`coefficient_table` takes."""
    return np.array([tuple(c) for c in channels], dtype=np.int64).reshape(-1, 6)


def coefficients(
    term: str, a: Channel, b: Channel, j2: int, t2: int, x_max: int
) -> dict[tuple[int, ...], float]:
    """Coefficients ``C_a`` of ``term`` between channels ``a`` (bra) and ``b`` (ket), ``J=j2/2``.

    One pair through :func:`coefficient_table`, as a dictionary of the non-zero values by radial
    function label (:func:`radial_keys`).
    """
    row = coefficient_table(term, channel_array([a]), channel_array([b]), j2, t2, x_max)[0]
    return {key: float(v) for key, v in zip(radial_keys(term, x_max), row, strict=True) if v != 0.0}
