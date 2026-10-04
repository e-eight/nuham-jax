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
The keys of the returned dictionaries label the radial functions (see :func:`radial_keys`).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from math import comb, factorial, sqrt
from typing import NamedTuple

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


def _dims(a: Channel, b: Channel) -> float:
    """``hat(j) hat(j') hat(J3) hat(J3') hat(l') hat(L')`` shared by all terms."""
    return sqrt(
        (2 * a.j12 + 1)
        * (2 * b.j12 + 1)
        * (a.j3 + 1)
        * (b.j3 + 1)
        * (2 * b.l12 + 1)
        * (2 * b.l3 + 1)
    )


def _nine_pair(
    a: Channel, b: Channel, row_a: tuple[int, int, int], row_b: tuple[int, int, int]
) -> float:
    """Product of the 9j symbols that recouple ``(l s) j`` and ``(L 1/2) J3`` to rank ``Z``."""
    v, k, z = row_a
    r, kk, zz = row_b
    return nine(
        2 * v, 2 * k, 2 * z, 2 * a.l12, 2 * a.s12, 2 * a.j12, 2 * b.l12, 2 * b.s12, 2 * b.j12
    ) * nine(2 * r, 2 * kk, 2 * zz, 2 * a.l3, 1, a.j3, 2 * b.l3, 1, b.j3)


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
    """Labels of the radial functions ``g_a`` of ``term`` (see ``local_force.radial.radial_function``)."""
    xs = range(x_max + 1)
    if term == "cE":
        return [(x,) for x in xs]
    if term == "cD":
        return [(k, x) for k in (0, 2) for x in xs]
    if term == "c1":
        return [(k3, x) for k3 in (0, 1) for x in xs]
    return [(k1, k2, k3, x) for k1 in (0, 2) for k2 in (0, 2) for k3 in range(k2 + 1) for x in xs]


def _add(out: dict[tuple[int, ...], float], key: tuple[int, ...], value: float) -> None:
    """Accumulate ``value`` under ``key`` if it is not exactly zero."""
    if value != 0.0:
        out[key] = out.get(key, 0.0) + value


def coefficients_ce(
    a: Channel, b: Channel, j2: int, t2: int, x_max: int
) -> dict[tuple[int, ...], float]:
    """Contact term, Eq. (15): keys ``(X,)`` for ``g = Z0(sqrt2 xi1) Z0,X(s, a)``."""
    if a.s12 != b.s12:
        return {}
    iso = tau2_dot_tau3(a.t12, b.t12, t2)
    if iso == 0.0:
        return {}
    pref = _lec_free_constants()["cE"] * iso * _dims(a, b)
    pref *= (-1.0) ** ((j2 - 1 + b.j3 - a.j3) // 2 + a.l12 + a.l3 + a.s12)
    out: dict[tuple[int, ...], float] = {}
    for x in range(max(abs(a.l12 - b.l12), abs(a.l3 - b.l3), abs(a.j12 - b.j12)), x_max + 1):
        if (x + a.l12 + b.l12) % 2 or (x + a.l3 + b.l3) % 2:
            continue
        w = (
            (-1.0) ** x
            * (2 * x + 1)
            * six(2 * x, 2 * b.l12, 2 * a.l12, 2 * a.s12, 2 * a.j12, 2 * b.j12)
            * six(2 * x, 2 * a.j12, 2 * b.j12, j2, b.j3, a.j3)
            * six(2 * x, 2 * b.l3, 2 * a.l3, 1, a.j3, b.j3)
            * cg0(b.l12, x, a.l12)
            * cg0(b.l3, x, a.l3)
        )
        _add(out, (x,), pref * w)
    return out


def coefficients_cd(
    a: Channel, b: Channel, j2: int, t2: int, x_max: int
) -> dict[tuple[int, ...], float]:
    """One-pion-exchange contact term, Eq. (38): keys ``(K, X)`` for ``g = f_K(sqrt2 xi1) Z0,X``."""
    iso = tau2_dot_tau3(a.t12, b.t12, t2)
    spin = spin_factor(a.s12, b.s12)
    if iso == 0.0 or spin == 0.0:
        return {}
    pref = _lec_free_constants()["cD"] * iso * spin * _dims(a, b)
    pref *= (-1.0) ** ((j2 - a.j3) // 2 + a.s12 + b.j12)
    out: dict[tuple[int, ...], float] = {}
    for k in (0, 2):
        fk = hat(k) * (-1.0) ** (k // 2) * cg0(1, 1, k)
        for v in range(abs(a.l12 - b.l12), a.l12 + b.l12 + 1):
            if (v + a.l12 + b.l12) % 2:
                continue
            for x in range(max(abs(a.l3 - b.l3), abs(v - k)), min(a.l3 + b.l3, v + k, x_max) + 1):
                if (x + a.l3 + b.l3) % 2 or (x + k + v) % 2:
                    continue
                wv = (
                    (-1.0) ** v
                    * hat(v)
                    * cg0(v, b.l12, a.l12)
                    * (2 * x + 1)
                    * cg0(x, k, v)
                    * cg0(x, b.l3, a.l3)
                )
                if wv == 0.0:
                    continue
                for z in range(
                    max(abs(v - 1), abs(a.j12 - b.j12), abs(x - 1), abs(a.j3 - b.j3) // 2),
                    min(v + 1, a.j12 + b.j12, x + 1, (a.j3 + b.j3) // 2) + 1,
                ):
                    wz = (
                        (2 * z + 1)
                        * _nine_pair(a, b, (v, 1, z), (x, 1, z))
                        * six(2 * z, 2 * a.j12, 2 * b.j12, j2, b.j3, a.j3)
                        * six(2 * v, 2, 2 * z, 2, 2 * x, 2 * k)
                    )
                    _add(out, (k, x), pref * fk * wv * wz)
    return out


def coefficients_c1(
    a: Channel, b: Channel, j2: int, t2: int, x_max: int
) -> dict[tuple[int, ...], float]:
    """Two-pion exchange c1, Eq. (42): keys ``(K3, X)``, ``g = s^K3 a^(1-K3) f1 f1,X``."""
    iso = tau2_dot_tau3(a.t12, b.t12, t2)
    spin = spin_factor(a.s12, b.s12)
    if iso == 0.0 or spin == 0.0:
        return {}
    pref = _lec_free_constants()["c1"] * iso * spin * _dims(a, b)
    pref *= (-1.0) ** ((j2 - a.j3) // 2 + a.s12 + b.j12)
    out: dict[tuple[int, ...], float] = {}
    for k3 in (0, 1):
        k4 = 1 - k3
        fk34 = sqrt(comb(3, 2 * k3) * (2 * k4 + 1))
        for x in range(x_max + 1):
            total = 0.0
            for r in range(max(abs(a.l3 - b.l3), abs(x - k4)), min(a.l3 + b.l3, x + k4) + 1):
                if (x + k4 + r) % 2 or (r + a.l3 + b.l3) % 2:
                    continue
                for v in range(abs(a.l12 - b.l12), a.l12 + b.l12 + 1):
                    if (v + a.l12 + b.l12) % 2:
                        continue
                    fvr = (
                        hat(v) * hat(r) * cg0(v, b.l12, a.l12) * cg0(r, b.l3, a.l3) * cg0(x, k4, r)
                    )
                    if fvr == 0.0:
                        continue
                    for y in range(
                        max(
                            abs(v - 1),
                            abs(a.j12 - b.j12),
                            abs(r - 1),
                            abs(a.j3 - b.j3) // 2,
                            abs(x - k3),
                        ),
                        min(v + 1, a.j12 + b.j12, r + 1, (a.j3 + b.j3) // 2, x + k3) + 1,
                    ):
                        if (x + k3 + y) % 2:
                            continue
                        fy = (
                            (-1.0) ** y
                            * hat(y)
                            * cg0(y, 1, v)
                            * _nine_pair(a, b, (v, 1, y), (r, 1, y))
                            * six(2 * y, 2 * a.j12, 2 * b.j12, j2, b.j3, a.j3)
                            * cg0(x, k3, y)
                            * six(2 * k4, 2, 2 * k3, 2 * y, 2 * x, 2 * r)
                        )
                        total += fvr * fy
            _add(out, (k3, x), pref * fk34 * (2 * x + 1) * total)
    return out


def _tpe_ranks() -> list[tuple[int, int, float]]:
    """Pairs ``(K1, K2)`` with weight ``(-1)^((K1+K2)/2) hat(K1) hat(K2) <1010|K10><1010|K20>``."""
    out = []
    for k1 in (0, 2):
        for k2 in (0, 2):
            out.append(
                (
                    k1,
                    k2,
                    (-1.0) ** ((k1 + k2) // 2) * hat(k1) * hat(k2) * cg0(1, 1, k1) * cg0(1, 1, k2),
                )
            )
    return out


def _coefficients_tpe(
    term: str, a: Channel, b: Channel, j2: int, t2: int, x_max: int
) -> dict[tuple[int, ...], float]:
    """c3 (Eq. 47) and c4 (Eq. 49): same coordinate expansion, different spin-isospin algebra."""
    if term == "c3":
        iso = tau2_dot_tau3(a.t12, b.t12, t2)
        spin = spin_factor(a.s12, b.s12)
        if iso == 0.0 or spin == 0.0:
            return {}
        pref = iso * spin
        phase = (-1.0) ** ((j2 - a.j3) // 2 + a.s12 + b.j12)
        dims = _dims(a, b)
    else:
        iso = tau_cross(a.t12, b.t12, t2)
        if iso == 0.0:
            return {}
        pref = iso * sqrt((2 * a.s12 + 1) * (2 * b.s12 + 1))
        phase = (-1.0) ** ((j2 + a.j3) // 2 + b.j12)
        dims = _dims(a, b)
    pref *= _lec_free_constants()[term] * dims * phase
    out: dict[tuple[int, ...], float] = {}
    for k1, k2, w12 in _tpe_ranks():
        if w12 == 0.0:
            continue
        for k3 in range(k2 + 1):
            k4 = k2 - k3
            fk34 = sqrt(comb(2 * k2 + 1, 2 * k3) * (2 * k4 + 1))
            for x in range(x_max + 1):
                total = 0.0
                for r in range(max(abs(a.l3 - b.l3), abs(x - k4)), min(a.l3 + b.l3, x + k4) + 1):
                    if (x + k4 + r) % 2 or (r + a.l3 + b.l3) % 2:
                        continue
                    for v in range(abs(a.l12 - b.l12), a.l12 + b.l12 + 1):
                        if (v + a.l12 + b.l12) % 2:
                            continue
                        fvr = (
                            (-1.0) ** (v + r)
                            * hat(v)
                            * hat(r)
                            * cg0(v, b.l12, a.l12)
                            * cg0(r, b.l3, a.l3)
                            * cg0(x, k4, r)
                        )
                        if fvr == 0.0:
                            continue
                        for y in range(
                            max(abs(v - k1), abs(r - k2), abs(x - k3)),
                            min(v + k1, r + k2, x + k3) + 1,
                        ):
                            if (x + k3 + y) % 2:
                                continue
                            fy = (
                                hat(y)
                                * cg0(y, k1, v)
                                * cg0(x, k3, y)
                                * six(2 * k3, 2 * k4, 2 * k2, 2 * r, 2 * y, 2 * x)
                            )
                            if fy == 0.0:
                                continue
                            total += fvr * fy * _tpe_spin_sum(term, a, b, j2, k1, k2, v, r, y)
                _add(out, (k1, k2, k3, x), pref * w12 * fk34 * (2 * x + 1) * total)
    return out


def _tpe_spin_sum(
    term: str, a: Channel, b: Channel, j2: int, k1: int, k2: int, v: int, r: int, y: int
) -> float:
    """Sum over the spin-orbit recoupling rank ``Z`` (and ``K4`` for c4) at fixed ``v, r, y``."""
    total = 0.0
    zlo = max(abs(a.j12 - b.j12), abs(r - 1), abs(a.j3 - b.j3) // 2, abs(y - 1))
    zhi = min(a.j12 + b.j12, r + 1, (a.j3 + b.j3) // 2, y + 1)
    if term == "c3":
        zlo, zhi = max(zlo, abs(v - 1)), min(zhi, v + 1)
    for z in range(zlo, zhi + 1):
        fz = (
            (-1.0) ** z
            * (2 * z + 1)
            * six(2 * z, 2 * a.j12, 2 * b.j12, j2, b.j3, a.j3)
            * six(2, 2, 2 * k2, 2 * r, 2 * y, 2 * z)
        )
        if fz == 0.0:
            continue
        if term == "c3":
            total += (
                fz * _nine_pair(a, b, (v, 1, z), (r, 1, z)) * six(2, 2, 2 * k1, 2 * v, 2 * y, 2 * z)
            )
        else:
            nine_orbit = nine(2 * r, 2, 2 * z, 2 * a.l3, 1, a.j3, 2 * b.l3, 1, b.j3)
            spin = 0.0
            for k5 in range(
                max(abs(a.s12 - b.s12), abs(v - z), abs(k1 - 1)),
                min(a.s12 + b.s12, v + z, k1 + 1, 2) + 1,
            ):
                spin += (
                    (2 * k5 + 1)
                    * nine(
                        2 * v,
                        2 * k5,
                        2 * z,
                        2 * a.l12,
                        2 * a.s12,
                        2 * a.j12,
                        2 * b.l12,
                        2 * b.s12,
                        2 * b.j12,
                    )
                    * nine(1, 1, 2 * a.s12, 1, 1, 2 * b.s12, 2, 2, 2 * k5)
                    * six(2 * k5, 2, 2 * k1, 2 * y, 2 * v, 2 * z)
                    * six(2, 2, 2, 2, 2 * k5, 2 * k1)
                )
            total += fz * nine_orbit * spin
    return total


COEFFICIENT_FUNCTIONS: dict[str, Callable[..., dict[tuple[int, ...], float]]] = {
    "cE": coefficients_ce,
    "cD": coefficients_cd,
    "c1": coefficients_c1,
    "c3": lambda a, b, j2, t2, x_max: _coefficients_tpe("c3", a, b, j2, t2, x_max),
    "c4": lambda a, b, j2, t2, x_max: _coefficients_tpe("c4", a, b, j2, t2, x_max),
}


def coefficients(
    term: str, a: Channel, b: Channel, j2: int, t2: int, x_max: int
) -> dict[tuple[int, ...], float]:
    """Coefficients ``C_a`` of ``term`` between channels ``a`` (bra) and ``b`` (ket), ``J=j2/2``."""
    return COEFFICIENT_FUNCTIONS[term](a, b, j2, t2, x_max)
