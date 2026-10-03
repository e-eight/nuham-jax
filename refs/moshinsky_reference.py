"""Reference implementation of the Moshinsky bracket by the explicit Kamuntavicius sum.

NOT PART OF THE PACKAGE. This is a close transcription of the sum used by NuHamil's ``gmosh``
(Kamuntavicius et al., Nucl. Phys. A 695, 191 (2001)), kept only as an independent check of the
rotation-generator tables in ``nuham.angmom.moshinsky_table``. It is slow (50-500 us per bracket)
and lives outside ``src/`` so the package does not ship it. Tests that use it skip when this
directory is absent.
"""

import math
from functools import cache

from nuham.angmom.racah import ninej


@cache
def _log_fact(n: int) -> float:
    """Return ``log(n!)``."""
    return math.lgamma(n + 1)

@cache
def cg_zero(la: int, lb: int, l: int) -> float:
    """Clebsch-Gordan coefficient ``<la 0 lb 0 | l 0>`` for integer (not doubled) arguments.

    Parameters
    ----------
    la, lb, l
        Orbital angular momenta.

    Returns
    -------
    float
        The coefficient; zero unless the triangle condition holds and ``la + lb + l`` is even.
    """
    if l < abs(la - lb) or l > la + lb or (la + lb + l) % 2:
        return 0.0
    g = (la + lb + l) // 2
    log_val = (
        0.5
        * (
            math.log(2 * l + 1)
            + _log_fact(2 * g - 2 * la)
            + _log_fact(2 * g - 2 * lb)
            + _log_fact(2 * g - 2 * l)
            - _log_fact(2 * g + 1)
        )
        + _log_fact(g)
        - _log_fact(g - la)
        - _log_fact(g - lb)
        - _log_fact(g - l)
    )
    return (-1) ** (g - l) * math.exp(log_val)


def _double_factorial(n: int) -> int:
    """Return ``n!!`` with ``n!! = 1`` for ``n <= 1``."""
    result = 1
    while n > 1:
        result *= n
        n -= 2
    return result


def _ratio(i: int, j: int, k: int) -> float:
    """Return ``i!! / (j!! k!!)`` as a float."""
    return _double_factorial(i) / (_double_factorial(j) * _double_factorial(k))


def _outside_triangle(a: int, b: int, c: int) -> bool:
    """Return whether ``c`` is outside the range allowed by ``a`` and ``b``."""
    return c < abs(a - b) or c > a + b


def _g(e1: int, l1: int, ea: int, la: int, eb: int, lb: int) -> float:
    """Single-pair coefficient entering the bracket sum."""
    if _outside_triangle(la, lb, l1):
        return 0.0
    return cg_zero(la, lb, l1) * math.sqrt(
        (2 * la + 1)
        * (2 * lb + 1)
        * _ratio(e1 - l1, ea - la, eb - lb)
        * _ratio(e1 + l1 + 1, ea + la + 1, eb + lb + 1)
    )


@cache
def bracket(
    n_cm: int,
    l_cm: int,
    n_rel: int,
    l_rel: int,
    n1: int,
    l1: int,
    n2: int,
    l2: int,
    lam: int,
    d: float = 1.0,
) -> float:
    """Moshinsky bracket ``<n_cm l_cm, n_rel l_rel; lam | n1 l1, n2 l2; lam>``.

    ``d`` is the mass ratio of the two particles; ``d = 1`` is the equal-mass case used for two
    nucleons. The phase is ``(-1)^(n1 + n2 + n_rel + n_cm)`` times the Kamuntavicius sum.

    Parameters
    ----------
    n_cm, l_cm
        Centre-of-mass radial and orbital quantum numbers.
    n_rel, l_rel
        Relative radial and orbital quantum numbers.
    n1, l1, n2, l2
        Quantum numbers of the two particles.
    lam
        Total orbital angular momentum.
    d
        Mass ratio.

    Returns
    -------
    float
        The bracket; zero unless energy and the angular triangles are consistent.
    """
    e_cm = 2 * n_cm + l_cm
    e_rel = 2 * n_rel + l_rel
    e1 = 2 * n1 + l1
    e2 = 2 * n2 + l2
    if (
        e_cm + e_rel != e1 + e2
        or _outside_triangle(l_cm, l_rel, lam)
        or _outside_triangle(l1, l2, lam)
    ):
        return 0.0
    scale = math.sqrt(d ** (e1 - e_rel) / (1 + d) ** (e1 + e2))
    sign = 1.0
    total = 0.0
    for ed in range(min(e_rel, e2) + 1):
        eb = e_rel - ed
        ec = e2 - ed
        ea = e1 - e_rel + ed
        if ea < 0:
            sign *= -d
            continue
        for ld in range(ed, -1, -2):
            for lb in range(eb, -1, -2):
                if _outside_triangle(ld, lb, l_rel):
                    continue
                for lc in range(ec, -1, -2):
                    if _outside_triangle(ld, lc, l2):
                        continue
                    for la in range(ea, -1, -2):
                        if _outside_triangle(la, lb, l1) or _outside_triangle(la, l_cm, lc):
                            continue
                        total += (
                            sign
                            * scale
                            * ninej(
                                2 * la,
                                2 * lb,
                                2 * l1,
                                2 * lc,
                                2 * ld,
                                2 * l2,
                                2 * l_cm,
                                2 * l_rel,
                                2 * lam,
                            )
                            * _g(e1, l1, ea, la, eb, lb)
                            * _g(e2, l2, ec, lc, ed, ld)
                            * _g(e_cm, l_cm, ea, la, ec, lc)
                            * _g(e_rel, l_rel, eb, lb, ed, ld)
                        )
        sign *= -d
    return total * (-1) ** (n1 + n2 + n_rel + n_cm)
