"""Wigner 6j and 9j symbols.

Angular momenta of the 6j and 9j symbols are passed as *doubled integers* (``2j``), so
half-integers need no floating point. Results are cached; this is host-side NumPy-free code.
"""

import math
from functools import cache


@cache
def _log_fact(n: int) -> float:
    """Return ``log(n!)``."""
    return math.lgamma(n + 1)


def _log_triangle(a: int, b: int, c: int) -> float | None:
    """Log of the triangle coefficient for doubled arguments, or ``None`` if not allowed."""
    if (a + b + c) % 2 or c < abs(a - b) or c > a + b:
        return None
    return 0.5 * (
        _log_fact((a + b - c) // 2)
        + _log_fact((a - b + c) // 2)
        + _log_fact((-a + b + c) // 2)
        - _log_fact((a + b + c) // 2 + 1)
    )


@cache
def sixj(j1: int, j2: int, j3: int, j4: int, j5: int, j6: int) -> float:
    """Wigner 6j symbol ``{j1 j2 j3; j4 j5 j6}`` for doubled arguments.

    Parameters
    ----------
    j1, j2, j3, j4, j5, j6
        Twice the angular momenta.

    Returns
    -------
    float
        The symbol, zero if any triangle condition fails.
    """
    tris = [
        _log_triangle(j1, j2, j3),
        _log_triangle(j1, j5, j6),
        _log_triangle(j4, j2, j6),
        _log_triangle(j4, j5, j3),
    ]
    if any(t is None for t in tris):
        return 0.0
    pre = sum(t for t in tris if t is not None)
    a = (j1 + j2 + j3) // 2
    b = (j1 + j5 + j6) // 2
    c = (j4 + j2 + j6) // 2
    d = (j4 + j5 + j3) // 2
    e = (j1 + j2 + j4 + j5) // 2
    f = (j2 + j3 + j5 + j6) // 2
    g = (j3 + j1 + j6 + j4) // 2
    total = 0.0
    for t in range(max(a, b, c, d), min(e, f, g) + 1):
        log_term = (
            pre
            + _log_fact(t + 1)
            - _log_fact(t - a)
            - _log_fact(t - b)
            - _log_fact(t - c)
            - _log_fact(t - d)
            - _log_fact(e - t)
            - _log_fact(f - t)
            - _log_fact(g - t)
        )
        total += (-1) ** t * math.exp(log_term)
    return total


@cache
def ninej(a: int, b: int, c: int, d: int, e: int, f: int, g: int, h: int, i: int) -> float:
    """Wigner 9j symbol ``{a b c; d e f; g h i}`` for doubled arguments, as a sum of 6j products.

    Parameters
    ----------
    a, b, c, d, e, f, g, h, i
        Twice the angular momenta, row by row.

    Returns
    -------
    float
        The symbol.
    """
    lo = max(abs(a - i), abs(d - h), abs(b - f))
    hi = min(a + i, d + h, b + f)
    total = 0.0
    for x in range(lo, hi + 1, 2):
        total += (
            (-1) ** x
            * (x + 1)
            * sixj(a, b, c, f, i, x)
            * sixj(d, e, f, b, x, h)
            * sixj(g, h, i, x, a, d)
        )
    return total
