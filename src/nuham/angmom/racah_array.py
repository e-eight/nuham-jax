"""Vectorized Wigner 6j and 9j symbols for arrays of doubled-integer arguments."""

import math

import numpy as np
from numpy.typing import ArrayLike

_MAX_ARG = 2048
_LOG_FACT = np.array([math.lgamma(i + 1) for i in range(_MAX_ARG)])


def _log_triangle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Log triangle coefficient and a validity mask for doubled arguments."""
    valid = ((a + b + c) % 2 == 0) & (c >= np.abs(a - b)) & (c <= a + b)
    safe = lambda x: np.where(valid, x, 0)  # noqa: E731
    val = 0.5 * (
        _LOG_FACT[safe((a + b - c) // 2)]
        + _LOG_FACT[safe((a - b + c) // 2)]
        + _LOG_FACT[safe((-a + b + c) // 2)]
        - _LOG_FACT[safe((a + b + c) // 2 + 1)]
    )
    return np.where(valid, val, 0.0), valid


def _masked(use: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Replace entries outside ``use`` by 0 so they are valid table indices."""
    return np.where(use, x, 0)


def sixj_array(
    j1: ArrayLike,
    j2: ArrayLike,
    j3: ArrayLike,
    j4: ArrayLike,
    j5: ArrayLike,
    j6: ArrayLike,
) -> np.ndarray:
    """Wigner 6j symbols ``{j1 j2 j3; j4 j5 j6}`` for broadcastable integer arrays (doubled).

    Parameters
    ----------
    j1, j2, j3, j4, j5, j6
        Twice the angular momenta; broadcast against each other.

    Returns
    -------
    np.ndarray
        The symbols, zero wherever a triangle condition fails.
    """
    j1, j2, j3, j4, j5, j6 = (np.asarray(x, dtype=np.int64) for x in np.broadcast_arrays(
        j1, j2, j3, j4, j5, j6
    ))  # fmt: skip
    t1, v1 = _log_triangle(j1, j2, j3)
    t2, v2 = _log_triangle(j1, j5, j6)
    t3, v3 = _log_triangle(j4, j2, j6)
    t4, v4 = _log_triangle(j4, j5, j3)
    valid = v1 & v2 & v3 & v4
    pre = t1 + t2 + t3 + t4
    a = (j1 + j2 + j3) // 2
    b = (j1 + j5 + j6) // 2
    c = (j4 + j2 + j6) // 2
    d = (j4 + j5 + j3) // 2
    e = (j1 + j2 + j4 + j5) // 2
    f = (j2 + j3 + j5 + j6) // 2
    g = (j3 + j1 + j6 + j4) // 2
    t_min = np.maximum(np.maximum(a, b), np.maximum(c, d))
    t_max = np.minimum(np.minimum(e, f), g)
    span = np.where(valid, t_max - t_min, -1)
    out = np.zeros(j1.shape)
    for k in range(int(span.max()) + 1 if span.size else 0):
        t = np.where(span >= k, t_min + k, 0)
        use = span >= k
        log_term = (
            pre
            + _LOG_FACT[_masked(use, t + 1)]
            - _LOG_FACT[_masked(use, t - a)]
            - _LOG_FACT[_masked(use, t - b)]
            - _LOG_FACT[_masked(use, t - c)]
            - _LOG_FACT[_masked(use, t - d)]
            - _LOG_FACT[_masked(use, e - t)]
            - _LOG_FACT[_masked(use, f - t)]
            - _LOG_FACT[_masked(use, g - t)]
        )
        out += np.where(use, np.where(t % 2 == 0, 1.0, -1.0) * np.exp(log_term), 0.0)
    return out


def ninej_array(
    a: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
    d: ArrayLike,
    e: ArrayLike,
    f: ArrayLike,
    g: ArrayLike,
    h: ArrayLike,
    i: ArrayLike,
) -> np.ndarray:
    """Wigner 9j symbols ``{a b c; d e f; g h i}`` for broadcastable doubled-integer arrays.

    Evaluated as the sum over ``x`` of ``(-1)^x (x+1)`` times three 6j symbols.

    Parameters
    ----------
    a, b, c, d, e, f, g, h, i
        Twice the angular momenta, row by row; broadcast against each other.

    Returns
    -------
    np.ndarray
        The symbols.
    """
    a, b, c, d, e, f, g, h, i = (
        np.asarray(x, dtype=np.int64) for x in np.broadcast_arrays(a, b, c, d, e, f, g, h, i)
    )
    lo = np.maximum(np.maximum(np.abs(a - i), np.abs(d - h)), np.abs(b - f))
    hi = np.minimum(np.minimum(a + i, d + h), b + f)
    span = np.where(hi >= lo, (hi - lo) // 2, -1)
    out = np.zeros(a.shape)
    for k in range(int(span.max()) + 1 if span.size else 0):
        x = np.where(span >= k, lo + 2 * k, 0)
        term = (
            np.where(x % 2 == 0, 1.0, -1.0)
            * (x + 1)
            * sixj_array(a, b, c, f, i, x)
            * sixj_array(d, e, f, b, x, h)
            * sixj_array(g, h, i, x, a, d)
        )
        out += np.where(span >= k, term, 0.0)
    return out
