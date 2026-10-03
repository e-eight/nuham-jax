import random

import pytest
from sympy import Rational
from sympy.physics.wigner import wigner_6j, wigner_9j

from nuham.angmom.racah import ninej, sixj


def half(x: int) -> Rational:
    return Rational(x, 2)


def sympy_value(func, args: list[int]) -> float:
    """Sympy raises on triangle violations where the symbol is zero."""
    try:
        return float(func(*[half(x) for x in args]))
    except ValueError:
        return 0.0


def test_sixj_matches_sympy() -> None:
    rng = random.Random(1)
    checked = 0
    while checked < 150:
        a = [rng.randint(0, 8) for _ in range(6)]
        want = sympy_value(wigner_6j, a)
        assert sixj(*a) == pytest.approx(want, abs=1e-13)
        checked += want != 0.0


def test_ninej_matches_sympy() -> None:
    rng = random.Random(2)
    checked = 0
    while checked < 60:
        a = [rng.randint(0, 6) for _ in range(9)]
        want = sympy_value(wigner_9j, a)
        assert ninej(*a) == pytest.approx(want, abs=1e-13)
        checked += want != 0.0


def test_sixj_orthogonality() -> None:
    # sum_x (2x+1)(2j+1) {a b x; c d j} {a b x; c d j'} = delta_{jj'}
    a, b, c, d = 3, 2, 4, 5
    js = [j for j in range(0, 20) if sixj(a, b, 2, c, d, j) != 0.0]
    for j in js:
        total = sum(
            (x + 1) * (j + 1) * sixj(a, b, x, c, d, j) ** 2 for x in range(abs(a - b), a + b + 1, 2)
        )
        assert total == pytest.approx(1.0, abs=1e-12)
