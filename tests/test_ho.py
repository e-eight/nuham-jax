import math

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest

import nuham  # noqa: F401  (enables float64)
from nuham.constants import oscillator_length_sq
from nuham.kernels.ho import momentum_radial, radial_functions

mp.mp.dps = 60


def exact_g(n: int, l: int, x: float) -> float:
    x = mp.mpf(x)
    a = l + mp.mpf(1) / 2
    val = (
        mp.sqrt(2 * mp.factorial(n) / mp.gamma(n + a + 1))
        * x ** (l + 1)
        * mp.exp(-x * x / 2)
        * mp.laguerre(n, a, x * x)
    )
    return float(val)


@pytest.mark.parametrize(("l", "n_max"), [(0, 50), (1, 50), (8, 46), (20, 40)])
def test_matches_mpmath(l: int, n_max: int) -> None:
    xs = np.linspace(0.05, 13.0, 25)
    ref = np.array([[exact_g(n, l, x) for x in xs] for n in range(n_max + 1)])
    got = np.asarray(radial_functions(xs, l, n_max))
    assert np.abs(got - ref).max() < 1e-14


@pytest.mark.parametrize("n_max", [0, 1, 2])
def test_small_n_max_shapes(n_max: int) -> None:
    out = radial_functions(jnp.linspace(0.1, 3.0, 7), 2, n_max)
    assert out.shape == (n_max + 1, 7)


def test_traced_l_matches_static() -> None:
    xs = jnp.linspace(0.1, 8.0, 30)
    ls = jnp.arange(5)
    batched = jax.vmap(lambda l: radial_functions(xs, l, 12))(ls)
    for l in range(5):
        np.testing.assert_allclose(batched[l], radial_functions(xs, l, 12), atol=1e-15)


@pytest.mark.parametrize("hw", [10.0, 16.0, 40.0])
def test_momentum_orthonormal_on_wide_mesh(hw: int) -> None:
    b = math.sqrt(oscillator_length_sq(hw))
    k, w = np.polynomial.legendre.leggauss(400)
    k, w = 15.0 * (k + 1.0), 15.0 * w
    r = np.asarray(momentum_radial(k, 3, 12, b))
    np.testing.assert_allclose((r * w) @ r.T, np.eye(13), atol=1e-12)


def test_momentum_phase() -> None:
    b = 1.7
    k = np.array([0.3, 1.1])
    plain = np.asarray(radial_functions(b * k, 1, 3)) * math.sqrt(b)
    got = np.asarray(momentum_radial(k, 1, 3, b))
    np.testing.assert_allclose(got, plain * np.array([1, -1, 1, -1])[:, None], atol=1e-15)
