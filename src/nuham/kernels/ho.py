"""Harmonic-oscillator radial functions via the normalized three-term recurrence in n.

The functions are built directly in normalized form, so the large unnormalized Laguerre
polynomials never appear. All arrays are float64 (enabled in ``nuham``).
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike


@partial(jax.jit, static_argnames=("n_max",))
def radial_functions(x: ArrayLike, l: ArrayLike, n_max: int) -> jax.Array:
    """Dimensionless HO radial functions ``g_nl(x)`` for ``n = 0..n_max``.

    ``g_nl(x) = sqrt(2 n! / Gamma(n + l + 3/2)) x^(l+1) exp(-x^2/2) L_n^(l+1/2)(x^2)``,
    normalized so that the integral of ``g^2`` over ``x`` from 0 to infinity is 1.

    Parameters
    ----------
    x
        Points with ``x > 0``, shape ``(M,)``.
    l
        Orbital angular momentum. A scalar, possibly traced, so it may be vmapped.
    n_max
        Largest radial quantum number (static).

    Returns
    -------
    jax.Array
        Shape ``(n_max + 1, M)``.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    alpha = jnp.asarray(l, dtype=jnp.float64) + 0.5
    x2 = x * x
    log_g0 = (
        0.5 * jnp.log(2.0) - 0.5 * lax.lgamma(alpha + 1.0) + (alpha + 0.5) * jnp.log(x) - 0.5 * x2
    )
    g0 = jnp.exp(log_g0)
    if n_max == 0:
        return g0[None, :]
    g1 = g0 * (alpha + 1.0 - x2) / jnp.sqrt(alpha + 1.0)
    if n_max == 1:
        return jnp.stack([g0, g1])

    def step(carry: tuple[jax.Array, jax.Array], m: jax.Array) -> tuple[tuple, jax.Array]:
        prev, prev2 = carry
        d = jnp.sqrt((m + 1.0) * (m + alpha + 1.0))
        nxt = ((2.0 * m + alpha + 1.0 - x2) / d) * prev - jnp.sqrt(m * (m + alpha)) / d * prev2
        return (nxt, prev), nxt

    ms = jnp.arange(1, n_max, dtype=jnp.float64)
    _, rest = lax.scan(step, (g1, g0), ms)
    return jnp.concatenate([g0[None, :], g1[None, :], rest], axis=0)


@partial(jax.jit, static_argnames=("n_max",))
def momentum_radial(k: ArrayLike, l: ArrayLike, n_max: int, b: ArrayLike) -> jax.Array:
    """HO radial functions in momentum space, NuHamil convention.

    ``R_nl(k) = (-1)^n sqrt(b) g_nl(b k)``, normalized so that the integral of ``R^2`` over ``k``
    is 1 (the factor ``k`` of the three-dimensional measure is already inside ``g``).

    Parameters
    ----------
    k
        Momenta in fm^-1, shape ``(M,)``.
    l
        Orbital angular momentum.
    n_max
        Largest radial quantum number (static).
    b
        Oscillator length in fm.

    Returns
    -------
    jax.Array
        Shape ``(n_max + 1, M)``.
    """
    b = jnp.asarray(b, dtype=jnp.float64)
    g = radial_functions(b * jnp.asarray(k, dtype=jnp.float64), l, n_max)
    phase = jnp.where(jnp.arange(n_max + 1) % 2 == 0, 1.0, -1.0)
    return jnp.sqrt(b) * phase[:, None] * g
