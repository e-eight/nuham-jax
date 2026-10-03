"""Matrix exponential of a rotation generator, using only matrix products."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

_TAYLOR_TERMS = 20
_TARGET_NORM = 0.25


@partial(jax.jit, static_argnames=("squarings",))
def _exp_scaled(a: Array, squarings: int) -> Array:
    """Taylor series of ``exp(a / 2^squarings)`` followed by ``squarings`` squarings."""
    x = a / 2.0**squarings
    term = jnp.eye(a.shape[0], dtype=a.dtype)
    total = term
    for k in range(1, _TAYLOR_TERMS + 1):
        term = term @ x / k
        total = total + term
    for _ in range(squarings):
        total = total @ total
    return total


def expm_antisymmetric(a: Array) -> Array:
    """Exponential of a real matrix by scaling, a degree-20 Taylor series and squaring.

    Accurate to about 1e-15 for the generators used here; ``jax.scipy.linalg.expm`` was
    measured to lose up to 3e-10 for these matrices, which are normal with integer spectrum.

    Parameters
    ----------
    a
        Square float64 matrix, typically antisymmetric (so the result is orthogonal).

    Returns
    -------
    Array
        ``exp(a)``, same shape as ``a``.
    """
    norm = float(jnp.max(jnp.sum(jnp.abs(a), axis=1)))
    squarings = max(0, int(jnp.ceil(jnp.log2(max(norm, 1e-300) / _TARGET_NORM))))
    return _exp_scaled(a, squarings)
