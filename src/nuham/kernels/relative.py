"""Relative-coordinate operators in the HO basis."""

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def kinetic_matrix(n: ArrayLike, l: ArrayLike, hw: float) -> Array:
    """Relative kinetic energy ``T_rel`` between states ``(n, l)``, in MeV.

    Diagonal in ``l`` and tridiagonal in ``n``. The off-diagonal sign is positive, which matches
    the ``(-1)^n`` phase of the momentum-space radial functions.

    Parameters
    ----------
    n, l
        Integer arrays of equal length describing the basis states.
    hw
        Oscillator energy in MeV.

    Returns
    -------
    Array
        Symmetric matrix of shape ``(len(n), len(n))``.
    """
    n = jnp.asarray(n)
    l = jnp.asarray(l)
    n1, n2 = n[:, None], n[None, :]
    l1 = l[:, None]
    same_l = l1 == l[None, :]
    lo = jnp.minimum(n1, n2)
    diag = 2.0 * n1 + l1 + 1.5
    off = jnp.sqrt((lo + 1.0) * (lo + l1 + 1.5))
    x = jnp.where(n1 == n2, diag, jnp.where(jnp.abs(n1 - n2) == 1, off, 0.0))
    return 0.5 * hw * jnp.where(same_l, x, 0.0)
