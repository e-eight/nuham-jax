"""Projection of momentum-space partial-wave matrix elements onto the HO relative basis."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from nuham.kernels.ho import momentum_radial


def weighted_radial(k: ArrayLike, w: ArrayLike, l: int, n_max: int, b: ArrayLike) -> Array:
    """Quadrature-weighted radial functions ``k_i w_i R_nl(k_i)``, shape ``(n_max + 1, N)``."""
    k = jnp.asarray(k, dtype=jnp.float64)
    return momentum_radial(k, l, n_max, b) * (k * jnp.asarray(w, dtype=jnp.float64))[None, :]


@partial(jax.jit, static_argnames=("ls", "n2max"))
def project_channel(
    k: Array, w: Array, v_mom: Array, ls: tuple[int, ...], n2max: int, b: ArrayLike
) -> Array:
    """Project one channel's momentum-space matrix onto HO relative states.

    ``V_HO[n'l', nl] = sum_ij F_n'l'(k_i) V(k_i, k_j) F_nl(k_j)`` with ``F = k w R``.
    States are ordered l-major (all n of the first ``l``, then the next), with
    ``2n + l <= n2max``.

    Parameters
    ----------
    k, w
        Quadrature mesh and weights, shape ``(N,)``.
    v_mom
        Momentum-space matrix, shape ``(len(ls) * N, len(ls) * N)``, blocks ordered like ``ls``.
    ls
        Orbital angular momenta of the channel (static).
    n2max
        Truncation on ``2n + l`` (static).
    b
        Oscillator length in fm.

    Returns
    -------
    Array
        Matrix of shape ``(D, D)`` in MeV, with ``D`` the number of retained states.
    """
    n_mesh = k.shape[0]
    f = [weighted_radial(k, w, l, (n2max - l) // 2, b) for l in ls]
    rows = []
    for i, fi in enumerate(f):
        cols = []
        for j, fj in enumerate(f):
            block = v_mom[i * n_mesh : (i + 1) * n_mesh, j * n_mesh : (j + 1) * n_mesh]
            cols.append(fi @ block @ fj.T)
        rows.append(cols)
    return jnp.block(rows)
