"""Projection of local radial kernels onto products of Jacobi HO radial functions (pure kernels)."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from nuham.kernels.ho import radial_functions


@partial(jax.jit, static_argnames=("l_max", "n_max"))
def orbital_table(nu: ArrayLike, xi: ArrayLike, xi_w: ArrayLike, l_max: int, n_max: int) -> Array:
    """Weighted HO radial functions ``sqrt(w_i) u_{nl}(xi_i)`` for ``l <= l_max``, ``n <= n_max``.

    ``u = r R`` is normalised to ``int u^2 dr = 1`` and positive near the origin, so that
    ``sum_i table[l, n, i] table[l, n', i]`` is the quadrature overlap (1 up to mesh error).

    Parameters
    ----------
    nu
        Oscillator parameter ``2 m hw / (hbar c)^2`` in fm^-2.
    xi, xi_w
        Radial mesh (fm) and Gauss weights.

    Returns
    -------
    Array
        Shape ``(l_max + 1, n_max + 1, len(xi))``.
    """
    xi = jnp.asarray(xi, dtype=jnp.float64)
    sw = jnp.sqrt(jnp.asarray(xi_w, dtype=jnp.float64))
    nu = jnp.asarray(nu, dtype=jnp.float64)
    x = jnp.sqrt(nu) * xi
    tables = jax.vmap(lambda l: radial_functions(x, l, n_max))(jnp.arange(l_max + 1))
    return tables * nu**0.25 * sw


@jax.jit
def project_pair(
    u12_bra: Array, u3_bra: Array, u12_ket: Array, u3_ket: Array, grid: Array
) -> Array:
    """Matrix elements of the kernel ``grid[i, j]`` between product states of two channels.

    Returns ``M[n12, n3, n45, n6] = sum_ij u12_bra[n12, i] u12_ket[n45, i] grid[i, j] u3_bra[n3, j]
    u3_ket[n6, j]`` with the Gauss weights already inside the orbital tables.
    """
    first = jnp.einsum("ai,bi,ij->abj", u12_bra, u12_ket, grid)
    return jnp.einsum("abj,cj,dj->acbd", first, u3_bra, u3_ket)


@jax.jit
def project_pairs_indexed(
    orbitals: Array, lb: Array, l3b: Array, lk: Array, l3k: Array, grids: Array
) -> Array:
    """Batched :func:`project_pair` over a chunk of channel pairs; the orbitals are picked here.

    ``orbitals`` is the table of :func:`orbital_table`; ``lb, l3b, lk, l3k`` give the ``l`` of
    each pair's four orbitals and ``grids`` the kernel of each pair. Picking the orbital rows
    inside the compiled kernel replaced four eager gathers per chunk, which made 120 s of
    dispatch overhead in a GH200 run.
    """
    return jax.vmap(project_pair)(orbitals[lb], orbitals[l3b], orbitals[lk], orbitals[l3k], grids)
