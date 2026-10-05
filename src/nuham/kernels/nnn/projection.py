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
def project_weighted_pairs(
    orbitals: Array,
    lb: Array,
    l3b: Array,
    lk: Array,
    l3k: Array,
    weights: Array,
    radial: Array,
) -> Array:
    """Matrix elements of the weighted radial kernels of a chunk of channel pairs.

    The kernel of pair ``p`` is ``sum_a weights[p, a] radial[a]`` (the radial functions on the
    ``xi`` mesh, flattened), so the coefficients are all that travels to the device per chunk and
    the table ``radial`` stays there. The orbital rows are picked here too: both replaced host
    products and eager gathers that took more than half of the force time of a GH200 run once the
    angular coefficients were vectorised.

    Parameters
    ----------
    orbitals
        Table of :func:`orbital_table`.
    lb, l3b, lk, l3k
        ``l`` of each pair's four orbitals, shape ``(P,)``.
    weights
        Coefficients, shape ``(P, n_radial)``.
    radial
        Radial functions, shape ``(n_radial, n_xi * n_xi)``.

    Returns
    -------
    Array
        ``M[p, n12, n3, n45, n6]`` as :func:`project_pair`, shape ``(P, ...)``.
    """
    n_xi = orbitals.shape[-1]
    grids = (weights @ radial).reshape(-1, n_xi, n_xi)
    return jax.vmap(project_pair)(orbitals[lb], orbitals[l3b], orbitals[lk], orbitals[l3k], grids)
