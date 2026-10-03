"""Contraction of block-diagonal relative interactions with transformation coefficients."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array


@jax.jit
def congruence_sum(t_blocks: Array, v_blocks: Array) -> Array:
    """Return ``sum_b T_b^T V_b T_b`` for a stack of blocks.

    The sum over blocks and rows collapses into one dense matrix product, so the cost is one
    GEMM of size ``P x (n_blocks * r) x P`` plus a batch of small ``r x r`` products. Zero-padded
    rows and blocks contribute nothing.

    Parameters
    ----------
    t_blocks
        Transformation coefficients, shape ``(n_blocks, r, P)``.
    v_blocks
        Relative interaction blocks, shape ``(n_blocks, r, r)``.

    Returns
    -------
    Array
        Matrix of shape ``(P, P)``.
    """
    w = jnp.einsum("brs,bsq->brq", v_blocks, t_blocks)
    return jnp.einsum("brp,brq->pq", t_blocks, w)


@partial(jax.jit, donate_argnums=0)
def accumulate_blocks(total: Array, t_rows: Array, row_index: Array, v_blocks: Array) -> Array:
    """Return ``total + sum_b T_b^T V_b T_b`` with the stacks gathered on the device.

    ``T_b = t_rows[row_index[b]]``, so the coefficient matrix is uploaded once per channel and
    only the small index and block arrays change between calls. ``total`` is donated and updated
    in place.

    Parameters
    ----------
    total
        Running sum, shape ``(P, P)``; its buffer is reused for the result.
    t_rows
        Coefficients of all relative-CM rows, shape ``(n_rows, P)``; any row that padding indices
        point at must be zero.
    row_index
        Integer array of shape ``(n_blocks, r)`` selecting the rows of each block.
    v_blocks
        Relative interaction blocks, shape ``(n_blocks, r, r)``.

    Returns
    -------
    Array
        The updated sum, shape ``(P, P)``.
    """
    return total + congruence_sum(t_rows[row_index], v_blocks)


@partial(jax.jit, donate_argnums=0)
def accumulate_blocks_multi(
    total: Array, t_rows: Array, row_index: Array, v_blocks: Array
) -> Array:
    """Like :func:`accumulate_blocks` for several parameter sets that share the coefficients.

    The coefficient stack is gathered once and used for every set, so the dominant product is one
    GEMM of size ``P x (n_blocks * r) x (K * P)``.

    Parameters
    ----------
    total
        Running sums, shape ``(K, P, P)``; the buffer is reused for the result.
    t_rows
        Coefficients of all relative-CM rows, shape ``(n_rows, P)``.
    row_index
        Integer array of shape ``(n_blocks, r)`` selecting the rows of each block.
    v_blocks
        Relative interaction blocks of every set, shape ``(K, n_blocks, r, r)``.

    Returns
    -------
    Array
        The updated sums, shape ``(K, P, P)``.
    """
    t = t_rows[row_index]
    w = jnp.einsum("kbrs,bsq->kbrq", v_blocks, t)
    return total + jnp.einsum("brp,kbrq->kpq", t, w)
