"""Contraction of block-diagonal relative interactions with transformation coefficients."""

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
