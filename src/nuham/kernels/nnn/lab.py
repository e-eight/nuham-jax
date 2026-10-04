"""Contraction of a Jacobi operator with the 3N transformation coefficients, one block pair.

The laboratory matrix of a channel is ``6 sum_g T_g^T V_g T_g`` over the centre-of-mass labels
``g = (jrel, ncm, lcm)``. Its block between the lab states of oscillator quanta ``N_b`` and
``N_k`` only involves the coefficient rows ``T_b`` and ``T_k`` of those two blocks, and the rows of
a block are the concatenation of the groups ``g``, so the sum over ``g`` is the product
``T_b^T V_bd T_k`` with ``V_bd`` block diagonal in ``g``. ``V_bd`` is small (the rows of one ``N``
block: thousands at the production size), so it is built dense on the device from the Jacobi
operators and the two big products run as plain GEMMs. This differs from
:func:`nuham.kernels.nn.lab.accumulate_blocks`, whose blocks have one common size and share one
coefficient matrix over all lab states.
"""

import jax
import jax.numpy as jnp
from jax import Array


@jax.jit
def congruence_block(
    t_bra: Array, t_ket: Array, v_flat: Array, src: Array, rows: Array, cols: Array
) -> Array:
    """Return ``T_b^T V_bd T_k`` for one block pair.

    Parameters
    ----------
    t_bra, t_ket
        Coefficient rows of the two blocks, shapes ``(R_b, C_b)`` and ``(R_k, C_k)``; padding rows
        and columns are zero.
    v_flat
        All Jacobi operators raveled and concatenated, with a zero as the last element.
    src
        Index into ``v_flat`` of every non-zero entry of ``V_bd``; padding entries point at the
        trailing zero.
    rows, cols
        Position of each entry in ``V_bd``, shape as ``src`` (padding entries may be anything).

    Returns
    -------
    Array
        Matrix of shape ``(C_b, C_k)``.
    """
    v_bd = jnp.zeros((t_bra.shape[0], t_ket.shape[0]), v_flat.dtype).at[rows, cols].add(v_flat[src])
    return t_bra.T @ (v_bd @ t_ket)


@jax.jit
def place_block(out: Array, block: Array, i: Array, j: Array) -> Array:
    """Return ``out`` with ``block`` written at ``[i, j]`` and its transpose at ``[j, i]``."""
    out = jax.lax.dynamic_update_slice(out, block, (i, j))
    return jax.lax.dynamic_update_slice(out, block.T, (j, i))
