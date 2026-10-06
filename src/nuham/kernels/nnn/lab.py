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

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from nuham.kernels.compile import NO_AUTOTUNE


@partial(jax.jit, static_argnames=("shape",))
def block_diagonal_operator(
    v_flat: Array, src: Array, rows: Array, cols: Array, shape: tuple[int, int]
) -> Array:
    """Return ``V_bd`` of the given ``shape``, built from the Jacobi operators on the device.

    Parameters
    ----------
    v_flat
        All Jacobi operators raveled and concatenated, with a zero as the last element.
    src
        Index into ``v_flat`` of every non-zero entry of ``V_bd``; padding entries point at the
        trailing zero.
    rows, cols
        Position of each entry in ``V_bd``, shape as ``src`` (padding entries may be anything).
    shape
        ``(R_b, R_k)``.
    """
    return jnp.zeros(shape, v_flat.dtype).at[rows, cols].add(v_flat[src])


@partial(jax.jit, static_argnames=("dtype",), compiler_options=NO_AUTOTUNE)
def congruence_block(
    t_bra: Array, t_ket: Array, v_bd: Array, scale: float, dtype: type = jnp.float64
) -> Array:
    """Return ``scale * T_b^T V_bd T_k`` as ``dtype``.

    The coefficient rows ``t_bra`` and ``t_ket`` have shapes ``(R_b, C_b)`` and ``(R_k, C_k)``
    (zero padding allowed) and ``v_bd`` has shape ``(R_b, R_k)``. The result has shape
    ``(C_b, C_k)``. Building ``V_bd`` is a separate kernel so that the compiled shapes of the two
    steps do not multiply.
    """
    return (scale * (t_bra.T @ (v_bd @ t_ket))).astype(dtype)
