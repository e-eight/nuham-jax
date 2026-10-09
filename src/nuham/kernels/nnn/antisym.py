"""Antisymmetrisation ``C W C^T`` of 3N Jacobi operators on the device (pure kernels).

``C`` stacks the antisymmetriser's ``cfp`` rows and is block-diagonal in the oscillator quanta
``N``. It is used as one dense matrix: two GEMMs do about 40 times the flops of a loop over the
blocks, but have one compiled shape per channel (the block shapes differ in every channel, and a
loop over them recompiled for each), and a device does them in about a second at the largest
channel.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from nuham.kernels.compile import NO_AUTOTUNE


@partial(jax.jit, static_argnames="shape")
def dense_from_coo(rows: Array, cols: Array, values: Array, shape: tuple[int, int]) -> Array:
    """Dense matrix with ``values`` added at ``(rows, cols)``, so duplicate entries sum.

    Entries with an index outside ``shape`` are dropped, which lets callers pad the lists to a
    fixed length (fewer compiled shapes).
    """
    return jnp.zeros(shape, dtype=values.dtype).at[rows, cols].add(values, mode="drop")


@partial(jax.jit, static_argnames="symmetric", compiler_options=NO_AUTOTUNE)
def antisymmetrize(w: Array, c: Array, symmetric: bool = True) -> Array:
    """Return ``C W C^T``, symmetrised against rounding when ``w`` is symmetric.

    Parameters
    ----------
    w
        NAS matrix, shape ``(n_nas, n_nas)``.
    c
        Dense block-diagonal ``cfp``, shape ``(n_as, n_nas)``.
    symmetric
        ``w`` is symmetric; ``False`` keeps the product as it is (the frequency overlap of
        :mod:`nuham.nnn.frequency` is not symmetric, and symmetrising would drop its odd part).
    """
    out = (c @ w) @ c.T
    return 0.5 * (out + out.T) if symmetric else out
