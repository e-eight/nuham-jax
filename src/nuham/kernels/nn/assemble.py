"""Device-side assembly of the four ``me2j`` values per tuple from the lab matrices.

All matrices of a run sit in three flat arrays (one per isospin case), so the kernel has a static
shape and compiles once per run, not once per ``(J, parity)`` matrix shape.
"""

import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import Array

_SQRT2 = math.sqrt(2.0)


@jax.jit
def assemble_values(
    v: Sequence[Array],
    pos: Array,
    dim: Array,
    off: Array,
    j2: Array,
    a: Array,
    b: Array,
    c: Array,
    d: Array,
    jj: Array,
    g: Array,
) -> Array:
    """Return the ``T = 0``, nn, pn ``T = 1`` and pp values of a batch of tuples, shape ``(N, 4)``.

    The arithmetic follows :func:`nuham.nn.me2j.assemble`: the four ``Tz = 0`` elements combined
    with the exchange phases, and the nn and pp elements scaled by the normalisation factors.

    Parameters
    ----------
    v
        Three flat arrays holding the row-major lab matrices of all ``(J, parity)`` groups for
        ``Tz = 0``, ``+1`` and ``-1`` (in this order), each with at least one element.
    pos
        Index of each orbit pair within the pair basis of a matrix, ``-1`` outside it; shape
        ``(3, G, n_orb * n_orb)``, int32.
    dim, off
        Side length and start offset in ``v`` of every matrix, shape ``(3, G)``, int64.
    j2
        Twice ``j`` of every orbit, shape ``(n_orb,)``.
    a, b, c, d, jj, g
        Orbit indices, ``J`` and group index of every tuple, shape ``(N,)``. Padding tuples may
        use any valid indices; their values are meaningless.

    Returns
    -------
    Array
        Values in the un-normalised Darmstadt convention, float64.
    """
    n_orb = j2.shape[0]

    def element(t: int, x: Array, y: Array, z: Array, w: Array) -> Array:
        i = pos[t, g, x * n_orb + y]
        k = pos[t, g, z * n_orb + w]
        ok = (i >= 0) & (k >= 0)
        idx = off[t, g] + i.astype(jnp.int64) * dim[t, g] + k
        return jnp.where(ok, v[t][jnp.where(ok, idx, 0)], 0.0)

    def phase(x: Array, y: Array) -> Array:
        m = (j2[x] + j2[y]) // 2 - jj
        return -(1.0 - 2.0 * (m & 1).astype(jnp.float64))

    ph_ab = phase(a, b)
    ph_cd = phase(c, d)
    nf = jnp.where(a == b, _SQRT2, 1.0) * jnp.where(c == d, _SQRT2, 1.0)
    x1 = element(0, a, b, c, d)
    x2 = element(0, b, a, d, c) * ph_ab * ph_cd
    x3 = element(0, a, b, d, c) * ph_cd
    x4 = element(0, b, a, c, d) * ph_ab
    return jnp.stack(
        [
            0.5 * (x1 + x2 - x3 - x4),
            element(1, a, b, c, d) * nf,
            0.5 * (x1 + x2 + x3 + x4),
            element(2, a, b, c, d) * nf,
        ],
        axis=1,
    )
