"""Device-side text formatting of ``me2j`` values as fixed-width ``f16.8`` fields.

The integer arithmetic and lookup tables are those of the host formatter
(:func:`nuham.nn.me2j._format_lines`), so away from rounding ties the text is identical to Python's
``%16.8f``. Values whose scaled fraction is within a few ulp of 0.5 are returned as a short list
for the host to format with Python itself.
"""

from functools import cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

LIMIT = 1.0e5  # every value below this magnitude fits a 16-character field


@cache
def tables() -> tuple[Array, Array]:
    """Return the digit tables as device arrays: ``(frac, head)``.

    ``frac[i]`` holds the four ASCII digits of ``i`` (0..9999), shape ``(10000, 4)``. ``head[i]``
    holds the first eight characters of a field: the right-aligned integer part and the decimal
    point, signed when ``i >= 100000`` (which encodes the negative value ``-(i - 100000)``), shape
    ``(200000, 8)``.
    """
    frac = np.frombuffer(
        "".join(f"{i:04d}" for i in range(10_000)).encode("ascii"), dtype=np.uint8
    ).reshape(-1, 4)
    head = np.frombuffer(
        "".join(
            f"{'-' if neg else ''}{i}".rjust(7) + "."
            for neg in (False, True)
            for i in range(100_000)
        ).encode("ascii"),
        dtype=np.uint8,
    ).reshape(-1, 8)
    return jnp.asarray(frac), jnp.asarray(head)


@partial(jax.jit, static_argnames=("capacity",))
def format_fields(
    values: Array, n_valid: Array, frac: Array, head: Array, capacity: int = 4096
) -> tuple[Array, Array, Array, Array, Array]:
    """Format values as 16-byte fields and list the ones the host must format itself.

    Parameters
    ----------
    values
        Values to format, any shape; flattened in C order.
    n_valid
        Number of leading values (of the flattened array) that are real; the rest is padding and
        is formatted as zero.
    frac, head
        The tables of :func:`tables`.
    capacity
        Length of the returned list of flagged values (static).

    Returns
    -------
    tuple[Array, Array, Array, Array, Array]
        ``fields`` of shape ``(M, 16)``, uint8; the number of flagged values (near-ties); the
        number of unformattable values (non-finite or beyond ``LIMIT``); and the first
        ``capacity`` flagged indices (``-1`` for unused entries) with their values.
    """
    x = values.ravel()
    real = jnp.arange(x.shape[0]) < n_valid
    x = jnp.where(real, x, 0.0)
    unformattable = real & ~(jnp.abs(x) < LIMIT)
    x = jnp.where(unformattable, 0.0, x)
    scaled = jnp.abs(x) * 1.0e8
    q = jnp.rint(scaled).astype(jnp.int64)
    near = jnp.abs(jnp.abs(scaled - jnp.floor(scaled)) - 0.5) <= 8.0 * jnp.spacing(scaled)
    whole, frac_part = jnp.divmod(q, 100_000_000)
    hi, lo = jnp.divmod(frac_part, 10_000)
    head_index = whole + 100_000 * jnp.signbit(x).astype(jnp.int64)
    fields = jnp.concatenate([head[head_index], frac[hi], frac[lo]], axis=1)
    flagged = near & real
    (index,) = jnp.nonzero(flagged, size=capacity, fill_value=-1)
    return (
        fields,
        flagged.sum(),
        unformattable.sum(),
        index.astype(jnp.int32),
        jnp.where(index >= 0, x[jnp.maximum(index, 0)], 0.0),
    )
