"""Overlap of harmonic-oscillator radial functions of two frequencies (pure kernel).

Changing the oscillator frequency of a Jacobi basis is a scale transformation of both Jacobi
vectors. It is diagonal in the angular quantum numbers and couples only the radial ones, so one
matrix per orbital angular momentum ``l`` describes it; ``nnn/frequency.py`` scatters it into the
states of a channel. ``docs/spec-5e-hw-conversion.md`` has the sources.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from nuham.kernels.ho import radial_functions


@partial(jax.jit, static_argnames=("n_max",))
def radial_overlap(
    l: ArrayLike, n_max: int, b_from: ArrayLike, b_to: ArrayLike, r: ArrayLike, w: ArrayLike
) -> Array:
    """Return ``O[n, n'] = <n l, b_from | n' l, b_to>`` for ``n, n' <= n_max``.

    ``sum_i w_i u_n(r_i; b_from) u_n'(r_i; b_to)`` with the coordinate-space radial functions
    ``u(r; b) = g(r / b) / sqrt(b)`` of :func:`nuham.kernels.ho.radial_functions` (no ``(-1)^n``
    phase; all functions are positive at the origin), normalised to 1 on ``(0, inf)``.

    Parameters
    ----------
    l
        Orbital angular momentum.
    n_max
        Largest radial quantum number (static).
    b_from, b_to
        Oscillator lengths in fm of the original and of the target basis.
    r, w
        Radial mesh in fm and its Gauss weights; it must reach beyond the functions of ``n_max``.
    """
    r = jnp.asarray(r, dtype=jnp.float64)
    w = jnp.asarray(w, dtype=jnp.float64)
    b_from = jnp.asarray(b_from, dtype=jnp.float64)
    b_to = jnp.asarray(b_to, dtype=jnp.float64)
    u_from = radial_functions(r / b_from, l, n_max) / jnp.sqrt(b_from)
    u_to = radial_functions(r / b_to, l, n_max) / jnp.sqrt(b_to)
    return (u_from * w) @ u_to.T
