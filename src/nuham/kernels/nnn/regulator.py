"""Non-local regulator of the LNL three-nucleon force in the Jacobi HO basis (pure kernel).

``F(p, k) = exp(-((p^2 + k^2) / 2 * (hbar c)^2 / Lambda^2)^n)`` depends on the momenta ``p`` and
``k`` conjugate to the two Jacobi vectors only through the hyperradius, so it is diagonal in the
angular quantum numbers of a state and couples only the radial ones. The kernel returns its
matrix for one pair of orbital angular momenta ``(l12, l3)``; ``nnn/regulator.py`` scatters it
into the states of a channel.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from nuham.constants import HBARC
from nuham.kernels.ho import momentum_radial


@partial(jax.jit, static_argnames=("n_max", "power"))
def regulator_table(
    l12: ArrayLike,
    l3: ArrayLike,
    n_max: int,
    b: ArrayLike,
    cutoff: ArrayLike,
    power: int,
    p: ArrayLike,
    w: ArrayLike,
) -> Array:
    """Return ``T[a, a', c, c'] = <a l12, c l3| F |a' l12, c' l3>`` for ``a, c <= n_max``.

    ``sum_ik w_i w_k u_a(p_i) u_a'(p_i) u_c(p_k) u_c'(p_k) F(p_i, p_k)`` with the momentum-space
    HO functions of :func:`nuham.kernels.ho.momentum_radial` (phase ``(-1)^n`` included, which
    does not cancel between different ``n``). One mesh serves both momenta.

    Parameters
    ----------
    l12, l3
        Orbital angular momenta of the two Jacobi vectors.
    n_max
        Largest radial quantum number (static).
    b
        Oscillator length of the Jacobi basis in fm, ``1 / sqrt(2 m_red hw / (hbar c)^2)``.
    cutoff
        Non-local cutoff Lambda in MeV.
    power
        Exponent ``n`` of the regulator (static).
    p, w
        Momentum mesh in fm^-1 and its Gauss weights.
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    w = jnp.asarray(w, dtype=jnp.float64)
    cutoff = jnp.asarray(cutoff, dtype=jnp.float64)
    f = jnp.exp(-((0.5 * (p[:, None] ** 2 + p[None, :] ** 2) * HBARC**2 / cutoff**2) ** power))
    u12 = momentum_radial(p, l12, n_max, b)
    u3 = momentum_radial(p, l3, n_max, b)
    pair12 = u12[:, None, :] * u12[None, :, :] * w
    pair3 = u3[:, None, :] * u3[None, :, :] * w
    first = jnp.einsum("abi,ik->abk", pair12, f)
    return jnp.einsum("abk,cdk->abcd", first, pair3)
