"""Radial functions of the local chiral N2LO three-nucleon force (pure kernels, no I/O).

Every function is a regulated Fourier-Bessel transform ``(1 / 2 pi^2) sum_p wf(p) p^m j_k(p r)``
on an explicit momentum mesh, in fm units (momenta in fm^-1). The mesh weights ``wf`` carry the
regulator and the pion propagator, so the same kernels give the delta functions, the Yukawa
tensors and their partial-wave expansions in the angle between the two Jacobi vectors.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike

from nuham.constants import HBARC

_X_MIN = 1e-9
_RESCALE = 1e150
_SERIES_X = 1.0
_SERIES_TERMS = 14


def local_regulator(p: ArrayLike, cutoff_mev: float, power: int) -> jax.Array:
    """Return ``exp(-(p hbar c / Lambda)^(2 power))`` for momenta ``p`` in fm^-1."""
    p = jnp.asarray(p, dtype=jnp.float64)
    return jnp.exp(-((p * HBARC / cutoff_mev) ** (2 * power)))


def _jn_over_xn_series(order: int, x: jax.Array) -> jax.Array:
    """Power series of ``j_order(x) / x^order``, accurate for ``x <= 1``."""
    double_fact = 1.0
    for m in range(1, 2 * order + 2, 2):
        double_fact *= m
    term = jnp.ones_like(x) / double_fact
    total = term
    x2 = x * x
    for k in range(1, _SERIES_TERMS):
        term = -term * x2 / (2.0 * k * (2.0 * order + 2.0 * k + 1.0))
        total = total + term
    return total


def jn_over_xn(order: int, x: ArrayLike) -> jax.Array:
    """Return ``j_order(x) / x^order`` for ``order`` 0, 1 or 2 without small-argument cancellation.

    The closed forms lose digits as ``x -> 0``, so a power series is used for ``x <= 1``.
    """
    x = jnp.maximum(jnp.asarray(x, dtype=jnp.float64), _X_MIN)
    s, c = jnp.sin(x), jnp.cos(x)
    if order == 0:
        closed = s / x
    elif order == 1:
        closed = (s / x - c) / x**2
    elif order == 2:
        closed = ((3.0 / x**2 - 1.0) * s / x - 3.0 * c / x**2) / x**2
    else:
        raise ValueError("order must be 0, 1 or 2")
    return jnp.where(x <= _SERIES_X, _jn_over_xn_series(order, jnp.minimum(x, _SERIES_X)), closed)


@partial(jax.jit, static_argnames=("n_max", "n_start"))
def _spherical_jn_miller(x: jax.Array, n_max: int, n_start: int) -> jax.Array:
    """Spherical Bessel functions ``j_0..j_n_max`` by Miller's downward recurrence.

    The sum rule ``sum (2n+1) j_n^2 = 1`` fixes the scale and the closed forms of ``j_0`` and
    ``j_1`` fix the sign. Values are rescaled whenever they pass 1e150 so that small arguments
    cannot overflow; the rescale count of each step is kept to put all orders on one scale.
    """
    x = jnp.maximum(x, _X_MIN)
    one = jnp.ones_like(x)

    def body(carry: tuple, n: jax.Array) -> tuple[tuple, tuple]:
        above, cur, shifts, ssum = carry  # f_{n+1}, f_n, rescale count, sum_{m >= n} (2m+1) f_m^2
        prev = (2.0 * n + 1.0) / x * cur - above
        big = jnp.abs(prev) > _RESCALE
        scale = jnp.where(big, 1.0 / _RESCALE, 1.0)
        prev, cur = prev * scale, cur * scale
        ssum = ssum * scale * scale + (2.0 * n - 1.0) * prev * prev
        shifts = shifts + big.astype(jnp.int32)
        return (cur, prev, shifts, ssum), (prev, shifts)

    init = (0.0 * one, one, jnp.zeros(x.shape, dtype=jnp.int32), (2.0 * n_start + 1.0) * one)
    ns = jnp.arange(n_start, 0, -1, dtype=jnp.float64)
    (_, _, shifts_end, ssum), (ys, ys_shifts) = lax.scan(body, init, ns)
    # ys[i] is f_{n_start-1-i}
    sel = n_start - 1 - jnp.arange(n_max + 1)
    vals = ys[sel] * jnp.exp((ys_shifts[sel] - shifts_end[None]) * jnp.log(_RESCALE))
    vals = vals / jnp.sqrt(ssum)
    exact0 = jnp.sin(x) / x
    exact1 = (exact0 - jnp.cos(x)) / x
    if n_max >= 1:
        ref = jnp.where(jnp.abs(exact0) >= jnp.abs(exact1), exact0 * vals[0], exact1 * vals[1])
    else:
        ref = exact0 * vals[0]
    return vals * jnp.where(ref < 0.0, -1.0, 1.0)


def spherical_jn(x: ArrayLike, n_max: int) -> jax.Array:
    """Spherical Bessel functions ``j_n(x)`` for ``n = 0..n_max`` at points ``x > 0``.

    Parameters
    ----------
    x
        Arguments, any shape. Not traced: the start order of the recurrence follows ``max(x)``.
    n_max
        Largest order (static).

    Returns
    -------
    jax.Array
        Shape ``(n_max + 1, *x.shape)``.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    n_start = n_max + int(float(jnp.max(x))) + 60
    out = _spherical_jn_miller(x.reshape(-1), n_max, n_start)
    return out.reshape((n_max + 1, *x.shape))


def legendre_table(c: ArrayLike, x_max: int) -> jax.Array:
    """Legendre polynomials ``P_0..P_x_max`` at ``c``, shape ``(x_max + 1, *c.shape)``."""
    c = jnp.asarray(c, dtype=jnp.float64)
    p0, p1 = jnp.ones_like(c), c
    if x_max == 0:
        return p0[None]

    def step(carry: tuple[jax.Array, jax.Array], n: jax.Array) -> tuple[tuple, jax.Array]:
        pm, pc = carry
        pn = ((2.0 * n + 1.0) * c * pc - n * pm) / (n + 1.0)
        return (pc, pn), pn

    _, rest = lax.scan(step, (p0, p1), jnp.arange(1, x_max, dtype=jnp.float64))
    return jnp.concatenate([p0[None], p1[None], rest], axis=0)


_INV_2PI2 = 1.0 / (2.0 * jnp.pi**2)


def radial_transform(
    r: ArrayLike, p: ArrayLike, wf: ArrayLike, order: int, power: int
) -> jax.Array:
    """Return ``(1 / 2 pi^2) sum_p wf(p) p^power j_order(p r)`` for orders 0-2 (shape of ``r``)."""
    r = jnp.asarray(r, dtype=jnp.float64)
    p = jnp.asarray(p, dtype=jnp.float64)
    x = r[..., None] * p
    j = jn_over_xn(order, x) * x**order
    return _INV_2PI2 * jnp.sum(jnp.asarray(wf) * p**power * j, axis=-1)


def radial_transform_over_r(
    r: ArrayLike, p: ArrayLike, wf: ArrayLike, order: int, power: int
) -> jax.Array:
    """Return ``(1/2pi^2) sum_p wf p^power j_order(p r) / r^order``, finite at ``r -> 0``."""
    r = jnp.asarray(r, dtype=jnp.float64)
    p = jnp.asarray(p, dtype=jnp.float64)
    j = jn_over_xn(order, r[..., None] * p)
    return _INV_2PI2 * jnp.sum(jnp.asarray(wf) * p ** (power + order) * j, axis=-1)


def bilocal_transform(
    s: ArrayLike, a: ArrayLike, p: ArrayLike, wf: ArrayLike, power: int, x_max: int
) -> jax.Array:
    """Coefficients ``(1/2pi^2) sum_p wf p^power j_x(p s) j_x(p a)``, ``x = 0..x_max``.

    These are the Legendre coefficients ``c_x`` of ``g(|a - s|) = sum_x (2x+1) c_x P_x(cos)``
    when ``g`` is the transform of the same weights with order 0 (addition theorem).

    Returns
    -------
    jax.Array
        Shape ``(x_max + 1, len(s), len(a))``.
    """
    s = jnp.asarray(s, dtype=jnp.float64)
    a = jnp.asarray(a, dtype=jnp.float64)
    p = jnp.asarray(p, dtype=jnp.float64)
    js = spherical_jn(s[:, None] * p[None, :], x_max)
    ja = spherical_jn(a[:, None] * p[None, :], x_max)
    weights = _INV_2PI2 * jnp.asarray(wf) * p**power
    return jnp.einsum("xip,p,xjp->xij", js, weights, ja)


def legendre_projected_over_r(
    s: ArrayLike,
    a: ArrayLike,
    cos: ArrayLike,
    cos_w: ArrayLike,
    p: ArrayLike,
    wf: ArrayLike,
    order: int,
    power: int,
    x_max: int,
) -> jax.Array:
    """Legendre coefficients ``(1/2) int dc P_x(c) g(r)`` of ``g(r) = T_order(r) / r^order``.

    ``r = sqrt(s^2 + a^2 - 2 s a c)`` is the distance between the two vectors and ``T_order`` the
    radial transform of :func:`radial_transform_over_r`. The loop over ``s`` keeps memory at
    ``len(a) * len(cos) * len(p)``.

    Returns
    -------
    jax.Array
        Shape ``(x_max + 1, len(s), len(a))``.
    """
    s = jnp.asarray(s, dtype=jnp.float64)
    a = jnp.asarray(a, dtype=jnp.float64)
    cos = jnp.asarray(cos, dtype=jnp.float64)
    leg = legendre_table(cos, x_max) * (0.5 * jnp.asarray(cos_w))  # (X, nc)

    def one_s(si: jax.Array) -> jax.Array:
        r = jnp.sqrt(
            jnp.maximum(si**2 + a[:, None] ** 2 - 2.0 * si * a[:, None] * cos[None, :], 0.0)
        )
        g = radial_transform_over_r(r, p, wf, order, power)  # (na, nc)
        return jnp.einsum("xc,ac->xa", leg, g)

    out = lax.map(one_s, s)  # (ns, X, na)
    return jnp.transpose(out, (1, 0, 2))
