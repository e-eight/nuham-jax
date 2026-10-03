"""Similarity renormalization group flow of a Hamiltonian ``H = T + V`` with generator ``T``."""

from functools import partial

import diffrax
import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array

from nuham.constants import HBARC, M_REDUCED_PN


def srg_alpha(lam: float, mass: float = M_REDUCED_PN) -> float:
    """Return the final flow parameter ``s = 1 / lc^4`` in MeV^-2 for the cutoff ``lam`` in fm^-1.

    ``lc = lam * hbar c / sqrt(2 mass)`` is the cutoff in sqrt(MeV); this is ``s = 1/lam^4`` in
    fm^4 expressed in energy units.
    """
    lc = lam * HBARC / (2.0 * mass) ** 0.5
    return lc**-4


def _commutator_rhs(h: Array, d: Array, e_up: Array, e_low: Array) -> Array:
    """Right-hand side ``[[T, H], H]`` using the tridiagonal structure of ``T``.

    With ``eta = TH - HT = A - A^T`` for ``A = TH``, and ``H`` symmetric, the right-hand side is
    ``eta H + (eta H)^T``, so only one dense matrix product is needed.
    """
    a = (
        d[:, None] * h
        + e_up[:, None] * jnp.roll(h, -1, axis=0)
        + e_low[:, None] * jnp.roll(h, 1, axis=0)
    )
    m = (a - a.T) @ h
    return m + m.T


@partial(jax.jit, static_argnames=("max_steps",))
def srg_flow(
    h0: Array,
    t_diag: Array,
    t_off: Array,
    s_final: float,
    rtol: float = 1e-10,
    atol: float = 1e-10,
    max_steps: int = 100_000,
) -> Array:
    """Integrate ``dH/ds = [[T, H], H]`` from ``s = 0`` to ``s_final`` for one channel.

    ``T`` is symmetric tridiagonal, given by its diagonal and first off-diagonal; entries of
    ``t_off`` that cross a block boundary (a change of ``l``) must be zero. Zero-padded rows and
    columns do not evolve, so batches of different sizes can be padded to a common size.
    Uses an adaptive eighth-order Runge-Kutta method with a max-norm error control.

    Parameters
    ----------
    h0
        Initial Hamiltonian, shape ``(n, n)``.
    t_diag
        Diagonal of ``T``, shape ``(n,)``.
    t_off
        ``T[i, i+1]``, shape ``(n - 1,)``.
    s_final
        Final flow parameter, in units inverse to those of ``h0`` squared.
    rtol, atol
        Tolerances in the units of ``h0``.
    max_steps
        Step limit (static); exceeding it raises.

    Returns
    -------
    Array
        Evolved Hamiltonian, shape ``(n, n)``.
    """
    zero = jnp.zeros(1, dtype=h0.dtype)
    e_up = jnp.concatenate([t_off, zero])
    e_low = jnp.concatenate([zero, t_off])
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda _t, h, _args: _commutator_rhs(h, t_diag, e_up, e_low)),
        diffrax.Dopri8(),
        t0=0.0,
        t1=s_final,
        dt0=None,
        y0=h0,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol, norm=optx.max_norm),
        max_steps=max_steps,
        saveat=diffrax.SaveAt(t1=True),
    )
    return solution.ys[0]


@partial(jax.jit, static_argnames=("max_steps",))
def srg_evolve_batch(
    h0: Array,
    t_diag: Array,
    t_off: Array,
    hw: float | Array,
    alpha: float | Array,
    rtol: float = 1e-10,
    atol: float = 1e-10,
    max_steps: int = 100_000,
) -> Array:
    """Evolve a batch of channels, in units of ``hw`` so that the tolerances are well scaled.

    Each batch entry has its own ``hw`` and ``alpha``, so channels of several parameter sets can
    share one batch. Entries keep their own adaptive step sequence, so a batched result equals
    the one from a batch of its own.

    Parameters
    ----------
    h0
        Initial Hamiltonians in MeV, shape ``(B, n, n)``.
    t_diag, t_off
        Diagonals of ``T`` in MeV, shapes ``(B, n)`` and ``(B, n - 1)``.
    hw
        Oscillator energy in MeV: a scalar, or one value per entry, shape ``(B,)``.
    alpha
        Final flow parameter in MeV^-2: a scalar, or one value per entry.
    rtol, atol
        Tolerances; ``atol`` is in MeV.
    max_steps
        Step limit per channel (static).

    Returns
    -------
    Array
        Evolved Hamiltonians in MeV, shape ``(B, n, n)``.
    """
    batch = h0.shape[0]
    hw = jnp.broadcast_to(jnp.asarray(hw, dtype=h0.dtype), (batch,))
    alpha = jnp.broadcast_to(jnp.asarray(alpha, dtype=h0.dtype), (batch,))
    flow = jax.vmap(srg_flow, in_axes=(0, 0, 0, 0, None, 0, None))
    out = flow(
        h0 / hw[:, None, None],
        t_diag / hw[:, None],
        t_off / hw[:, None],
        alpha * hw**2,
        rtol,
        atol / hw,
        max_steps,
    )
    return out * hw[:, None, None]
