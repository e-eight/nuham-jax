"""SRG evolution of the 3N Jacobi Hamiltonian and the stored three-body operator (Phase 5.3).

For one channel ``H0 = T + 3 V_NN(bare) + V_3N(bare)`` is flowed with generator ``T`` in the full
space ``N <= nmax`` of the ramp, and the stored operator is ``V3 = H(s) - T - 3 V_NN(s)``, the
embedded 2-body-evolved NN force being subtracted. The flow runs in units of ``hw`` so the
tolerances are well scaled, as in the NN flow.
"""

import jax.numpy as jnp
import numpy as np

from nuham.kernels.srg import srg_alpha, srg_flow_dense
from nuham.nnn.hamiltonian import IsospinNN, kinetic_jacobi, nn_jacobi
from nuham.nnn.jacobi import JacobiChannel


def evolve_hamiltonian(
    h0: np.ndarray,
    t: np.ndarray,
    hw: float,
    lam: float,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> np.ndarray:
    """Flow ``h0`` to the cutoff ``lam`` (fm^-1) with generator ``t``; all matrices in MeV."""
    out = srg_flow_dense(
        jnp.asarray(h0 / hw), jnp.asarray(t / hw), srg_alpha(lam) * hw**2, rtol, atol / hw
    )
    return np.asarray(out) * hw


def three_body_operator(
    channel: JacobiChannel,
    hw: float,
    lam: float,
    bare: IsospinNN,
    evolved: IsospinNN,
    v3n: np.ndarray | None = None,
    e3max: int | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> np.ndarray:
    """Return ``V3 = H(s) - T - 3 V_NN(s)`` of ``channel`` in MeV, cut to ``N <= e3max``.

    Parameters
    ----------
    channel
        Jacobi basis up to the ramp's ``nmax``; the flow runs in all of it.
    hw, lam
        Oscillator energy (MeV) and SRG cutoff (fm^-1).
    bare, evolved
        Isospin NN potentials from :func:`nuham.nnn.hamiltonian.isospin_nn`.
    v3n
        Bare genuine 3N force in the AS basis of ``channel``; ``None`` for the NN-only operator.
    e3max
        Keep only the leading block with ``N <= e3max`` (AS states are ordered by ``N``, so the cut
        is a principal sub-matrix); ``None`` keeps everything.
    rtol, atol
        Flow tolerances; ``atol`` in MeV.
    """
    t = kinetic_jacobi(channel, hw)
    h0 = t + nn_jacobi(channel, bare)
    if v3n is not None:
        h0 = h0 + v3n
    v3 = evolve_hamiltonian(h0, t, hw, lam, rtol, atol) - t - nn_jacobi(channel, evolved)
    if e3max is None:
        return v3
    keep = sum(len(b.cfp) for b in channel.blocks if b.n_quanta <= e3max)
    return v3[:keep, :keep]
