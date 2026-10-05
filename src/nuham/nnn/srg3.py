"""SRG evolution of the 3N Jacobi Hamiltonian and the stored three-body operator (Phase 5.3).

For one channel ``H0 = T + 3 V_NN(bare) + V_3N(bare)`` is flowed with generator ``T`` in the full
space ``N <= nmax`` of the ramp, and the stored operator is ``V3 = H(s) - T - 3 V_NN(s)``, the
embedded 2-body-evolved NN force being subtracted. The flow runs in units of ``hw`` so the
tolerances are well scaled, as in the NN flow.
"""

import jax
import jax.numpy as jnp
import numpy as np

from nuham.kernels.srg import srg_alpha, srg_flow_dense_stats
from nuham.kernels.transfer import to_host
from nuham.nnn.hamiltonian import IsospinNN, kinetic_jacobi, nn_jacobi
from nuham.nnn.jacobi import JacobiChannel
from nuham.timing import StageTimer, stage


def evolve_hamiltonian(
    h0: np.ndarray | jax.Array,
    t: np.ndarray | jax.Array,
    hw: float,
    lam: float,
    rtol: float = 1e-10,
    atol: float = 1e-10,
    stats: dict[str, int] | None = None,
) -> jax.Array:
    """Flow ``h0`` to the cutoff ``lam`` (fm^-1) with generator ``t``; all matrices in MeV.

    The result stays on the device.

    If ``stats`` is given it receives the number of integrator steps as ``stats["steps"]``.
    """
    out, steps = srg_flow_dense_stats(
        jnp.asarray(h0) / hw, jnp.asarray(t) / hw, srg_alpha(lam) * hw**2, rtol, atol / hw
    )
    if stats is not None:
        stats["steps"] = int(steps)
    return out * hw


def three_body_operator(
    channel: JacobiChannel,
    hw: float,
    lam: float,
    bare: IsospinNN,
    evolved: IsospinNN,
    v3n: np.ndarray | jax.Array | None = None,
    e3max: int | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-10,
    timer: StageTimer | None = None,
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
    timer
        Optional stage timer; records ``embed`` (the matrices in the AS basis) and ``flow``.
    """
    with stage(timer, "embed"):
        t = kinetic_jacobi(channel, hw)
        h0 = t + nn_jacobi(channel, bare)
        if v3n is not None:
            h0 = h0 + jnp.asarray(v3n)
        h0.block_until_ready()  # the device is asynchronous: charge the work to its own stage
    with stage(timer, "flow"):
        evolved_h = evolve_hamiltonian(h0, t, hw, lam, rtol, atol)
        evolved_h.block_until_ready()
    with stage(timer, "embed"):
        v3 = evolved_h - t - nn_jacobi(channel, evolved)
        v3.block_until_ready()
    if e3max is not None:
        keep = sum(len(b.cfp) for b in channel.blocks if b.n_quanta <= e3max)
        v3 = v3[:keep, :keep]
    with stage(timer, "to_host"):
        return to_host(v3)
