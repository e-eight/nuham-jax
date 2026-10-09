"""Hamiltonian of the 3N system in the antisymmetrised Jacobi basis (Phase 5.3).

``H0 = T + 3 V_NN + V_3N`` with the kinetic energy and the NN force embedded from the pair (1,2)
into the non-antisymmetrised (NAS) basis and then antisymmetrised, ``C^T W C``
(:func:`nuham.nnn.local_force.assembly.antisymmetrize`, on the device). The NN force is the
isospin-averaged charge-independent one without Coulomb: ``t12 = 0`` takes the pn channel,
``t12 = 1`` the mean of pp, pn and nn. Relative ``j12`` above the file's ``jmax`` has no NN
force. Nothing here depends on the SRG flow; :mod:`nuham.nnn.srg3` evolves the matrices.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import jax
import numpy as np
import scipy.sparse as sp

from nuham.nn.input_file import MomentumNN
from nuham.nn.relative import RelativeChannel, evolve_potentials, evolve_relative_nn_many
from nuham.nnn.jacobi import JacobiChannel
from nuham.nnn.local_force.assembly import antisymmetrize


@dataclass(frozen=True)
class IsospinChannel:
    """Relative NN potential of one ``(j12, parity, s12, t12)`` channel, charge-averaged.

    Attributes
    ----------
    n, l
        Quantum numbers of the relative basis states (l-major).
    v
        Potential in MeV, shape ``(len(n), len(n))``.
    """

    n: np.ndarray
    l: np.ndarray
    v: np.ndarray

    @cached_property
    def index(self) -> dict[tuple[int, int], int]:
        """Row of the potential for each ``(n, l)`` state."""
        return {(int(n), int(l)): i for i, (n, l) in enumerate(zip(self.n, self.l, strict=True))}


N2MAX_FLOW = 100
"""Relative space of the NN flow; a smaller one deviates from NuHamil (see the SRG study)."""

JMAX_NN = 8
"""Largest relative ``J`` of the NN force; higher ones are zero."""

IsospinNN = dict[tuple[int, int, int, int], IsospinChannel]
"""Isospin channels keyed by ``(j12, parity, s12, t12)`` with ``t12`` the true isospin."""


def isospin_channels(channels: Sequence[RelativeChannel]) -> IsospinNN:
    """Combine ``(J, parity, S, Tz)`` channels into charge-independent isospin channels.

    In a channel of given ``(J, parity, S)`` the isospin follows from ``l + s + t`` odd, so every
    channel has one ``t12``. ``t12 = 1`` averages the pp, pn and nn potentials with equal weights
    (NuHamil's convention for the 3N force); ``t12 = 0`` is the pn one.
    """
    groups: dict[tuple[int, int, int], list[RelativeChannel]] = {}
    for ch in channels:
        groups.setdefault((ch.j, ch.parity, ch.s), []).append(ch)
    out: IsospinNN = {}
    for (j, parity, s), members in groups.items():
        t = (int(parity < 0) + s + 1) % 2
        want = {-1, 0, 1} if t == 1 else {0}
        chosen = [c for c in members if c.tz in want]
        if {c.tz for c in chosen} != want:
            raise ValueError(f"channel J={j}, parity={parity}, S={s} lacks charges {want}")
        v = sum(c.v for c in chosen) / len(chosen)
        out[j, parity, s, t] = IsospinChannel(chosen[0].n, chosen[0].l, np.asarray(v))
    return out


def isospin_nn(
    nn: MomentumNN,
    hw: float,
    lam: float | None,
    n2max: int = N2MAX_FLOW,
    jmax: int = JMAX_NN,
) -> tuple[IsospinNN, IsospinNN]:
    """Return ``(bare, evolved)`` isospin-averaged NN potentials, without Coulomb.

    As in NuHamil the charge average is taken on the bare potentials and the average is what the
    SRG flows (the flow is nonlinear, so averaging the flowed pp, pn and nn would differ at the
    1e-4 level). They depend only on ``(hw, lam)``, so one pair serves every 3N channel (NuHamil
    recomputes it per channel). The evolved one is the 2-body flow in the full ``n2max`` space,
    subtracted from the 3-body flow at the end.
    """
    (bare_rel,) = evolve_relative_nn_many(nn, [(hw, None)], n2max=n2max, jmax=jmax, coulomb=False)
    bare = isospin_channels(bare_rel)
    if not lam:
        return bare, bare
    keys = list(bare)
    flowed = evolve_potentials([(bare[k].n, bare[k].l, bare[k].v) for k in keys], hw, lam)
    return bare, {
        k: IsospinChannel(bare[k].n, bare[k].l, v) for k, v in zip(keys, flowed, strict=True)
    }


def nas_of(channel: JacobiChannel) -> np.ndarray:
    """NAS states of all blocks of ``channel`` in the order used by :func:`antisymmetrize`."""
    return np.concatenate([b.nas for b in channel.blocks])


def kinetic_nas(nas: np.ndarray, hw: float) -> sp.csr_array:
    """Sparse ``t12 + t3`` on the NAS states: ``(2n + l + 3/2) hw/2`` plus the ``n -> n + 1`` links.

    Both Jacobi coordinates use the same ``hw`` and the same tridiagonal formula as the relative
    kinetic energy of two nucleons. States missing from ``nas`` (above the truncation) are dropped.
    """
    rows = nas.tolist()
    where = {tuple(r): i for i, r in enumerate(rows)}
    ii: list[int] = []
    jj: list[int] = []
    vv: list[float] = []
    for i, r in enumerate(rows):
        n12, l12, n3, l3 = r[0], r[1], r[5], r[6]
        ii.append(i)
        jj.append(i)
        vv.append(0.5 * hw * (2 * n12 + l12 + 2 * n3 + l3 + 3.0))
        for col, n, l in ((0, n12, l12), (5, n3, l3)):
            up = list(r)
            up[col] += 1
            k = where.get(tuple(up))
            if k is not None:
                off = 0.5 * hw * np.sqrt((n + 1.0) * (n + l + 1.5))
                ii += [i, k]
                jj += [k, i]
                vv += [off, off]
    return sp.csr_array((vv, (ii, jj)), shape=(len(rows), len(rows)))


def nn_nas(nas: np.ndarray, iso: IsospinNN) -> sp.csr_array:
    """Sparse NAS matrix of one NN potential acting on the pair (1,2), identity on the spectator.

    Matrix elements couple states that agree in ``(j12, s12, t12, n3, l3, j3)`` and in the parity
    of ``l12``; ``j12`` above the potential's range gives zero.
    """
    groups: dict[tuple[int, ...], list[int]] = {}
    for i, r in enumerate(nas.tolist()):
        n12, l12, s12, j12, t12, n3, l3, j3 = r
        groups.setdefault((j12, l12 % 2, s12, t12, n3, l3, j3), []).append(i)
    ii: list[np.ndarray] = []
    jj: list[np.ndarray] = []
    vv: list[np.ndarray] = []
    for (j12, lpar, s12, t12, *_), members in groups.items():
        channel = iso.get((j12, 1 - 2 * lpar, s12, t12))
        if channel is None:
            continue
        rows = nas[members]
        pos = np.array([channel.index.get((int(n), int(l)), -1) for n, l in rows[:, :2]])
        if (pos < 0).any():
            raise ValueError("NAS state outside the relative space of the NN potential")
        members_arr = np.array(members)
        ii.append(np.repeat(members_arr, len(members_arr)))
        jj.append(np.tile(members_arr, len(members_arr)))
        vv.append(channel.v[np.ix_(pos, pos)].ravel())
    shape = (len(nas), len(nas))
    if not ii:
        return sp.csr_array(shape)
    return sp.csr_array((np.concatenate(vv), (np.concatenate(ii), np.concatenate(jj))), shape=shape)


def kinetic_jacobi(channel: JacobiChannel, hw: float) -> jax.Array:
    """Relative kinetic energy of three nucleons in the AS basis of ``channel`` (MeV), on device."""
    return antisymmetrize(kinetic_nas(nas_of(channel), hw), channel)


def nn_jacobi(channel: JacobiChannel, iso: IsospinNN) -> jax.Array:
    """``3 C^T V_NN C``: the NN force summed over the three pairs, in the AS basis (device, MeV)."""
    return 3.0 * antisymmetrize(nn_nas(nas_of(channel), iso), channel)
