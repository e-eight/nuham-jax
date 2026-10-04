"""Assembly of the local N2LO 3N force in the antisymmetrised Jacobi basis (Phase 5.2.4).

For one ``(J, P, T)`` the non-antisymmetrised (NAS) matrix is
``W = 3 sum_t LEC_t V_t`` with ``V_t`` the representative-term matrices of
:mod:`nuham.nnn.local_force.angular` projected on HO states, and the antisymmetrised matrix is
``cfp W cfp^T`` block by block in ``N`` (the factor 3 replaces the sum over the three permutations,
checked in ``docs/study-3nf-cartesian.md``). The LECs are in GeV^-1 for c1, c3, c4.
"""

from collections import defaultdict

import numpy as np

from nuham.constants import HBARC, M_REDUCED_PN
from nuham.kernels.nnn.projection import orbital_table, project_pair
from nuham.nnn.jacobi import JacobiChannel
from nuham.nnn.local_force.angular import TERMS, Channel, coefficients
from nuham.nnn.local_force.radial import LocalMeshes, RadialTables, radial_function


def nas_matrix(
    nas: np.ndarray,
    j2: int,
    t2: int,
    hw: float,
    meshes: LocalMeshes,
    tables: RadialTables,
    lecs: dict[str, float],
) -> np.ndarray:
    """Return ``W = 3 sum_t LEC_t V_t`` on the NAS states ``nas``.

    Rows of ``nas`` are ``n12 l12 s12 j12 t12 n3 l3 j3`` (``nnn.jacobi.NAS_FIELDS``).

    The radial tables must have been built for ``x_max >= max(N) + 2``.
    """
    x_max = tables.zx.shape[0] - 1
    nu = 2.0 * M_REDUCED_PN * hw / HBARC**2
    l_max = int(nas[:, [1, 6]].max())
    n_max = int(nas[:, [0, 5]].max())
    orbitals = np.asarray(orbital_table(nu, meshes.xi, meshes.xi_w, l_max, n_max))
    by_channel: dict[Channel, list[int]] = defaultdict(list)
    for i, row in enumerate(nas):
        by_channel[Channel(row[1], row[2], row[3], row[4], row[6], row[7])].append(i)
    channels = sorted(by_channel)
    radial: dict[str, dict[tuple[int, ...], np.ndarray]] = {t: {} for t in TERMS}  # filled lazily
    out = np.zeros((len(nas), len(nas)))
    for ib, cb in enumerate(channels):
        rows = np.array(by_channel[cb])
        for ck in channels[: ib + 1]:
            cols = np.array(by_channel[ck])
            grid = 0.0
            for term in TERMS:
                if lecs.get(term, 0.0) == 0.0:
                    continue
                for key, c in coefficients(term, cb, ck, j2, t2, x_max).items():
                    g = radial[term].get(key)
                    if g is None:
                        g = radial[term][key] = radial_function(term, key, tables, meshes)
                    grid = grid + 3.0 * lecs[term] * c * g
            if isinstance(grid, float):
                continue
            m = np.asarray(
                project_pair(
                    orbitals[cb.l12], orbitals[cb.l3], orbitals[ck.l12], orbitals[ck.l3], grid
                )
            )
            block = m[nas[rows, 0][:, None], nas[rows, 5][:, None], nas[cols, 0], nas[cols, 5]]
            out[np.ix_(rows, cols)] = block
            out[np.ix_(cols, rows)] = block.T
    return out


def antisymmetrize(w: np.ndarray, channel: JacobiChannel, blocks: list | None = None) -> np.ndarray:
    """Return ``cfp W cfp^T`` block by block in ``N`` (``blocks`` default: those of ``channel``)."""
    blocks = list(channel.blocks) if blocks is None else blocks
    off = np.concatenate([[0], np.cumsum([len(b.nas) for b in blocks])])
    off_as = np.concatenate([[0], np.cumsum([len(b.cfp) for b in blocks])])
    out = np.zeros((off_as[-1], off_as[-1]))
    for i, bi in enumerate(blocks):
        for k, bk in enumerate(blocks[: i + 1]):
            v = bi.cfp @ w[off[i] : off[i + 1], off[k] : off[k + 1]] @ bk.cfp.T
            out[off_as[i] : off_as[i + 1], off_as[k] : off_as[k + 1]] = v
            out[off_as[k] : off_as[k + 1], off_as[i] : off_as[i + 1]] = v.T
    return out


def local_3nf_matrix(
    channel: JacobiChannel,
    hw: float,
    meshes: LocalMeshes,
    tables: RadialTables,
    lecs: dict[str, float],
) -> np.ndarray:
    """Bare local N2LO 3N force in the antisymmetrised Jacobi basis of ``channel`` (MeV)."""
    nas = np.concatenate([b.nas for b in channel.blocks])
    w = nas_matrix(nas, channel.j2, channel.t2, hw, meshes, tables, lecs)
    return antisymmetrize(w, channel)
