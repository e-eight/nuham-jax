"""Matrix of the LNL non-local regulator on the NAS states of a Jacobi channel (host driver).

NuHamil's mesh and scale are used (500 Gauss points on ``[0, 150 / sqrt(hw)]`` fm^-1, the
Jacobi oscillator length of the channel's ``hw``); ``docs/spec-5b2-lnl-regulator.md`` has the
sources and the convention.
"""

from collections import defaultdict
from collections.abc import Callable

import numpy as np
import scipy.sparse as sp

from nuham.constants import M_REDUCED_PN, oscillator_length_sq
from nuham.kernels.nnn.regulator import regulator_table
from nuham.nnn.local_force.radial import gauss_legendre

N_POINTS = 500
P_MAX_FACTOR = 150.0
"""NuHamil's ``pmax = 25 * 6 / sqrt(hw)`` in fm^-1, ``hw`` in MeV."""


def block_matrix(nas: np.ndarray, table_of: Callable[[int, int], np.ndarray]) -> sp.csr_array:
    """Scatter radial tables into the sparse NAS matrix of an operator that is diagonal in angles.

    The operator is nonzero only between NAS states of equal ``l12, s12, j12, t12, l3, j3``
    (rows of ``nas`` are ``n12 l12 s12 j12 t12 n3 l3 j3``, ``nnn.jacobi.NAS_FIELDS``), where it
    equals ``table_of(l12, l3)[n12, n12', n3, n3']``. Each table is requested once per pair
    ``(l12, l3)``.
    The regulator and the frequency overlap of :mod:`nuham.nnn.frequency` are such operators.
    """
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for i, (_, l12, s12, j12, t12, _, l3, j3) in enumerate(nas):
        groups[l12, l3, s12, j12, t12, j3].append(i)
    tables: dict[tuple[int, int], np.ndarray] = {}
    rows, cols, vals = [], [], []
    for (l12, l3, *_), members in groups.items():
        if (l12, l3) not in tables:
            tables[l12, l3] = table_of(l12, l3)
        ix = np.array(members)
        n12, n3 = nas[ix, 0], nas[ix, 5]
        block = tables[l12, l3][n12[:, None], n12[None, :], n3[:, None], n3[None, :]]
        rows.append(np.broadcast_to(ix[:, None], block.shape).ravel())
        cols.append(np.broadcast_to(ix[None, :], block.shape).ravel())
        vals.append(block.ravel())
    shape = (len(nas), len(nas))
    return sp.coo_array(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape
    ).tocsr()


def regulator_matrix(
    nas: np.ndarray,
    hw: float,
    cutoff: float,
    power: int,
    n_points: int = N_POINTS,
    p_max: float | None = None,
) -> sp.csr_array:
    """Return the regulator ``R`` on the NAS states ``nas`` as a sparse host matrix.

    Rows of ``nas`` are ``n12 l12 s12 j12 t12 n3 l3 j3`` (``nnn.jacobi.NAS_FIELDS``). ``R`` is
    nonzero only between states of equal ``l12, s12, j12, t12, l3, j3``, so it is block diagonal
    after a permutation and sparse; :func:`nuham.nnn.local_force.assembly.antisymmetrize` takes
    it as it is (a dense ``R`` and one device scatter per block cost ``blocks * n_nas^2``).
    ``p_max`` defaults to NuHamil's ``150 / sqrt(hw)``.

    Parameters
    ----------
    cutoff
        Non-local cutoff in MeV.
    power
        Regulator exponent ``n``.
    """
    p, w = gauss_legendre(n_points, 0.0, P_MAX_FACTOR / np.sqrt(hw) if p_max is None else p_max)
    b = float(
        np.sqrt(oscillator_length_sq(hw, 2.0 * M_REDUCED_PN))
    )  # Jacobi nu = 2 m_red hw / hc^2
    n_max = int(max(nas[:, 0].max(), nas[:, 5].max()))
    return block_matrix(
        nas, lambda l12, l3: np.asarray(regulator_table(l12, l3, n_max, b, cutoff, power, p, w))
    )
