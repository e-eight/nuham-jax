"""Frequency conversion of the 3N Jacobi operator (host driver, gap K).

NuHamil's ``hw_target`` option makes the Jacobi operator at one oscillator frequency ``hw`` and
converts it to another, ``V3 -> U^T V3 U`` in the truncated antisymmetrised space, ``U`` being the
antisymmetrised product of the radial overlaps of the two Jacobi coordinates; the truncated
conversion differs from an operator made at the target frequency directly.
``docs/spec-5e-hw-conversion.md`` has the sources.
"""

import numpy as np
import scipy.sparse as sp

from nuham.constants import M_REDUCED_PN, oscillator_length_sq
from nuham.kernels.nnn.frequency import radial_overlap
from nuham.nnn.local_force.radial import gauss_legendre
from nuham.nnn.regulator import block_matrix

N_POINTS = 500
R_MAX = 25.0
"""NuHamil's radial mesh for the overlaps: 500 Gauss points on ``[0, 25]`` fm."""


def jacobi_length(hw: float) -> float:
    """Oscillator length in fm of the Jacobi coordinates, ``b = 1 / sqrt(2 m_red hw / hc^2)``."""
    return float(np.sqrt(oscillator_length_sq(hw, 2.0 * M_REDUCED_PN)))


def overlap_tables(
    n_max: int, hw: float, hw_target: float, l_max: int, n_points: int = N_POINTS
) -> np.ndarray:
    """Return ``O[l, n, n']`` (``l <= l_max``, ``n, n' <= n_max``); rows ``hw``, columns target."""
    r, w = gauss_legendre(n_points, 0.0, R_MAX)
    b_from, b_to = jacobi_length(hw), jacobi_length(hw_target)
    return np.stack(
        [np.asarray(radial_overlap(l, n_max, b_from, b_to, r, w)) for l in range(l_max + 1)]
    )


def frequency_overlap(
    nas: np.ndarray, hw: float, hw_target: float, n_points: int = N_POINTS
) -> sp.csr_array:
    """Return ``W`` on the NAS states ``nas``: the overlap of the frequencies before ``C W C^T``.

    Rows of ``nas`` are ``n12 l12 s12 j12 t12 n3 l3 j3`` (``nnn.jacobi.NAS_FIELDS``). ``W`` is
    nonzero between states of equal ``l12, s12, j12, t12, l3, j3`` with the value
    ``O_{l12}(n12, n12') O_{l3}(n3, n3')``; its rows belong to ``hw`` and its columns to
    ``hw_target``. :func:`nuham.nnn.local_force.assembly.antisymmetrize` gives ``U``, and the
    converted operator is ``U^T V U``.
    """
    n_max = int(max(nas[:, 0].max(), nas[:, 5].max()))
    l_max = int(max(nas[:, 1].max(), nas[:, 6].max()))
    o = overlap_tables(n_max, hw, hw_target, l_max, n_points)
    return block_matrix(nas, lambda l12, l3: np.einsum("ab,cd->abcd", o[l12], o[l3]))
