"""Radial tables of the local N2LO three-nucleon force on explicit meshes (Phase 5.2.2).

The force is a sum of terms ``sum_a C_a(alpha, alpha') g_a(xi1, xi2)``. The ``g_a`` below depend
only on the cutoff, the regulator power and the pion mass, not on channel, LECs or ``hw``, so they
are computed once. Two mesh choices exist: NuHamil's hard-coded meshes (to reproduce its numbers)
and converged meshes sized from the cutoff.
"""

from dataclasses import dataclass

import numpy as np

from nuham.constants import HBARC, M_PI, M_REDUCED_PN
from nuham.kernels.nnn.radial import (
    bilocal_transform,
    legendre_projected_over_r,
    local_regulator,
    radial_transform,
)


@dataclass(frozen=True)
class LocalMeshes:
    """Gauss-Legendre meshes: Jacobi radius ``xi`` (fm), momentum ``p`` (fm^-1) and ``cos``."""

    xi: np.ndarray
    xi_w: np.ndarray
    p: np.ndarray
    p_w: np.ndarray
    cos: np.ndarray
    cos_w: np.ndarray


@dataclass(frozen=True)
class RadialTables:
    """Universal radial tables on the ``xi1 x xi2`` mesh (indices ``i`` for xi1, ``j`` for xi2).

    ``z0[i]`` and ``fk[k, i]`` (``k`` = 0, 1, 2) depend on ``r12 = sqrt2 xi1``; ``zx[x, i, j]``,
    ``fkx[k, x, i, j]`` are the partial-wave coefficients in the angle between ``s = xi1 / sqrt2``
    and ``a = sqrt(3/2) xi2``.
    """

    z0: np.ndarray
    fk: np.ndarray
    zx: np.ndarray
    fkx: np.ndarray


def gauss_legendre(n: int, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``n`` Gauss-Legendre nodes and weights on ``[lo, hi]``."""
    x, w = np.polynomial.legendre.leggauss(n)
    half = 0.5 * (hi - lo)
    return lo + half * (x + 1.0), half * w


def nuhamil_meshes() -> LocalMeshes:
    """NuHamil's meshes: 80 points on [0, 15] fm, 100 on [0, 8] fm^-1, 80 cosines."""
    xi, xi_w = gauss_legendre(80, 0.0, 15.0)
    p, p_w = gauss_legendre(100, 0.0, 8.0)
    c, c_w = gauss_legendre(80, -1.0, 1.0)
    return LocalMeshes(xi, xi_w, p, p_w, c, c_w)


def xi_mesh_size(
    n3max: int, hw: float, margin: float = 1.22, per_fm: float = 7.0
) -> tuple[int, float]:
    """Return ``(n_xi, xi_max)`` for a radial mesh that holds all HO states up to ``n3max``.

    The outer turning point of a state with ``2n + l = N`` is ``sqrt((2N + 3) / nu)`` with
    ``nu = 2 m hw / (hbar c)^2``; a mesh ending 22% beyond it with 7 points per fm reproduces the
    matrix elements to 1e-14 (``docs/study-local-3nf.md``). NuHamil's fixed 80 points to 15 fm fall
    short of this beyond about ``n3max * hw`` = 600 MeV (an error of order 1e-9 MeV).
    """
    nu = 2.0 * M_REDUCED_PN * hw / HBARC**2
    xi_max = margin * np.sqrt((2 * n3max + 3) / nu)
    return int(np.ceil(per_fm * xi_max)), float(xi_max)


def converged_meshes(
    cutoff_mev: float,
    power: int = 2,
    n_xi: int = 80,
    xi_max: float = 15.0,
    n_p: int = 300,
    n_cos: int = 120,
    p_max: float | None = None,
) -> LocalMeshes:
    """Meshes sized from the cutoff: the p mesh ends where the regulator is below 1e-17."""
    if p_max is None:
        p_max = (np.log(1e17)) ** (1.0 / (2 * power)) * cutoff_mev / HBARC
    xi, xi_w = gauss_legendre(n_xi, 0.0, xi_max)
    p, p_w = gauss_legendre(n_p, 0.0, p_max)
    c, c_w = gauss_legendre(n_cos, -1.0, 1.0)
    return LocalMeshes(xi, xi_w, p, p_w, c, c_w)


def radial_tables(meshes: LocalMeshes, cutoff_mev: float, power: int, x_max: int) -> RadialTables:
    """Return z0, fk, zx and fkx for ``x = 0..x_max`` (NuHamil uses ``x_max = N3max + 2``)."""
    mu = M_PI / HBARC
    s = meshes.xi / np.sqrt(2.0)
    a = np.sqrt(1.5) * meshes.xi
    r12 = np.sqrt(2.0) * meshes.xi
    reg = np.asarray(local_regulator(meshes.p, cutoff_mev, power))
    wd = meshes.p_w * reg
    wy = wd / (meshes.p**2 + mu**2)
    z0 = radial_transform(r12, meshes.p, wd, 0, 2)
    fk = np.stack(
        [
            radial_transform(r12, meshes.p, wy, 0, 4),
            radial_transform(r12, meshes.p, wy, 1, 3),
            radial_transform(r12, meshes.p, wy, 2, 4),
        ]
    )
    zx = bilocal_transform(s, a, meshes.p, wd, 2, x_max)
    fkx0 = bilocal_transform(s, a, meshes.p, wy, 4, x_max)
    fkx1 = legendre_projected_over_r(s, a, meshes.cos, meshes.cos_w, meshes.p, wy, 1, 3, x_max)
    fkx2 = legendre_projected_over_r(s, a, meshes.cos, meshes.cos_w, meshes.p, wy, 2, 4, x_max)
    return RadialTables(
        np.asarray(z0),
        np.asarray(fk),
        np.asarray(zx),
        np.stack([fkx0, np.asarray(fkx1), np.asarray(fkx2)]),
    )


def radial_function(
    term: str, key: tuple[int, ...], tables: RadialTables, meshes: LocalMeshes
) -> np.ndarray:
    """Radial function ``g_a(xi1, xi2)`` of ``term`` on the mesh, shape ``(n_xi, n_xi)``.

    The key is the one used by :func:`nuham.nnn.local_force.angular.radial_keys`: ``(X,)`` for cE,
    ``(K, X)`` for cD, ``(K3, X)`` for c1 and ``(K1, K2, K3, X)`` for c3 and c4.
    ``s = xi1 / sqrt2`` and
    ``a = sqrt(3/2) xi2`` are the two vectors whose distance enters the pion and contact functions.
    """
    s = (meshes.xi / np.sqrt(2.0))[:, None]
    a = (np.sqrt(1.5) * meshes.xi)[None, :]
    if term == "cE":
        return tables.z0[:, None] * tables.zx[key[0]]
    if term == "cD":
        return tables.fk[key[0]][:, None] * tables.zx[key[1]]
    if term == "c1":
        k3, x = key
        return s**k3 * a ** (1 - k3) * tables.fk[1][:, None] * tables.fkx[1, x]
    k1, k2, k3, x = key
    return s**k3 * a ** (k2 - k3) * tables.fk[k1][:, None] * tables.fkx[k2, x]
