"""Tests of the LNL non-local regulator matrix (gap D, step D1)."""

import numpy as np
import pytest
from scipy.special import spherical_jn

import nuham  # noqa: F401  (enables float64)
from nuham.angmom.moshinsky_table import MoshinskyTable
from nuham.constants import HBARC, M_REDUCED_PN, oscillator_length_sq
from nuham.kernels.ho import momentum_radial, radial_functions
from nuham.kernels.nnn.regulator import regulator_table
from nuham.nnn.jacobi import build_channel
from nuham.nnn.local_force.assembly import antisymmetrize
from nuham.nnn.local_force.radial import gauss_legendre
from nuham.nnn.regulator import N_POINTS, P_MAX_FACTOR, regulator_matrix

HW = 16.0
B = float(np.sqrt(oscillator_length_sq(HW, 2.0 * M_REDUCED_PN)))
P_MAX = P_MAX_FACTOR / np.sqrt(HW)


@pytest.fixture(scope="module")
def table() -> MoshinskyTable:
    """Moshinsky table for the Jacobi basis (d = 1/3)."""
    return MoshinskyTable(d=1 / 3)


@pytest.fixture(scope="module")
def mesh() -> tuple[np.ndarray, np.ndarray]:
    """NuHamil's momentum mesh at the test hw."""
    return gauss_legendre(N_POINTS, 0.0, P_MAX)


def test_momentum_functions_are_orthonormal_on_the_mesh(mesh) -> None:
    """The regulator integrals need ``sum_i w_i u_n u_n' = delta`` up to n = 20, l = 40."""
    p, w = mesh
    for l in (0, 5, 40):
        u = np.asarray(momentum_radial(p, l, 20, B))
        assert np.abs((u * w) @ u.T - np.eye(21)).max() < 1e-10


def test_infinite_cutoff_gives_identity(mesh) -> None:
    p, w = mesh
    t = np.asarray(regulator_table(3, 2, 10, B, 1e9, 2, p, w))
    ident = np.einsum("ab,cd->abcd", np.eye(11), np.eye(11))
    assert np.abs(t - ident).max() < 1e-9


def test_power_one_factorises(mesh) -> None:
    """exp(-(p^2 + k^2) c) = exp(-p^2 c) exp(-k^2 c): ``R`` is a product of 1D integrals."""
    p, w = mesh
    cutoff = 500.0
    t = np.asarray(regulator_table(2, 1, 8, B, cutoff, 1, p, w))
    # An independent mesh and a direct evaluation of each factor.
    q, v = gauss_legendre(1500, 0.0, 1.3 * P_MAX)
    g = np.exp(-0.5 * q**2 * HBARC**2 / cutoff**2)

    def one(l: int) -> np.ndarray:
        u = np.asarray(momentum_radial(q, l, 8, B))
        return (u * v * g) @ u.T

    assert np.abs(t - np.einsum("ab,cd->abcd", one(2), one(1))).max() < 1e-12


def test_table_is_symmetric_and_bounded(mesh) -> None:
    p, w = mesh
    t = np.asarray(regulator_table(2, 4, 10, B, 500.0, 2, p, w))
    flat = t.transpose(0, 2, 1, 3).reshape(121, 121)
    assert np.abs(flat - flat.T).max() < 1e-14
    eig = np.linalg.eigvalsh(flat)
    assert eig.min() > 0.0 and eig.max() < 1.0 + 1e-10


@pytest.mark.parametrize(("j2", "parity", "t2"), [(1, 1, 1), (3, -1, 1), (5, 1, 3)])
def test_commutes_with_the_antisymmetriser(
    j2: int, parity: int, t2: int, table: MoshinskyTable
) -> None:
    """F depends on the hyperradius only, so ``R A = A R`` with ``A = C^T C``.

    A wrong phase, normalisation or scale of the momentum functions breaks this.
    """
    channel = build_channel(j2, parity, t2, 8, table)
    nas = np.concatenate([b.nas for b in channel.blocks])
    r = np.asarray(regulator_matrix(nas, HW, 500.0, 2))
    cfp = np.zeros((channel.n_as, channel.n_nas))
    i = k = 0
    for b in channel.blocks:
        cfp[i : i + len(b.cfp), k : k + len(b.nas)] = b.cfp
        i, k = i + len(b.cfp), k + len(b.nas)
    a = cfp.T @ cfp
    assert np.abs(r - r.T).max() < 1e-13
    assert np.abs(r @ a - a @ r).max() < 1e-9
    fa = np.asarray(antisymmetrize(r, channel))
    eig = np.linalg.eigvalsh(fa)
    assert eig.min() > 0.0 and eig.max() < 1.0 + 1e-9
    assert np.abs(fa - cfp @ r @ cfp.T).max() < 1e-12


@pytest.mark.parametrize("l", [0, 1, 2, 5])
def test_momentum_functions_are_the_fourier_transform_of_the_position_ones(l: int) -> None:
    """``u(p) = p sqrt(2/pi) int r u(r) j_l(pr) dr`` up to one n-independent factor ``i^l``.

    The position functions are the positive-at-the-origin ones of the force projection, so
    this fixes the ``(-1)^n`` phase of ``R`` relative to the basis the force is built in (it
    changes the antisymmetrised regulator by 0.1, so a commutator test cannot see it).
    """
    n_max = 6
    r, wr = gauss_legendre(3000, 0.0, 40.0)
    p, _ = gauss_legendre(40, 0.1, 2.5)
    nu = 1.0 / B**2
    u_r = nu**0.25 * np.asarray(radial_functions(np.sqrt(nu) * r, l, n_max))
    bessel = spherical_jn(l, np.outer(p, r))
    ft = np.sqrt(2.0 / np.pi) * p[None, :] * ((u_r * wr * r) @ bessel.T)
    ratio = ft / np.asarray(momentum_radial(p, l, n_max, B))
    assert np.abs(np.abs(ratio) - 1.0).max() < 1e-9
    assert np.abs(ratio - ratio[0, 0]).max() < 1e-9
