"""Tests of the local-3NF radial kernels and tables (Phase 5.2.2)."""

import numpy as np
import pytest
from scipy.special import spherical_jn as scipy_jn

import nuham  # noqa: F401  (enables float64)
from nuham.constants import HBARC, M_PI
from nuham.kernels.nnn_radial import (
    bilocal_transform,
    jn_over_xn,
    legendre_projected_over_r,
    legendre_table,
    local_regulator,
    radial_transform,
    radial_transform_over_r,
    spherical_jn,
)
from nuham.local3n import converged_meshes, nuhamil_meshes, radial_tables

MU = M_PI / HBARC


def test_spherical_jn_matches_scipy() -> None:
    x = np.concatenate([np.logspace(-4, 0, 15), np.linspace(1.0, 300.0, 300)])
    ref = np.array([scipy_jn(n, x) for n in range(46)])
    assert np.abs(np.asarray(spherical_jn(x, 45)) - ref).max() < 1e-14


@pytest.mark.parametrize("order", [0, 1, 2])
def test_jn_over_xn_small_argument(order: int) -> None:
    x = np.logspace(-6, 2, 120)
    ref = scipy_jn(order, x) / x**order
    assert np.abs(np.asarray(jn_over_xn(order, x)) / ref - 1.0).max() < 5e-14


def test_legendre_table() -> None:
    c = np.linspace(-1, 1, 9)
    ref = np.array([np.polynomial.legendre.Legendre.basis(n)(c) for n in range(12)])
    assert np.abs(np.asarray(legendre_table(c, 11)) - ref).max() < 1e-14


def test_yukawa_tensor_closed_form() -> None:
    """f1 is minus the Yukawa derivative; a huge cutoff leaves only a 1e-5 residue."""
    lam = 3000.0
    m = converged_meshes(lam, 2, n_p=1500)
    wy = m.p_w * np.asarray(local_regulator(m.p, lam, 2)) / (m.p**2 + MU**2)
    r = np.array([2.0, 3.0, 5.0])
    f1 = np.asarray(radial_transform(r, m.p, wy, 1, 3))
    exact = np.exp(-MU * r) * (1.0 + MU * r) / (4.0 * np.pi * r**2)
    assert np.abs(f1 / exact - 1.0).max() < 1e-5  # (m_pi / Lambda)^4 = 4.5e-6 from the regulator
    f0 = np.asarray(radial_transform(r, m.p, wy, 0, 4))
    assert np.abs(f0 / (-(MU**2) * np.exp(-MU * r) / (4.0 * np.pi * r)) - 1.0).max() < 1e-5


@pytest.fixture(scope="module")
def local_setup() -> tuple:
    m = nuhamil_meshes()
    reg = np.asarray(local_regulator(m.p, 400.0, 2))
    return m, m.p_w * reg, m.p_w * reg / (m.p**2 + MU**2)


def test_addition_theorem_for_zx_and_fkx0(local_setup: tuple) -> None:
    """sum_x (2x+1) c_x P_x(cos) must reproduce the transform at the distance |a - s|."""
    m, wd, wy = local_setup
    s, a = np.array([0.5, 2.0, 4.0]), np.array([1.0, 3.0, 5.5])
    c = np.array([-0.9, 0.1, 0.8])
    x_max = 110  # p s reaches 8 * 4 = 32; x must exceed it
    legendre = np.asarray(legendre_table(c, x_max))
    r13 = np.sqrt(
        s[:, None, None] ** 2 + a[None, :, None] ** 2 - 2 * s[:, None, None] * a[None, :, None] * c
    )
    for wf, power in ((wd, 2), (wy, 4)):
        cx = np.asarray(bilocal_transform(s, a, m.p, wf, power, x_max))
        recon = np.einsum("x,xij,xc->ijc", 2 * np.arange(x_max + 1) + 1, cx, legendre)
        ref = np.asarray(radial_transform(r13, m.p, wf, 0, power))
        assert np.abs(recon - ref).max() < 1e-13


@pytest.mark.parametrize(("order", "power"), [(1, 3), (2, 4)])
def test_legendre_projection_reconstructs_vector_and_tensor(
    local_setup: tuple, order: int, power: int
) -> None:
    m, _, wy = local_setup
    s, a = np.array([0.5, 2.0, 4.0]), np.array([1.0, 3.0, 5.5])
    c = np.array([-0.9, 0.1, 0.8])
    x_max = 60
    cx = np.asarray(legendre_projected_over_r(s, a, m.cos, m.cos_w, m.p, wy, order, power, x_max))
    recon = np.einsum(
        "x,xij,xc->ijc", 2 * np.arange(x_max + 1) + 1, cx, np.asarray(legendre_table(c, x_max))
    )
    r13 = np.sqrt(
        s[:, None, None] ** 2 + a[None, :, None] ** 2 - 2 * s[:, None, None] * a[None, :, None] * c
    )
    assert (
        np.abs(recon - np.asarray(radial_transform_over_r(r13, m.p, wy, order, power))).max()
        < 1e-13
    )


def test_nuhamil_meshes_are_converged_at_cutoff_400() -> None:
    """The tables on NuHamil's meshes agree with a much finer momentum and cosine mesh to 1e-11."""
    lam, x_max = 400.0, 12
    t0 = radial_tables(nuhamil_meshes(), lam, 2, x_max)
    fine = converged_meshes(lam, 2, n_p=600, n_cos=200)
    t1 = radial_tables(fine, lam, 2, x_max)
    for a, b in ((t0.z0, t1.z0), (t0.fk, t1.fk), (t0.zx, t1.zx), (t0.fkx, t1.fkx)):
        assert np.abs(a - b).max() < 1e-11 * np.abs(b).max()


def test_table_shapes() -> None:
    m = converged_meshes(500.0, 2, n_xi=10, n_p=60, n_cos=20)
    t = radial_tables(m, 500.0, 2, 5)
    assert t.z0.shape == (10,)
    assert t.fk.shape == (3, 10)
    assert t.zx.shape == (6, 10, 10)
    assert t.fkx.shape == (3, 6, 10, 10)
