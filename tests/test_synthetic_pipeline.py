"""Whole-pipeline checks on a synthetic interaction; no NuHamil data needed.

Several of these are exact physics identities, so they test the transformation and the
isospin assembly independently of NuHamil.
"""

import math
from pathlib import Path

import numpy as np
import pytest
from synthetic import synthetic_nn, write_nn_mom

import nuham  # noqa: F401
from nuham.angmom.moshinsky_table import MoshinskyTable
from nuham.basis.lab import lab_channels, make_orbits
from nuham.basis.relative import allowed_l, relative_states
from nuham.constants import HBARC, M_REDUCED_PN, oscillator_length_sq
from nuham.kernels.nn.projection import project_channel
from nuham.kernels.nn.relative import kinetic_matrix
from nuham.nn.generate import generate_me2j
from nuham.nn.input_file import read_nn_mom
from nuham.nn.lab_transform import lab_channel_matrices
from nuham.nn.me2j import read_me2j, tuple_chunks
from nuham.nn.relative import RelativeChannel, evolve_relative_nn

HW = 16.0
EMAX, E2MAX = 3, 6


@pytest.fixture(scope="module")
def nn():
    return synthetic_nn()


@pytest.fixture(scope="module")
def mosh() -> MoshinskyTable:
    return MoshinskyTable()


def test_binary_round_trip(tmp_path: Path, nn) -> None:
    write_nn_mom(tmp_path / "x.bin", nn)
    back = read_nn_mom(tmp_path / "x.bin")
    np.testing.assert_array_equal(back.k, nn.k)
    np.testing.assert_array_equal(back.w, nn.w)
    assert back.jmax == nn.jmax and len(back.channels) == len(nn.channels)
    for a, b in zip(back.channels, nn.channels, strict=True):
        assert (a.j, a.parity, a.s, a.tz, a.ls) == (b.j, b.parity, b.s, b.tz, b.ls)
        np.testing.assert_array_equal(a.v, b.v)


def test_synthetic_interaction_is_charge_independent(nn) -> None:
    by_key = {(c.j, c.parity, c.s, c.tz): c for c in nn.channels}
    for (j, parity, s, tz), ch in by_key.items():
        if tz == 1:
            np.testing.assert_array_equal(ch.v, by_key[(j, parity, s, -1)].v)
        if tz == 0 and (ch.ls[0] + s) % 2 == 0:
            np.testing.assert_array_equal(ch.v, by_key[(j, parity, s, 1)].v)


@pytest.mark.parametrize("l", [0, 2])
def test_projection_reproduces_the_kinetic_energy(l: int) -> None:
    # The kinetic energy is diagonal in momentum, so on a quadrature mesh it is
    # T(k) / (w k^2) on the diagonal; projecting it must give the tridiagonal HO matrix.
    x, w = np.polynomial.legendre.leggauss(300)
    k, w = 10.0 * (x + 1.0), 10.0 * w
    v = np.diag(HBARC**2 * k**2 / (2 * M_REDUCED_PN) / (w * k**2))
    got = np.asarray(project_channel(k, w, v, (l,), 12, math.sqrt(oscillator_length_sq(HW))))
    n, ll = relative_states((l,), 12)
    np.testing.assert_allclose(got, np.asarray(kinetic_matrix(n, ll, HW)), atol=1e-8)


@pytest.fixture(scope="module")
def evolved(nn):
    return evolve_relative_nn(nn, HW, 2.0, n2max=30, jmax=3, coulomb=True)


def test_srg_preserves_every_spectrum(nn, evolved) -> None:
    bare = evolve_relative_nn(nn, HW, None, n2max=30, jmax=3, coulomb=True)
    worst = 0.0
    for ev, b in zip(evolved, bare, strict=True):
        t = np.asarray(kinetic_matrix(ev.n, ev.l, HW))
        worst = max(worst, np.abs(np.linalg.eigvalsh(t + ev.v) - np.linalg.eigvalsh(t + b.v)).max())
        np.testing.assert_allclose(ev.v, ev.v.T, atol=1e-8)
    assert worst < 1e-6


def test_transforming_the_identity_gives_the_identity(mosh: MoshinskyTable) -> None:
    # If V_rel = 1 on every relative state, the lab matrix is T^T T, which must be the identity on
    # the (normalised, antisymmetrised or proton-neutron) pair basis: this tests the orthonormality
    # of the full coefficient set (brackets, 6j, 9j, normalisations) with no interaction at all.
    relative = {}
    for j in range(E2MAX + 2):
        for parity in (1, -1):
            for s in (0, 1):
                for tz in (-1, 0, 1):
                    ls = allowed_l(j, s, parity, tz)
                    if ls:
                        n, l = relative_states(ls, E2MAX)
                        relative[(j, parity, s, tz)] = RelativeChannel(
                            j, parity, s, tz, n, l, np.eye(len(n))
                        )
    orb = make_orbits(EMAX)
    for j, parity in lab_channels(orb, E2MAX):
        for tz, m in lab_channel_matrices(orb, E2MAX, j, parity, relative, mosh).items():
            np.testing.assert_allclose(
                m.v, np.eye(len(m.pair_a)), atol=1e-11, err_msg=f"{j, parity, tz}"
            )


def test_lab_matrices_are_symmetric(evolved, mosh: MoshinskyTable) -> None:
    relative = {(c.j, c.parity, c.s, c.tz): c for c in evolved}
    orb = make_orbits(EMAX)
    for j, parity in lab_channels(orb, E2MAX):
        for m in lab_channel_matrices(orb, E2MAX, j, parity, relative, mosh).values():
            np.testing.assert_allclose(m.v, m.v.T, atol=1e-10)


@pytest.fixture(scope="module")
def tuples() -> int:
    return sum(len(c) for _, _, c, _, _ in tuple_chunks(make_orbits(EMAX), E2MAX))


def _generate(tmp_path: Path, nn, coulomb: bool) -> np.ndarray:
    out = tmp_path / "x.me2j.gz"
    generate_me2j(nn, out, HW, 2.0, EMAX, E2MAX, coulomb=coulomb, n2max=30)
    return read_me2j(out).reshape(-1, 4)


def test_charge_independent_interaction_gives_equal_isospin_columns(
    tmp_path: Path, nn, tuples: int
) -> None:
    vals = _generate(tmp_path, nn, coulomb=False)
    assert len(vals) == tuples
    nn_col, pn_t1, pp_col = vals[:, 1], vals[:, 2], vals[:, 3]
    assert np.abs(nn_col).max() > 0.1 and np.abs(vals[:, 0]).max() > 0.1
    np.testing.assert_allclose(nn_col, pp_col, atol=1e-8)
    np.testing.assert_allclose(nn_col, pn_t1, atol=1e-8)


def test_coulomb_splits_pp_but_not_nn_from_pn(tmp_path: Path, nn) -> None:
    vals = _generate(tmp_path, nn, coulomb=True)
    assert np.abs(vals[:, 3] - vals[:, 1]).max() > 0.05
    np.testing.assert_allclose(vals[:, 1], vals[:, 2], atol=1e-8)


def test_forbidden_entries_are_exactly_zero(tmp_path: Path, nn) -> None:
    vals = _generate(tmp_path, nn, coulomb=False)
    k = 0
    for a, b, _c, _d, js in tuple_chunks(make_orbits(EMAX), E2MAX):
        for j in js:
            if a == b and j % 2 == 0:
                assert vals[k, 0] == 0.0
            if a == b and j % 2 == 1:
                assert np.all(vals[k, 1:] == 0.0)
            k += 1


def test_batched_assemble_equals_per_pair_reference(
    nn, mosh, monkeypatch: pytest.MonkeyPatch
) -> None:
    from assemble_reference import assemble_per_pair

    from nuham.nn.me2j import LabChannel, assemble

    relative = {(c.j, c.parity, c.s, c.tz): c for c in evolve_relative_nn(nn, HW, 2.0, n2max=24)}
    orb = make_orbits(EMAX)
    channels = {
        jp: LabChannel(lab_channel_matrices(orb, E2MAX, *jp, relative, mosh), len(orb))
        for jp in lab_channels(orb, E2MAX)
    }
    want = np.concatenate(list(assemble_per_pair(orb, E2MAX, channels)))
    got = np.concatenate(list(assemble(orb, E2MAX, channels)))
    np.testing.assert_array_equal(got, want)
    # Tiny batches must give the same values as one big batch.
    monkeypatch.setattr("nuham.nn.me2j._BATCH_TUPLES", 50)
    small = np.concatenate(list(assemble(orb, E2MAX, channels)))
    np.testing.assert_array_equal(small, want)
