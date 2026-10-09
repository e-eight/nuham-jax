"""Frequency conversion of the 3N Jacobi operator (gap K, steps K1-K3)."""

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

import nuham  # noqa: F401  (enables float64)
from nuham.angmom.moshinsky_table import MoshinskyTable
from nuham.nn.input_file import read_nn_mom
from nuham.nnn.frequency import frequency_overlap, jacobi_length, overlap_tables
from nuham.nnn.generate import LocalForce, jacobi_operators
from nuham.nnn.hamiltonian import kinetic_nas
from nuham.nnn.jacobi import build_channel
from nuham.nnn.local_force.assembly import antisymmetrize

HW, HW_TARGET = 30.0, 16.0


def test_equal_frequencies_give_the_identity() -> None:
    o = overlap_tables(20, 16.0, 16.0, 40)
    assert np.abs(o - np.eye(21)).max() < 1e-12


@pytest.mark.parametrize("l", [0, 3, 12])
def test_lowest_state_closed_form(l: int) -> None:
    """``<0 l, b1 | 0 l, b2> = (2 b1 b2 / (b1^2 + b2^2))^(l + 3/2)``."""
    b1, b2 = jacobi_length(HW), jacobi_length(HW_TARGET)
    o = overlap_tables(6, HW, HW_TARGET, 12)
    assert abs(o[l, 0, 0] - (2 * b1 * b2 / (b1**2 + b2**2)) ** (l + 1.5)) < 1e-12


@pytest.mark.parametrize("l", [0, 2, 10])
def test_rows_of_the_lowest_states_are_normalised(l: int) -> None:
    """The functions of one frequency are complete: ``sum_n' O[n, n']^2 = 1`` for a low ``n``."""
    o = overlap_tables(40, HW, HW_TARGET, 10)[l]
    assert np.abs((o[:4] ** 2).sum(axis=1) - 1.0).max() < 1e-9
    assert np.abs((o[:, :4] ** 2).sum(axis=0) - 1.0).max() < 1e-9


def test_conversion_there_and_back_is_the_identity_on_low_states() -> None:
    forth = overlap_tables(40, HW, HW_TARGET, 4)
    back = overlap_tables(40, HW_TARGET, HW, 4)
    for l in range(5):
        assert np.abs((forth[l] @ back[l])[:4, :4] - np.eye(4)).max() < 1e-8


@pytest.fixture(scope="module")
def channel():
    return build_channel(3, -1, 1, 8, MoshinskyTable(d=1 / 3))


def test_overlap_matrix_structure(channel) -> None:
    """``W`` is nonzero only between states of equal ``l12, s12, j12, t12, l3, j3``."""
    nas = np.concatenate([b.nas for b in channel.blocks])
    w = frequency_overlap(nas, HW, HW_TARGET).tocoo()
    same = nas[w.row][:, [1, 2, 3, 4, 6, 7]] == nas[w.col][:, [1, 2, 3, 4, 6, 7]]
    assert same.all()
    assert abs(frequency_overlap(nas, 16.0, 16.0) - np.eye(len(nas))).max() < 1e-12


def test_conversion_carries_the_kinetic_energy_to_the_target_basis(channel) -> None:
    """``W^T T(hw) W = T(hw_target)`` on the low states: a physical operator in a new basis.

    This fixes the orientation of ``W`` (rows the original frequency, columns the target): the
    transposed matrix misses by several MeV. The truncation only spoils the states near the edge.
    """
    nas = np.concatenate([b.nas for b in channel.blocks])
    w = frequency_overlap(nas, 20.0, 16.0)
    moved = (w.T @ kinetic_nas(nas, 20.0) @ w).toarray()
    low = np.flatnonzero(2 * (nas[:, 0] + nas[:, 5]) + nas[:, 1] + nas[:, 6] <= 2)
    exact = kinetic_nas(nas, 16.0).toarray()
    assert np.abs(moved[np.ix_(low, low)] - exact[np.ix_(low, low)]).max() < 2e-2
    wrong = (w @ kinetic_nas(nas, 20.0) @ w.T).toarray()
    assert np.abs(wrong[np.ix_(low, low)] - exact[np.ix_(low, low)]).max() > 1.0


def _antisymmetrized_overlap(channel, nas, hw: float, hw_target: float) -> np.ndarray:
    """``U = C W C^T`` through the package's ``antisymmetrize``, keeping the odd part of ``W``."""
    w = frequency_overlap(nas, hw, hw_target)
    return np.asarray(antisymmetrize(w, channel, symmetric=False))


def test_antisymmetrized_overlap_is_the_explicit_cfp_product(channel) -> None:
    """``W`` is not symmetric: symmetrising ``C W C^T`` (right for the forces) drops its odd part.

    The overlap of the lowest states of ``J = 1/2``, ``T = 1/2`` is ``U[1, 0] = -0.0197`` in
    NuHamil's debug output; that odd element was the first thing a symmetrised product lost.
    """
    nas = np.concatenate([b.nas for b in channel.blocks])
    c = sp.block_diag([b.cfp for b in channel.blocks]).toarray()
    w = frequency_overlap(nas, 30.0, 16.0)
    u = _antisymmetrized_overlap(channel, nas, 30.0, 16.0)
    assert np.abs(u - c @ w.toarray() @ c.T).max() < 1e-12
    assert np.abs(u - u.T).max() > 1e-3


def test_converted_channel_is_orthogonal_on_the_low_states(channel) -> None:
    """``U^T U`` is the identity on states far below the truncation and not above it."""
    nas = np.concatenate([b.nas for b in channel.blocks])
    u = _antisymmetrized_overlap(channel, nas, 20.0, 16.0)
    low = sum(len(b.cfp) for b in channel.blocks if b.n_quanta <= 2)
    gram = u.T @ u
    assert np.abs(gram[:low, :low] - np.eye(low)).max() < 0.05
    assert np.abs(gram - np.eye(len(gram))).max() > 0.05


@pytest.fixture(scope="module")
def nn(nn_bin: Path):
    return read_nn_mom(nn_bin)


def _ops(nn, hw: float, hw_target: float | None) -> dict:
    ops, _ = jacobi_operators(nn, hw, 2.0, 3, "flat8", LocalForce(), js=[1, 3], hw_target=hw_target)
    return {k: np.asarray(v) for k, v in ops.items()}


def test_conversion_to_the_same_frequency_changes_nothing(nn) -> None:
    plain = _ops(nn, 16.0, None)
    same = _ops(nn, 16.0, 16.0)
    assert all(np.array_equal(plain[k], same[k]) for k in plain)


def test_conversion_is_u_transpose_v_u_in_the_full_channel(nn) -> None:
    """With no cut (``e3max`` = ramp) the stored operator is ``U^T V U`` of the unconverted one."""
    ops, chans = jacobi_operators(nn, 20.0, 2.0, 3, "flat3", LocalForce(), js=[3])
    conv, _ = jacobi_operators(nn, 20.0, 2.0, 3, "flat3", LocalForce(), js=[3], hw_target=16.0)
    for key, v in ops.items():
        nas = np.concatenate([b.nas for b in chans[key].blocks])
        u = _antisymmetrized_overlap(chans[key], nas, 20.0, 16.0)
        assert np.abs(conv[key] - u.T @ v @ u).max() < 1e-9
        assert np.abs(conv[key] - v).max() > 1e-3
        assert np.abs(conv[key] - conv[key].T).max() < 1e-9
