from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import nuham  # noqa: F401
from nuham.angmom.moshinsky_table import MoshinskyTable
from nuham.basis.lab import lab_channels, make_orbits, ordered_pairs
from nuham.interactions.nn_file import read_nn_mom
from nuham.kernels.lab import congruence_sum
from nuham.lab_transform import (
    enumerate_rows,
    lab_matrix,
    lab_matrix_reference,
    t_matrix,
)
from nuham.relative_nn import evolve_relative_nn


def test_congruence_sum_matches_loop_with_padding() -> None:
    rng = np.random.default_rng(0)
    t = rng.normal(size=(5, 4, 7))
    v = rng.normal(size=(5, 4, 4))
    t[2, 2:] = 0.0  # padded rows
    t[4] = 0.0  # padded block
    got = np.asarray(congruence_sum(jnp.asarray(t), jnp.asarray(v)))
    want = sum(t[b].T @ v[b] @ t[b] for b in range(5))
    np.testing.assert_allclose(got, want, atol=1e-12)


@pytest.fixture(scope="module")
def setup(nn_bin: Path):
    nn = read_nn_mom(nn_bin)
    relative = {(c.j, c.parity, c.s, c.tz): c for c in evolve_relative_nn(nn, 16.0, None, n2max=40)}
    return relative, make_orbits(4), MoshinskyTable()


def test_matches_reference_in_every_channel(setup) -> None:
    relative, orb, mosh = setup
    e2max = 8
    for j, parity in lab_channels(orb, e2max):
        a, b = ordered_pairs(orb, e2max, j, parity)
        rows = enumerate_rows(j, parity, e2max, 8)
        t = t_matrix(rows, a, b, orb, j, mosh)
        for tz in (-1, 0, 1):
            new = lab_matrix(rows, t, a, b, j, parity, tz, relative)
            ref = lab_matrix_reference(rows, t, a, b, j, parity, tz, relative)
            np.testing.assert_array_equal(new.pair_a, ref.pair_a)
            np.testing.assert_allclose(new.v, ref.v, atol=1e-11)


def test_empty_pair_basis_gives_empty_matrix(setup) -> None:
    # With e2max = 0 the only pair is 0s1/2 0s1/2; for odd J it is excluded for pp and nn.
    relative, orb, mosh = setup
    a, b = ordered_pairs(orb, 0, 1, 1)
    rows = enumerate_rows(1, 1, 0, 8)
    t = t_matrix(rows, a, b, orb, 1, mosh)
    assert lab_matrix(rows, t, a, b, 1, 1, 1, relative).v.shape == (0, 0)
    assert lab_matrix(rows, t, a, b, 1, 1, 0, relative).v.shape == (1, 1)


def test_vectorized_t_matrix_matches_scalar_reference(setup) -> None:
    from nuham.lab_transform import t_matrix_reference

    _, orb, mosh = setup
    e2max = 8
    for j, parity in lab_channels(orb, e2max):
        a, b = ordered_pairs(orb, e2max, j, parity)
        rows = enumerate_rows(j, parity, e2max, 8)
        np.testing.assert_allclose(
            t_matrix(rows, a, b, orb, j, mosh),
            t_matrix_reference(rows, a, b, orb, j, mosh),
            atol=1e-12,
        )


def test_accumulate_blocks_matches_loop_and_updates_in_place() -> None:
    from nuham.kernels.lab import accumulate_blocks

    rng = np.random.default_rng(3)
    t_rows = rng.normal(size=(40, 9))
    t_rows[39] = 0.0  # the zero row that padding indices point at
    index = rng.integers(0, 39, size=(6, 3))
    index[2, 1:] = 39
    index[5] = 39
    v = rng.normal(size=(6, 3, 3))
    start = rng.normal(size=(9, 9))
    got = np.asarray(
        accumulate_blocks(
            jnp.asarray(start), jnp.asarray(t_rows), jnp.asarray(index), jnp.asarray(v)
        )
    )
    want = start + sum(t_rows[index[b]].T @ v[b] @ t_rows[index[b]] for b in range(6))
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_device_input_gives_the_same_lab_matrix(setup) -> None:
    relative, orb, mosh = setup
    e2max = 8
    a, b = ordered_pairs(orb, e2max, 2, 1)
    rows = enumerate_rows(2, 1, e2max, 8)
    t = t_matrix(rows, a, b, orb, 2, mosh)
    for tz in (-1, 0, 1):
        host = lab_matrix(rows, t, a, b, 2, 1, tz, relative)
        dev = lab_matrix(rows, jnp.asarray(t), a, b, 2, 1, tz, relative)
        np.testing.assert_allclose(dev.v, host.v, atol=1e-13)
