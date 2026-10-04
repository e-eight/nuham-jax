"""H0 = T + 3 V_NN (+ V_3N) in the AS Jacobi basis against NuHamil's logged eigenvalues (Phase 5.3).

The eigenvalues of H0 are gauge invariant, so our own AS basis can be compared with the lowest five
printed by NuHamil per channel (flat8 ramp, hw 16, no flow yet). NuHamil printed six decimals and
read the NN force from its 7-digit cache in all channels but the first.
"""

import re
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from ref_nnn_projection import read_jacobi_matrix

import nuham  # noqa: F401  (enables float64)
from nuham.nn.input_file import read_nn_mom
from nuham.nnn.hamiltonian import isospin_nn, kinetic_jacobi, nn_jacobi
from nuham.nnn.jacobi import build_channel, jacobi_channels
from nuham.nnn.local_force.assembly import antisymmetrize, local_3nf_matrix
from nuham.nnn.local_force.radial import nuhamil_meshes, radial_tables
from nuham.nnn.srg3 import evolve_hamiltonian, three_body_operator

DATA = Path(__file__).resolve().parents[1] / "refs" / "work" / "nnn_e4"
LECS = {"c1": -0.81, "c3": -3.2, "c4": 5.4, "cD": -0.2, "cE": 0.098}
HW = 16.0


def logged_eigenvalues(name: str) -> list[np.ndarray]:
    """Lowest five eigenvalues of the original H per channel, in the log's order."""
    path = DATA / name
    if not path.exists():
        pytest.skip(f"{path} not found")
    pattern = re.compile(r"Eigen values of original 3-body H:\s+(.*)")
    return [np.array(m.group(1).split(), dtype=float) for m in map(pattern.search, open(path)) if m]


@pytest.fixture(scope="module")
def iso(nn_bin: Path) -> tuple:
    return isospin_nn(read_nn_mom(nn_bin), HW, 2.0)


def test_kinetic_is_twice_the_pair_kinetic() -> None:
    """On antisymmetric states ``t12 + t3`` and ``2 t12`` give the same matrix (spec 5c, 2.3)."""
    channel = build_channel(3, 1, 1, 8)
    nas = np.concatenate([b.nas for b in channel.blocks])
    both = kinetic_jacobi(channel, HW)
    np.testing.assert_allclose(both, 2.0 * antisymmetrize(_pair_only(nas), channel), atol=1e-11)


def _pair_only(nas: np.ndarray) -> sp.csr_array:
    """``t12`` alone on the NAS states (the n3 links and the n3 part of the diagonal removed)."""
    where = {tuple(r): i for i, r in enumerate(nas.tolist())}
    ii, jj, vv = [], [], []
    for i, r in enumerate(nas.tolist()):
        ii.append(i)
        jj.append(i)
        vv.append(0.5 * HW * (2 * r[0] + r[1] + 1.5))
        up = list(r)
        up[0] += 1
        k = where.get(tuple(up))
        if k is not None:
            off = 0.5 * HW * np.sqrt((r[0] + 1.0) * (r[0] + r[1] + 1.5))
            ii += [i, k]
            jj += [k, i]
            vv += [off, off]
    return sp.csr_array((vv, (ii, jj)), shape=(len(nas), len(nas)))


def test_kinetic_is_block_tridiagonal_in_n() -> None:
    channel = build_channel(1, 1, 1, 8)
    t = kinetic_jacobi(channel, HW)
    n_of = np.concatenate([[b.n_quanta] * len(b.cfp) for b in channel.blocks])
    far = np.abs(n_of[:, None] - n_of[None, :]) > 2
    assert np.abs(t[far]).max() < 1e-12
    np.testing.assert_allclose(t, t.T, atol=1e-12)


def test_h0_without_3nf_matches_nuhamil(iso: tuple) -> None:
    logged = logged_eigenvalues("log_ind.txt")
    bare, _ = iso
    for (j2, parity, t2, nmax), want in zip(jacobi_channels(5, "flat8"), logged, strict=True):
        channel = build_channel(j2, parity, t2, nmax)
        h = kinetic_jacobi(channel, HW) + nn_jacobi(channel, bare)
        got = np.linalg.eigvalsh(h)[:5]
        assert np.abs(got - want).max() < 5e-6, (j2, parity, t2)


@pytest.mark.parametrize("channel_index", [0, 1, 2, 3, 4, 12])
def test_h0_with_3nf_matches_nuhamil(iso: tuple, channel_index: int) -> None:
    logged = logged_eigenvalues("log_3n_e4.txt")
    bare, _ = iso
    meshes = nuhamil_meshes()
    tables = radial_tables(meshes, 400.0, 2, 10)
    j2, parity, t2, nmax = jacobi_channels(11, "flat8")[channel_index]
    channel = build_channel(j2, parity, t2, nmax)
    h = (
        kinetic_jacobi(channel, HW)
        + nn_jacobi(channel, bare)
        + local_3nf_matrix(channel, HW, meshes, tables, LECS)
    )
    got = np.linalg.eigvalsh(h)[:5]
    assert np.abs(got - logged[channel_index]).max() < 5e-6


def test_flow_keeps_spectrum_and_v3_limits(iso: tuple) -> None:
    """The flow is unitary, and V3 is the bare 3N force without flow or zero for no force at all."""
    bare, evolved = iso
    channel = build_channel(1, 1, 1, 8)
    t = kinetic_jacobi(channel, HW)
    h0 = t + nn_jacobi(channel, bare)
    h = evolve_hamiltonian(h0, t, HW, 2.0, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(np.linalg.eigvalsh(h), np.linalg.eigvalsh(h0), atol=1e-8)
    np.testing.assert_allclose(h, h.T, atol=1e-11)
    v3n = np.diag(np.arange(len(h0), dtype=float))
    # evolved = bare is the lambda -> infinity limit: no flow, V3 is the genuine force itself.
    same = three_body_operator(channel, HW, 1e6, bare, bare, v3n, atol=1e-12)
    np.testing.assert_allclose(same, v3n, atol=1e-8)
    assert np.abs(three_body_operator(channel, HW, 2.0, bare, evolved)).max() > 1.0
    cut = three_body_operator(channel, HW, 2.0, bare, evolved, e3max=4)
    full = three_body_operator(channel, HW, 2.0, bare, evolved)
    np.testing.assert_array_equal(cut, full[: len(cut), : len(cut)])
    assert 0 < len(cut) < len(full)


FULL_BIN = (
    "NNN-full_j{j}p{s}t{t}_Nmax8_Local2_400_c1_-0.81_c3_-3.2_c4_5.4_cD_-0.2_cE_0.098_srg2_hw16.bin"
)


@pytest.mark.parametrize("channel_index", [0, 1, 2, 4, 12, 13])
def test_v3_spectrum_matches_nuhamil(iso: tuple, channel_index: int) -> None:
    """The whole V3 (spectrum, trace, Frobenius norm) equals NuHamil's stored NNN-full matrix."""
    bare, evolved = iso
    j2, parity, t2, nmax = jacobi_channels(11, "flat8")[channel_index]
    path = DATA / "ops" / FULL_BIN.format(j=j2, s="+" if parity > 0 else "-", t=t2)
    if not path.exists():
        pytest.skip(f"{path} not found")
    ref = read_jacobi_matrix(str(path))
    meshes = nuhamil_meshes()
    tables = radial_tables(meshes, 400.0, 2, 10)
    channel = build_channel(j2, parity, t2, nmax)
    v3n = local_3nf_matrix(channel, HW, meshes, tables, LECS)
    v3 = three_body_operator(channel, HW, 2.0, bare, evolved, v3n)
    assert v3.shape == ref.shape
    assert np.abs(np.linalg.eigvalsh(v3) - np.linalg.eigvalsh(ref)).max() < 5e-6
    assert abs(np.trace(v3) - np.trace(ref)) < 2e-5
