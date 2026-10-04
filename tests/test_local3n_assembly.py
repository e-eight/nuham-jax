"""Assembly of the local N2LO 3NF in the antisymmetrised Jacobi basis against NuHamil (Phase 5.2.4).

Reference: NuHamil with ``renorm="bare"`` stores ``H - T - 3 V_NN``. The run without the genuine
force (``NNN-ind``) holds only that subtraction's noise (up to 5e-6 in one channel), so the genuine
bare V3 is the difference of the two files. NuHamil's ``cfp`` files fix the AS gauge element by
element; with our own (arbitrary but orthonormal) AS basis the spectrum must agree.
"""

from pathlib import Path

import numpy as np
import pytest
from cfp_reader import read_cfp
from ref_nnn_projection import read_jacobi_matrix

import nuham  # noqa: F401  (enables float64)
from nuham.nnn.jacobi import JacobiBlock, JacobiChannel, build_channel
from nuham.nnn.local_force.assembly import local_3nf_matrix
from nuham.nnn.local_force.radial import nuhamil_meshes, radial_tables

DATA = Path(__file__).resolve().parents[1] / "refs" / "work" / "nnn3f"
LECS = {"c1": -0.81, "c3": -3.2, "c4": 5.4, "cD": -0.2, "cE": 0.098}
FULL = (
    "NNN-full_j{j}p{s}t{t}_Nmax6_Local2_400_c1_-0.81_c3_-3.2_c4_5.4_cD_-0.2_cE_0.098_bare_hw16.bin"
)
IND = "NNN-ind_j{j}p{s}t{t}_Nmax6_bare_hw16.bin"
CASES = [(1, 1, 1), (1, 1, 3), (3, -1, 1), (5, 1, 1), (11, -1, 3)]


@pytest.fixture(scope="module")
def tables() -> tuple:
    meshes = nuhamil_meshes()
    return meshes, radial_tables(meshes, 400.0, 2, 8)


def reference(j2: int, parity: int, t2: int) -> np.ndarray:
    """Genuine bare V3 in NuHamil's AS gauge, or skip if the oracle data is absent."""
    s = "+" if parity > 0 else "-"
    full = DATA / "n400_N6_bare" / "ops" / FULL.format(j=j2, s=s, t=t2)
    ind = DATA / "n400_N6_bare_ind" / "ops" / IND.format(j=j2, s=s, t=t2)
    if not (full.exists() and ind.exists()):
        pytest.skip("NuHamil bare 3N oracle files (refs/work/nnn3f) are absent")
    return read_jacobi_matrix(str(full)) - read_jacobi_matrix(str(ind))


@pytest.mark.parametrize(("j2", "parity", "t2"), CASES)
def test_matches_nuhamil_with_its_cfp(tables: tuple, j2: int, parity: int, t2: int) -> None:
    meshes, tab = tables
    ref = reference(j2, parity, t2)
    s = "+" if parity > 0 else "-"
    (_, blocks) = read_cfp(DATA / "n400_N6" / "cfp" / f"cfp_A3_j{j2}p{s}t{t2}_Nmax6.bin")
    channel = JacobiChannel(
        j2, parity, t2, 6, tuple(JacobiBlock(n, nas, c) for n, nas, c in blocks)
    )
    v = local_3nf_matrix(channel, 16.0, meshes, tab, LECS)
    assert v.shape == ref.shape
    assert np.abs(v - ref).max() < 1e-10


@pytest.mark.parametrize(("j2", "parity", "t2"), [(1, 1, 1), (3, -1, 1)])
def test_spectrum_with_our_cfp(tables: tuple, j2: int, parity: int, t2: int) -> None:
    meshes, tab = tables
    ref = reference(j2, parity, t2)
    v = local_3nf_matrix(build_channel(j2, parity, t2, 6), 16.0, meshes, tab, LECS)
    assert np.abs(np.linalg.eigvalsh(v) - np.linalg.eigvalsh(ref)).max() < 1e-10
    assert np.abs(v - v.T).max() < 1e-12


def test_workers_and_extra_ranks_change_nothing(tables: tuple) -> None:
    """Worker processes and a larger ``x_max`` (no silent rank truncation) change nothing."""
    meshes, tab = tables
    channel = build_channel(1, 1, 1, 6)
    base = local_3nf_matrix(channel, 16.0, meshes, tab, LECS)
    parallel = local_3nf_matrix(channel, 16.0, meshes, tab, LECS, workers=2)
    assert np.abs(parallel - base).max() == 0.0
    wide = local_3nf_matrix(channel, 16.0, meshes, radial_tables(meshes, 400.0, 2, 12), LECS)
    assert np.abs(wide - base).max() < 1e-13
