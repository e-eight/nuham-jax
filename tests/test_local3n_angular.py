"""Angular coefficients of the local N2LO 3NF against NuHamil's per-term NAS matrices (Phase 5.2.3).

The oracle files are the committed ``save_3nf_before_lec`` outputs in ``refs/work/nnn3f`` (hw 16,
N3max 6). Each term must reproduce every element of the non-antisymmetrised Jacobi matrix.
"""

from pathlib import Path

import numpy as np
import pytest
from ref_nnn_projection import (
    TERM_ORDER,
    lower_to_full,
    nas_blocks_in_file,
    nas_term_matrix,
    read_hopw,
)
from sympy.physics.wigner import clebsch_gordan

import nuham  # noqa: F401  (enables float64)
from nuham.nnn.local_force.angular import cg0, radial_keys, tau2_dot_tau3, tau_cross
from nuham.nnn.local_force.radial import nuhamil_meshes, radial_tables

DATA = Path(__file__).resolve().parents[1] / "refs" / "work" / "nnn3f"


def oracle_file(run: str, j2: int, parity: int, t2: int) -> Path:
    """Path of the saved per-term file, skipping the test if the oracle data is absent."""
    sign = "+" if parity > 0 else "-"
    nmax = 6
    files = sorted((DATA / run / "ops").glob(f"N2LO_3NF_HOPW_J{j2}P{sign}T{t2}_Nmax{nmax}_*.bin"))
    if not files:
        pytest.skip("NuHamil oracle files (refs/work/nnn3f) are absent")
    return files[0]


@pytest.mark.parametrize(
    ("a", "b", "c"), [(1, 1, 0), (1, 1, 2), (2, 2, 2), (2, 3, 1), (4, 2, 6), (3, 3, 4)]
)
def test_cg0_matches_sympy(a: int, b: int, c: int) -> None:
    ref = float(clebsch_gordan(a, b, c, 0, 0, 0))
    assert cg0(a, b, c) == pytest.approx(ref, abs=1e-14)


def test_isospin_factors() -> None:
    # six times the factor is the matrix element of tau_2 . tau_3: +1 for the T = 3/2 state
    assert 6.0 * tau2_dot_tau3(1, 1, 3) == pytest.approx(1.0, abs=1e-14)
    assert tau_cross(1, 1, 3) == pytest.approx(0.0, abs=1e-14)  # c4 vanishes for T = 3/2


def test_radial_keys_count() -> None:
    assert len(radial_keys("cE", 8)) == 9
    assert len(radial_keys("cD", 8)) == 18
    assert len(radial_keys("c1", 8)) == 18
    assert len(radial_keys("c3", 8)) == 9 * (1 + 3 + 1 + 3)  # K1, K2 in {0, 2} with K3 = 0..K2


@pytest.fixture(scope="module")
def tables() -> tuple:
    meshes = nuhamil_meshes()
    return meshes, {lam: radial_tables(meshes, lam, 2, 8) for lam in (400.0, 500.0)}


@pytest.mark.parametrize(
    ("run", "lam", "j2", "parity", "t2"),
    [
        ("n400_N6", 400.0, 1, 1, 1),
        ("n400_N6", 400.0, 1, 1, 3),
        ("n400_N6", 400.0, 3, -1, 1),
        ("n400_N6", 400.0, 5, 1, 1),
        ("n400_N6", 400.0, 11, -1, 3),
        ("n500_N6", 500.0, 1, 1, 1),
    ],
)
def test_terms_match_nuhamil(
    tables: tuple, run: str, lam: float, j2: int, parity: int, t2: int
) -> None:
    meshes, by_lambda = tables
    header, data = read_hopw(str(oracle_file(run, j2, parity, t2)))
    nmax, dim = header[3], int(round((np.sqrt(8 * header[4] + 1) - 1) / 2))
    blocks = nas_blocks_in_file(j2, parity, t2, nmax, dim)
    for k, term in enumerate(TERM_ORDER):
        _, matrix = nas_term_matrix(
            term, j2, parity, t2, nmax, meshes, by_lambda[lam], 16.0, n_blocks=blocks
        )
        ref = lower_to_full(data[k])
        assert matrix.shape == ref.shape
        assert np.abs(matrix - ref).max() < 1e-11, term
