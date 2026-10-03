from pathlib import Path

import numpy as np
import pytest

import nuham  # noqa: F401
from nuham.angmom.moshinsky_table import MoshinskyTable
from nuham.basis.lab import make_orbits
from nuham.generate import generate_me2j
from nuham.interactions.nn_file import read_nn_mom
from nuham.me2j import read_me2j, tuple_chunks

REF = Path(__file__).parent.parent / "refs" / "work"

# name, hw, lambda, emax, e2max, tolerance (MeV). The bare and the lambda=1.8 references were
# made from NuHamil's 7-digit relative cache, which limits agreement to ~5e-6; the others
# projected the .bin directly and are limited by NuHamil's own VODE error (~3e-7).
CASES = [
    ("srg2.0_hw16_emax4_e2max8", 16.0, 2.0, 4, 8, 1e-6),
    ("srg2.0_hw20_emax4_e2max8", 20.0, 2.0, 4, 8, 1e-6),
    ("bare2.0_hw16_emax4_e2max8", 16.0, None, 4, 8, 5e-6),
    ("srg1.8_hw16_emax6_e2max12", 16.0, 1.8, 6, 12, 5e-6),
]


@pytest.fixture(scope="module")
def nn(nn_bin: Path):
    return read_nn_mom(nn_bin)


@pytest.fixture(scope="module")
def mosh() -> MoshinskyTable:
    return MoshinskyTable()


@pytest.mark.parametrize(("name", "hw", "lam", "emax", "e2max", "tol"), CASES)
def test_matches_nuhamil_me2j(
    tmp_path: Path, nn, mosh: MoshinskyTable, name: str, hw: float, lam, emax: int, e2max: int,
    tol: float,
) -> None:  # fmt: skip
    ref_path = REF / f"TwBME-HO_NN-only_N3LO_EM500_{name}.me2j.gz"
    if not ref_path.exists():
        pytest.skip("reference file missing")
    out = tmp_path / "out.me2j.gz"
    generate_me2j(nn, out, hw, lam, emax, e2max, mosh=mosh)
    got, ref = read_me2j(out), read_me2j(ref_path)
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < tol


def test_forbidden_entries_vanish(tmp_path: Path, nn, mosh: MoshinskyTable) -> None:
    out = tmp_path / "out.me2j.gz"
    generate_me2j(nn, out, 16.0, 2.0, 4, 8, mosh=mosh)
    vals = read_me2j(out).reshape(-1, 4)
    orb = make_orbits(4)
    k = 0
    for a, b, _cs, _ds, js in tuple_chunks(orb, 8):
        for j in js:
            if a == b and j % 2 == 0:
                assert vals[k, 0] == 0.0  # T = 0 needs odd J for identical orbits
            if a == b and j % 2 == 1:
                assert vals[k, 1] == 0.0 and vals[k, 3] == 0.0 and vals[k, 2] == 0.0
            k += 1
