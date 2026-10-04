from pathlib import Path

import numpy as np
import pytest

import nuham  # noqa: F401
from nuham.angmom.moshinsky_table import MoshinskyTable
from nuham.basis.lab import make_orbits
from nuham.nn.generate import generate_me2j
from nuham.nn.input_file import read_nn_mom
from nuham.nn.me2j import read_me2j, tuple_chunks

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


# Other interactions of NuHamil's input directory at hw 16, lambda 2.0, emax 4: the interaction is a
# plug-in, so no code changes were needed. The references come from NuHamil (VODE error ~3e-7);
# the observed maximum differences were 1.1e-7 to 6.5e-7 MeV, and all 11 other files agreed
# (docs/imsrg-check.md).
OTHER_INTERACTIONS = ["LO_EMN500", "N4LO_EMN500", "N2LO_opt", "N2LO_sat", "N3LO_texas"]


@pytest.mark.parametrize("interaction", OTHER_INTERACTIONS)
def test_other_interactions_match_nuhamil(
    tmp_path: Path, interaction_bin, mosh: MoshinskyTable, interaction: str
) -> None:
    ref_path = REF / f"TwBME-HO_NN-only_{interaction}_srg2.0_hw16_emax4_e2max8.me2j.gz"
    if not ref_path.exists():
        pytest.skip("reference file missing")
    nn = read_nn_mom(interaction_bin(interaction))
    out = tmp_path / "out.me2j.gz"
    generate_me2j(nn, out, 16.0, 2.0, 4, 8, mosh=mosh)
    got, ref = read_me2j(out), read_me2j(ref_path)
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 1e-6
