"""Shared fixtures: locations of NuHamil inputs and reference outputs."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REF_WORK = ROOT / "refs" / "work"
sys.path.insert(0, str(ROOT / "refs"))  # optional reference implementations (not the package)


@pytest.fixture(scope="session")
def nn_bin() -> Path:
    """Path of the EM500 momentum-space file; skips the test if the NuHamil checkout is absent."""
    base = Path(os.environ.get("NUHAMIL_PUBLIC", ROOT.parent / "NuHamil-public"))
    path = base / "input_nn_files" / "N3LO_EM500_kmax8_N100_Jmax8.bin"
    if not path.exists():
        pytest.skip(f"{path} not found (set NUHAMIL_PUBLIC)")
    return path


@pytest.fixture(scope="session")
def rel_cache_hw16() -> Path:
    """NuHamil's bare relative-space cache for hw = 16 MeV."""
    path = REF_WORK / "NNint_A2_rel_N3LO_EM500_bare_Nmax100_hw16.gz"
    if not path.exists():
        pytest.skip(f"{path} not found")
    return path
