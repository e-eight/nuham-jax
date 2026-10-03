"""Shared fixtures: locations of NuHamil inputs and reference outputs."""

import os
import sys
from pathlib import Path

import pytest

from nuham.fetch import data_dir, fetch_nn_file, nn_file_name

ROOT = Path(__file__).resolve().parent.parent
REF_WORK = ROOT / "refs" / "work"
sys.path.insert(0, str(ROOT / "refs"))  # optional reference implementations (not the package)
sys.path.insert(0, str(Path(__file__).parent))  # test helpers such as ``synthetic``


@pytest.fixture(scope="session")
def nn_bin() -> Path:
    """Path of the EM500 momentum-space file, or skip the test.

    Looked up in ``$NUHAMIL_PUBLIC/input_nn_files``, a sibling ``NuHamil-public`` checkout, and the
    download cache. Set ``NUHAM_FETCH=1`` to download it into the cache when it is missing.
    """
    name = nn_file_name("N3LO_EM500")
    base = Path(os.environ.get("NUHAMIL_PUBLIC", ROOT.parent / "NuHamil-public"))
    for path in (base / "input_nn_files" / name, data_dir() / name):
        if path.exists():
            return path
    if os.environ.get("NUHAM_FETCH") == "1":
        return fetch_nn_file("N3LO_EM500")
    pytest.skip(
        f"{name} not found; run `uv run python -m nuham.fetch`, set NUHAMIL_PUBLIC, "
        "or set NUHAM_FETCH=1"
    )


@pytest.fixture(scope="session")
def rel_cache_hw16() -> Path:
    """NuHamil's bare relative-space cache for hw = 16 MeV."""
    path = REF_WORK / "NNint_A2_rel_N3LO_EM500_bare_Nmax100_hw16.gz"
    if not path.exists():
        pytest.skip(f"{path} not found")
    return path
