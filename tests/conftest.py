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


def _find_nn_file(interaction: str) -> Path | None:
    """Locate a momentum-space file in ``$NUHAMIL_PUBLIC``, the sibling checkout or the cache."""
    name = nn_file_name(interaction)
    base = Path(os.environ.get("NUHAMIL_PUBLIC", ROOT.parent / "NuHamil-public"))
    for path in (base / "input_nn_files" / name, data_dir() / name):
        if path.exists():
            return path
    return None


@pytest.fixture(scope="session")
def nn_bin() -> Path:
    """Path of the EM500 momentum-space file, or skip the test.

    Looked up in ``$NUHAMIL_PUBLIC/input_nn_files``, a sibling ``NuHamil-public`` checkout, and the
    download cache. Set ``NUHAM_FETCH=1`` to download it into the cache when it is missing.
    """
    found = _find_nn_file("N3LO_EM500")
    if found is not None:
        return found
    if os.environ.get("NUHAM_FETCH") == "1":
        return fetch_nn_file("N3LO_EM500")
    pytest.skip(
        f"{nn_file_name('N3LO_EM500')} not found; run `uv run python -m nuham.fetch`, set "
        "NUHAMIL_PUBLIC, or set NUHAM_FETCH=1"
    )


@pytest.fixture(scope="session")
def interaction_bin():
    """Return a function ``name -> Path`` of any NuHamil momentum-space file; it skips if missing.

    Only the EM500 file can be downloaded (``nuham.fetch``); the others are looked up in the same
    places as :func:`nn_bin` and their tests skip when absent.
    """

    def find(name: str) -> Path:
        found = _find_nn_file(name)
        if found is None:
            pytest.skip(f"{nn_file_name(name)} not found (set NUHAMIL_PUBLIC)")
        return found

    return find


@pytest.fixture(scope="session")
def rel_cache_hw16() -> Path:
    """NuHamil's bare relative-space cache for hw = 16 MeV."""
    path = REF_WORK / "NNint_A2_rel_N3LO_EM500_bare_Nmax100_hw16.gz"
    if not path.exists():
        pytest.skip(f"{path} not found")
    return path
