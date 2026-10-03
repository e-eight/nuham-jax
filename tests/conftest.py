"""Shared fixtures: locations of NuHamil inputs and reference outputs."""

import os
import sys
from pathlib import Path

import pytest

from nuham.fetch import fetch_nn_file, find_nn_file, nn_file_name

ROOT = Path(__file__).resolve().parent.parent
REF_WORK = ROOT / "refs" / "work"
sys.path.insert(0, str(ROOT / "refs"))  # optional reference implementations (not the package)
sys.path.insert(0, str(Path(__file__).parent))  # test helpers such as ``synthetic``


@pytest.fixture(scope="session")
def interaction_bin():
    """Return a function ``name -> Path`` of a NuHamil momentum-space file; it skips if missing.

    The lookup and the download live in :mod:`nuham.fetch`; this only turns "not found" into a
    skip. A missing file is downloaded (pinned commit, SHA-256 checked) when ``NUHAM_FETCH=1``.
    """

    def find(name: str) -> Path:
        found = find_nn_file(name)
        if found is None and os.environ.get("NUHAM_FETCH") == "1":
            found = fetch_nn_file(name)
        if found is None:
            pytest.skip(
                f"{nn_file_name(name)} not found; run `uv run python -m nuham.fetch {name}`, set "
                "NUHAMIL_PUBLIC, or set NUHAM_FETCH=1"
            )
        return found

    return find


@pytest.fixture(scope="session")
def nn_bin(interaction_bin) -> Path:
    """Path of the EM500 momentum-space file, or skip the test (see :func:`interaction_bin`)."""
    return interaction_bin("N3LO_EM500")


@pytest.fixture(scope="session")
def rel_cache_hw16() -> Path:
    """NuHamil's bare relative-space cache for hw = 16 MeV."""
    path = REF_WORK / "NNint_A2_rel_N3LO_EM500_bare_Nmax100_hw16.gz"
    if not path.exists():
        pytest.skip(f"{path} not found")
    return path
