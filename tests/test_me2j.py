import gzip
from pathlib import Path

import numpy as np
import pytest

from nuham.me2j import read_me2j, write_me2j

REF = Path(__file__).parent.parent / "refs" / "work"


def test_round_trip_and_layout(tmp_path: Path) -> None:
    values = np.linspace(-3.0, 4.0, 27)
    path = tmp_path / "x.me2j.gz"
    chunks = iter([values[:7], values[7:20], values[20:]])
    assert write_me2j(path, chunks, header="hdr") == 27
    with gzip.open(path, "rt") as fh:
        lines = fh.read().split("\n")
    assert lines[0] == "hdr"
    assert [len(x) for x in lines[1:4]] == [160, 160, 112]
    np.testing.assert_allclose(read_me2j(path), values, atol=5e-9)


def test_reads_nuhamil_file() -> None:
    ref = read_me2j(REF / "TwBME-HO_NN-only_N3LO_EM500_srg2.0_hw16_emax4_e2max8.me2j.gz")
    assert len(ref) == 4 * 6834
    assert ref[0] == np.float64(ref[0]) and abs(ref[1] - -15.12895093) < 1e-8


def test_threaded_writer_matches_direct_formatting_across_flushes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nuham.me2j._FLUSH_VALUES", 50)
    rng = np.random.default_rng(1)
    chunks = [rng.normal(scale=5, size=(int(k), 4)) for k in rng.integers(1, 40, size=30)]
    flat = np.concatenate([c.ravel() for c in chunks])
    path = tmp_path / "x.me2j"
    assert write_me2j(path, iter(chunks), header="h") == len(flat)
    lines = path.read_text().split("\n")
    assert lines[0] == "h"
    want = ["".join(f"{x:16.8f}" for x in flat[i : i + 10]) for i in range(0, len(flat), 10)]
    assert lines[1:-1] == want


def test_writer_failure_is_raised(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        write_me2j(tmp_path / "missing_dir" / "x.me2j.gz", iter([np.ones((3, 4))]))


def test_compression_level_is_a_parameter(tmp_path: Path) -> None:
    values = np.linspace(0.0, 1.0, 400)
    sizes = []
    for level in (1, 9):
        path = tmp_path / f"l{level}.me2j.gz"
        write_me2j(path, iter([values]), compresslevel=level)
        np.testing.assert_allclose(read_me2j(path), values, atol=5e-9)
        sizes.append(path.stat().st_size)
    assert sizes[1] <= sizes[0]
