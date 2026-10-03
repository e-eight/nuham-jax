import gzip
from pathlib import Path

import numpy as np

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
