import gzip
from pathlib import Path

import numpy as np
import pytest

from nuham.me2j import _format_lines, _format_lines_python, read_me2j, write_me2j

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
    path = REF / "TwBME-HO_NN-only_N3LO_EM500_srg2.0_hw16_emax4_e2max8.me2j.gz"
    if not path.exists():
        pytest.skip("reference file missing")
    ref = read_me2j(path)
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


def _adversarial_values() -> np.ndarray:
    """Values that stress the vectorized formatter: ties, near-ties, signs, magnitudes."""
    rng = np.random.default_rng(1)
    ties = (np.arange(0, 400) * 2 + 1) / 2.0e8  # exact decimal ties; few are exact in binary
    exact = np.array([0.5e-8, 1.5e-8, 2.5e-8, 0.125, 0.5, 1.0, 99999.5, 0.999999995, 9.999999995])
    powers = np.concatenate([10.0 ** np.arange(-9, 5), 10.0 ** np.arange(0, 5) - 1e-8])
    random = np.concatenate(
        [rng.normal(size=4000), rng.normal(size=4000) * 100, rng.uniform(-1, 1, 4000) * 1e-6]
    )
    near = np.round(rng.normal(size=2000) * 10, 8) + 5e-9 * rng.choice([-1, 1], 2000)
    base = np.concatenate([ties, exact, powers, random, near, [0.0, -0.0, 1e-12, -1e-12]])
    both = np.concatenate([base, -base])
    return both[: len(both) // 10 * 10]


def test_vectorized_format_is_byte_identical_to_python() -> None:
    values = _adversarial_values()
    assert _format_lines(values) == _format_lines_python(values).encode("ascii")


def test_format_falls_back_for_non_finite_and_large_values() -> None:
    values = np.array([np.nan, np.inf, -np.inf, 1e5, -1e7, 123456.789, 0.0, 1.0, 2.0, 3.0])
    assert _format_lines(values) == _format_lines_python(values).encode("ascii")
