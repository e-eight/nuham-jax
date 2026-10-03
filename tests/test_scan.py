"""Scans must reproduce separate single-set runs; no NuHamil data needed."""

from pathlib import Path

import numpy as np
import pytest
from synthetic import synthetic_nn

import nuham  # noqa: F401
from nuham.angmom.moshinsky_table import MoshinskyTable
from nuham.generate import generate_me2j
from nuham.me2j import read_me2j
from nuham.relative_nn import evolve_relative_nn, evolve_relative_nn_many
from nuham.scan import ScanPoint, generate_scan
from nuham.timing import StageTimer

EMAX, E2MAX = 3, 6
SETS = [(16.0, 2.0), (20.0, 1.8), (12.0, None), (16.0, 1.6)]


@pytest.fixture(scope="module")
def nn():
    return synthetic_nn()


@pytest.fixture(scope="module")
def mosh() -> MoshinskyTable:
    return MoshinskyTable()


def test_many_relative_equals_single_runs(nn) -> None:
    many = evolve_relative_nn_many(nn, SETS, n2max=24)
    for (hw, lam), got in zip(SETS, many, strict=True):
        single = evolve_relative_nn(nn, hw, lam, n2max=24)
        assert len(got) == len(single)
        for a, b in zip(got, single, strict=True):
            assert (a.j, a.parity, a.s, a.tz) == (b.j, b.parity, b.s, b.tz)
            np.testing.assert_allclose(a.v, b.v, atol=1e-9)


def test_flow_group_size_does_not_change_results(nn) -> None:
    together = evolve_relative_nn_many(nn, SETS, n2max=24, flow_sets=None)
    split = evolve_relative_nn_many(nn, SETS, n2max=24, flow_sets=1)
    for a, b in zip(together, split, strict=True):
        for x, y in zip(a, b, strict=True):
            np.testing.assert_allclose(x.v, y.v, atol=1e-9)


@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_scan_matches_separate_runs(tmp_path: Path, nn, mosh, batch_size: int) -> None:
    paths = [tmp_path / f"scan{i}.me2j.gz" for i in range(len(SETS))]
    points = [ScanPoint(hw, lam, path) for (hw, lam), path in zip(SETS, paths, strict=True)]
    counts = generate_scan(nn, points, EMAX, E2MAX, mosh=mosh, n2max=24, batch_size=batch_size)
    for i, (hw, lam) in enumerate(SETS):
        ref = tmp_path / f"ref{i}.me2j.gz"
        n = generate_me2j(nn, ref, hw, lam, EMAX, E2MAX, mosh=mosh, n2max=24)
        assert counts[i] == n
        np.testing.assert_allclose(read_me2j(paths[i]), read_me2j(ref), atol=1e-8)


def test_compute_only_points_write_nothing(tmp_path: Path, nn, mosh) -> None:
    points = [ScanPoint(16.0, 2.0, None), ScanPoint(20.0, 2.0, tmp_path / "x.me2j.gz")]
    counts = generate_scan(nn, points, EMAX, E2MAX, mosh=mosh, n2max=24)
    assert counts[0] == 0 and counts[1] > 0
    assert [p.name for p in tmp_path.iterdir()] == ["x.me2j.gz"]


def test_timer_accumulates_over_points(tmp_path: Path, nn, mosh) -> None:
    timer = StageTimer()
    points = [ScanPoint(hw, 2.0, tmp_path / f"t{i}.me2j") for i, hw in enumerate((14.0, 18.0))]
    generate_scan(nn, points, EMAX, E2MAX, mosh=mosh, n2max=24, batch_size=1, timer=timer)
    assert timer.calls["srg"] == 2
    assert all(v >= 0.0 for v in timer.seconds.values())  # write is not double-subtracted


def test_invalid_batch_size(nn) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        generate_scan(nn, [], EMAX, E2MAX, batch_size=0)
