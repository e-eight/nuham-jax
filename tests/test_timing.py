"""Stage timers and compute-only mode must not change what ``generate_me2j`` produces."""

from pathlib import Path

import numpy as np
import pytest
from synthetic import synthetic_nn

import nuham  # noqa: F401
from nuham.angmom.moshinsky_table import MoshinskyTable
from nuham.nn.generate import generate_me2j
from nuham.nn.me2j import read_me2j
from nuham.timing import StageTimer, stage


def test_stage_timer_accumulates_and_wrap_excludes_consumer() -> None:
    timer = StageTimer()
    with timer.stage("a"):
        pass
    with timer.stage("a"):
        pass
    assert timer.calls["a"] == 2
    assert list(timer.wrap("w", iter(range(3)))) == [0, 1, 2]
    assert timer.calls["w"] == 4  # one call per item plus the one that finds the end
    with stage(None, "x"):
        pass


def test_timer_and_compute_only_do_not_change_output(tmp_path: Path) -> None:
    nn, mosh = synthetic_nn(), MoshinskyTable()
    plain, timed = tmp_path / "plain.me2j", tmp_path / "timed.me2j"
    n = generate_me2j(nn, plain, 16.0, 2.0, 3, 6, mosh=mosh)
    timer = StageTimer()
    assert generate_me2j(nn, timed, 16.0, 2.0, 3, 6, mosh=mosh, timer=timer) == n
    np.testing.assert_array_equal(read_me2j(plain), read_me2j(timed))
    assert plain.read_bytes() == timed.read_bytes()
    assert set(timer.seconds) == {
        "project", "srg", "lab_coeffs", "lab_contract", "assemble", "write",
    }  # fmt: skip
    assert all(v >= 0.0 for v in timer.seconds.values())
    compute_only = StageTimer()
    assert generate_me2j(nn, None, 16.0, 2.0, 3, 6, mosh=mosh, timer=compute_only) == 0
    assert "write" not in compute_only.seconds


@pytest.mark.parametrize("lam", [None, 2.0])
def test_bare_and_evolved_both_record_srg_stage(lam) -> None:
    timer = StageTimer()
    generate_me2j(synthetic_nn(), None, 16.0, lam, 2, 4, mosh=MoshinskyTable(), timer=timer)
    assert "srg" in timer.seconds
