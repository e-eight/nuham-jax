"""The 3N driver's plumbing: worker counts, config line, stop points, stage log (Phase 5.4).

A worker count that silently stayed at 1 once made a GPU run six times slower, so these tests
check that every option reaches the code that uses it.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import nuham  # noqa: F401  (enables float64)
from nuham.nn.input_file import read_nn_mom
from nuham.nnn import generate
from nuham.nnn.generate import LocalForce, generate_me3j, jacobi_operators
from nuham.runtime import default_workers


@pytest.fixture(scope="module")
def nn(nn_bin: Path):
    return read_nn_mom(nn_bin)


@pytest.fixture
def force_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Replace the 3NF assembly by a recorder that returns zeros of the right size."""
    calls: list[dict] = []

    def fake(channel, hw, meshes, tables, lecs, *, workers=1, timer=None):
        calls.append({"workers": workers, "n": channel.n_as})
        return np.zeros((channel.n_as, channel.n_as))

    monkeypatch.setattr(generate, "local_3nf_matrix", fake)
    return calls


def test_workers_reach_the_force(nn, force_calls: list[dict]) -> None:
    jacobi_operators(nn, 16.0, 2.0, 2, "flat2", LocalForce(), workers=3)
    assert force_calls and {c["workers"] for c in force_calls} == {3}


def test_default_workers_is_every_cpu(nn, force_calls: list[dict]) -> None:
    generate_me3j(nn, None, 16.0, 2.0, 2, 4, 2, "flat2", LocalForce(), stop_after="jacobi")
    assert {c["workers"] for c in force_calls} == {default_workers()}


def test_one_worker_on_a_big_machine_warns(
    nn, force_calls: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(generate, "default_workers", lambda: 32)
    with pytest.warns(UserWarning, match="workers=1 with 32 CPUs"):
        generate_me3j(nn, None, 16.0, 2.0, 2, 4, 2, "flat2", LocalForce(), workers=1,
                      stop_after="jacobi")  # fmt: skip


def test_bad_stop_point_is_rejected(nn) -> None:
    with pytest.raises(ValueError, match="stop_after"):
        generate_me3j(nn, None, 16.0, 2.0, 2, 4, 2, "flat2", stop_after="later")


def test_stage_log_has_one_record_per_channel(nn, force_calls: list[dict], tmp_path: Path) -> None:
    log = tmp_path / "stages.jsonl"
    timer = generate.StageTimer()
    generate_me3j(nn, None, 16.0, 2.0, 2, 4, 2, "flat2", LocalForce(), timer, workers=2,
                  stop_after="jacobi", stage_log=log)  # fmt: skip
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == len(force_calls) > 0
    assert all(r["kind"] == "jacobi" and r["n_as"] > 0 for r in records)
    assert set(records[0]["seconds"]) == {"jacobi_basis", "force", "embed", "flow", "to_host"}
    # the run total is the sum of the channels
    total = sum(r["seconds"]["flow"] for r in records)
    assert timer.seconds["flow"] == pytest.approx(total, abs=1e-4 * len(records))


def run_cli(monkeypatch: pytest.MonkeyPatch, capsys, *argv: str) -> tuple[dict, dict]:
    """Run ``main`` with a recorder in place of the generator; return its config and call."""
    seen: dict = {}

    def fake(*args, **kwargs):
        seen["args"] = args
        return 0

    monkeypatch.setattr(generate, "generate_me3j", fake)
    monkeypatch.setattr(generate, "read_nn_mom", lambda path: None)
    monkeypatch.setattr(generate, "enable_compilation_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        sys, "argv",
        ["generate", "nn.bin", "out.me3j.gz", "--hw", "16", "--lam", "2", "--emax", "4",
         "--e2max", "8", "--e3max", "4", "--ramp", "flat8", *argv],
    )  # fmt: skip
    generate.main()
    line = next(x for x in capsys.readouterr().out.splitlines() if x.startswith("config: "))
    return json.loads(line[len("config: ") :]), seen


def test_cli_reports_and_forwards_its_settings(monkeypatch, capsys) -> None:
    config, seen = run_cli(monkeypatch, capsys)
    assert config["workers"] == default_workers()
    assert config["compress_threads"] == min(8, default_workers())
    # last positional arguments of generate_me3j: workers, cache, stop, log, js
    assert seen["args"][-5] == default_workers()
    config, seen = run_cli(monkeypatch, capsys, "--workers", "5", "--stop-after", "lab",
                           "--compress-threads", "3")  # fmt: skip
    assert (config["workers"], config["compress_threads"], config["stop_after"]) == (5, 3, "lab")
    assert seen["args"][-5] == 5 and seen["args"][-3] == "lab"


def test_finished_channels_are_cached_and_resumed(
    nn, force_calls: list[dict], tmp_path: Path
) -> None:
    first, _ = jacobi_operators(nn, 16.0, 2.0, 3, "flat3", LocalForce(), cache_dir=tmp_path)
    calls = len(force_calls)
    assert calls > 0
    second, _ = jacobi_operators(nn, 16.0, 2.0, 3, "flat3", LocalForce(), cache_dir=tmp_path)
    assert len(force_calls) == calls  # nothing was recomputed
    assert first.keys() == second.keys()
    assert all(np.array_equal(first[k], second[k]) for k in first)
    # other physics never reads the cache
    jacobi_operators(nn, 20.0, 2.0, 3, "flat3", LocalForce(), cache_dir=tmp_path)
    assert len(force_calls) == 2 * calls


def test_js_and_force_cut_select_channels(nn, force_calls: list[dict]) -> None:
    ops, _ = jacobi_operators(nn, 16.0, 2.0, 3, "flat3", None, js=[3])
    assert {j2 for j2, _, _ in ops} == {3}
    jacobi_operators(nn, 16.0, 2.0, 3, "flat3", LocalForce())
    everything = len(force_calls)
    force_calls.clear()
    jacobi_operators(nn, 16.0, 2.0, 3, "flat3", LocalForce(jmax3=3))
    assert 0 < len(force_calls) < everything
    with pytest.raises(ValueError, match="js"):
        generate_me3j(nn, None, 16.0, 2.0, 2, 4, 2, "flat2", js=[1], stop_after="write")


def test_operators_are_not_reused_with_another_basis(
    nn, force_calls: list[dict], tmp_path: Path
) -> None:
    """A cached V3 belongs to the exact ``cfp`` it was made in: a rebuilt basis recomputes it."""
    jacobi_operators(nn, 16.0, 2.0, 3, "flat3", LocalForce(), cache_dir=tmp_path)
    calls = len(force_calls)
    path = next(tmp_path.glob("jacobi_j3_p+_t1_N3.npz"))
    with np.load(path) as data:
        arrays = dict(data)
    key = next(k for k in arrays if k.startswith("cfp"))
    arrays[key] = -arrays[key]  # the same space in another gauge
    np.savez(path, **arrays)
    jacobi_operators(nn, 16.0, 2.0, 3, "flat3", LocalForce(), cache_dir=tmp_path)
    assert len(force_calls) == calls + 1  # only the channel with the changed basis


def test_basis_cache_of_another_version_is_rebuilt(tmp_path) -> None:
    from nuham.nnn import jacobi

    built = jacobi.load_or_build_channel(3, 1, 1, 4, cache_dir=tmp_path)
    path = next(tmp_path.glob("*.npz"))
    with np.load(path) as data:
        arrays = {k: v for k, v in data.items() if k != "version"}
    key = next(k for k in arrays if k.startswith("cfp"))
    arrays[key] = arrays[key] * 0.0  # stale content under an unversioned file
    np.savez(path, **arrays)
    again = jacobi.load_or_build_channel(3, 1, 1, 4, cache_dir=tmp_path)
    for a, b in zip(again.blocks, built.blocks, strict=True):
        np.testing.assert_array_equal(a.cfp, b.cfp)
