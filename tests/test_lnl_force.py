"""The LNL option of the 3N driver (gap D, step D2): wiring, cache keys, presets."""

import argparse
from pathlib import Path

import numpy as np
import pytest

import nuham  # noqa: F401  (enables float64)
from nuham.nn.input_file import read_nn_mom
from nuham.nnn.generate import PRESETS, LocalForce, _force_key, force_from_args, jacobi_operators


@pytest.fixture(scope="module")
def nn(nn_bin: Path):
    return read_nn_mom(nn_bin)


def _ops(nn, force: LocalForce) -> dict:
    ops, _ = jacobi_operators(nn, 16.0, 2.0, 3, "flat3", force, js=[1, 3])
    return {k: np.asarray(v) for k, v in ops.items()}


def test_infinite_nonlocal_cutoff_is_the_local_force(nn) -> None:
    local = _ops(nn, LocalForce())
    lnl = _ops(nn, LocalForce(nonlocal_cutoff=1e9))
    assert max(np.abs(lnl[k] - local[k]).max() for k in local) < 1e-8


def test_finite_nonlocal_cutoff_changes_the_force(nn) -> None:
    local = _ops(nn, LocalForce())
    lnl = _ops(nn, LocalForce(nonlocal_cutoff=500.0))
    assert max(np.abs(lnl[k] - local[k]).max() for k in local) > 1e-2


def test_local_cache_keys_are_unchanged() -> None:
    """``nonlocal_cutoff=None`` must not alter ``repr`` (and so the keys of finished caches)."""
    assert repr(LocalForce()) == (
        "LocalForce(lecs={'c1': -0.81, 'c3': -3.2, 'c4': 5.4, 'cD': -0.2, 'cE': 0.098}, "
        "cutoff=400.0, power=2, jmax3=None)"
    )
    assert _force_key(LocalForce()) == LocalForce()
    assert _force_key(LocalForce(nonlocal_cutoff=500.0)) != _force_key(
        LocalForce(nonlocal_cutoff=400.0)
    )
    assert _force_key(None) is None


def _args(**given) -> argparse.Namespace:
    keys = ("c1", "c3", "c4", "cD", "cE", "cutoff", "cutoff_nl", "power")
    values = {k: None for k in keys} | {"preset": "3NF400", "j3max_3nf": None} | given
    return argparse.Namespace(**values)


def test_presets_and_overrides() -> None:
    assert force_from_args(_args()) == PRESETS["3NF400"]
    lnl = force_from_args(_args(preset="N3LOEMlnl"))
    assert (lnl.cutoff, lnl.nonlocal_cutoff, lnl.power) == (650.0, 500.0, 2)
    assert (lnl.lecs["cD"], lnl.lecs["cE"]) == (0.7, -0.06)
    over = force_from_args(_args(preset="N3LOEMlnl", cutoff_nl=450.0, cD=0.1, j3max_3nf=5))
    assert (over.nonlocal_cutoff, over.lecs["cD"], over.jmax3) == (450.0, 0.1, 5)
    assert force_from_args(_args(cutoff_nl=500.0)).cutoff == 400.0
