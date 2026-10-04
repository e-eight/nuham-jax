"""The device formatter and the fields writer must write exactly the text the host writes."""

import gzip
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from synthetic import synthetic_nn
from test_me2j import _adversarial_values

import nuham  # noqa: F401
import nuham.nn.me2j as me2j
from nuham.angmom.moshinsky_table import MoshinskyTable
from nuham.basis.lab import lab_channels, make_orbits
from nuham.kernels.nn.format import format_fields, tables
from nuham.nn.lab_transform import lab_channel_matrices
from nuham.nn.me2j import (
    LabChannel,
    _format_lines_python,
    assemble,
    format_device,
    write_channels,
    write_me2j,
    write_me2j_fields,
)
from nuham.nn.relative import evolve_relative_nn
from nuham.runtime import WriterOptions

EMAX, E2MAX = 3, 6


def _patched_fields(values: np.ndarray) -> np.ndarray:
    """Run the kernel and apply the host patch for flagged values, as :func:`format_device`.

    The capacity covers every value, so the adversarial inputs (thousands of ties) never overflow.
    """
    frac, head = tables()
    fields, n_flagged, n_bad, index, flagged = format_fields(
        jnp.asarray(values), values.size, frac, head, values.size
    )
    assert int(n_bad) == 0
    out = np.asarray(fields).copy()
    at, vals = np.asarray(index)[: int(n_flagged)], np.asarray(flagged)[: int(n_flagged)]
    for i, x in zip(at, vals, strict=True):
        out[i] = np.frombuffer(f"{x:16.8f}".encode("ascii"), dtype=np.uint8)
    return out


def test_kernel_text_is_byte_identical_to_python() -> None:
    values = _adversarial_values()
    fields = _patched_fields(values)
    text = _format_lines_python(values).replace("\n", "")
    assert fields.tobytes().decode("ascii") == text


def test_kernel_matches_python_on_random_values_of_many_scales() -> None:
    rng = np.random.default_rng(5)
    for scale in (1e-8, 1e-3, 1.0, 50.0, 1e4):
        v = rng.normal(size=100_000) * scale
        v = np.concatenate([v, np.round(v, 8) + 5e-9 * rng.choice([-1, 1], v.size)])
        v = v[: len(v) // 10 * 10]
        assert _patched_fields(v).tobytes().decode("ascii") == _format_lines_python(v).replace(
            "\n", ""
        )


def test_padding_is_formatted_as_zero_and_counts_are_reported() -> None:
    frac, head = tables()
    values = jnp.asarray([1.5, -2.25, np.nan, 1e9, 7.0, 9.0])
    fields, n_flagged, n_bad, _, _ = format_fields(values, 2, frac, head, 16)
    assert int(n_bad) == 0 and int(n_flagged) == 0  # the NaN and 1e9 are padding here
    assert bytes(np.asarray(fields[2])).decode() == f"{0.0:16.8f}"
    _, _, n_bad_all, _, _ = format_fields(values, 6, frac, head, 16)
    assert int(n_bad_all) == 2


@pytest.fixture(scope="module")
def setup():
    relative = {
        (c.j, c.parity, c.s, c.tz): c
        for c in evolve_relative_nn(synthetic_nn(), 16.0, 2.0, n2max=24)
    }
    orb = make_orbits(EMAX)
    mosh = MoshinskyTable()
    channels = {
        jp: LabChannel(lab_channel_matrices(orb, E2MAX, *jp, relative, mosh, None, True), len(orb))
        for jp in lab_channels(orb, E2MAX)
    }
    return orb, channels


def _text(path: Path) -> bytes:
    with gzip.open(path, "rb") as fh:
        return fh.read()


@pytest.mark.parametrize("batch_tuples", [37, 1000, 1 << 20])
def test_fields_writer_equals_host_writer(tmp_path: Path, setup, batch_tuples: int) -> None:
    orb, channels = setup
    a, b = tmp_path / "a.me2j.gz", tmp_path / "b.me2j.gz"
    n = write_me2j(a, assemble(orb, E2MAX, channels))
    m = write_me2j_fields(b, format_device(orb, E2MAX, channels, batch_tuples))
    assert n == m
    assert _text(a) == _text(b)  # includes windows that split lines and the final partial line


def test_patching_and_capacity_overflow(
    tmp_path: Path, setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    orb, channels = setup
    real = me2j.format_fields

    def flag_every_seventh(values, n_valid, frac, head, capacity):
        fields, _, n_bad, _, _ = real(values, n_valid, frac, head, capacity)
        x = np.asarray(values).ravel()[: int(n_valid)]
        at = np.arange(0, len(x), 7)
        index = np.full(capacity, -1, dtype=np.int32)
        flagged = np.zeros(capacity)
        index[: min(capacity, len(at))] = at[:capacity]
        flagged[: min(capacity, len(at))] = x[at[:capacity]]
        return fields, jnp.asarray(len(at)), n_bad, jnp.asarray(index), jnp.asarray(flagged)

    monkeypatch.setattr(me2j, "format_fields", flag_every_seventh)
    monkeypatch.setattr(me2j, "_FLAG_CAPACITY", 4)  # forces the redo with more room
    a, b = tmp_path / "a.me2j", tmp_path / "b.me2j"
    write_me2j(a, assemble(orb, E2MAX, channels))
    write_me2j_fields(b, format_device(orb, E2MAX, channels, 500))
    assert a.read_bytes() == b.read_bytes()


def test_unformattable_value_raises(setup, monkeypatch: pytest.MonkeyPatch) -> None:
    orb, channels = setup
    real = me2j.assemble_values
    monkeypatch.setattr(me2j, "assemble_values", lambda *a: real(*a).at[0, 0].set(1e7))
    with pytest.raises(ValueError, match="cannot format"):
        list(format_device(orb, E2MAX, channels))


def test_write_channels_option_combinations(tmp_path: Path, setup) -> None:
    orb, channels = setup
    want = tmp_path / "w.me2j"
    write_channels(want, orb, E2MAX, channels)
    for assembler, formatter in [("device", "host"), ("device", "device")]:
        out = tmp_path / f"{assembler}_{formatter}.me2j"
        write_channels(
            out, orb, E2MAX, channels, WriterOptions(assembler=assembler, formatter=formatter)
        )
        assert out.read_bytes() == want.read_bytes()
    with pytest.raises(ValueError, match="needs assembler"):
        write_channels(
            tmp_path / "x.me2j",
            orb,
            E2MAX,
            channels,
            WriterOptions(assembler="numpy", formatter="device"),
        )
    with pytest.raises(ValueError, match="unknown assembler"):
        WriterOptions(assembler="gpu")


def test_threaded_compression_gives_a_valid_stream_with_the_same_text(tmp_path: Path) -> None:
    pytest.importorskip("isal")
    values = np.random.default_rng(3).normal(scale=4, size=200_000)
    a, b = tmp_path / "a.me2j.gz", tmp_path / "b.me2j.gz"
    write_me2j(a, iter([values]), compressor="gzip")
    write_me2j(b, iter([values]), compressor="isal", compress_threads=4)
    assert _text(a) == _text(b)
