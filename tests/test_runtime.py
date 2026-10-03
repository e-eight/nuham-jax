from pathlib import Path

import jax
import pytest

from nuham.runtime import default_cache_dir, enable_compilation_cache


def test_default_cache_dir_follows_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert default_cache_dir() == tmp_path / "nuham" / "jax"


def test_enable_compilation_cache_sets_jax_config(tmp_path: Path) -> None:
    previous = jax.config.jax_compilation_cache_dir
    try:
        used = enable_compilation_cache(tmp_path / "cc")
        assert used.is_dir()
        assert jax.config.jax_compilation_cache_dir == str(tmp_path / "cc")
    finally:
        jax.config.update("jax_compilation_cache_dir", previous)


def test_default_flow_sets_follows_backend(monkeypatch) -> None:
    import jax

    from nuham.runtime import default_flow_sets

    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    assert default_flow_sets() == 1
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    assert default_flow_sets() is None


def test_writer_options_default_by_backend(monkeypatch) -> None:
    import jax

    from nuham.runtime import WriterOptions

    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    cpu = WriterOptions().resolve()
    assert (cpu.assembler, cpu.formatter, cpu.compress_threads) == ("numpy", "host", 1)
    assert not cpu.on_device
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    gpu = WriterOptions().resolve()
    assert (gpu.assembler, gpu.formatter) == ("device", "device")
    assert gpu.compress_threads is not None and 1 <= gpu.compress_threads <= 4
    assert gpu.on_device
    # explicit choices win, and resolving twice changes nothing
    mixed = WriterOptions(assembler="device", formatter="host", compress_threads=2).resolve()
    assert (mixed.assembler, mixed.formatter, mixed.compress_threads) == ("device", "host", 2)
    assert gpu.resolve() == gpu
    # on a CPU the device assembler alone does not switch the formatter on
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    assert WriterOptions(assembler="device").resolve().formatter == "host"


def test_writer_options_validation() -> None:
    import pytest

    from nuham.runtime import WriterOptions

    with pytest.raises(ValueError, match="unknown compressor"):
        WriterOptions(compressor="zstd")
    with pytest.raises(ValueError, match="unknown formatter"):
        WriterOptions(formatter="gpu")
    with pytest.raises(ValueError, match="at least 1"):
        WriterOptions(compress_threads=0)
    with pytest.raises(ValueError, match="needs assembler"):
        WriterOptions(assembler="numpy", formatter="device").resolve()


def test_writer_arguments_round_trip() -> None:
    import argparse

    from nuham.runtime import WriterOptions, add_writer_arguments, writer_options_from_args

    parser = argparse.ArgumentParser()
    add_writer_arguments(parser)
    assert writer_options_from_args(parser.parse_args([])) == WriterOptions()
    args = parser.parse_args(
        ["--gzip-level", "2", "--compressor", "isal", "--compress-threads", "3",
         "--assembler", "device", "--formatter", "device"]
    )  # fmt: skip
    assert writer_options_from_args(args) == WriterOptions(2, "isal", 3, "device", "device")
