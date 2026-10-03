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
