import hashlib
from pathlib import Path

import pytest

from nuham import fetch


@pytest.fixture
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake 'remote' directory served over file:// with a registered checksum."""
    remote = tmp_path / "remote"
    remote.mkdir()
    payload = b"not a real potential" * 100
    (remote / fetch.nn_file_name("TEST")).write_bytes(payload)
    monkeypatch.setattr(fetch, "_URL", f"file://{remote}/{{name}}_kmax8_N100_Jmax8.bin")
    monkeypatch.setitem(fetch.SHA256, "TEST", hashlib.sha256(payload).hexdigest())
    return remote


def test_downloads_and_verifies(served: Path, tmp_path: Path) -> None:
    target = fetch.fetch_nn_file("TEST", tmp_path / "cache")
    assert target.read_bytes() == (served / fetch.nn_file_name("TEST")).read_bytes()
    assert target.stat().st_mode & 0o044  # readable by others, not a private temp file
    assert not list((tmp_path / "cache").glob("*.part"))


def test_reuses_a_valid_file_without_the_network(
    served: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = fetch.fetch_nn_file("TEST", tmp_path / "cache")
    monkeypatch.setattr(fetch, "_URL", "file:///nonexistent/{name}")
    assert fetch.fetch_nn_file("TEST", tmp_path / "cache") == first


def test_checksum_mismatch_is_rejected_and_cleaned_up(
    served: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(fetch.SHA256, "TEST", "0" * 64)
    with pytest.raises(ValueError, match="checksum"):
        fetch.fetch_nn_file("TEST", tmp_path / "cache")
    assert list((tmp_path / "cache").iterdir()) == []


def test_corrupt_cached_file_is_replaced(served: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / fetch.nn_file_name("TEST")).write_bytes(b"corrupt")
    path = fetch.fetch_nn_file("TEST", cache)
    assert path.read_bytes() != b"corrupt"


def test_unknown_name_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown"):
        fetch.fetch_nn_file("NOPE", tmp_path)


def test_default_cache_follows_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert fetch.data_dir() == tmp_path / "nuham" / "data"


def test_checksum_table_is_well_formed() -> None:
    assert "N3LO_EM500" in fetch.SHA256
    assert all(len(h) == 64 and int(h, 16) >= 0 for h in fetch.SHA256.values())


def test_find_nn_file_searches_the_directories_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    public, cache = tmp_path / "public", tmp_path / "cache"
    (public / "input_nn_files").mkdir(parents=True)
    cache.mkdir()
    monkeypatch.setenv("NUHAMIL_PUBLIC", str(public))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    name = fetch.nn_file_name("TEST")
    assert fetch.find_nn_file("TEST") is None
    (cache / "nuham" / "data").mkdir(parents=True)
    (cache / "nuham" / "data" / name).write_bytes(b"cached")
    assert fetch.find_nn_file("TEST") == cache / "nuham" / "data" / name
    (public / "input_nn_files" / name).write_bytes(b"checkout")
    assert fetch.find_nn_file("TEST") == public / "input_nn_files" / name  # the checkout wins


def test_checksums_match_the_local_checkout_when_present() -> None:
    found = {name: fetch.find_nn_file(name) for name in fetch.SHA256}
    if not any(found.values()):
        pytest.skip("no local NuHamil checkout")
    for name, path in found.items():
        if path is not None and path.parent != fetch.data_dir():
            assert fetch._sha256(path) == fetch.SHA256[name], name
