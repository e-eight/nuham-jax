"""Download NuHamil's momentum-space NN input files (the data, not the code)."""

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

NUHAMIL_COMMIT = "4d310c14970d9b47fa6f40dfde78d03b61afb526"
"""NuHamil-public commit that the reference outputs in ``refs/work`` were produced with."""

_URL = (
    "https://raw.githubusercontent.com/Takayuki-Miyagi/NuHamil-public/"
    f"{NUHAMIL_COMMIT}/input_nn_files/{{name}}_kmax8_N100_Jmax8.bin"
)

SHA256 = {
    "N3LO_EM500": "0f3af5611bdc0d187120ebf3e2cb1312d094b6b738c5a1593a4914344ac0f64a",
}
"""Expected SHA-256 of each downloadable file."""


def data_dir() -> Path:
    """Return ``$XDG_CACHE_HOME/nuham/data`` (``~/.cache/nuham/data`` by default)."""
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "nuham" / "data"


def nn_file_name(name: str) -> str:
    """Return the file name NuHamil uses for the interaction ``name``."""
    return f"{name}_kmax8_N100_Jmax8.bin"


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_nn_file(
    name: str = "N3LO_EM500", directory: str | Path | None = None, force: bool = False
) -> Path:
    """Download a momentum-space NN file from NuHamil's public repository, once.

    The file is fetched from a pinned commit, checked against its SHA-256, and written atomically.
    An existing file with the right checksum is reused.

    Parameters
    ----------
    name
        Interaction name; only names listed in :data:`SHA256` are supported.
    directory
        Target directory; defaults to :func:`data_dir`.
    force
        Download again even if a valid file exists.

    Returns
    -------
    Path
        Path of the verified file.

    Raises
    ------
    ValueError
        If ``name`` is unknown or the downloaded file has the wrong checksum.
    """
    if name not in SHA256:
        raise ValueError(f"unknown interaction {name!r}; known: {sorted(SHA256)}")
    target_dir = Path(directory) if directory is not None else data_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / nn_file_name(name)
    if not force and target.exists() and _sha256(target) == SHA256[name]:
        return target
    with tempfile.NamedTemporaryFile(dir=target_dir, delete=False, suffix=".part") as tmp:
        part = Path(tmp.name)
        with urllib.request.urlopen(_URL.format(name=name), timeout=60) as response:  # noqa: S310
            for block in iter(lambda: response.read(1 << 20), b""):
                tmp.write(block)
    if _sha256(part) != SHA256[name]:
        part.unlink()
        raise ValueError(f"checksum mismatch for {name}; the download was discarded")
    part.chmod(0o644)
    part.replace(target)
    return target


def main() -> None:
    """Command-line entry point: ``python -m nuham.fetch [name]``."""
    parser = argparse.ArgumentParser(description="Download a NuHamil momentum-space NN file.")
    parser.add_argument("name", nargs="?", default="N3LO_EM500", choices=sorted(SHA256))
    parser.add_argument("--directory", default=None, help=f"default: {data_dir()}")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(fetch_nn_file(args.name, args.directory, args.force))


if __name__ == "__main__":
    main()
