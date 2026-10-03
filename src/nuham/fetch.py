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
    "DN2LOGO394": "d6827ce86966a23b22906e340251752cb6c2f8daed2de587b5dfa25d4f6d08b7",
    "DN2LOGO450": "00a1d1fdff11e6830bd33b9af69fe1fe61974c79b6712a9ec0b6c736317f8005",
    "DNLOGO450": "d16410e87ec3237d19de638db0cf8f9b1123187f12e2114d2b0557b5b1c609a0",
    "LO_EMN500": "cde610ce00b72e509145b315f9308f98ba7a372a8b48afc66bdcaedb28fa7752",
    "N2LO_EMN500": "0bae8478611ca3c100e201146fc6bab8b34c2f9eb442e72f5e04d13eb157fc5d",
    "N2LO_opt": "f42a98a1866fe417e57421752cf7e7b7b112e6a4f97708badf4261f734de01ec",
    "N2LO_sat": "c013b2b592bb3c21e09442cfd2cf0c464013156fbcf3cd2d3835ff9fd8000331",
    "N3LO_EM500": "0f3af5611bdc0d187120ebf3e2cb1312d094b6b738c5a1593a4914344ac0f64a",
    "N3LO_EMN500": "9c8edc61c48c4feb16229dce4ce17b1d937ac1fa90171b6bfca29b5fedd5d011",
    "N3LO_texas": "538e812f870e09d9dbddfc197647d50bcd5b2a148a5c5a81f371ae23ceee025d",
    "N4LO_EMN500": "e4d11f7de1e9809fa33542eddd81a79c83ccb35f6d74a64b9d84d5e4ec8424e2",
    "NLO_EMN500": "0240dd202faa9d523f8b093b22c49467ff0ffd2c73fa5711867806975f9e4557",
}
"""Expected SHA-256 of each downloadable file."""


def data_dir() -> Path:
    """Return ``$XDG_CACHE_HOME/nuham/data`` (``~/.cache/nuham/data`` by default)."""
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "nuham" / "data"


def search_dirs() -> list[Path]:
    """Return the directories searched for momentum-space files, in order.

    ``$NUHAMIL_PUBLIC/input_nn_files``, a ``NuHamil-public`` checkout next to this repository, and
    the download cache :func:`data_dir`.
    """
    sibling = Path(__file__).resolve().parents[3] / "NuHamil-public"
    base = Path(os.environ.get("NUHAMIL_PUBLIC", sibling))
    return [base / "input_nn_files", data_dir()]


def find_nn_file(name: str) -> Path | None:
    """Return the path of the momentum-space file of ``name`` if it exists locally, else ``None``.

    Nothing is downloaded and no checksum is checked (a local checkout is trusted); use
    :func:`fetch_nn_file` to download a verified copy into the cache.
    """
    for directory in search_dirs():
        path = directory / nn_file_name(name)
        if path.exists():
            return path
    return None


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
