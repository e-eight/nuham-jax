"""Compression speed and size of the ``me2j`` text with ``gzip`` and, if installed, ``isal``.

Generates a plain-text me2j sample with the package (emax 8 by default), then times each
compressor on it and checks that the output decompresses to the same text. ``isal`` is optional
(``uv run --with isal ...``); it implements igzip, which writes standard gzip streams.

    uv run --with isal python benchmarks/compress_bench.py NN.bin [--emax 8]
"""

import argparse
import gzip
import json
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import nuham  # noqa: F401  sets JAX allocation defaults before the first device use
from nuham.angmom.moshinsky_table import MoshinskyTable
from nuham.nn.generate import generate_me2j
from nuham.nn.input_file import read_nn_mom
from nuham.runtime import enable_compilation_cache


def best_of(
    compress: Callable[[bytes], bytes], data: bytes, repeats: int = 3
) -> tuple[float, bytes]:
    """Return the fastest wall time of ``compress(data)`` over ``repeats`` runs, and its output."""
    best, out = float("inf"), b""
    for _ in range(repeats):
        start = time.perf_counter()
        out = compress(data)
        best = min(best, time.perf_counter() - start)
    return best, out


def main() -> None:
    """Time every available compressor and print a JSON record."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("nn_file")
    p.add_argument("--emax", type=int, default=8)
    p.add_argument("--bracket-cache", default=None)
    args = p.parse_args()
    enable_compilation_cache()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.me2j"
        generate_me2j(
            read_nn_mom(args.nn_file), path, 16.0, 2.0, args.emax, 2 * args.emax,
            mosh=MoshinskyTable(args.bracket_cache),
        )  # fmt: skip
        data = path.read_bytes()
    candidates = {
        f"gzip level {lv}": (lambda d, lv=lv: gzip.compress(d, compresslevel=lv))
        for lv in (1, 2, 6)
    }
    try:
        from isal import igzip

        for lv in (0, 1, 2, 3):
            candidates[f"isal level {lv}"] = lambda d, lv=lv: igzip.compress(d, compresslevel=lv)
    except ImportError:
        pass
    rows = []
    for name, compress in candidates.items():
        seconds, out = best_of(compress, data)
        rows.append(
            {
                "compressor": name,
                "mb_per_s": len(data) / seconds / 1e6,
                "size_mb": len(out) / 1e6,
                "roundtrip_ok": gzip.decompress(out) == data,
            }
        )
    print(json.dumps({"text_mb": len(data) / 1e6, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
