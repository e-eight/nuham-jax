"""Parser for NuHamil's bare relative-space cache ``NNint_A2_rel_*.gz`` (7-digit text)."""

import gzip
from pathlib import Path

import numpy as np


def read_rel_cache(
    path: Path,
) -> dict[tuple[int, int, int, int], dict[tuple[int, int, int, int], float]]:
    """Return ``{(J, parity, S, Tz): {(n1, l1, n2, l2): value}}``; lower triangle as stored."""
    with gzip.open(path, "rt") as fh:
        fh.readline()
        rows = np.array(fh.read().split(), dtype=float).reshape(-1, 9)
    out: dict = {}
    for r in rows:
        key = tuple(int(x) for x in r[:4])
        out.setdefault(key, {})[tuple(int(x) for x in r[4:8])] = r[8]
    return out
