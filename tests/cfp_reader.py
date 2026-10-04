"""Reader for NuHamil's ``cfp_A3_*.bin`` Jacobi caches (test helper, format in docs/spec-5a)."""

from pathlib import Path

import numpy as np


def read_cfp(
    path: Path,
) -> tuple[tuple[int, int, int, int], list[tuple[int, np.ndarray, np.ndarray]]]:
    """Parse a cache file.

    Returns
    -------
    tuple
        ``((2J, P, 2T, Nmax), blocks)`` with ``blocks`` a list of ``(N, nas, cfp)`` for the N
        blocks that hold states; ``nas`` is ``(n_nas, 8)`` and ``cfp`` is ``(n_as, n_nas)``.
    """
    raw = np.fromfile(path, dtype=np.uint8)
    pos = 0

    def take(dtype: str, count: int) -> np.ndarray:
        nonlocal pos
        width = np.dtype(dtype).itemsize * count
        out = raw[pos : pos + width].view(dtype)
        pos += width
        return out

    j2, parity, t2, nmax = (int(x) for x in take("<i4", 4))
    blocks = []
    for _ in range(nmax + 1):
        n_nas, n_as, n_quanta = (int(x) for x in take("<i4", 3))
        nas = take("<i4", 8 * n_nas).reshape(n_nas, 8).astype(np.int64) if n_nas else None
        cfp = take("<f8", n_as * n_nas).reshape(n_as, n_nas) if n_as else None
        if nas is not None and cfp is not None:
            blocks.append((n_quanta, nas, cfp.copy()))
    assert pos == raw.size, "cfp file not fully parsed"
    return (j2, parity, t2, nmax), blocks
