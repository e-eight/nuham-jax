"""Row-packed lower triangle of a symmetric matrix: the storage of the lab matrices.

Element ``(i, j)`` with ``i >= j`` is at ``i (i + 1) / 2 + j``. A symmetric matrix needs half the
memory this way (the lab matrices of the production file are 45 GiB in single precision as full
squares), and every lookup goes through :func:`packed_index`, which orders the two indices.
"""

import numpy as np


def packed_size(n: int) -> int:
    """Return the number of stored elements of an ``n x n`` symmetric matrix."""
    return n * (n + 1) // 2


def packed_index(i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Position of element ``(i, j)`` (either order) in the packed array."""
    hi, lo = np.maximum(i, j).astype(np.int64), np.minimum(i, j).astype(np.int64)
    return hi * (hi + 1) // 2 + lo


def pack_lower(m: np.ndarray) -> np.ndarray:
    """Packed lower triangle of the square matrix ``m`` (its upper triangle is ignored)."""
    rows, cols = np.tril_indices(len(m))
    return m[rows, cols]


def unpack_lower(packed: np.ndarray, n: int) -> np.ndarray:
    """Symmetric ``n x n`` matrix of a packed lower triangle."""
    rows, cols = np.tril_indices(n)
    out = np.zeros((n, n), dtype=packed.dtype)
    out[rows, cols] = packed
    out[cols, rows] = packed
    return out
