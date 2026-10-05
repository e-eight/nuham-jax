"""Device antisymmetrisation and scatter of the 3N Jacobi operators against plain NumPy."""

import numpy as np
import scipy.sparse as sp

import nuham  # noqa: F401  (enables float64)
from nuham.kernels.nnn.projection import scatter_pair_blocks
from nuham.nnn.jacobi import JacobiBlock, JacobiChannel
from nuham.nnn.local_force.assembly import antisymmetrize


def _channel(sizes: list[tuple[int, int]], rng: np.random.Generator) -> JacobiChannel:
    """Channel with random ``cfp`` blocks of shape ``(n_as, n_nas)`` (the basis is irrelevant)."""
    blocks = tuple(
        JacobiBlock(n, np.zeros((nas, 8), dtype=int), rng.normal(size=(n_as, nas)))
        for n, (n_as, nas) in enumerate(sizes)
    )
    return JacobiChannel(1, 1, 1, len(sizes) - 1, blocks)


def _reference(w: np.ndarray, channel: JacobiChannel) -> np.ndarray:
    """``C W C^T`` with the block-diagonal ``C`` built explicitly."""
    c = np.zeros((channel.n_as, channel.n_nas))
    r = k = 0
    for b in channel.blocks:
        c[r : r + len(b.cfp), k : k + b.cfp.shape[1]] = b.cfp
        r, k = r + len(b.cfp), k + b.cfp.shape[1]
    return c @ w @ c.T


def test_dense_and_sparse_equal_the_explicit_product() -> None:
    rng = np.random.default_rng(1)
    channel = _channel([(2, 5), (3, 7), (1, 4), (4, 9)], rng)
    w = rng.normal(size=(channel.n_nas, channel.n_nas))
    w = w + w.T
    ref = _reference(w, channel)
    np.testing.assert_allclose(antisymmetrize(w, channel), ref, atol=1e-12)
    sparse = sp.csr_array(np.where(np.abs(w) > 1.0, w, 0.0))
    np.testing.assert_allclose(
        antisymmetrize(sparse, channel), _reference(sparse.toarray(), channel), atol=1e-12
    )


def test_scatter_fills_both_triangles_once_and_drops_padding() -> None:
    n = 6
    n12 = np.array([0, 1, 0, 1, 0, 1, 0])
    n3 = np.array([0, 0, 1, 1, 0, 0, 0])
    m = np.arange(2 * 2 * 2 * 2 * 2, dtype=float).reshape(2, 2, 2, 2, 2)
    rows = np.array([[0, 1, n], [2, 3, 4]], dtype=np.int32)
    cols = np.array([[2, 3, n], [2, 3, 4]], dtype=np.int32)
    mirror = np.array([True, False])
    out = np.asarray(
        scatter_pair_blocks(np.zeros((n, n)), m, rows, cols, mirror, n12, n3)  # type: ignore[arg-type]
    )
    want = np.zeros((n, n))
    for p in range(2):
        for a in rows[p][rows[p] < n]:
            for b in cols[p][cols[p] < n]:
                want[a, b] = m[p, n12[a], n3[a], n12[b], n3[b]]
                if mirror[p]:
                    want[b, a] = want[a, b]
    np.testing.assert_array_equal(out, want)
