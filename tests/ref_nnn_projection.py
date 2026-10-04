"""Slow reference projection of the local-3NF angular coefficients onto NAS HO states (test helper).

Builds the non-antisymmetrised Jacobi matrix of one term for one ``(J, P, T)`` by plain Gauss
quadrature on the mesh, channel pair by channel pair, in the order of ``nnn.jacobi.nas_states``.
"""

import numpy as np

import nuham  # noqa: F401  (enables float64)
from nuham.constants import HBARC, M_REDUCED_PN
from nuham.kernels.ho import radial_functions
from nuham.nnn.jacobi import nas_states
from nuham.nnn.local_force.angular import Channel, coefficients
from nuham.nnn.local_force.radial import LocalMeshes, RadialTables, radial_function


def nas_term_matrix(
    term: str,
    j2: int,
    parity: int,
    t2: int,
    n3max: int,
    meshes: LocalMeshes,
    tables: RadialTables,
    hw: float,
    n_blocks: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(states, matrix)`` for the NAS blocks ``n_blocks`` (default: all N to ``n3max``)."""
    nu = 2.0 * M_REDUCED_PN * hw / HBARC**2
    if n_blocks is None:
        n_blocks = [n for n in range(n3max + 1) if (-1) ** n == parity]
    states = np.concatenate([nas_states(j2, parity, t2, n) for n in n_blocks])
    x_max = tables.zx.shape[0] - 1
    xi, w = meshes.xi, meshes.xi_w
    nmax = n3max // 2
    # u_{n l}(xi) = nu^(1/4) g_{n l}(sqrt(nu) xi): normalised with int u^2 dxi = 1
    u = {
        l: np.asarray(radial_functions(np.sqrt(nu) * xi, l, nmax)) * nu**0.25
        for l in range(n3max + 1)
    }
    chan_of = [Channel(s[1], s[2], s[3], s[4], s[6], s[7]) for s in states]
    chans = sorted(set(chan_of))
    matrix = np.zeros((len(states), len(states)))
    for ib, cb in enumerate(chans):
        sb = [i for i, c in enumerate(chan_of) if c == cb]
        for ck in chans[: ib + 1]:
            sk = [i for i, c in enumerate(chan_of) if c == ck]
            coef = coefficients(term, cb, ck, j2, t2, x_max)
            if not coef:
                continue
            grid = sum(c * radial_function(term, key, tables, meshes) for key, c in coef.items())
            grid = grid * w[:, None] * w[None, :]
            for i in sb:
                ub = np.outer(u[cb.l12][states[i, 0]], u[cb.l3][states[i, 5]])
                for k in sk:
                    uk = np.outer(u[ck.l12][states[k, 0]], u[ck.l3][states[k, 5]])
                    matrix[i, k] = matrix[k, i] = float(np.sum(ub * grid * uk))
    return states, matrix


TERM_ORDER = ("c1", "c3", "c4", "cD", "cE")


def read_hopw(path: str) -> tuple[tuple[int, ...], np.ndarray]:
    """Read a NuHamil ``N2LO_3NF_HOPW`` file: header ``(2J, P, 2T, Nmax, n)`` and five triangles."""
    raw = open(path, "rb").read()
    j2, parity, t2, nmax = (int(v) for v in np.frombuffer(raw[:16], dtype=np.int32))
    n = int(np.frombuffer(raw[16:24], dtype=np.int64)[0])
    data = np.frombuffer(raw[24:], dtype=np.float64).reshape(5, n)
    return (j2, parity, t2, nmax, n), data


def lower_to_full(v: np.ndarray) -> np.ndarray:
    """Unpack a row-wise lower triangle into a symmetric matrix."""
    dim = int(round((np.sqrt(8 * len(v) + 1) - 1) / 2))
    m = np.zeros((dim, dim))
    m[np.tril_indices(dim)] = v
    return m + m.T - np.diag(np.diag(m))


def nas_blocks_in_file(j2: int, parity: int, t2: int, nmax: int, dim: int) -> list[int]:
    """The N blocks NuHamil kept (it drops a block without antisymmetric states), by dimension."""
    from itertools import combinations

    blocks = [n for n in range(nmax + 1) if (-1) ** n == parity]
    sizes = {n: len(nas_states(j2, parity, t2, n)) for n in blocks}
    for r in range(len(blocks), 0, -1):
        for sub in combinations(blocks, r):
            if sum(sizes[n] for n in sub) == dim:
                return list(sub)
    raise ValueError("no block subset matches the file dimension")


def read_jacobi_matrix(path: str) -> np.ndarray:
    """Read a NuHamil ``NNN-full`` or ``NNN-ind`` file: ``int32 n`` then ``n x n`` doubles."""
    raw = open(path, "rb").read()
    n = int(np.frombuffer(raw[:4], dtype=np.int32)[0])
    return np.frombuffer(raw[4:], dtype=np.float64).reshape(n, n)
