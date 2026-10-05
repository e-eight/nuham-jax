"""Three-nucleon Jacobi HO basis: channels, pair-antisymmetric states and the antisymmetriser.

The isospin-formalism basis is labelled by total ``(J, P, T)`` (``J`` and ``T`` doubled in the
API, as in the rest of the package) and by the oscillator quanta ``N = 2 n12 + l12 + 2 n3 + l3``.
For every ``(J, P, T, N)`` the matrix of the three-body antisymmetriser ``A`` is built in the
space of states that are already antisymmetric in the pair (1,2) and reduced to an orthonormal
basis of its range; those vectors are the rows of ``cfp``, an isometry from the antisymmetric
(AS) subspace into the non-antisymmetrised (NAS) states. Nothing here depends on ħω or on the
interaction.

The AS basis is an arbitrary orthonormal basis of the range of ``A``. Only ``C^T C = A`` and
quantities built from it are physical; :func:`antisymmetric_basis` makes the choice a
deterministic function of ``A``, not equal to NuHamil's rows.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg

from nuham.angmom.moshinsky_table import MoshinskyTable
from nuham.angmom.racah_array import ninej_array, sixj_array

NAS_FIELDS = ("n12", "l12", "s12", "j12", "t12", "n3", "l3", "j3")
"""Columns of a NAS state array; ``j3`` is doubled, the others are true values."""


@dataclass(frozen=True)
class JacobiBlock:
    """One ``(J, P, T, N)`` block of the Jacobi basis.

    Attributes
    ----------
    n_quanta
        Oscillator quanta ``N`` of the block.
    nas
        Integer array ``(n_nas, 8)`` of NAS quantum numbers, columns :data:`NAS_FIELDS`.
    cfp
        ``(n_as, n_nas)`` matrix; rows are orthonormal AS states in the NAS basis.
    """

    n_quanta: int
    nas: np.ndarray
    cfp: np.ndarray


@dataclass(frozen=True)
class JacobiChannel:
    """All blocks of one ``(J, P, T)`` channel up to ``nmax``.

    Blocks whose AS dimension is zero are dropped (NuHamil drops their NAS states too), so
    ``n_nas`` counts only NAS states of blocks that hold at least one AS state.
    """

    j2: int
    parity: int
    t2: int
    nmax: int
    blocks: tuple[JacobiBlock, ...]

    @property
    def n_nas(self) -> int:
        """Number of NAS states kept."""
        return sum(len(b.nas) for b in self.blocks)

    @property
    def n_as(self) -> int:
        """Number of AS states."""
        return sum(len(b.cfp) for b in self.blocks)


def ramp_nmax(j2: int, ramp: str) -> int:
    """Largest ``N`` for channels of total angular momentum ``J = j2 / 2``.

    ``"flatN"`` gives ``N`` for every ``J``; ``"rampN0-j1-N1-j2-N2..."`` starts at ``N0`` and
    switches to ``N_k`` when ``2J > j_k`` (thresholds are doubled ``J``, later pairs win).
    """
    if ramp.startswith("flat"):
        return int(ramp[4:])
    if ramp.startswith("ramp"):
        parts = [int(x) for x in ramp[4:].split("-")]
        nmax = parts[0]
        for threshold, value in zip(parts[1::2], parts[2::2], strict=True):
            if j2 > threshold:
                nmax = value
        return nmax
    raise ValueError(f"unknown ramp {ramp!r}")


def jacobi_channels(jmax3: int, ramp: str) -> list[tuple[int, int, int, int]]:
    """Channels ``(2J, P, 2T, nmax)`` in NuHamil's order: T outermost, then J, then P = +1, -1.

    ``jmax3`` is the largest doubled ``J``; ``T`` is 1/2 and 3/2.
    """
    return [
        (j2, parity, t2, ramp_nmax(j2, ramp))
        for t2 in (1, 3)
        for j2 in range(1, jmax3 + 1, 2)
        for parity in (1, -1)
    ]


def _triangle_fails(a: int, b: int, c: int) -> bool:
    """Return True if doubled ``a, b, c`` violate the triangle rule or the parity of the sum."""
    return c < abs(a - b) or c > a + b or (a + b + c) % 2 != 0


def nas_states(j2: int, parity: int, t2: int, n_quanta: int) -> np.ndarray:
    """Pair-antisymmetric NAS states of one ``(J, P, T, N)`` block in NuHamil's order.

    Loop order (outer to inner): ``n12``, ``n3``, ``l12``, ``s12``, ``j12``, ``t12``, ``j3``.
    The order fixes the index of a state and is part of the file-compatible interface.

    Returns
    -------
    np.ndarray
        Integer array ``(n, 8)`` with columns :data:`NAS_FIELDS`; empty ``(0, 8)`` if none.
    """
    if (-1) ** n_quanta != parity:
        return np.zeros((0, 8), dtype=np.int64)
    rows: list[tuple[int, ...]] = []
    for n12 in range(n_quanta // 2 + 1):
        for n3 in range((n_quanta - 2 * n12) // 2 + 1):
            rest = n_quanta - 2 * n12 - 2 * n3
            for l12 in range(rest + 1):
                l3 = rest - l12
                for s12 in (0, 1):
                    for j12 in range(abs(l12 - s12), l12 + s12 + 1):
                        for t12 in (0, 1):
                            if _triangle_fails(2 * t12, 1, t2) or (-1) ** (l12 + s12 + t12) != -1:
                                continue
                            for j3 in range(abs(2 * l3 - 1), 2 * l3 + 2, 2):
                                if _triangle_fails(2 * j12, j3, j2):
                                    continue
                                rows.append((n12, l12, s12, j12, t12, n3, l3, j3))
    return np.array(rows, dtype=np.int64).reshape(-1, 8)


def antisymmetrizer(
    nas: np.ndarray, j2: int, t2: int, n_quanta: int, table: MoshinskyTable
) -> np.ndarray:
    """Matrix of ``A = (1 - P13 - P23) / 3`` on the NAS states of one block.

    ``A = (1 - 2 X) / 3`` where ``X`` is the matrix of the transposition (2 3). It is built as a
    sum over the total spin ``S`` and the coupled orbital momentum ``lam`` of products of
    per-state 9j factors with the ``d = 1/3`` Moshinsky bracket ``<(n12 l12)(n3 l3); lam |
    (n45 l45)(n6 l6); lam>``, so each term is an outer product times a gathered bracket block.

    Parameters
    ----------
    nas
        NAS states from :func:`nas_states`.
    j2, t2
        Doubled total ``J`` and ``T`` of the channel.
    n_quanta
        ``N`` of the block (all states must have it).
    table
        :class:`MoshinskyTable` with ``d = 1/3``.

    Returns
    -------
    np.ndarray
        Dense symmetric ``(n, n)`` matrix.
    """
    n12, l12, s12, jp, t12, n3, l3, j3 = nas.T
    row = table.row_index(n_quanta)
    col = table.col_index(n_quanta)
    idx_row = np.array([row[(a, b, c, d)] for a, b, c, d in zip(n12, l12, n3, l3, strict=True)])
    idx_col = np.array([col[(a, b, c, d)] for a, b, c, d in zip(n12, l12, n3, l3, strict=True)])

    spin = np.zeros((2, 2, 2))  # [S index, s_a, s_b] with the sqrt((2 s_a + 1)(2 s_b + 1)) factor
    for si, s2 in enumerate((1, 3)):
        for sa in (0, 1):
            for sb in (0, 1):
                spin[si, sa, sb] = np.sqrt((2 * sa + 1) * (2 * sb + 1)) * sixj_array(
                    1, 1, 2 * sa, 1, s2, 2 * sb
                )
    iso = np.zeros((2, 2))
    for ta in (0, 1):
        for tb in (0, 1):
            iso[ta, tb] = np.sqrt((2 * ta + 1) * (2 * tb + 1)) * sixj_array(
                1, 1, 2 * ta, 1, t2, 2 * tb
            )
    phase = (-1.0) ** (s12 + t12)

    x = np.zeros((len(nas), len(nas)))
    for si, s2 in enumerate((1, 3)):
        spin_ab = spin[si][np.ix_(s12, s12)]
        for lam in range(max(abs(j2 - s2) // 2, 0), min(n_quanta, (j2 + s2) // 2) + 1):
            g = np.sqrt((2 * jp + 1) * (j3 + 1) * (2 * lam + 1) * (s2 + 1)) * ninej_array(
                1, j3, 2 * l3, s2, j2, 2 * lam, 2 * s12, 2 * jp, 2 * l12
            )
            if not g.any():
                continue
            bracket = table.gather(n_quanta, lam, idx_row, idx_col)
            x += np.outer(g, g) * spin_ab * bracket
    x *= np.outer(phase, phase) * iso[np.ix_(t12, t12)]
    return (np.eye(len(nas)) - 2.0 * x) / 3.0


def antisymmetric_basis(a: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the range of the projector ``a`` as rows ``(n_as, n_nas)``.

    The trace of ``a`` is the integer AS dimension and ``a`` must be symmetric and idempotent
    (asserted to 1e-9; NuHamil only checks its eigenvalues to 1e-2). The basis is the first
    ``n_as`` columns of the column-pivoted QR of ``a`` with ``R``'s diagonal positive, so on one
    platform it is a function of ``a`` (an eigensolver would return an arbitrary basis of the
    degenerate unit eigenspace). It is not portable: the columns of the real projectors tie in
    norm (relative gaps of 0 to 1e-16 at the pivot choice), so another LAPACK, or a device, may
    pivot differently and give another gauge. Any orthonormal basis of the same space is equally
    valid, and cached operators are keyed on the exact ``cfp`` (:func:`basis_digest`).

    Raises
    ------
    ValueError
        If ``a`` is not a projector within tolerance.
    """
    n_as = int(round(float(np.trace(a))))
    defect = max(np.abs(a - a.T).max(), np.abs(a @ a - a).max()) if len(a) else 0.0
    if defect > 1e-9 or abs(np.trace(a) - n_as) > 1e-9:
        raise ValueError(
            f"antisymmetriser is not a projector: defect {defect:.2e}, trace {np.trace(a):.9f}"
        )
    if n_as == 0:
        return np.zeros((0, len(a)))
    q, r, _ = scipy.linalg.qr(a, pivoting=True)
    rows = q[:, :n_as].T
    return rows * np.sign(np.diag(r)[:n_as])[:, None]


def build_channel(
    j2: int, parity: int, t2: int, nmax: int, table: MoshinskyTable | None = None
) -> JacobiChannel:
    """Jacobi basis of one ``(J, P, T)`` channel for ``N <= nmax``.

    Parameters
    ----------
    j2, parity, t2
        Doubled ``J``, parity ``+-1`` and doubled ``T``.
    nmax
        Largest ``N``. ``A`` conserves ``N``, so the result for a smaller ``nmax`` is the set of
        low blocks of the result for a larger one.
    table
        Bracket table with ``d = 1/3``; a fresh in-memory one is made if omitted.
    """
    table = MoshinskyTable(d=1 / 3) if table is None else table
    blocks = []
    for n_quanta in range(nmax + 1):
        nas = nas_states(j2, parity, t2, n_quanta)
        if len(nas) == 0:
            continue
        cfp = antisymmetric_basis(antisymmetrizer(nas, j2, t2, n_quanta, table))
        if len(cfp) == 0:
            continue
        blocks.append(JacobiBlock(n_quanta, nas, cfp))
    return JacobiChannel(j2, parity, t2, nmax, tuple(blocks))


BASIS_VERSION = 1
"""Bump when a change of the antisymmetriser or of the basis choice (pivoting, signs) changes the
gauge of ``cfp``; cached bases of another version are rebuilt."""


def basis_digest(channel: JacobiChannel) -> str:
    """Short digest of the exact states and ``cfp`` of ``channel``.

    The operators in the AS basis (``V3``) are only meaningful together with the ``cfp`` they were
    made with, so cached operators are keyed on this digest.
    """
    h = hashlib.sha1()
    for b in channel.blocks:
        h.update(repr((b.n_quanta, b.nas.shape, b.cfp.shape)).encode())
        h.update(np.ascontiguousarray(b.nas).tobytes())
        h.update(np.ascontiguousarray(b.cfp).tobytes())
    return h.hexdigest()[:16]


def load_or_build_channel(
    j2: int,
    parity: int,
    t2: int,
    nmax: int,
    table: MoshinskyTable | None = None,
    cache_dir: str | Path | None = None,
) -> JacobiChannel:
    """Return :func:`build_channel`'s result, read from ``cache_dir`` when it is there.

    The basis does not depend on ``hw``, the SRG cutoff or the interaction, so one cache serves
    every run (NuHamil's ``cfp`` files play the same role). The file is one ``.npz`` per channel
    and ``nmax``; it is written after a build and carries :data:`BASIS_VERSION` (a file of another
    version is rebuilt). Without ``cache_dir`` this is ``build_channel``.
    """
    if cache_dir is None:
        return build_channel(j2, parity, t2, nmax, table)
    path = Path(cache_dir) / f"jacobi_j{j2}_p{'+' if parity > 0 else '-'}_t{t2}_N{nmax}.npz"
    if path.exists():
        with np.load(path) as data:
            if "version" in data and int(data["version"]) == BASIS_VERSION:
                blocks = tuple(
                    JacobiBlock(int(n), data[f"nas{n}"], data[f"cfp{n}"]) for n in data["quanta"]
                )
                return JacobiChannel(j2, parity, t2, nmax, blocks)
    channel = build_channel(j2, parity, t2, nmax, table)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "version": np.array(BASIS_VERSION),
        "quanta": np.array([b.n_quanta for b in channel.blocks], dtype=np.int64),
    }
    for b in channel.blocks:
        arrays[f"nas{b.n_quanta}"], arrays[f"cfp{b.n_quanta}"] = b.nas, b.cfp
    np.savez(path.with_suffix(".tmp.npz"), **arrays)  # ty: ignore[invalid-argument-type]
    path.with_suffix(".tmp.npz").rename(path)
    return channel
