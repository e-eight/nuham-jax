"""Reader for NuHamil's momentum-space NN partial-wave files (``*_kmax8_N100_Jmax8.bin``)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nuham.basis.relative import allowed_l


@dataclass(frozen=True)
class MomentumChannel:
    """One (J, parity, S, Tz) channel of the momentum-space potential.

    Attributes
    ----------
    j, parity, s, tz
        Channel quantum numbers; ``tz`` is -1 for pp, 0 for pn, +1 for nn.
    ls
        Orbital angular momenta; the blocks of ``v`` follow this order.
    v
        Unweighted ``V(k', k)`` in MeV fm^3, shape ``(len(ls) * N, len(ls) * N)``.
    """

    j: int
    parity: int
    s: int
    tz: int
    ls: tuple[int, ...]
    v: np.ndarray


@dataclass(frozen=True)
class MomentumNN:
    """A momentum-space NN interaction on a Gauss-Legendre mesh.

    Attributes
    ----------
    k, w
        Mesh points (fm^-1) and weights, shape ``(N,)``.
    jmax
        Largest total angular momentum in the file.
    channels
        Channels in file order.
    """

    k: np.ndarray
    w: np.ndarray
    jmax: int
    channels: tuple[MomentumChannel, ...]


def read_nn_mom(path: str | Path) -> MomentumNN:
    """Read a NuHamil momentum-space NN file.

    The layout is a raw little-endian stream without record markers: ``int32`` mesh size,
    ``Jmax`` and channel count; ``float64`` mesh points and weights; then per channel the
    ``int32`` values ``J, parity, S, Tz, ndim`` followed by ``ndim**2`` ``float64`` values.

    Parameters
    ----------
    path
        File to read.

    Returns
    -------
    MomentumNN
        The mesh and all channels.

    Raises
    ------
    ValueError
        If a channel dimension disagrees with its allowed ``l`` values, or data trails.
    """
    raw = Path(path).read_bytes()
    n_mesh, jmax, n_chan = np.frombuffer(raw, "<i4", 3)
    off = 12
    k = np.frombuffer(raw, "<f8", n_mesh, off).copy()
    off += 8 * n_mesh
    w = np.frombuffer(raw, "<f8", n_mesh, off).copy()
    off += 8 * n_mesh
    channels = []
    for _ in range(n_chan):
        j, parity, s, tz, ndim = (int(v) for v in np.frombuffer(raw, "<i4", 5, off))
        off += 20
        v = np.frombuffer(raw, "<f8", ndim * ndim, off).reshape(ndim, ndim).copy()
        off += 8 * ndim * ndim
        ls = allowed_l(j, s, parity, tz)
        if ndim != len(ls) * n_mesh:
            raise ValueError(
                f"channel {(j, parity, s, tz)}: dimension {ndim} != {len(ls)} * {n_mesh}"
            )
        channels.append(MomentumChannel(j, parity, s, tz, ls, v))
    if off != len(raw):
        raise ValueError(f"{len(raw) - off} unread bytes at end of {path}")
    return MomentumNN(k, w, int(jmax), tuple(channels))
