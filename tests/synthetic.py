"""A synthetic, charge-independent NN interaction in NuHamil's momentum-space format.

It lets the whole pipeline be tested without any NuHamil data. The potential is a sum of two
separable Gaussian terms per channel with coefficients that depend only on ``(J, S, T, l, l')`` and
not on ``Tz``, so pp, nn and the ``T = 1`` part of pn are identical.
"""

import zlib
from pathlib import Path

import numpy as np

from nuham.basis.relative import allowed_l
from nuham.interactions.nn_file import MomentumChannel, MomentumNN

_TERMS = ((1.5, -120.0), (3.0, 80.0))  # (width in fm^-1, strength in MeV fm^3)


def _coefficient(j: int, s: int, isospin: int, term: int, l_lo: int, l_hi: int) -> float:
    """Deterministic pseudo-random coefficient; the diagonal is O(1), the off-diagonal small."""
    seed = zlib.crc32(repr((j, s, isospin, term, l_lo, l_hi)).encode())
    value = np.random.default_rng(seed).uniform(-1.0, 1.0)
    return 1.0 + 0.5 * value if l_lo == l_hi else 0.3 * value


def _phi(k: np.ndarray, l: int, width: float) -> np.ndarray:
    """Separable radial form ``(k/width)^l exp(-(k/width)^2)``."""
    return (k / width) ** l * np.exp(-((k / width) ** 2))


def synthetic_nn(n_mesh: int = 40, kmax: float = 8.0, jmax: int = 3) -> MomentumNN:
    """Build the synthetic interaction on a Gauss-Legendre mesh.

    Parameters
    ----------
    n_mesh
        Number of mesh points.
    kmax
        Largest momentum in fm^-1.
    jmax
        Largest total angular momentum.

    Returns
    -------
    MomentumNN
        Channels in the same layout NuHamil's files use (``V`` symmetric, blocks ordered by ``l``).
    """
    x, w = np.polynomial.legendre.leggauss(n_mesh)
    k, w = 0.5 * kmax * (x + 1.0), 0.5 * kmax * w
    channels = []
    for tz in (-1, 0, 1):
        for j in range(jmax + 1):
            for parity in (1, -1):
                for s in (0, 1):
                    ls = allowed_l(j, s, parity, tz)
                    if not ls:
                        continue
                    isospin = 1 if (ls[0] + s) % 2 == 0 else 0
                    v = np.zeros((len(ls) * n_mesh, len(ls) * n_mesh))
                    for t, (width, strength) in enumerate(_TERMS):
                        for a, la in enumerate(ls):
                            for b, lb in enumerate(ls):
                                c = _coefficient(j, s, isospin, t, min(la, lb), max(la, lb))
                                block = (
                                    strength * c * np.outer(_phi(k, la, width), _phi(k, lb, width))
                                )
                                v[a * n_mesh : (a + 1) * n_mesh, b * n_mesh : (b + 1) * n_mesh] += (
                                    block
                                )
                    channels.append(MomentumChannel(j, parity, s, tz, ls, v))
    return MomentumNN(k, w, jmax, tuple(channels))


def write_nn_mom(path: Path, nn: MomentumNN) -> None:
    """Write ``nn`` in NuHamil's raw little-endian ``.bin`` layout."""
    with Path(path).open("wb") as fh:
        fh.write(np.array([len(nn.k), nn.jmax, len(nn.channels)], dtype="<i4").tobytes())
        fh.write(nn.k.astype("<f8").tobytes())
        fh.write(nn.w.astype("<f8").tobytes())
        for ch in nn.channels:
            fh.write(np.array([ch.j, ch.parity, ch.s, ch.tz, len(ch.v)], dtype="<i4").tobytes())
            fh.write(ch.v.astype("<f8").tobytes())
