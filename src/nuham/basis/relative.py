"""Quantum-number bookkeeping for the two-nucleon relative HO basis (pn formalism)."""

import numpy as np


def allowed_l(j: int, s: int, parity: int, tz: int) -> tuple[int, ...]:
    """Orbital angular momenta ``l`` that couple with spin ``s`` to ``j`` in a channel.

    For identical nucleons (``tz = +-1``) only ``(-1)^(l+s) = +1`` survives antisymmetry.

    Parameters
    ----------
    j, s
        Total and spin angular momentum.
    parity
        ``+1`` or ``-1``; fixes the parity of ``l``.
    tz
        ``-1`` for pp, ``0`` for pn, ``+1`` for nn.

    Returns
    -------
    tuple[int, ...]
        Allowed ``l`` in ascending order (one or two values).
    """
    return tuple(
        l
        for l in range(abs(j - s), j + s + 1)
        if (-1) ** l == parity and not (abs(tz) == 1 and (-1) ** (l + s) == -1)
    )


def relative_states(ls: tuple[int, ...], n2max: int) -> tuple[np.ndarray, np.ndarray]:
    """Relative states ``(n, l)`` of a channel with ``2n + l <= n2max``, ordered l-major.

    Parameters
    ----------
    ls
        Orbital angular momenta of the channel.
    n2max
        Truncation on ``2n + l``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Integer arrays ``(n, l)``, each of length equal to the channel dimension.
    """
    n = [i for l in ls for i in range((n2max - l) // 2 + 1)]
    l_all = [l for l in ls for _ in range((n2max - l) // 2 + 1)]
    return np.asarray(n, dtype=np.int64), np.asarray(l_all, dtype=np.int64)
