"""The per-pair ``assemble`` that the batched one in ``nuham.nn.me2j`` replaced, kept as a check."""

import math
from collections.abc import Iterator

import numpy as np

from nuham.basis.lab import Orbits
from nuham.nn.me2j import LabChannel, _phase, tuple_chunks


def assemble_per_pair(
    orb: Orbits, e2max: int, channels: dict[tuple[int, int], LabChannel]
) -> Iterator[np.ndarray]:
    """Yield the four values per tuple, one chunk per ``(a, b)`` pair, shape ``(n, 4)``."""
    for a, b, c, d, jj in tuple_chunks(orb, e2max):
        vals = np.zeros((len(c), 4))
        par = (-1) ** (orb.l[a] + orb.l[b])
        for j in np.unique(jj):
            sel = np.flatnonzero(jj == j)
            ch = channels[(int(j), par)]
            cs, ds = c[sel], d[sel]
            nf = (math.sqrt(2.0) if a == b else 1.0) * np.where(cs == ds, math.sqrt(2.0), 1.0)
            ph_ab = _phase(orb, a, b, int(j))
            ph_cd = _phase(orb, cs, ds, int(j))
            x1 = ch.element(0, a, b, cs, ds)
            x2 = ch.element(0, b, a, ds, cs) * ph_ab * ph_cd
            x3 = ch.element(0, a, b, ds, cs) * ph_cd
            x4 = ch.element(0, b, a, cs, ds) * ph_ab
            vals[sel, 0] = 0.5 * (x1 + x2 - x3 - x4)
            vals[sel, 1] = ch.element(1, a, b, cs, ds) * nf
            vals[sel, 2] = 0.5 * (x1 + x2 + x3 + x4)
            vals[sel, 3] = ch.element(-1, a, b, cs, ds) * nf
        yield vals
