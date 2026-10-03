"""The Darmstadt ``me2j`` format: tuple enumeration, assembly of the four isospin values, I/O."""

import gzip
import math
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np

from nuham.basis.lab import Orbits
from nuham.lab_transform import LabMatrix
from nuham.timing import StageTimer

HEADER = "NN int. calculated by nuham (JAX)"
_LINE = "%16.8f" * 10


def unordered_pairs(orb: Orbits, e2max: int) -> np.ndarray:
    """Pairs ``(a, b)`` with ``b <= a`` and ``e_a + e_b <= e2max``, in file order, shape (n, 2)."""
    out = [(a, b) for a in range(len(orb)) for b in range(a + 1) if orb.e[a] + orb.e[b] <= e2max]
    return np.array(out, dtype=np.int64)


def tuple_chunks(
    orb: Orbits, e2max: int
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]]:
    """Enumerate the ``(a, b, c, d, J)`` tuples of the file, one chunk per ``(a, b)``.

    The tuples satisfy ``(c, d) <= (a, b)`` in file order, equal parity of ``l_a + l_b`` and
    ``l_c + l_d``, and ``J`` runs over all couplings of both pairs.

    Yields
    ------
    tuple[int, int, np.ndarray, np.ndarray, np.ndarray]
        ``a``, ``b`` and arrays ``c``, ``d``, ``J`` of equal length, in file order.
    """
    pairs = unordered_pairs(orb, e2max)
    lsum = orb.l[pairs[:, 0]] + orb.l[pairs[:, 1]]
    jlo = np.abs(orb.j2[pairs[:, 0]] - orb.j2[pairs[:, 1]])
    jhi = orb.j2[pairs[:, 0]] + orb.j2[pairs[:, 1]]
    for u, (a, b) in enumerate(pairs):
        v = np.flatnonzero((lsum[: u + 1] - lsum[u]) % 2 == 0)
        j_min = np.maximum(jlo[v], jlo[u]) // 2
        j_max = np.minimum(jhi[v], jhi[u]) // 2
        count = np.maximum(j_max - j_min + 1, 0)
        which = np.repeat(np.arange(len(v)), count)
        offset = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
        yield int(a), int(b), pairs[v][which, 0], pairs[v][which, 1], j_min[which] + offset


class LabChannel:
    """Lab matrices of one ``(J, parity)`` with lookup tables from an orbit pair to its index."""

    def __init__(self, matrices: dict[int, LabMatrix], n_orbits: int) -> None:
        """Index the pair bases of the pp (``-1``), pn (``0``) and nn (``+1``) matrices."""
        self.matrices = matrices
        self.position: dict[int, np.ndarray] = {}
        for tz, m in matrices.items():
            pos = np.full((n_orbits, n_orbits), -1, dtype=np.int64)
            pos[m.pair_a, m.pair_b] = np.arange(len(m.pair_a))
            self.position[tz] = pos

    def element(
        self, tz: int, a: int | np.ndarray, b: int | np.ndarray, c: np.ndarray, d: np.ndarray
    ) -> np.ndarray:
        """Matrix elements ``<ab|V|cd>`` for arrays ``c, d``; zero outside the pair basis.

        ``a`` and ``b`` are scalars or arrays of the same shape as ``c`` and ``d``.
        """
        pos = self.position[tz]
        i = np.broadcast_to(pos[a, b], c.shape)
        k = pos[c, d]
        out = np.zeros(k.shape)
        ok = (i >= 0) & (k >= 0)
        out[ok] = self.matrices[tz].v[i[ok], k[ok]]
        return out


def _phase(orb: Orbits, x: int | np.ndarray, y: int | np.ndarray, j: int) -> np.ndarray | float:
    """Exchange phase ``-(-1)^(j_x + j_y - J)`` of the pn lab states."""
    return -((-1.0) ** ((orb.j2[x] + orb.j2[y]) // 2 - j))


_BATCH_TUPLES = 1 << 18


def assemble(
    orb: Orbits,
    e2max: int,
    channels: dict[tuple[int, int], LabChannel],
) -> Iterator[np.ndarray]:
    """Yield the four ``me2j`` values per tuple in file order, shape ``(n, 4)`` per chunk.

    The columns are ``T = 0``, nn, pn ``T = 1`` and pp, in the un-normalised Darmstadt convention.
    Consecutive ``(a, b)`` pairs are batched into chunks of about ``_BATCH_TUPLES`` tuples and each
    ``(J, parity)`` is handled with one set of array operations per chunk, which is far fewer small
    numpy calls than one per pair and ``J``.
    """
    pending: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []
    count = 0
    for chunk in tuple_chunks(orb, e2max):
        pending.append(chunk)
        count += len(chunk[2])
        if count >= _BATCH_TUPLES:
            yield _assemble_batch(orb, channels, pending)
            pending, count = [], 0
    if pending:
        yield _assemble_batch(orb, channels, pending)


def _assemble_batch(
    orb: Orbits,
    channels: dict[tuple[int, int], LabChannel],
    pending: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Values of the tuples of several consecutive ``(a, b)`` pairs, shape ``(n, 4)``."""
    sizes = [len(p[2]) for p in pending]
    aa = np.repeat([p[0] for p in pending], sizes)
    bb = np.repeat([p[1] for p in pending], sizes)
    c = np.concatenate([p[2] for p in pending])
    d = np.concatenate([p[3] for p in pending])
    jj = np.concatenate([p[4] for p in pending])
    odd = (orb.l[aa] + orb.l[bb]) % 2
    group = 2 * jj + odd
    vals = np.zeros((len(c), 4))
    for g in np.unique(group):
        sel = np.flatnonzero(group == g)
        j, par = int(g) // 2, -1 if g % 2 else 1
        ch = channels[(j, par)]
        a, b, cs, ds = aa[sel], bb[sel], c[sel], d[sel]
        nf = np.where(a == b, math.sqrt(2.0), 1.0) * np.where(cs == ds, math.sqrt(2.0), 1.0)
        ph_ab = _phase(orb, a, b, j)
        ph_cd = _phase(orb, cs, ds, j)
        x1 = ch.element(0, a, b, cs, ds)
        x2 = ch.element(0, b, a, ds, cs) * ph_ab * ph_cd
        x3 = ch.element(0, a, b, ds, cs) * ph_cd
        x4 = ch.element(0, b, a, cs, ds) * ph_ab
        vals[sel, 0] = 0.5 * (x1 + x2 - x3 - x4)
        vals[sel, 1] = ch.element(1, a, b, cs, ds) * nf
        vals[sel, 2] = 0.5 * (x1 + x2 + x3 + x4)
        vals[sel, 3] = ch.element(-1, a, b, cs, ds) * nf
    return vals


_FLUSH_VALUES = 400_000


def _format_lines_python(values: np.ndarray) -> str:
    """Format a multiple of 10 values as ``10f16.8`` lines with Python's own ``%`` operator."""
    return "".join(_LINE % tuple(r) + "\n" for r in values.reshape(-1, 10).tolist())


@cache
def _tables() -> tuple[np.ndarray, np.ndarray]:
    """Lookup tables for the formatter, as explicitly little-endian words.

    ``frac[i]`` holds the four ASCII digits of ``i`` (0..9999); ``head[i]`` holds the first eight
    characters of a field: the right-aligned integer part (signed when ``i >= 100000``, which
    encodes a negative value ``-(i - 100000)``) and the decimal point.
    """
    frac = np.frombuffer(
        "".join(f"{i:04d}" for i in range(10_000)).encode("ascii"), dtype="<u4"
    ).copy()
    head = np.frombuffer(
        "".join(
            f"{'-' if neg else ''}{i}".rjust(7) + "."
            for neg in (False, True)
            for i in range(100_000)
        ).encode("ascii"),
        dtype="<u8",
    ).copy()
    return frac, head


# Below this magnitude every value fits ``f16.8`` (a rounding carry and a sign included).
_VECTOR_LIMIT = 1.0e5


def _format_lines(values: np.ndarray) -> bytes:
    """Format a multiple of 10 values as ``10f16.8`` lines, each ending in a newline.

    The output is byte-identical to Python's ``%16.8f``, which rounds the exact binary value
    half-to-even. Here the value is scaled by 1e8 and rounded as an integer, which is only
    reliable away from ties: scaling has an error of about one ulp of the product, so any value
    whose scaled fraction lies within a few ulp of 0.5 is formatted by Python instead. Chunks
    with non-finite or very large values take the Python path entirely.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    if not (np.abs(x) < _VECTOR_LIMIT).all():
        return _format_lines_python(x).encode("ascii")
    scaled = np.abs(x) * 1.0e8
    q = np.rint(scaled).astype(np.int64)
    near_tie = np.abs(np.abs(scaled - np.floor(scaled)) - 0.5) <= 4.0 * np.spacing(scaled)
    frac_table, head_table = _tables()
    whole, frac = np.divmod(q, 100_000_000)
    hi, lo = np.divmod(frac, 10_000)
    buf = np.empty((len(x), 16), dtype=np.uint8)
    words = buf.view("<u8")
    words[:, 0] = head_table[whole + 100_000 * np.signbit(x)]
    halves = buf.view("<u4")
    halves[:, 2] = frac_table[hi]
    halves[:, 3] = frac_table[lo]
    for i in np.flatnonzero(near_tie):
        buf[i] = np.frombuffer((f"{x[i]:16.8f}").encode("ascii"), dtype=np.uint8)
    lines = np.empty((len(x) // 10, 161), dtype=np.uint8)
    lines[:, :160] = buf.reshape(-1, 160)
    lines[:, 160] = ord("\n")
    return lines.tobytes()


def write_me2j(
    path: str | Path,
    chunks: Iterator[np.ndarray],
    header: str = HEADER,
    compresslevel: int = 1,
) -> int:
    """Write values as gzip text: one header line, then Fortran ``10f16.8`` (10 values per line).

    Formatting runs in the calling thread; compression and disk writes run in a background
    thread behind a bounded queue, so they overlap with the computation that produces ``chunks``.

    Parameters
    ----------
    path
        Output file; a ``.gz`` suffix gives gzip, anything else plain text.
    chunks
        Arrays of values, flattened in order.
    header
        Free-text first line.
    compresslevel
        gzip level for ``.gz`` files. Level 1 compresses at about 150 MB/s and gives files about
        10% larger than level 4 (70 MB/s), so it is the default.

    Returns
    -------
    int
        The number of values written.
    """
    pending: list[np.ndarray] = []
    n_pending = 0
    total = 0
    queue: Queue[bytes | None] = Queue(maxsize=8)
    failure: list[BaseException] = []

    def drain() -> None:
        try:
            fh = (
                gzip.open(path, "wb", compresslevel=compresslevel)
                if str(path).endswith(".gz")
                else open(path, "wb")
            )
            with fh:
                while (item := queue.get()) is not None:
                    fh.write(item)
        except BaseException as exc:  # noqa: BLE001  re-raised in the caller
            failure.append(exc)
            while queue.get() is not None:
                pass

    worker = Thread(target=drain, daemon=True)
    worker.start()
    queue.put((header + "\n").encode("ascii"))
    try:
        for chunk in chunks:
            pending.append(np.asarray(chunk).ravel())
            n_pending += pending[-1].size
            if n_pending >= _FLUSH_VALUES:
                data = np.concatenate(pending)
                full = len(data) // 10 * 10
                queue.put(_format_lines(data[:full]))
                total += full
                pending, n_pending = [data[full:]], len(data) - full
        data = np.concatenate(pending) if pending else np.empty(0)
        full = len(data) // 10 * 10
        if full:
            queue.put(_format_lines(data[:full]))
        if len(data) > full:
            queue.put(("".join(f"{x:16.8f}" for x in data[full:].tolist()) + "\n").encode("ascii"))
        total += len(data)
    finally:
        queue.put(None)
        worker.join()
    if failure:
        raise failure[0]
    return total


def write_me2j_timed(
    path: str | Path, chunks: Iterator[np.ndarray], timer: StageTimer, compresslevel: int = 1
) -> int:
    """Write like :func:`write_me2j` and record ``assemble`` and ``write`` on ``timer``.

    ``assemble`` is the time spent producing chunks; ``write`` is the rest of the call (formatting
    and the final flush; compression overlaps in its thread and shows up as back-pressure). Safe to
    call repeatedly with the same timer.
    """
    before = timer.seconds.get("assemble", 0.0)
    with timer.stage("write"):
        n = write_me2j(path, timer.wrap("assemble", chunks), compresslevel=compresslevel)
    timer.seconds["write"] -= timer.seconds["assemble"] - before
    return n


def read_me2j(path: str | Path) -> np.ndarray:
    """Read all values of an ``me2j`` file (gzip or plain), skipping the header line."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        fh.readline()
        return np.array(fh.read().split(), dtype=np.float64)
