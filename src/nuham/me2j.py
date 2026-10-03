"""The Darmstadt ``me2j`` format: tuple enumeration, assembly of the four isospin values, I/O."""

import gzip
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from queue import Queue
from threading import Thread
from types import ModuleType
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

from nuham.basis.lab import Orbits
from nuham.kernels.assemble import assemble_values
from nuham.kernels.format import format_fields
from nuham.kernels.format import tables as format_tables
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


@dataclass(frozen=True)
class _FlatChannels:
    """The lab matrices of all groups in the flat layout of :func:`assemble_values`."""

    v: tuple[Array, Array, Array]
    pos: Array
    dim: Array
    off: Array
    group: np.ndarray


def _flatten_channels(channels: dict[tuple[int, int], LabChannel], n_orb: int) -> _FlatChannels:
    """Concatenate the matrices of all ``(J, parity)`` groups into one flat array per ``Tz``.

    Host matrices are uploaded; device matrices (``on_device=True`` in the lab contraction) are
    concatenated where they are, with no host round trip.

    ``group[J, odd]`` is the group index of total ``J`` and parity ``(-1)^odd`` (``-1`` if absent).
    """
    keys = sorted(channels)
    j_max = max(j for j, _ in keys)
    group = np.full((j_max + 1, 2), -1, dtype=np.int32)
    pos = np.full((3, len(keys), n_orb * n_orb), -1, dtype=np.int32)
    dim = np.zeros((3, len(keys)), dtype=np.int64)
    off = np.zeros((3, len(keys)), dtype=np.int64)
    flats = []
    for t, tz in enumerate((0, 1, -1)):
        parts, cursor = [jnp.zeros(1)], 1  # one spare element keeps masked gathers in range
        for g, key in enumerate(keys):
            ch = channels[key]
            group[key[0], 0 if key[1] == 1 else 1] = g
            m = ch.matrices[tz].v
            dim[t, g], off[t, g] = m.shape[0], cursor
            pos[t, g] = ch.position[tz].ravel()
            parts.append(jnp.asarray(m).ravel())
            cursor += m.size
        flats.append(jnp.concatenate(parts))
    return _FlatChannels(
        (flats[0], flats[1], flats[2]), jnp.asarray(pos), jnp.asarray(dim), jnp.asarray(off), group
    )


def _device_windows(
    orb: Orbits,
    e2max: int,
    channels: dict[tuple[int, int], LabChannel],
    batch_tuples: int,
    post: Callable[[Array, int], Any],
) -> Iterator[tuple[Any, int]]:
    """Run the assembly kernel over fixed windows of tuples and yield ``(post(values, n), n)``.

    The tuples of the file are cut into windows of ``batch_tuples`` (the last one is padded to a
    power of two), so the kernel compiles for few shapes. ``post`` is called with the device array
    of values ``(N, 4)`` and the number ``n`` of real tuples as soon as the window is launched, so
    more device work can be queued behind the kernel. The host builds the index arrays of the next
    window while the device works on the current one; results are yielded one window late.
    """
    flat = _flatten_channels(channels, len(orb))
    j2 = jnp.asarray(orb.j2, dtype=jnp.int32)
    l_orb = np.asarray(orb.l)

    def launch(cols: np.ndarray, size: int) -> tuple[Any, int]:
        n = cols.shape[1]
        padded = np.zeros((6, size), dtype=np.int32)
        padded[:, :n] = cols
        a, b, c, d, jj, g = (jnp.asarray(row) for row in padded)
        values = assemble_values(flat.v, flat.pos, flat.dim, flat.off, j2, a, b, c, d, jj, g)
        return post(values, n), n

    def columns(chunks: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
        sizes = [len(ch[2]) for ch in chunks]
        aa = np.repeat([ch[0] for ch in chunks], sizes)
        bb = np.repeat([ch[1] for ch in chunks], sizes)
        jj = np.concatenate([ch[4] for ch in chunks])
        g = flat.group[jj, (l_orb[aa] + l_orb[bb]) % 2]
        return np.stack(
            [aa, bb, np.concatenate([ch[2] for ch in chunks]),
             np.concatenate([ch[3] for ch in chunks]), jj, g]
        ).astype(np.int32)  # fmt: skip

    inflight: tuple[Any, int] | None = None
    carry = np.zeros((6, 0), dtype=np.int32)
    pending: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []
    count = 0

    def flush(final: bool) -> Iterator[tuple[Any, int]]:
        nonlocal inflight, carry, pending, count
        if pending:
            carry = np.concatenate([carry, columns(pending)], axis=1)
            pending, count = [], 0
        while carry.shape[1] >= batch_tuples or (final and carry.shape[1] > 0):
            take = min(batch_tuples, carry.shape[1])
            size = batch_tuples if take == batch_tuples else max(1024, 1 << (take - 1).bit_length())
            current = launch(carry[:, :take], size)
            carry = carry[:, take:]
            if inflight is not None:
                yield inflight
            inflight = current
        if final and inflight is not None:
            yield inflight
            inflight = None

    for chunk in tuple_chunks(orb, e2max):
        pending.append(chunk)
        count += len(chunk[2])
        if count >= batch_tuples:
            yield from flush(final=False)
    yield from flush(final=True)


def assemble_device(
    orb: Orbits,
    e2max: int,
    channels: dict[tuple[int, int], LabChannel],
    batch_tuples: int = 1 << 20,
) -> Iterator[np.ndarray]:
    """Yield the same values as :func:`assemble`, computed by a JAX kernel on the default device.

    Parameters
    ----------
    orb, e2max, channels
        As in :func:`assemble`. The matrices may be host or device arrays.
    batch_tuples
        Tuples per kernel call.

    Yields
    ------
    np.ndarray
        Values of shape ``(n, 4)``; the windows are not aligned with the ``(a, b)`` pairs.
    """
    for values, n in _device_windows(orb, e2max, channels, batch_tuples, lambda v, _n: v):
        yield np.asarray(values)[:n]


_FLAG_CAPACITY = 4096


def format_device(
    orb: Orbits,
    e2max: int,
    channels: dict[tuple[int, int], LabChannel],
    batch_tuples: int = 1 << 20,
) -> Iterator[np.ndarray]:
    """Yield the file's values as 16-character ``f16.8`` text fields, formatted on the device.

    The values never leave the device: each window is assembled and formatted there and only
    the bytes (16 per value) come back. The few values near a rounding tie, which the device
    cannot decide exactly, are formatted here with Python's own ``%``, so the text is identical to
    :func:`write_me2j`'s. Use :func:`write_me2j_fields` to write it.

    Parameters
    ----------
    orb, e2max, channels, batch_tuples
        As in :func:`assemble_device`.

    Yields
    ------
    np.ndarray
        Fields of shape ``(4 n, 16)``, uint8, in file order.

    Raises
    ------
    ValueError
        If a value is not finite or has magnitude of 1e5 or more (it would not fit a field).
    """
    frac, head = format_tables()

    def post(values: Array, n: int) -> tuple[Array, tuple[Array, ...]]:
        return values, format_fields(values, 4 * n, frac, head, _FLAG_CAPACITY)

    for (values, formatted), n in _device_windows(orb, e2max, channels, batch_tuples, post):
        fields_dev, n_flagged, n_bad, index, flagged = formatted
        if int(n_bad):
            raise ValueError("a value is not finite or has magnitude >= 1e5; cannot format it")
        n_flagged = int(n_flagged)
        if n_flagged > _FLAG_CAPACITY:  # rare: redo with room for every flagged value
            size = 1 << (n_flagged - 1).bit_length()
            fields_dev, _, _, index, flagged = format_fields(values, 4 * n, frac, head, size)
        fields = np.asarray(fields_dev)[: 4 * n]
        if n_flagged:
            fields = fields.copy()  # the device buffer's host view is read-only
        for i, x in zip(
            np.asarray(index)[:n_flagged], np.asarray(flagged)[:n_flagged], strict=True
        ):
            fields[i] = np.frombuffer(f"{x:16.8f}".encode("ascii"), dtype=np.uint8)
        yield fields


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


@cache
def _igzip() -> ModuleType | None:
    """Return ``isal.igzip`` if the optional ``isal`` package is installed, else ``None``."""
    try:
        from isal import igzip
    except ImportError:
        return None
    return igzip


def _open_gzip(
    path: str | Path, compresslevel: int, compressor: str, threads: int = 1
) -> gzip.GzipFile:
    """Open ``path`` for binary gzip output with the compressor chosen by ``compressor``.

    ``"gzip"`` is the standard library, ``"isal"`` is ISA-L's igzip (levels 0-3; needs the
    optional ``isal`` package) and ``"auto"`` uses igzip when it is installed and the level allows,
    else ``gzip``. With igzip, ``threads > 1`` compresses blocks in parallel threads
    (``isal.igzip_threaded``) into one valid stream of the same size. All write standard gzip
    streams, but not byte-identical ones.
    """
    if compressor not in ("auto", "gzip", "isal"):
        raise ValueError(f"unknown compressor {compressor!r}; use 'auto', 'gzip' or 'isal'")
    igzip = None if compressor == "gzip" else _igzip()
    if compressor == "isal" and igzip is None:
        raise ImportError("compressor='isal' needs the optional package: uv sync --extra fast")
    if compressor == "isal" and compresslevel > 3:
        raise ValueError("isal supports compression levels 0-3")
    if igzip is not None and compresslevel <= 3:
        if threads > 1:
            from isal import igzip_threaded

            return igzip_threaded.open(path, "wb", compresslevel=compresslevel, threads=threads)
        return igzip.open(path, "wb", compresslevel=compresslevel)
    return gzip.open(path, "wb", compresslevel=compresslevel)


def _write_stream(
    path: str | Path,
    header: str,
    compresslevel: int,
    compressor: str,
    threads: int,
    produce: Callable[[Callable[[bytes | memoryview], None]], int],
) -> int:
    """Run ``produce(put)`` in the caller while a background thread compresses and writes.

    ``put`` queues bytes behind a bounded queue; ``produce`` returns the number of values it
    wrote. A ``.gz`` suffix gives gzip output, anything else plain text. A failure in the writer
    thread is re-raised here.
    """
    queue: Queue[bytes | memoryview | None] = Queue(maxsize=8)
    failure: list[BaseException] = []

    def drain() -> None:
        try:
            fh = (
                _open_gzip(path, compresslevel, compressor, threads)
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
        total = produce(queue.put)
    finally:
        queue.put(None)
        worker.join()
    if failure:
        raise failure[0]
    return total


def write_me2j(
    path: str | Path,
    chunks: Iterator[np.ndarray],
    header: str = HEADER,
    compresslevel: int = 1,
    compressor: str = "auto",
    compress_threads: int = 1,
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
    compressor
        ``"auto"`` (ISA-L's igzip if the optional ``isal`` package is installed, else ``gzip``),
        ``"gzip"`` or ``"isal"``; see :func:`_open_gzip`. On the GH200 igzip made the write stage
        3.5x shorter at emax 12 (``docs/benchmarks.md``).
    compresslevel
        gzip level for ``.gz`` files. Level 1 compresses at about 150 MB/s and gives files about
        10% larger than level 4 (70 MB/s), so it is the default.
    compress_threads
        Threads for igzip's parallel compressor (1 is the single-threaded igzip).

    Returns
    -------
    int
        The number of values written.
    """

    def produce(put: Callable[[bytes | memoryview], None]) -> int:
        pending: list[np.ndarray] = []
        n_pending = 0
        total = 0
        for chunk in chunks:
            pending.append(np.asarray(chunk).ravel())
            n_pending += pending[-1].size
            if n_pending >= _FLUSH_VALUES:
                data = np.concatenate(pending)
                full = len(data) // 10 * 10
                put(_format_lines(data[:full]))
                total += full
                pending, n_pending = [data[full:]], len(data) - full
        data = np.concatenate(pending) if pending else np.empty(0)
        full = len(data) // 10 * 10
        if full:
            put(_format_lines(data[:full]))
        if len(data) > full:
            put(("".join(f"{x:16.8f}" for x in data[full:].tolist()) + "\n").encode("ascii"))
        return total + len(data)

    return _write_stream(path, header, compresslevel, compressor, compress_threads, produce)


def write_me2j_fields(
    path: str | Path,
    fields: Iterator[np.ndarray],
    header: str = HEADER,
    compresslevel: int = 1,
    compressor: str = "auto",
    compress_threads: int = 1,
) -> int:
    """Write values that are already formatted, as the 16-byte fields of :func:`format_device`.

    Only the line layout is done here (ten fields and a newline per line), so the calling thread
    does little more than copy bytes. The arguments are those of :func:`write_me2j`.

    Returns
    -------
    int
        The number of values written.
    """

    def produce(put: Callable[[bytes | memoryview], None]) -> int:
        carry = np.empty((0, 16), dtype=np.uint8)
        total = 0
        for chunk in fields:
            data = np.concatenate([carry, chunk]) if len(carry) else chunk
            full = len(data) // 10 * 10
            if full:
                lines = np.empty((full // 10, 161), dtype=np.uint8)
                lines[:, :160] = data[:full].reshape(-1, 160)
                lines[:, 160] = ord("\n")
                put(memoryview(lines.reshape(-1)))
                total += full
            carry = data[full:].copy()
        if len(carry):
            put(carry.tobytes() + b"\n")
        return total + len(carry)

    return _write_stream(path, header, compresslevel, compressor, compress_threads, produce)


def write_me2j_timed(
    path: str | Path,
    chunks: Iterator[np.ndarray],
    timer: StageTimer,
    compresslevel: int = 1,
    compressor: str = "auto",
    compress_threads: int = 1,
    writer: Callable[..., int] = write_me2j,
) -> int:
    """Write like ``writer`` (:func:`write_me2j` or :func:`write_me2j_fields`) and time it.

    Records ``assemble`` (time spent producing chunks, which for the device path includes waiting
    for the device and copying the result) and ``write`` (the rest of the call: formatting and
    line layout, and the final flush; compression overlaps in its thread and shows up as
    back-pressure). Safe to call repeatedly with the same timer.
    """
    before = timer.seconds.get("assemble", 0.0)
    with timer.stage("write"):
        n = writer(
            path,
            timer.wrap("assemble", chunks),
            compresslevel=compresslevel,
            compressor=compressor,
            compress_threads=compress_threads,
        )
    timer.seconds["write"] -= timer.seconds["assemble"] - before
    return n


def write_channels(
    path: str | Path,
    orb: Orbits,
    e2max: int,
    channels: dict[tuple[int, int], LabChannel],
    assembler: str = "numpy",
    formatter: str = "host",
    compresslevel: int = 1,
    compressor: str = "auto",
    compress_threads: int = 1,
    timer: StageTimer | None = None,
) -> int:
    """Assemble the values of ``channels`` and write them as ``me2j``, with the chosen machinery.

    Parameters
    ----------
    path
        Output file.
    orb, e2max, channels
        As in :func:`assemble`.
    assembler
        ``"numpy"`` (:func:`assemble`) or ``"device"`` (:func:`assemble_device`, a JAX kernel; the
        lab matrices may then stay on the device).
    formatter
        ``"host"`` (:func:`write_me2j`) or ``"device"`` (:func:`format_device` with
        :func:`write_me2j_fields`; needs ``assembler="device"``).
    compresslevel, compressor, compress_threads
        See :func:`write_me2j`.
    timer
        Optional stage timer; records ``assemble`` and ``write``.

    Returns
    -------
    int
        The number of values written.
    """
    if assembler not in ("numpy", "device"):
        raise ValueError(f"unknown assembler {assembler!r}; use 'numpy' or 'device'")
    if formatter not in ("host", "device"):
        raise ValueError(f"unknown formatter {formatter!r}; use 'host' or 'device'")
    if formatter == "device" and assembler != "device":
        raise ValueError("formatter='device' needs assembler='device'")
    chunks: Iterator[np.ndarray]
    if formatter == "device":
        chunks, writer = format_device(orb, e2max, channels), write_me2j_fields
    elif assembler == "device":
        chunks, writer = assemble_device(orb, e2max, channels), write_me2j
    else:
        chunks, writer = assemble(orb, e2max, channels), write_me2j
    if timer is None:
        return writer(
            path,
            chunks,
            compresslevel=compresslevel,
            compressor=compressor,
            compress_threads=compress_threads,
        )
    return write_me2j_timed(
        path, chunks, timer, compresslevel, compressor, compress_threads, writer
    )


def read_me2j(path: str | Path) -> np.ndarray:
    """Read all values of an ``me2j`` file (gzip or plain), skipping the header line."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        fh.readline()
        return np.array(fh.read().split(), dtype=np.float64)
