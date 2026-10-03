"""Relative/CM to laboratory transformation of a relative-space interaction.

For a lab channel ``(J, parity)`` the transformation coefficients ``T`` map relative-CM states
``(S, j_rel, N_cm, L_cm, n_rel, l_rel)`` to pairs ``|a b; J>``. The interaction is block-diagonal
in ``(S, j_rel, N_cm, L_cm)``, so ``V_lab = sum_blocks T_b^T V_b T_b``.
"""

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array

from nuham.angmom.moshinsky_table import MoshinskyTable
from nuham.angmom.racah import ninej, sixj
from nuham.angmom.racah_array import ninej_array, sixj_array
from nuham.basis.lab import Orbits, ordered_pairs
from nuham.kernels.lab import accumulate_blocks
from nuham.kernels.transfer import to_host
from nuham.relative_nn import RelativeChannel
from nuham.timing import StageTimer, stage


@dataclass(frozen=True)
class RelCmRows:
    """Relative-CM states of a lab channel, as parallel integer arrays.

    Attributes
    ----------
    s, jrel, ncm, lcm, nrel, lrel
        Quantum numbers of each state.
    total
        ``2 n_cm + l_cm + 2 n_rel + l_rel``, the quanta that must equal ``e_a + e_b``.
    """

    s: np.ndarray
    jrel: np.ndarray
    ncm: np.ndarray
    lcm: np.ndarray
    nrel: np.ndarray
    lrel: np.ndarray
    total: np.ndarray


def enumerate_rows(j: int, parity: int, e2max: int, jrel_max: int) -> RelCmRows:
    """Enumerate relative-CM states that can couple to total ``J`` with the given lab parity.

    Parameters
    ----------
    j
        Total angular momentum.
    parity
        Lab parity ``(-1)^(l_rel + L_cm)``.
    e2max
        Bound on the total quanta.
    jrel_max
        Largest relative ``j`` with a nonzero interaction.

    Returns
    -------
    RelCmRows
        All states, in no particular order.
    """
    out: list[tuple[int, ...]] = []
    for s in (0, 1):
        for jrel in range(jrel_max + 1):
            for lrel in range(abs(jrel - s), jrel + s + 1):
                for lcm in range(abs(jrel - j), min(jrel + j, e2max) + 1):
                    if (-1) ** (lrel + lcm) != parity:
                        continue
                    for ncm in range((e2max - lcm) // 2 + 1):
                        for nrel in range((e2max - lcm - 2 * ncm - lrel) // 2 + 1):
                            total = 2 * ncm + lcm + 2 * nrel + lrel
                            out.append((s, jrel, ncm, lcm, nrel, lrel, total))
    cols = (np.array(c, dtype=np.int64) for c in (zip(*out, strict=True) if out else [[]] * 7))
    return RelCmRows(*cols)


def t_matrix_reference(
    rows: RelCmRows,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    orb: Orbits,
    j: int,
    mosh: MoshinskyTable,
) -> np.ndarray:
    """Slow scalar reference for :func:`t_matrix`, looping over every ``lam <= N`` and pair.

    ``T[r, p] = (-1)^(Lcm + lrel + S + J) sum_lam A(p, S, lam) B(r, lam) <bracket>`` with the 9j
    LS-to-jj factor ``A`` and the 6j recoupling factor ``B``; it is nonzero only where the
    relative-CM quanta equal ``e_a + e_b``.

    Parameters
    ----------
    rows
        Relative-CM states.
    pair_a, pair_b
        Orbit indices of the ordered lab pairs.
    orb
        Orbits.
    j
        Total angular momentum.
    mosh
        Bracket table.

    Returns
    -------
    np.ndarray
        Matrix of shape ``(len(rows.s), len(pair_a))``.
    """
    t = np.zeros((len(rows.s), len(pair_a)))
    pair_total = orb.e[pair_a] + orb.e[pair_b]
    for total in np.unique(rows.total):
        ri = np.flatnonzero(rows.total == total)
        pi = np.flatnonzero(pair_total == total)
        if len(pi) == 0:
            continue
        row_map = mosh.row_index(int(total))
        col_map = mosh.col_index(int(total))
        cr = np.array([row_map[(rows.ncm[r], rows.lcm[r], rows.nrel[r], rows.lrel[r])] for r in ri])
        pa, pb = pair_a[pi], pair_b[pi]
        cc = np.array(
            [col_map[(orb.n[x], orb.l[x], orb.n[y], orb.l[y])] for x, y in zip(pa, pb, strict=True)]
        )
        for lam in range(int(total) + 1):
            b_fac = np.array(
                [
                    (-1.0) ** (rows.lcm[r] + rows.lrel[r] + rows.s[r] + j)
                    * math.sqrt((2 * lam + 1) * (2 * rows.jrel[r] + 1))
                    * sixj(
                        2 * rows.lcm[r],
                        2 * rows.lrel[r],
                        2 * lam,
                        2 * rows.s[r],
                        2 * j,
                        2 * rows.jrel[r],
                    )
                    for r in ri
                ]
            )
            if not b_fac.any():
                continue
            a_fac = np.zeros((2, len(pi)))
            for s in (0, 1):
                for k, (x, y) in enumerate(zip(pa, pb, strict=True)):
                    a_fac[s, k] = math.sqrt(
                        (2 * lam + 1) * (2 * s + 1) * (orb.j2[x] + 1) * (orb.j2[y] + 1)
                    ) * ninej(
                        2 * orb.l[x],
                        2 * orb.l[y],
                        2 * lam,
                        1,
                        1,
                        2 * s,
                        orb.j2[x],
                        orb.j2[y],
                        2 * j,
                    )
            mat = mosh.gather(int(total), lam, cr, cc)
            t[np.ix_(ri, pi)] += b_fac[:, None] * mat * a_fac[rows.s[ri]]
    return t


def t_matrix(
    rows: RelCmRows,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    orb: Orbits,
    j: int,
    mosh: MoshinskyTable,
) -> np.ndarray:
    """Return the transformation coefficients without the pp/nn antisymmetry factors.

    ``T[r, p] = (-1)^(Lcm + lrel + S + J) sum_lam A(p, S, lam) B(r, lam) <bracket>`` with the 9j
    LS-to-jj factor ``A`` and the 6j recoupling factor ``B``; it is nonzero only where the
    relative-CM quanta equal ``e_a + e_b``. Both factors vanish unless ``|J - S| <= lam <= J + S``,
    so only ``lam = J - 1, J, J + 1`` are summed, and ``A`` and ``B`` are evaluated for all pairs
    and rows at once with the array Racah symbols.

    Parameters
    ----------
    rows
        Relative-CM states.
    pair_a, pair_b
        Orbit indices of the ordered lab pairs.
    orb
        Orbits.
    j
        Total angular momentum.
    mosh
        Bracket table.

    Returns
    -------
    np.ndarray
        Matrix of shape ``(len(rows.s), len(pair_a))``.
    """
    t = np.zeros((len(rows.s), len(pair_a)))
    pair_total = orb.e[pair_a] + orb.e[pair_b]
    la, lb = orb.l[pair_a], orb.l[pair_b]
    ja, jb = orb.j2[pair_a], orb.j2[pair_b]
    sign = np.where((rows.lcm + rows.lrel + rows.s + j) % 2 == 0, 1.0, -1.0)
    b_all: dict[int, np.ndarray] = {}
    a_all: dict[int, np.ndarray] = {}
    for lam in range(max(0, j - 1), j + 2):
        b_all[lam] = (
            sign
            * np.sqrt((2 * lam + 1) * (2 * rows.jrel + 1))
            * sixj_array(2 * rows.lcm, 2 * rows.lrel, 2 * lam, 2 * rows.s, 2 * j, 2 * rows.jrel)
        )
        a_all[lam] = np.stack(
            [
                np.sqrt((2 * lam + 1) * (2 * s + 1) * (ja + 1) * (jb + 1))
                * ninej_array(2 * la, 2 * lb, 2 * lam, 1, 1, 2 * s, ja, jb, 2 * j)
                for s in (0, 1)
            ]
        )
    for total in np.unique(rows.total):
        ri = np.flatnonzero(rows.total == total)
        pi = np.flatnonzero(pair_total == total)
        if len(pi) == 0:
            continue
        row_map = mosh.row_index(int(total))
        col_map = mosh.col_index(int(total))
        cr = np.array([row_map[(rows.ncm[r], rows.lcm[r], rows.nrel[r], rows.lrel[r])] for r in ri])
        cc = np.array(
            [
                col_map[(orb.n[x], orb.l[x], orb.n[y], orb.l[y])]
                for x, y in zip(pair_a[pi], pair_b[pi], strict=True)
            ]
        )
        for lam in b_all:
            if lam > total:
                continue
            b_fac = b_all[lam][ri]
            if not b_fac.any():
                continue
            mat = mosh.gather(int(total), lam, cr, cc)
            t[np.ix_(ri, pi)] += b_fac[:, None] * mat * a_all[lam][:, pi][rows.s[ri]]
    return t


@dataclass(frozen=True)
class LabMatrix:
    """Lab-frame interaction of one ``(J, parity, Tz)`` channel.

    Attributes
    ----------
    pair_a, pair_b
        Orbit indices defining the pair basis (rows and columns of ``v``).
    v
        Symmetric matrix of ``<ab; J|V|cd; J>`` in MeV. For ``|Tz| = 1`` the states are normalised
        and antisymmetrised; for ``Tz = 0`` they are ordered proton-neutron products.
    """

    pair_a: np.ndarray
    pair_b: np.ndarray
    v: np.ndarray


def lab_matrix_reference(
    rows: RelCmRows,
    t_raw: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    j: int,
    parity: int,
    tz: int,
    relative: dict[tuple[int, int, int, int], RelativeChannel],
) -> LabMatrix:
    """Slow reference for :func:`lab_matrix`: one rank update of the full matrix per block.

    Parameters
    ----------
    rows, t_raw
        Output of :func:`enumerate_rows` and :func:`t_matrix` for this ``(J, parity)``.
    pair_a, pair_b
        The ordered pairs used to build ``t_raw``.
    j
        Total angular momentum.
    parity
        Lab parity.
    tz
        ``-1`` pp, ``0`` pn, ``+1`` nn.
    relative
        Relative channels keyed by ``(j, parity, s, tz)``.

    Returns
    -------
    LabMatrix
        The lab matrix over the pair basis appropriate to ``tz``.
    """
    if tz == 0:
        keep = np.ones(len(pair_a), dtype=bool)
    else:
        keep = (pair_a >= pair_b) & ~((pair_a == pair_b) & (j % 2 == 1))
    cols = np.flatnonzero(keep)
    scale = (
        np.where(pair_a[cols] == pair_b[cols], 1.0, math.sqrt(2.0)) if tz else np.ones(len(cols))
    )
    t = t_raw[:, cols] * scale[None, :]
    v_lab = np.zeros((len(cols), len(cols)))
    blocks: dict[tuple[int, int, int, int], list[int]] = {}
    for r in range(len(rows.s)):
        if tz != 0 and (rows.lrel[r] + rows.s[r]) % 2:
            continue
        blocks.setdefault((rows.s[r], rows.jrel[r], rows.ncm[r], rows.lcm[r]), []).append(r)
    index_cache: dict[tuple[int, int, int, int], dict[tuple[int, int], int]] = {}
    for (s, jrel, _ncm, _lcm), members in blocks.items():
        key = (jrel, (-1) ** rows.lrel[members[0]], s, tz)
        ch = relative.get(key)
        if ch is None:
            continue
        if key not in index_cache:
            index_cache[key] = {
                (int(n), int(l)): i for i, (n, l) in enumerate(zip(ch.n, ch.l, strict=True))
            }
        pos = [index_cache[key][(int(rows.nrel[r]), int(rows.lrel[r]))] for r in members]
        tb = t[members]
        v_lab += tb.T @ (ch.v[np.ix_(pos, pos)] @ tb)
    return LabMatrix(pair_a[cols], pair_b[cols], v_lab)


def _select(
    rows: RelCmRows,
    t_raw: np.ndarray | Array,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    j: int,
    tz: int,
) -> tuple[np.ndarray, np.ndarray | Array]:
    """Columns of the pair basis for ``tz`` and the coefficients restricted and scaled to them.

    Works on host or device arrays. For ``tz = 0`` every column is kept unscaled, so ``t_raw`` is
    returned as it is.
    """
    if tz == 0:
        return np.arange(len(pair_a)), t_raw
    keep = (pair_a >= pair_b) & ~((pair_a == pair_b) & (j % 2 == 1))
    cols = np.flatnonzero(keep)
    scale = np.where(pair_a[cols] == pair_b[cols], 1.0, math.sqrt(2.0))
    return cols, t_raw[:, cols] * scale[None, :]


def _blocks(
    rows: RelCmRows, tz: int, relative: dict[tuple[int, int, int, int], RelativeChannel]
) -> list[tuple[list[int], np.ndarray]]:
    """Row indices and relative interaction ``V_b`` of every block ``(S, j_rel, N_cm, L_cm)``."""
    members_of: dict[tuple[int, int, int, int], list[int]] = {}
    for r in range(len(rows.s)):
        if tz != 0 and (rows.lrel[r] + rows.s[r]) % 2:
            continue
        members_of.setdefault((rows.s[r], rows.jrel[r], rows.ncm[r], rows.lcm[r]), []).append(r)
    index_cache: dict[tuple[int, int, int, int], dict[tuple[int, int], int]] = {}
    out = []
    for (s, jrel, _ncm, _lcm), members in members_of.items():
        key = (jrel, (-1) ** rows.lrel[members[0]], s, tz)
        ch = relative.get(key)
        if ch is None:
            continue
        if key not in index_cache:
            index_cache[key] = {
                (int(n), int(l)): i for i, (n, l) in enumerate(zip(ch.n, ch.l, strict=True))
            }
        pos = [index_cache[key][(int(rows.nrel[r]), int(rows.lrel[r]))] for r in members]
        out.append((members, ch.v[np.ix_(pos, pos)]))
    return out


_PAD_P = 128
_PAD_ROWS = 256
_MAX_BLOCK_BYTES = 512 * 2**20


def lab_matrix(
    rows: RelCmRows,
    t_raw: np.ndarray | Array,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    j: int,
    parity: int,
    tz: int,
    relative: dict[tuple[int, int, int, int], RelativeChannel],
) -> LabMatrix:
    """Contract the relative interaction of one ``Tz`` with the transformation coefficients.

    The coefficients live on the device (JAX default device) for the whole call. Blocks are
    grouped by size and zero-padded into stacks of equal shape; each stack is gathered on the
    device and added into a running sum by :func:`nuham.kernels.lab.accumulate_blocks`, so the
    result crosses to the host once.

    Parameters
    ----------
    rows, t_raw
        Output of :func:`enumerate_rows` and :func:`t_matrix` for this ``(J, parity)``; ``t_raw``
        may already be a device array, which avoids re-uploading it for each ``tz``.
    pair_a, pair_b
        The ordered pairs used to build ``t_raw``.
    j
        Total angular momentum.
    parity
        Lab parity.
    tz
        ``-1`` pp, ``0`` pn, ``+1`` nn.
    relative
        Relative channels keyed by ``(j, parity, s, tz)``.

    Returns
    -------
    LabMatrix
        The lab matrix over the pair basis appropriate to ``tz``.
    """
    cols, t = _select(rows, t_raw, pair_a, pair_b, j, tz)
    dim = len(cols)
    if dim == 0:
        return LabMatrix(pair_a[cols], pair_b[cols], np.zeros((0, 0)))
    padded = -(-dim // _PAD_P) * _PAD_P
    n_rows = len(rows.s)
    t_rows = (
        jnp.zeros((-(-(n_rows + 1) // _PAD_ROWS) * _PAD_ROWS, padded))
        .at[:n_rows, :dim]
        .set(jnp.asarray(t))
    )
    by_size: dict[int, list[tuple[list[int], np.ndarray]]] = {}
    for members, v_b in _blocks(rows, tz, relative):
        size = max(4, 1 << (len(members) - 1).bit_length())
        by_size.setdefault(size, []).append((members, v_b))
    total = jnp.zeros((padded, padded))
    for size, group in by_size.items():
        chunk = max(1, _MAX_BLOCK_BYTES // (size * padded * 8))
        for start in range(0, len(group), chunk):
            part = group[start : start + chunk]
            n_pad = -(-len(part) // 64) * 64
            idx = np.full((n_pad, size), n_rows, dtype=np.int64)
            v_blocks = np.zeros((n_pad, size, size))
            for b, (members, v_b) in enumerate(part):
                idx[b, : len(members)] = members
                v_blocks[b, : len(members), : len(members)] = v_b
            total = accumulate_blocks(total, t_rows, jnp.asarray(idx), jnp.asarray(v_blocks))
    return LabMatrix(pair_a[cols], pair_b[cols], to_host(total[:dim, :dim]))


def lab_channel_matrices(
    orb: Orbits,
    e2max: int,
    j: int,
    parity: int,
    relative: dict[tuple[int, int, int, int], RelativeChannel],
    mosh: MoshinskyTable,
    timer: StageTimer | None = None,
) -> dict[int, LabMatrix]:
    """Lab matrices for ``tz = -1, 0, +1`` of one ``(J, parity)`` channel.

    An optional ``timer`` records ``lab_coeffs`` (coefficient build, including any bracket-table
    construction) and ``lab_contract`` (upload, contraction and copy back).
    """
    jrel_max = max(k[0] for k in relative)
    pair_a, pair_b = ordered_pairs(orb, e2max, j, parity)
    rows = enumerate_rows(j, parity, e2max, jrel_max)
    with stage(timer, "lab_coeffs"):
        t_dev = jnp.asarray(t_matrix(rows, pair_a, pair_b, orb, j, mosh))
    with stage(timer, "lab_contract"):
        return {
            tz: lab_matrix(rows, t_dev, pair_a, pair_b, j, parity, tz, relative)
            for tz in (-1, 0, 1)
        }
