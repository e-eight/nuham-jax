"""Relative/CM to laboratory transformation of a relative-space interaction.

For a lab channel ``(J, parity)`` the transformation coefficients ``T`` map relative-CM states
``(S, j_rel, N_cm, L_cm, n_rel, l_rel)`` to pairs ``|a b; J>``. The interaction is block-diagonal
in ``(S, j_rel, N_cm, L_cm)``, so ``V_lab = sum_blocks T_b^T V_b T_b``.
"""

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from nuham.angmom.moshinsky_table import MoshinskyTable
from nuham.angmom.racah import ninej, sixj
from nuham.basis.lab import Orbits, ordered_pairs
from nuham.kernels.lab import congruence_sum
from nuham.relative_nn import RelativeChannel


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
            mat = mosh.matrix(int(total), lam)[np.ix_(cr, cc)]
            t[np.ix_(ri, pi)] += b_fac[:, None] * mat * a_fac[rows.s[ri]]
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
    t_raw: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    j: int,
    tz: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Columns of the pair basis for ``tz`` and the coefficients restricted and scaled to them."""
    if tz == 0:
        keep = np.ones(len(pair_a), dtype=bool)
    else:
        keep = (pair_a >= pair_b) & ~((pair_a == pair_b) & (j % 2 == 1))
    cols = np.flatnonzero(keep)
    scale = (
        np.where(pair_a[cols] == pair_b[cols], 1.0, math.sqrt(2.0)) if tz else np.ones(len(cols))
    )
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
_MAX_BLOCK_BYTES = 512 * 2**20


def lab_matrix(
    rows: RelCmRows,
    t_raw: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    j: int,
    parity: int,
    tz: int,
    relative: dict[tuple[int, int, int, int], RelativeChannel],
) -> LabMatrix:
    """Contract the relative interaction of one ``Tz`` with the transformation coefficients.

    Blocks are grouped by size, zero-padded into stacks of equal shape and contracted with
    :func:`nuham.kernels.lab.congruence_sum`, so most of the work is a few large matrix products.

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
    cols, t = _select(rows, t_raw, pair_a, pair_b, j, tz)
    dim = len(cols)
    if dim == 0:
        return LabMatrix(pair_a[cols], pair_b[cols], np.zeros((0, 0)))
    padded = -(-dim // _PAD_P) * _PAD_P
    t_ext = np.zeros((len(rows.s) + 1, padded))
    t_ext[:-1, :dim] = t
    zero_row = len(rows.s)
    by_size: dict[int, list[tuple[list[int], np.ndarray]]] = {}
    for members, v_b in _blocks(rows, tz, relative):
        size = max(4, 1 << (len(members) - 1).bit_length())
        by_size.setdefault(size, []).append((members, v_b))
    v_lab = np.zeros((padded, padded))
    for size, group in by_size.items():
        chunk = max(1, _MAX_BLOCK_BYTES // (size * padded * 8))
        for start in range(0, len(group), chunk):
            part = group[start : start + chunk]
            n_pad = -(-len(part) // 64) * 64
            idx = np.full((n_pad, size), zero_row, dtype=np.int64)
            v_blocks = np.zeros((n_pad, size, size))
            for b, (members, v_b) in enumerate(part):
                idx[b, : len(members)] = members
                v_blocks[b, : len(members), : len(members)] = v_b
            v_lab += np.asarray(congruence_sum(jnp.asarray(t_ext[idx]), jnp.asarray(v_blocks)))
    return LabMatrix(pair_a[cols], pair_b[cols], v_lab[:dim, :dim])


def lab_channel_matrices(
    orb: Orbits,
    e2max: int,
    j: int,
    parity: int,
    relative: dict[tuple[int, int, int, int], RelativeChannel],
    mosh: MoshinskyTable,
) -> dict[int, LabMatrix]:
    """Lab matrices for ``tz = -1, 0, +1`` of one ``(J, parity)`` channel."""
    jrel_max = max(k[0] for k in relative)
    pair_a, pair_b = ordered_pairs(orb, e2max, j, parity)
    rows = enumerate_rows(j, parity, e2max, jrel_max)
    t_raw = t_matrix(rows, pair_a, pair_b, orb, j, mosh)
    return {
        tz: lab_matrix(rows, t_raw, pair_a, pair_b, j, parity, tz, relative) for tz in (-1, 0, 1)
    }
