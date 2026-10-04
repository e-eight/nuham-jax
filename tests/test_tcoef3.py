"""Jacobi-to-lab coefficients of three nucleons (step 5.1b): independent and NuHamil references."""

import numpy as np
import pytest
from ref3n import (
    REF_DIR,
    antisym_matrix,
    file_tuples,
    read_cfp_bin,
    read_me3j,
    read_op_bin,
)

from nuham.angmom.moshinsky_table import MoshinskyTable
from nuham.basis.lab import make_orbits
from nuham.jacobi3 import JacobiBlock, JacobiChannel
from nuham.tcoef3 import (
    jacobi_channels_for,
    jacobi_cm_rows,
    lab_scalar,
    lab_states,
    make_brackets,
    tcoef_as,
    tcoef_nas,
)

EMAX = 4


@pytest.fixture(scope="module")
def orbits():
    """Orbits up to emax 4."""
    return make_orbits(EMAX)


@pytest.fixture(scope="module")
def brackets():
    """Bracket tables with d = 1 and d = 2."""
    return make_brackets()


def test_lab_channel_sizes(orbits) -> None:
    """Lab space at emax/e2max/e3max = 4/8/4 has 22 channels and 715 states (spec 5d, 2.4)."""
    sizes = []
    for t2 in (1, 3):
        for j2 in range(1, 2 * EMAX + 4, 2):
            for parity in (1, -1):
                n = len(lab_states(orbits, j2, parity, t2, 2 * EMAX, EMAX))
                if n:
                    sizes.append(n)
    assert len(sizes) == 22
    assert sum(sizes) == 715
    assert max(sizes) == 99


@pytest.mark.parametrize(
    ("j2", "parity", "t2", "n"),
    [(1, 1, 1, 0), (1, -1, 1, 1), (3, -1, 1, 1), (3, 1, 3, 2), (1, 1, 1, 2), (3, -1, 1, 3)],
)
def test_antisymmetriser_matches_mscheme(orbits, brackets, j2, parity, t2, n) -> None:
    """``6 T_AS^T T_AS`` equals ``sum_P sgn(P) <abc|P|def>`` computed in the m-scheme.

    This fixes the factor 6, the phases of the coefficients, the d = 2 bracket and the use of
    ``cfp`` without any NuHamil input.
    """
    lab = lab_states(orbits, j2, parity, t2, 2 * EMAX, EMAX).select(n)
    rows = jacobi_cm_rows(j2, parity, t2, n)
    chans = jacobi_channels_for(j2, t2, EMAX)
    t_as, _ = tcoef_as(tcoef_nas(rows, lab, j2, orbits, brackets), rows, chans, t2)
    cols = [(lab.i1[c], lab.i2[c], lab.i3[c], lab.j12[c], lab.t12[c]) for c in range(len(lab))]
    np.testing.assert_allclose(
        6.0 * t_as.T @ t_as, antisym_matrix(orbits, cols, j2, t2), atol=1e-11
    )


def test_pauli_forbidden_block_has_no_antisymmetric_rows(orbits, brackets) -> None:
    """J = 3/2, T = 1/2 at N = 0: the only NAS state is Pauli forbidden."""
    rows = jacobi_cm_rows(3, 1, 1, 0)
    chans = jacobi_channels_for(3, 1, EMAX)
    lab = lab_states(orbits, 3, 1, 1, 2 * EMAX, EMAX).select(0)
    t_as, labels = tcoef_as(tcoef_nas(rows, lab, 3, orbits, brackets), rows, chans, 1)
    assert t_as.shape[0] == 0
    assert labels == []


@pytest.mark.parametrize(("j2", "parity", "t2", "n"), [(1, -1, 1, 1), (1, 1, 1, 2), (3, 1, 3, 2)])
def test_nas_matches_transcribed_sums(orbits, brackets, j2, parity, t2, n) -> None:
    """The vectorised coefficients equal an element-wise transcription of the nested sums."""
    ref = pytest.importorskip("tcoef3_reference")
    lab = lab_states(orbits, j2, parity, t2, 2 * EMAX, EMAX).select(n)
    rows = jacobi_cm_rows(j2, parity, t2, n)
    got = tcoef_nas(rows, lab, j2, orbits, brackets)
    for r in range(0, len(rows), 3):
        for c in range(len(lab)):
            col = (lab.i1[c], lab.i2[c], lab.i3[c], lab.j12[c], lab.t12[c])
            want = ref.tnas_element(tuple(rows.qn[r]), col, j2, orbits)
            assert got[r, c] == pytest.approx(want, abs=1e-12)


def test_d2_bracket_matches_kamuntavicius_sum() -> None:
    """The matrix-exponential table with d = 2 equals the explicit bracket sum."""
    ref = pytest.importorskip("moshinsky_reference")
    table = MoshinskyTable(d=2.0)
    for total in (2, 4, 5):
        rows, cols = table.row_index(total), table.col_index(total)
        for lam in range(total + 1):
            m = table.matrix(total, lam)
            for rq, i in rows.items():
                for cq, j in cols.items():
                    assert m[i, j] == pytest.approx(ref.bracket(*rq, *cq, lam, 2.0), abs=1e-12)


def test_reference_me3j_all_values(orbits, brackets) -> None:
    """All 24 890 values of NuHamil's 4/8/4 file from NuHamil's own Jacobi operator and cfp.

    The Jacobi operator and ``cfp`` are NuHamil's (same gauge), so this isolates the lab
    transformation: coefficients, factor 6 and the file order.
    """
    me3j = REF_DIR / "nnn_3NF400_e4.me3j.gz"
    if not me3j.exists():
        pytest.skip(f"{me3j} not found")
    chans, vjac = {}, {}
    for t2 in (1, 3):
        for j2 in range(1, 12, 2):
            for p in (1, -1):
                tag = f"j{j2}p{'+' if p == 1 else '-'}t{t2}"
                _, _, _, blocks = read_cfp_bin(REF_DIR / "cfp" / f"cfp_A3_{tag}_Nmax8.bin")
                op = next((REF_DIR / "ops").glob(f"NNN-full_{tag}_Nmax8*.bin"))
                full = read_op_bin(op)
                off, spans = 0, {}
                for n, _, cfp in blocks:
                    spans[n] = np.arange(off, off + len(cfp))
                    off += len(cfp)
                kept = [b for b in blocks if b[0] <= EMAX]
                idx = np.concatenate([spans[b[0]] for b in kept]) if kept else np.zeros(0, int)
                chans[(t2, j2, p)] = JacobiChannel(
                    j2, p, t2, EMAX, tuple(JacobiBlock(n, nas, cfp) for n, nas, cfp in kept)
                )
                vjac[(t2, j2, p)] = full[np.ix_(idx, idx)]
    labs = {}
    for t2 in (1, 3):
        for j2 in range(1, 2 * EMAX + 4, 2):
            for p in (1, -1):
                keys = [(jr, pp) for jr in range(1, 12, 2) for pp in (1, -1)]
                lab, mat = lab_scalar(
                    j2,
                    p,
                    t2,
                    orbits,
                    2 * EMAX,
                    EMAX,
                    {k: vjac[(t2, *k)] for k in keys},
                    {k: chans[(t2, *k)] for k in keys},
                    brackets,
                )
                index = {
                    (lab.i1[c], lab.i2[c], lab.i3[c], lab.j12[c], lab.t12[c]): c
                    for c in range(len(lab))
                }
                labs[(j2, p, t2)] = (mat, index)
    want = read_me3j(me3j)
    tuples = list(file_tuples(orbits, 2 * EMAX, EMAX))
    assert len(tuples) == len(want) == 24890
    got = np.zeros(len(want))
    for k, (a, b, c, d, e, f, j12, j45, jt, t12, t45, tt) in enumerate(tuples):
        parity = (-1) ** (orbits.l[a] + orbits.l[b] + orbits.l[c])
        entry = labs.get((jt, parity, tt))
        if entry is None:
            continue
        mat, index = entry
        i, j = index.get((a, b, c, j12, t12)), index.get((d, e, f, j45, t45))
        if i is not None and j is not None:
            got[k] = mat[i, j]
    np.testing.assert_allclose(got, want, atol=1e-8)
