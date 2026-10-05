"""The me3j file: tuple order and counts, and the whole pipeline against NuHamil (Phase 5.4).

The end-to-end tests run our chain (NN projection, isospin average, Jacobi Hamiltonian, local 3NF,
SRG flow, T-coefficients, lab contraction, file order) at emax/e2max/e3max = 4/8/4, hw 16,
lambda 2.0, ramp flat8 and compare every value with NuHamil's double-precision file.
"""

from pathlib import Path

import numpy as np
import pytest
from ref3n import file_tuples, read_me3j

import nuham  # noqa: F401  (enables float64)
from nuham.basis.lab import make_orbits
from nuham.nn.input_file import read_nn_mom
from nuham.nn.me2j import read_me2j
from nuham.nnn.generate import LocalForce, generate_me3j
from nuham.nnn.me3j import bra_tuples, count_values, triples

REF = Path(__file__).resolve().parents[1] / "refs" / "work" / "nnn_e4"


def test_counts_match_nuhamil() -> None:
    """24 890 values at 4/8/4 (NuHamil's file) and 456 320 at 6/12/6 (spec 5d 6.5)."""
    assert count_values(make_orbits(4), 8, 4) == 24890
    assert count_values(make_orbits(6), 12, 6) == 456320


def test_tuple_order_equals_the_nested_loops() -> None:
    """The vectorised enumeration reproduces NuHamil's loop order tuple by tuple."""
    orb = make_orbits(4)
    tri = triples(orb, 8, 4)
    got = []
    for bra in range(len(tri[0])):
        ket, j12, j45, jt, t12, t45, t2 = bra_tuples(orb, tri, bra)
        got += zip(
            np.full(len(ket), tri[0][bra]), np.full(len(ket), tri[1][bra]),
            np.full(len(ket), tri[2][bra]), tri[0][ket], tri[1][ket], tri[2][ket],
            j12, j45, jt, t12, t45, t2, strict=True,
        )  # fmt: skip
    want = list(file_tuples(orb, 8, 4))
    assert [tuple(int(x) for x in t) for t in got] == want


@pytest.mark.parametrize(
    ("reference", "genuine", "jmax3"),
    [("nnn_ind_e4.me3j.gz", False, 5), ("nnn_3NF400_e4.me3j.gz", True, None)],
)
def test_whole_file_matches_nuhamil(
    nn_bin: Path, tmp_path: Path, reference: str, genuine: bool, jmax3: int | None
) -> None:
    """NuHamil's NN-only file was made with jmax3 = 5 (J_rel <= 5/2), the 3NF400 one with 11."""
    want_path = REF / reference
    if not want_path.exists():
        pytest.skip(f"{want_path} not found")
    out = tmp_path / "ours.me3j.gz"
    n = generate_me3j(
        read_nn_mom(nn_bin),
        out,
        16.0,
        2.0,
        4,
        8,
        4,
        "flat8",
        LocalForce() if genuine else None,
        jmax3=jmax3,
    )
    want, got = read_me3j(want_path), read_me2j(out)
    assert n == len(want) == len(got) == 24890
    assert np.abs(got - want).max() < 1e-5
    assert (got[want == 0.0] == 0.0).all()


def test_parallel_values_equal_serial() -> None:
    """Forked workers give the same stream, in order, as one process."""
    from nuham.nnn.me3j import LabMatrix3, me3j_values
    from nuham.nnn.packed import pack_lower
    from nuham.nnn.tcoef import lab_states

    orb = make_orbits(6)
    rng = np.random.default_rng(3)
    labs = {}
    for t2 in (1, 3):
        for j2 in range(1, 16, 2):
            for p in (1, -1):
                lab = lab_states(orb, j2, p, t2, 12, 6)
                if len(lab):
                    m = rng.normal(size=(len(lab), len(lab)))
                    labs[j2, p, t2] = LabMatrix3.make(lab, pack_lower(m), len(orb))
    serial = np.concatenate(list(me3j_values(orb, 12, 6, labs)))
    parallel = np.concatenate(list(me3j_values(orb, 12, 6, labs, workers=3, pairs_per_task=500)))
    assert len(serial) == 456320
    np.testing.assert_array_equal(parallel, serial)


def test_fields_equal_formatted_values(tmp_path: Path) -> None:
    """Workers that format give a file with the same text as the plain writer."""
    from nuham.nnn.me3j import LabMatrix3, me3j_values, write_me3j, write_me3j_fields
    from nuham.nnn.packed import pack_lower
    from nuham.nnn.tcoef import lab_states

    orb = make_orbits(5)
    rng = np.random.default_rng(5)
    labs = {}
    for t2 in (1, 3):
        for j2 in range(1, 14, 2):
            for p in (1, -1):
                lab = lab_states(orb, j2, p, t2, 10, 5)
                if len(lab):
                    m = 3.0 * rng.normal(size=(len(lab), len(lab)))
                    labs[j2, p, t2] = LabMatrix3.make(lab, pack_lower(m), len(orb))
    write_me3j(tmp_path / "a.me3j", me3j_values(orb, 10, 5, labs))
    fields = me3j_values(orb, 10, 5, labs, workers=2, pairs_per_task=300, as_fields=True)
    write_me3j_fields(tmp_path / "b.me3j", fields)
    assert (tmp_path / "a.me3j").read_bytes() == (tmp_path / "b.me3j").read_bytes()
