import numpy as np

from nuham.angmom.racah import ninej, sixj
from nuham.angmom.racah_array import ninej_array, sixj_array


def test_sixj_array_matches_scalar_including_invalid() -> None:
    rng = np.random.default_rng(0)
    args = rng.integers(0, 9, size=(6, 40000))
    got = sixj_array(*args)
    want = np.array([sixj(*map(int, col)) for col in args.T])
    np.testing.assert_allclose(got, want, atol=1e-13)
    assert np.count_nonzero(want) > 500


def test_ninej_array_matches_scalar() -> None:
    rng = np.random.default_rng(1)
    args = rng.integers(0, 5, size=(9, 40000))
    got = ninej_array(*args)
    want = np.array([ninej(*map(int, col)) for col in args.T])
    np.testing.assert_allclose(got, want, atol=1e-13)
    assert np.count_nonzero(want) > 100


def test_broadcasting_and_empty() -> None:
    out = sixj_array(np.arange(6)[:, None], 2, 2, 2, 2, np.arange(6)[None, :])
    assert out.shape == (6, 6)
    assert sixj_array(np.empty(0, dtype=int), 1, 1, 1, 1, 1).shape == (0,)


def test_lab_type_nine_j_with_spin_half_rows() -> None:
    # The LS-to-jj coefficients used by the transformation: {la lb lam; 1/2 1/2 S; ja jb J}.
    la, lb, lam, s, ja, jb, jj = 3, 2, 3, 1, 5, 3, 3
    want = ninej(2 * la, 2 * lb, 2 * lam, 1, 1, 2 * s, ja, jb, 2 * jj)
    got = ninej_array(2 * la, 2 * lb, 2 * lam, 1, 1, 2 * s, ja, jb, 2 * jj)
    assert abs(float(got) - want) < 1e-14
