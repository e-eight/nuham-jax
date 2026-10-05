"""Guard of the cached ``V3``: a small evolved operator must keep its numbers.

Cached operators are keyed on ``CACHE_VERSION`` by hand, so a change of the physics code that moves
``V3`` (a bug fix, a new term, another generator) would otherwise let a resumed run read stale
files without notice. If this test fails: decide whether the change is intended, bump
``nuham.nnn.generate.CACHE_VERSION``, and regenerate the reference with
``uv run python tests/test_v3_reference.py``. The sampled elements depend on the AS gauge, so
a change of the basis (``BASIS_VERSION``) also needs a new reference.
"""

import json
from pathlib import Path

import numpy as np

import nuham  # noqa: F401  (enables float64)
from nuham.fetch import find_nn_file
from nuham.nn.input_file import read_nn_mom
from nuham.nnn.generate import LocalForce, jacobi_operators

REFERENCE = Path(__file__).parent / "data" / "v3_reference.json"
KEYS = [(1, 1, 1), (3, -1, 3)]


def small_v3(nn) -> dict[str, dict]:
    """Numbers of the evolved ``V3`` (hw 16, lambda 2, e3max 4, genuine 3NF400) of two channels."""
    ops, _ = jacobi_operators(
        nn, 16.0, 2.0, 4, "flat4", LocalForce(), rtol=1e-10, js=[k[0] for k in KEYS]
    )
    out = {}
    for key in KEYS:
        v = ops[key]
        n = len(v)
        picks = [(0, 0), (0, n - 1), (n - 1, n - 1), (n // 2, n // 3), (n // 3, n // 2)]
        out[repr(key)] = {
            "n": n,
            "trace": float(np.trace(v)),
            "frobenius": float(np.linalg.norm(v)),
            "elements": [float(v[i, j]) for i, j in picks],
        }
    return out


def test_small_v3_is_unchanged(nn_bin: Path) -> None:
    got = small_v3(read_nn_mom(nn_bin))
    want = json.loads(REFERENCE.read_text())
    for key, ref in want.items():
        assert got[key]["n"] == ref["n"]
        for name in ("trace", "frobenius"):
            assert abs(got[key][name] - ref[name]) < 1e-8, (
                f"V3 of {key} changed ({name}): bump CACHE_VERSION in nuham/nnn/generate.py "
                "and regenerate tests/data/v3_reference.json"
            )
        np.testing.assert_allclose(got[key]["elements"], ref["elements"], atol=1e-8)


if __name__ == "__main__":
    path = find_nn_file("N3LO_EM500")
    assert path is not None, "EM500 file not found; see nuham.fetch"
    REFERENCE.write_text(json.dumps(small_v3(read_nn_mom(path)), indent=1))
