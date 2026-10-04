"""Device benchmarks for the dense parts of the 3N chain (Phase 5.1/5.1b gates), as JSON.

Three independent measurements, chosen with a sub-command:

``qr``     pivoted QR (host SciPy; device ``jax.lax.linalg.qr(pivoting=True)``) and device ``eigh``
           of a random symmetric projector of rank ``n/3``, the shape of the antisymmetriser blocks
           (largest block at Nmax 40: about 10 400 states). Also checks that the device pivot order
           equals SciPy's, which keeps ``jacobi3.antisymmetric_basis`` deterministic.
``tcoef``  one ``T_NAS`` block of the J=3/2+, T=1/2 lab channel from the energy-blocked dense
           products of :func:`nuham.tcoef3.tcoef_nas_blocked` (host NumPy); ``--n 12`` is the
           N = 12 block of e3max 12, ``--reference`` adds the term-wise build and the difference.
``gemm``   float64 matrix-product rate for the shapes of the lab contraction ``T^T V T``.

    uv run python benchmarks/bench_3n_dense.py qr --sizes 2000 5000 10400 --json qr.json
    uv run python benchmarks/bench_3n_dense.py tcoef --n 10 12 16 --json tcoef.json
    uv run python benchmarks/bench_3n_dense.py gemm --json gemm.json
"""

import argparse
import json
import resource
import time
from collections.abc import Callable
from pathlib import Path

import jax
import numpy as np
import scipy.linalg
from run_nn import environment, peak_device_bytes

import nuham  # noqa: F401  sets JAX allocation defaults before the first device use
from nuham.basis.lab import make_orbits
from nuham.tcoef3 import (
    jacobi_cm_rows,
    lab_states,
    make_brackets,
    tcoef_nas,
    tcoef_nas_blocked,
)

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GEMM_SHAPES = [(2000, 8000), (4000, 16000), (8000, 32259), (10400, 32259)]


def timed(fn: Callable[[], object], repeat: int = 1) -> tuple[object, float]:
    """Run ``fn`` ``repeat`` times, blocking on device results; return last value, best time."""
    best, out = float("inf"), None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        best = min(best, time.perf_counter() - t0)
    return out, best


def attempt(fn: Callable[[], dict]) -> dict:
    """Run a measurement and record the exception text instead of aborting the whole benchmark."""
    try:
        return fn()
    except Exception as err:  # noqa: BLE001  a missing backend feature is a result, not a crash
        return {"error": f"{type(err).__name__}: {str(err)[:300]}"}


def random_projector(n: int, rank: int, seed: int = 0) -> jax.Array:
    """Symmetric idempotent ``(n, n)`` matrix with a random range of dimension ``rank``."""
    g = jax.random.normal(jax.random.key(seed), (n, rank), dtype=jnp.float64)
    q, _ = jnp.linalg.qr(g)
    return q @ q.T


def qr_record(n: int) -> dict:
    """Time the ways of getting an orthonormal basis of the range of a projector of size ``n``."""
    rank = n // 3
    a = random_projector(n, rank).block_until_ready()
    a_host = np.asarray(a)
    rec: dict = {"n": n, "rank": rank}

    def host_qr() -> dict:
        (q, _, p), t = timed(lambda: scipy.linalg.qr(a_host, pivoting=True))
        rec["_host_p"] = p
        return {
            "seconds": t,
            "range_error": float(np.abs(q[:, :rank] @ q[:, :rank].T - a_host).max()),
        }

    def dev_qr(use_magma: bool | None) -> dict:
        def run() -> tuple:
            return jax.lax.linalg.qr(a, pivoting=True, full_matrices=False, use_magma=use_magma)

        _, first = timed(run)  # includes compilation
        (q, _, p), t = timed(run)
        same = bool(np.array_equal(np.asarray(p), rec["_host_p"]))
        err = float(jnp.abs(q[:, :rank] @ q[:, :rank].T - a).max())
        return {
            "seconds": t,
            "first_call_seconds": first,
            "same_pivots_as_scipy": same,
            "range_error": err,
        }

    def dev_eigh() -> dict:
        (w, v), t = timed(lambda: jnp.linalg.eigh(a))
        (w, v), t = timed(lambda: jnp.linalg.eigh(a))
        u = v[:, w > 0.5]
        return {
            "seconds": t,
            "unit_eigenvalues": int(u.shape[1]),
            "range_error": float(jnp.abs(u @ u.T - a).max()),
        }

    rec["host_scipy_pivoted_qr"] = attempt(host_qr)
    rec["device_pivoted_qr_default"] = attempt(lambda: dev_qr(None))
    rec["device_pivoted_qr_magma"] = attempt(lambda: dev_qr(True))
    rec["device_eigh"] = attempt(dev_eigh)
    rec.pop("_host_p", None)
    rec["peak_device_bytes"] = peak_device_bytes()
    rec["peak_host_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    return rec


def tcoef_record(n: int, reference: bool = False) -> dict:
    """Time the energy-blocked ``T_NAS`` build of one block (quanta ``n``) of the J=3/2+ channel.

    With ``reference`` the term-wise :func:`nuham.tcoef3.tcoef_nas` is timed and compared too.
    """
    orb = make_orbits(n)
    br = make_brackets()
    cols = lab_states(orb, 3, 1, 1, 2 * n, n).select(n)
    rows = jacobi_cm_rows(3, 1, 1, n)
    rec: dict = {"n": n, "rows": len(rows), "cols": len(cols)}
    stats: dict[str, int] = {}
    t0 = time.perf_counter()
    blocked = tcoef_nas_blocked(rows, cols, 3, orb, br, stats)
    rec["blocked"] = {
        "seconds": time.perf_counter() - t0,
        "nonzeros": int((blocked != 0).sum()),
        "valid_fraction": stats["valid"] / max(stats["elements"], 1),
        **stats,
    }
    if reference:
        t0 = time.perf_counter()
        termwise = tcoef_nas(rows, cols, 3, orb, br)
        rec["termwise"] = {
            "seconds": time.perf_counter() - t0,
            "max_difference": float(np.abs(blocked - termwise).max()),
        }
    rec["peak_host_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    return rec


def gemm_record(m: int, k: int) -> dict:
    """float64 products ``(m, m) @ (m, k)`` and ``(k, m) @ (m, k)`` at a lab-contraction shape."""
    v = jnp.asarray(np.random.default_rng(0).standard_normal((m, m)))
    t = jnp.asarray(np.random.default_rng(1).standard_normal((m, k)))
    _, _ = timed(lambda: v @ t)
    _, t1 = timed(lambda: v @ t, 3)
    vt = (v @ t).block_until_ready()
    _, t2 = timed(lambda: t.T @ vt, 3)
    return {
        "m": m, "k": k,
        "vt_seconds": t1, "vt_tflops": 2 * m * m * k / t1 / 1e12,
        "ttvt_seconds": t2, "ttvt_tflops": 2 * k * m * k / t2 / 1e12,
    }  # fmt: skip


def main() -> None:
    """Run the chosen measurement and print and optionally save the record."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="what", required=True)
    q = sub.add_parser("qr")
    q.add_argument("--sizes", type=int, nargs="+", default=[2000, 5000, 10400])
    t = sub.add_parser("tcoef")
    t.add_argument("--n", type=int, nargs="+", default=[8, 10, 12])
    t.add_argument(
        "--reference", action="store_true", help="also time and compare the term-wise tcoef_nas"
    )
    sub.add_parser("gemm")
    for sp in sub.choices.values():
        sp.add_argument("--json", default=None, help="also write the record to this file")
    args = p.parse_args()
    if args.what == "qr":
        calls = [(qr_record, (n,)) for n in args.sizes]
    elif args.what == "tcoef":
        calls = [(tcoef_record, (n, args.reference)) for n in args.n]
    else:
        calls = [(gemm_record, mk) for mk in GEMM_SHAPES]
    rows = []
    for fn, fn_args in calls:
        rows.append(fn(*fn_args))
        print(json.dumps(rows[-1]), flush=True)
    record = {"benchmark": args.what, "environment": environment(), "rows": rows}
    if args.json:
        Path(args.json).write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
