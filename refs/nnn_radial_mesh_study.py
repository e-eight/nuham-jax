"""Mesh-convergence study of the local-3NF radial tables (Phase 5.2.2).

Errors are max |table - reference| / max |reference| on NuHamil's 80-point xi mesh, with
x_max = 42 (N3max 40). The reference uses a very fine momentum and cosine mesh.
"""

import time

import numpy as np

import nuham  # noqa: F401
from nuham.nnn.local_force.radial import LocalMeshes, converged_meshes, gauss_legendre, nuhamil_meshes, radial_tables

X_MAX = 42


def with_xi(m: LocalMeshes, base: LocalMeshes) -> LocalMeshes:
    return LocalMeshes(base.xi, base.xi_w, m.p, m.p_w, m.cos, m.cos_w)


def errors(t, ref) -> dict[str, float]:
    out = {"z0": t.z0 - ref.z0, "fk": t.fk - ref.fk, "zx": t.zx - ref.zx}
    sc = {"z0": ref.z0, "fk": ref.fk, "zx": ref.zx}
    for k, name in enumerate(("fkx0", "fkx1", "fkx2")):
        out[name] = t.fkx[k] - ref.fkx[k]
        sc[name] = ref.fkx[k]
    return {k: float(np.abs(v).max() / np.abs(sc[k]).max()) for k, v in out.items()}


base = nuhamil_meshes()
for lam in (400.0, 500.0, 650.0, 1000.0):
    ref_m = with_xi(converged_meshes(lam, 2, n_p=1600, n_cos=240, p_max=None), base)
    ref = radial_tables(ref_m, lam, 2, X_MAX)
    print(f"=== Lambda = {lam:.0f} MeV, reference n_p=1600 n_cos=240 pmax={ref_m.p.max():.2f}")
    cases = {"NuHamil (100p to 8, 80cos)": base}
    for n_p, n_cos in ((200, 120), (300, 120), (400, 160), (600, 160)):
        cases[f"converged n_p={n_p} n_cos={n_cos}"] = with_xi(converged_meshes(lam, 2, n_p=n_p, n_cos=n_cos), base)
    for name, m in cases.items():
        t0 = time.time()
        e = errors(radial_tables(m, lam, 2, X_MAX), ref)
        print(f"{name:34s} " + " ".join(f"{k}={v:.1e}" for k, v in e.items()) + f"  ({time.time() - t0:.1f} s)")
