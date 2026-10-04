"""xi-mesh convergence of the assembled local-3NF matrix at large N (Phase 5.2.5).

For one (J, P, T) at N3max N the AS matrix is built on several xi meshes (p and cosine meshes converged) and
compared with the finest one: max element difference, eigenvalue difference, and the HO overlap defect on
the mesh. Usage: python refs/nnn_xi_mesh_study.py N 2J P 2T workers [hw]
"""

import sys
import time

import numpy as np

import nuham  # noqa: F401
from nuham.constants import HBARC, M_REDUCED_PN
from nuham.kernels.nnn.projection import orbital_table
from nuham.nnn.jacobi import build_channel
from nuham.nnn.local_force.assembly import local_3nf_matrix
from nuham.nnn.local_force.radial import converged_meshes, nuhamil_meshes, radial_tables

nmax, j2, par, t2, workers = (int(a) for a in sys.argv[1:6])
hw_arg = float(sys.argv[6]) if len(sys.argv) > 6 else 16.0
LECS = {"c1": -0.81, "c3": -3.2, "c4": 5.4, "cD": -0.2, "cE": 0.098}
CUT, HW = 400.0, hw_arg
t0 = time.time()
channel = build_channel(j2, par, t2, nmax)
print(f"N3max {nmax} 2J={j2} P={par:+d} 2T={t2}: AS {channel.n_as}, NAS {channel.n_nas}, basis {time.time() - t0:.0f} s", flush=True)
edge = ((2 * nmax + 3) / (2.0 * M_REDUCED_PN * HW / HBARC**2)) ** 0.5
print(f"hw {HW}: outer turning point sqrt((2N+3)/nu) = {edge:.1f} fm")
SCALE = edge / 14.7  # the meshes below were chosen for hw 16, N3max 40
cases = {
    "NuHamil (80, 15 fm; p100 to 8)": nuhamil_meshes(),
    "80, 15 fm (converged p, cos)": converged_meshes(CUT, 2, n_xi=80, xi_max=15.0),
    "120, 15 fm": converged_meshes(CUT, 2, n_xi=120, xi_max=15.0),
    "120, 18 fm": converged_meshes(CUT, 2, n_xi=120, xi_max=18.0),
    "160, 18 fm": converged_meshes(CUT, 2, n_xi=160, xi_max=18.0),
    "200, 22 fm (reference)": converged_meshes(CUT, 2, n_xi=200, xi_max=22.0),
}
if SCALE > 1.0:  # a smaller hw widens the oscillator: stretch the large meshes with it
    cases = {k: v for k, v in cases.items() if "NuHamil" in k or "15 fm" in k}
    for n_xi, xi_max in ((120, 1.22 * edge), (160, 1.22 * edge), (200, 1.5 * edge)):
        cases[f"{n_xi}, {xi_max:.0f} fm" + (" (reference)" if n_xi == 200 else "")] = converged_meshes(
            CUT, 2, n_xi=n_xi, xi_max=xi_max
        )
nu = 2.0 * M_REDUCED_PN * HW / HBARC**2
n_max = nmax // 2
mats = {}
for name, m in cases.items():
    t = time.time()
    orb = np.asarray(orbital_table(nu, m.xi, m.xi_w, nmax, n_max))
    defect = max(
        np.abs(orb[l, : (nmax - l) // 2 + 1] @ orb[l, : (nmax - l) // 2 + 1].T - np.eye((nmax - l) // 2 + 1)).max()
        for l in range(nmax + 1)
    )
    tab = radial_tables(m, CUT, 2, nmax + 2)
    mats[name] = local_3nf_matrix(channel, HW, m, tab, LECS, workers=workers)
    print(f"  {name}: overlap defect {defect:.1e}, built in {time.time() - t:.0f} s", flush=True)
ref = mats[[k for k in mats if "reference" in k][0]]
ev_ref = np.linalg.eigvalsh(ref)
print("difference from the reference mesh (max element / max eigenvalue; matrix scale %.2f):" % np.abs(ref).max())
for name, v in mats.items():
    print(f"  {name:34s} {np.abs(v - ref).max():.2e}  {np.abs(np.linalg.eigvalsh(v) - ev_ref).max():.2e}")
