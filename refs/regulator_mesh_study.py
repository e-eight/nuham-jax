"""Mesh convergence of the LNL regulator matrix (gap D, D1): NuHamil's 500-point mesh against finer ones.

usage: uv run python refs/regulator_mesh_study.py [hw]
Prints, for N = 12, 24, 40 and every (l12, l3) pair, the largest difference over the states with
2 n12 + l12 + 2 n3 + l3 <= N (both bra and ket within N) between NuHamil's mesh and a reference mesh.
"""

import sys

import numpy as np

import nuham  # noqa: F401
from nuham.constants import M_REDUCED_PN, oscillator_length_sq
from nuham.kernels.nnn.regulator import regulator_table
from nuham.nnn.local_force.radial import gauss_legendre

hw = float(sys.argv[1]) if len(sys.argv) > 1 else 16.0
cutoff, power = 500.0, 2
b = float(np.sqrt(oscillator_length_sq(hw, 2.0 * M_REDUCED_PN)))
pmax = 150.0 / np.sqrt(hw)
meshes = {
    "nuhamil 500": gauss_legendre(500, 0.0, pmax),
    "250": gauss_legendre(250, 0.0, pmax),
    "1000": gauss_legendre(1000, 0.0, pmax),
    "1500, 2x pmax": gauss_legendre(1500, 0.0, 2 * pmax),
}
ref = meshes["1500, 2x pmax"]
for nn in (12, 24, 40):
    n_max = nn // 2
    worst = {k: 0.0 for k in meshes}
    where = {}
    n = np.arange(n_max + 1)
    for l12 in range(nn + 1):
        for l3 in range(nn + 1 - l12):
            if (l12 + l3) % 2:
                continue
            # axes: [n12, n12', n3, n3']; mask requires both (n12, n3) and (n12', n3') within N
            m = np.zeros((n_max + 1,) * 4, dtype=bool)
            for a in range(n_max + 1):
                for c in range(n_max + 1):
                    if 2 * a + l12 + 2 * c + l3 > nn:
                        continue
                    for a2 in range(n_max + 1):
                        for c2 in range(n_max + 1):
                            m[a, a2, c, c2] = 2 * a2 + l12 + 2 * c2 + l3 <= nn
            if not m.any():
                continue
            t_ref = np.asarray(regulator_table(l12, l3, n_max, b, cutoff, power, *ref))
            for name, (p, w) in meshes.items():
                t = np.asarray(regulator_table(l12, l3, n_max, b, cutoff, power, p, w))
                d = np.abs(t - t_ref)[m].max()
                if d > worst[name]:
                    worst[name], where[name] = d, (l12, l3)
    print(f"hw {hw}, N <= {nn}: " + ", ".join(f"{k}: {v:.2e} at {where.get(k)}" for k, v in worst.items()))
