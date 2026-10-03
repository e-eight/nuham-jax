import sys, time, math, pickle, os

sys.path.insert(0, "/home/soham/Projects/nuclear-physics/nuhamil-alt/refs")
import numpy as np

os.environ["ORACLE_SOLVER"] = "dop853"
import oracle_numpy as o
from concurrent.futures import ProcessPoolExecutor

NC = [20, 30, 40, 50, 60, 80]
E2 = [8, 12, 16, 20, 28]


def chan(args):
    hw, lam, key, st, V = args
    J, P, S, Z = key
    T = o.kinetic(st, hw)
    C = o.coulomb_matrix(st, hw) if Z == -1 else 0.0
    lc = lam * o.hc / math.sqrt(2 * o.mred)
    s = lc**-4
    N = np.array([2 * n + l for n, l in st])
    res = {}
    t = time.time()
    full = o.srg_flow(T + V + C, T, s, 1e-10, 1e-10) - T
    res[100] = (full, time.time() - t)
    for nc in NC:
        idx = np.where(N <= nc)[0]
        t = time.time()
        r = (
            o.srg_flow((T + V + C)[np.ix_(idx, idx)], T[np.ix_(idx, idx)], s, 1e-10, 1e-10)
            - T[np.ix_(idx, idx)]
        )
        res[nc] = (r, time.time() - t)
    errs = {}
    for nc in NC:
        idx = np.where(N <= nc)[0]
        for e2 in E2:
            if e2 > nc:
                continue
            sel = np.where(N <= e2)[0]
            sub_ref = full[np.ix_(sel, sel)]
            pos = {i: k for k, i in enumerate(idx)}
            ps = [pos[i] for i in sel]
            errs[(nc, e2)] = np.abs(res[nc][0][np.ix_(ps, ps)] - sub_ref).max()
    tim = {k: v[1] for k, v in res.items()}
    return key, errs, tim


if __name__ == "__main__":
    for hw, lam in [(16, 2.0), (16, 1.8), (16, 2.8), (24, 2.0), (40, 2.0), (10, 2.0)]:
        bare = o.read_rel_bin(hw)
        jobs = [(hw, lam, k, st, V) for k, (st, V) in bare.items() if k[0] <= 8]
        t = time.time()
        with ProcessPoolExecutor(12) as ex:
            out = list(ex.map(chan, jobs))
        print(
            f"hw={hw} lam={lam}  ({time.time() - t:.0f}s)  max abs err (MeV) in V sub-block, rows: Nflow, cols: e2max",
            flush=True,
        )
        print("  Nflow  " + "  ".join(f"e2={e:<3d}" for e in E2) + "   flow-time(sum,s)")
        tsum = {k: sum(x[2][k] for x in out) for k in [100] + NC}
        for nc in NC:
            row = []
            for e2 in E2:
                row.append(f"{max(x[1][(nc, e2)] for x in out):.1e}" if e2 <= nc else "   -   ")
            print(f"  {nc:4d}   " + " ".join(row) + f"    {tsum[nc]:.1f}")
        print(
            f"   100   (reference)                                       {tsum[100]:.1f}",
            flush=True,
        )
