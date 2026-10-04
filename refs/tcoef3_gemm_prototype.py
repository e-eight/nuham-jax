"""Throwaway prototype: the Jacobi->lab NAS coefficients as batched dense products (spec 5b/5d study).

For fixed (j12, jc) the sum over (lit, Lam) factorises into ``A[(pair, p), k] @ B[k, (c, q)]``, which is
the structure meant for a GPU (see docs/study-tcoef3.md). On the CPU it is slower than the term-wise
``nuham.tcoef3.tcoef_nas`` it is checked against. Run: ``python refs/tcoef3_gemm_prototype.py 8``.
"""

import sys, time, numpy as np
from nuham.basis.lab import make_orbits
from nuham.angmom.racah_array import sixj_array
import nuham.tcoef3 as tc
from nuham.tcoef3 import *


def tcoef_gemm(rows, cols, j2, orb, br, chunk=2_000_000, backend="numpy"):
    """Dense-product form; ``backend='jax'`` runs the products and the scatter on the default device."""
    dev = backend == "jax"
    if dev:
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
    t_host = time.perf_counter()
    n_r, n_c = len(rows), len(cols)
    out = np.zeros((n_r, n_c))
    outd = jnp.zeros(n_r * n_c) if dev else None
    qn = rows.qn
    pst, p_of_row = tc._unique_rows(qn[:, :5])
    qst, q_of_row = tc._unique_rows(qn[:, [5, 6, 7, 9, 10, 8]])
    pairs, pair_of_col = tc._unique_rows(np.stack([cols.i1, cols.i2, cols.j12], 1))
    corbs, c_of_col = np.unique(cols.i3, return_inverse=True)
    c_of_col = c_of_col.reshape(-1)
    rowmap = tc._index_map(np.stack([p_of_row, q_of_row], 1), np.arange(n_r), (len(pst), len(qst)))
    colmap = tc._index_map(
        np.stack([pair_of_col, c_of_col, cols.t12], 1), np.arange(n_c), (len(pairs), len(corbs), 2)
    )
    flat = out.reshape(-1)
    j12_p, t12_p = pst[:, 3], pst[:, 4]
    jab_pair = pairs[:, 2]
    jc_c = orb.j2[corbs]
    lcm_q, j3_q, jrel_q = qst[:, 4], qst[:, 2], qst[:, 5]
    n = int(cols.n[0])
    npair, npp, nc, nq = len(pairs), len(pst), len(corbs), len(qst)
    # dense left per lit, right per (lit,Lam)
    Lf = {}
    for lit in range(n + 1):
        lp, lq, lnit, lv = tc._left_terms(lit, pairs, pst, br, orb)
        if len(lv):
            d = np.zeros((npair, npp))
            d[lp, lq] = lv
            Lf[lit] = d
    K = []  # (lit, Lam)
    Rf = {}
    for lit in Lf:
        for lam in range(1, 2 * lit + int(jc_c.max()) + 2, 2):
            rc, rq, rn, rv = tc._right_terms(lit, lam, j2, corbs, qst, br, orb)
            if len(rv):
                d = np.zeros((nc, nq))
                d[rc, rq] = rv
                Rf[(lit, lam)] = d
                K.append((lit, lam))
    if not K:
        return out
    lits = np.array([k[0] for k in K])
    lams = np.array([k[1] for k in K])
    lits = np.array([k[0] for k in K])
    lams = np.array([k[1] for k in K])
    nK = len(K)
    LfK = np.stack([Lf[k[0]] for k in K])  # (K,npair,npp)
    RfK = np.stack([Rf[k] for k in K])  # (K,nc,nq)
    jabs = np.arange(int(jab_pair.max()) + 1)
    jcs = np.arange(int(jc_c.max()) + 1)
    j12s = np.arange(int(j12_p.max()) + 1)
    a1t = (
        sixj_array(
            2 * j12s[None, None, None, :],
            2 * lits[:, None, None, None],
            2 * jabs[None, :, None, None],
            jcs[None, None, :, None],
            j2,
            lams[:, None, None, None],
        )
        if False
        else None
    )
    a1t = sixj_array(
        2 * j12s[None, None, None, :],
        2 * lits[:, None, None, None],
        2 * jabs[None, :, None, None],
        jcs[None, None, :, None],
        j2,
        lams[:, None, None, None],
    )  # (K,jab,jc,j12)
    lcms = np.arange(int(lcm_q.max()) + 1)
    j3s = np.arange(int(j3_q.max()) + 1)
    jrels = np.arange(int(jrel_q.max()) + 1)
    flops = 0
    tcoef_gemm.stage = {"host_factors": time.perf_counter() - t_host, "product": 0.0}
    t_prod = time.perf_counter()
    for j12 in np.unique(j12_p):
        Pj = np.nonzero(j12_p == j12)[0]
        gt = (
            sixj_array(
                2 * lcm_q[None, :], j3_q[None, :], lams[:, None], 2 * j12, j2, jrel_q[None, :]
            )
            * (lams[:, None] + 1.0)
            * (-1.0) ** ((lams[:, None] - jrel_q[None, :]) // 2)
            * np.sqrt(jrel_q[None, :] + 1.0)
        )
        LfP = LfK[:, :, Pj]
        for jc in np.unique(jc_c):
            Cj = np.nonzero(jc_c == jc)[0]
            a1 = a1t[:, jab_pair, jc, j12]  # (K,npair)
            keep = np.nonzero(np.abs(a1).sum(1) > 0)[0]
            if len(keep) == 0:
                continue
            B = (RfK[keep][:, Cj, :] * gt[keep][:, None, :]).reshape(len(keep), -1)
            A = (
                (LfP[keep] * a1[keep][:, :, None])
                .transpose(1, 2, 0)
                .reshape(npair * len(Pj), len(keep))
            )
            flops += 2 * A.shape[0] * A.shape[1] * B.shape[1]
            step = max(1, chunk // max(B.shape[1], 1))
            qs = np.tile(np.arange(nq), len(Cj))[None, :]
            cc = np.repeat(Cj, nq)[None, :]
            if dev:
                Ad = jnp.asarray(A)
                Bd = jnp.asarray(B)
            for s0 in range(0, A.shape[0], step):
                blk = (Ad[s0 : s0 + step] @ Bd) if dev else A[s0 : s0 + step] @ B
                ii = np.arange(s0, min(s0 + step, A.shape[0]))
                pair_i = ii // len(Pj)
                p_i = Pj[ii % len(Pj)]
                ridx = rowmap[p_i[:, None], qs]
                cidx = colmap[pair_i[:, None], cc, t12_p[p_i][:, None]]
                ok = (ridx >= 0) & (cidx >= 0)
                if dev:
                    tgt = np.where(ok, ridx * n_c + cidx, n_r * n_c).reshape(-1)
                    outd = outd.at[jnp.asarray(tgt)].add(blk.reshape(-1), mode="drop")
                else:
                    flat[(ridx * n_c + cidx)[ok]] += blk[ok]
    if dev:
        out = np.asarray(outd.block_until_ready()).reshape(n_r, n_c)
    tcoef_gemm.stage["product"] = time.perf_counter() - t_prod
    tcoef_gemm.flops = flops
    return out


if __name__ == "__main__":
    j2, P, t2, n, emax = 3, 1, 1, int(sys.argv[1]), int(sys.argv[1])
    orb = make_orbits(emax)
    br = make_brackets()
    lab = lab_states(orb, j2, P, t2, 2 * emax, emax)
    cols = lab.select(n)
    rows = jacobi_cm_rows(j2, P, t2, n)
    tcoef_nas(rows, cols, j2, orb, br)
    t = time.process_time()
    A = tcoef_nas(rows, cols, j2, orb, br)
    ta = time.process_time() - t
    t = time.process_time()
    B = tcoef_gemm(rows, cols, j2, orb, br)
    tb = time.process_time() - t
    print(
        "terms",
        ta,
        "gemm",
        tb,
        "maxdiff",
        abs(A - B).max(),
        "gemm flops %.2e" % tcoef_gemm.flops,
        "nnz",
        int((A != 0).sum()),
    )
