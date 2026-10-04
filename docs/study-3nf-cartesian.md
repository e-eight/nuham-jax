# Study: Cartesian check of NuHamil's local N2LO terms (Phase 5.2.0 and 5.2.1)

Oracle data: `refs/work/nnn3f/` (`run_all.sh`; NuHamil with `save_3nf_before_lec=.true.`, hw 16, Lambda_local 400 or 500, power 2).
`n400_N6` and `n500_N6` (3NF400 and the 500 MeV set cD 0.83, cE -0.052, `ramp="flat6"`) are committed: the 24
`ops/N2LO_3NF_HOPW_*.bin` files (LEC-free NAS lower triangles, order c1 c3 c4 cD cE) each; `cfp/` only in `n400_N6` (it does not depend on the force). `n400_N8` (17 MB of
the same files) is kept locally, not committed. The Jacobi Nmax comes from the **ramp**, not from `N3max`: `flat8` with
`N3max=6` still writes Nmax 8 files.
Run the cases one at a time: three concurrent 16-thread NuHamil runs on 16 cores were about 40x slower per block.

`refs/nnn_cartesian.py` builds one representative term of cE, cD and c1 from the physics formulas alone (explicit 8x8 spin and
isospin matrices, Cartesian gradients of the regulated Yukawa, regulated deltas, product quadrature over both Jacobi vectors, own
radial tables) and compares it with NuHamil's stored `V_term`.

Results (max abs difference after fixing the factor and gauge below, MeV; all 1e-10 or better):

| channel (2J, P, 2T) | terms checked | states |
|---|---|---|
| 1+ T1/2, Lambda 400 | cE, cD, c1 | N <= 2 (14) |
| 1- T1/2, Lambda 400 | cE | N = 1 (6) |
| 1+ T1/2, Lambda 500 (N3max 6 set) | cE, cD, c1 | N <= 2 |
| 1+ T3/2, 3+ T1/2, 3+ T3/2 | cE, cD, c1 | N <= 2 |

Findings:
- **Factor 3 is right.** `V_term` equals one ordered physical term times 1 for cE (three pair terms, `sum_{j<k}`) and times 2 for
  cD and c1 (six ordered terms `sum_{i!=j!=k}`, 3 `V_term` = 6 terms). So NuHamil's `3 * V_term * LEC` is the full physical operator
  in antisymmetric states. cE, cD and c1 signs and constants (hc powers, 1e-3 for c1) are as in spec 5b.
- **Roles.** The representative has the pion on r_21, the contact (cD) or second delta (cE) on r_31, tau_2.tau_3 and sigma_2, sigma_3
  (c1: pions on r_21 and r_31); this follows from momentum conservation, not only from the Fortran.
- **Phase (the `(-1)^x` question).** NuHamil's NAS states equal the Cartesian ones, built as ((l12 s12)j12 (l3 1/2)j3)J with
  Y_l12 of xi1 = (r1-r2)/sqrt2 and positive-at-origin radial functions, times `(-1)^l12` (equivalently `(-1)^l3` up to a channel-wide sign:
  same product `s_a s_b`). The `src/` angular layer must give V in this gauge.
- **State lists drop empty blocks.** NuHamil's NAS list leaves out an N block with no antisymmetric state (e.g. the N=0 state
  of 1+ T3/2 and of 3+ T1/2: file dimension 49 and 151 against our 50 and 152). Indices of the remaining states keep our order.
- The 80-point r mesh and 100-point p mesh of NuHamil agree with converged tables to 1e-10 at N <= 2; large N is 5.2.5.

Not done: c3 and c4 (checked later against the oracle files; c4 needs the cross-product spin algebra), the pre-SRG antisymmetric
matrix gate (NuHamil stores only the post-SRG `NNN-full`; use `CFP^T (3 V LEC) CFP` eigenvalues against the log instead, in 5.2.4).
