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

# Phase 5.2.2: radial layer (`nuham.kernels.nnn_radial`, `nuham.local3n`)

`kernels/nnn_radial.py` holds the pure kernels (regulator, spherical Bessel functions of all orders by Miller's downward
recurrence with overflow rescaling, closed forms with a series near zero for orders 0-2, Legendre table, the p-sum transforms and the Legendre
projection of `f1/r`, `f2/r^2`). `local3n.py` builds the meshes (`nuhamil_meshes()` = NuHamil's 80 xi points to 15 fm, 100 p points to
8 fm^-1, 80 cosines; `converged_meshes(cutoff)` = p mesh to where the regulator is 1e-17, 300 p points, 120 cosines) and
`radial_tables` returns `z0, fk(0..2), zx(x), fkx(0..2, x)` on the xi x xi mesh. Differences from NuHamil's `precalculations`:
`f1` and `f2` are not recomputed for every x (the Legendre projection is one pass), and the p sums are batched on the device.
Timing at N3max 40 (x up to 42), CPU: 1.6 s cold, 0.5-1.2 s warm.

Gates (`tests/test_nnn_radial.py`):
- spherical `j_n` against SciPy to 1e-14 (n <= 45, x up to 300), small-argument forms to 5e-14 relative;
- Yukawa closed forms (`f1 = e^{-mu r}(1 + mu r)/(4 pi r^2)`, `f0 = -mu^2 e^{-mu r}/(4 pi r)`) at Lambda 3000 MeV: only the (m_pi/Lambda)^4 = 4.5e-6 regulator residue remains;
- addition theorem: `sum_x (2x+1) zx_x P_x(cos)` and the same for `fkx0` reproduce the transform at `|a - s|` to 1e-13 (so `zx` and `fkx0` carry no hidden `(-1)^x`);
- Legendre reconstruction of `fkx1` and `fkx2` to 1e-13;
- agreement with the independent Cartesian-check tables in `refs/nnn_cartesian.py` to 1e-10 absolute (those agree with NuHamil at N <= 2, section above).

Mesh convergence (`refs/nnn_radial_mesh_study.py`, output next to it; x up to 42, errors relative to the largest table value, reference 1600 p points and 240 cosines):
- NuHamil's meshes are converged to 1e-13 for Lambda = 400, 500 and 650 MeV, so the 80-point cosine and 100-point p meshes are not a limitation for the radial tables at N3max 40;
- at Lambda = 1000 MeV NuHamil's 8 fm^-1 p cut-off leaves 1e-3 (`exp(-(8 hc/1000)^4)` = 2e-3), as predicted in spec 5b;
- converged meshes with 200-600 p points and 120-160 cosines all reach the 1e-13 noise floor, so 300 and 120 are the defaults.
The xi mesh (80 points to 15 fm) only enters through the HO projection and is studied in 5.2.4 and 5.2.5.

# Phase 5.2.3: angular layer (`nuham.local3n_angular`, `tests/test_local3n_angular.py`)

`local3n_angular.coefficients(term, bra, ket, j2, t2, x_max)` returns, for two pair-antisymmetric channels
`(l12, s12, j12, t12; l3, j3)`, the coefficients `C_a` of the radial functions `g_a` (keys `(X)` for cE, `(K, X)` for
cD, `(K3, X)` for c1, `(K1, K2, K3, X)` for c3 and c4; `local3n.radial_function` builds `g_a` from the 5.2.2 tables). All
constants are in (LEC = 1, MeV, fm), c1, c3, c4 for 1 GeV^-1. The formulas are Eqs. (15), (38), (42), (47) and (49) of Navratil
(Few-Body Syst. 41, 117), read from the arXiv text; a hand check shows they are the same sums as NuHamil's
`NNNForceLocal.F90` (same 6j and 9j triads; Nav07's `[C(2K2+1, 2K3)]^(1/2) hat(K2-K3)` equals NuHamil's `fk3k4`). Because
that agrees term by term, no slow Fortran transcription was added to `refs/`: NuHamil's saved matrices are the cross-check.

Gate: `tests/ref_nnn_projection.py` projects the coefficients with plain Gauss quadrature on NuHamil's meshes and compares with
the saved per-term NAS matrices. Maximum absolute difference (MeV, matrices up to 0.6):
- N3max 6, Lambda 400 and 500 (the 500 MeV set): all 24 (J, P, T) blocks, all five terms, <= 5e-13;
- N3max 8, Lambda 400: all 24 blocks, all five terms, <= 4e-13.
This includes c3 and c4 (c4 is zero for T = 3/2 as it must be) and settles the sign and phase conventions: the NAS gauge is Nav07's
(the `(-1)^l12` of the Cartesian check above is consistent with Nav07 defining xi2 with the opposite sign to the one used there, which gives `(-1)^l3`; not checked separately). The paper's printed phase `(-1)^(J-J'+s+j')` is
garbled in the arXiv text; the Fortran's `(-1)^((Jtot-j3)/2 + s12 + j45)` (c4: `(Jtot+j3)/2 + j45`) is what reproduces the files.

Cost: the loops are scalar Python over cached 6j/9j symbols (0.3 s per N3max 6 channel for all terms, a minute for the 24 channels at
N3max 8). That is fine for validation, not for N3max 40: 5.2.4 and 5.2.5 vectorise it on `racah_array` and group the intermediate
sums (the `(v, r, y, z)` loops factor over the bra and ket orbital labels).
