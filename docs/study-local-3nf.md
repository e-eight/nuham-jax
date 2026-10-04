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

# Phase 5.2.2: radial layer (`nuham.kernels.nnn.radial`, `nuham.nnn.local_force.radial`)

`kernels/nnn/radial.py` holds the pure kernels (regulator, spherical Bessel functions of all orders by Miller's downward
recurrence with overflow rescaling, closed forms with a series near zero for orders 0-2, Legendre table, the p-sum transforms and the Legendre
projection of `f1/r`, `f2/r^2`). `nnn/local_force/radial.py` builds the meshes (`nuhamil_meshes()` = NuHamil's 80 xi points to 15 fm, 100 p points to
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

# Phase 5.2.3: angular layer (`nuham.nnn.local_force.angular`, `tests/test_local3n_angular.py`)

`nnn.local_force.angular.coefficients(term, bra, ket, j2, t2, x_max)` returns, for two pair-antisymmetric channels
`(l12, s12, j12, t12; l3, j3)`, the coefficients `C_a` of the radial functions `g_a` (keys `(X)` for cE, `(K, X)` for
cD, `(K3, X)` for c1, `(K1, K2, K3, X)` for c3 and c4; `local_force.radial.radial_function` builds `g_a` from the 5.2.2 tables). All
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

# Phase 5.2.4: projection and assembly (`nuham.nnn.local_force.assembly`, `kernels/nnn_projection.py`)

`nas_matrix` sums, for every pair of channels, the five terms times `3 LEC` into one grid `sum_a C_a g_a(xi1, xi2)` and projects it once
with `kernels.nnn.projection.project_pair` (two einsums over weighted HO orbital tables, static shapes: orbitals are padded to
`n_max + 1`). `antisymmetrize` then forms `cfp W cfp^T` block by block in N, with the 5.1 `cfp` or NuHamil's.
`local_3nf_matrix(channel, hw, meshes, tables, lecs)` returns the bare genuine V3 in the AS Jacobi basis, MeV.

Reference for the AS-matrix gate: NuHamil with `renorm="bare"` stores `H - T - 3 V_NN`, which is the bare genuine 3NF
(`n400_N6_bare`). The same run without the genuine force (`n400_N6_bare_ind`) shows that the subtraction itself is not exactly zero:
all channels are at 1e-14 except 1/2+ T1/2, where one element is 4.7e-6 (a NuHamil artefact of the 3 V_NN embedding, independent of the 3NF).
The 3NF matrix is therefore `NNN-full - NNN-ind`.

Results, N3max 6, hw 16, 3NF400 (LECs of the `3NF400` set):
- with NuHamil's `cfp` the matrix equals the stored one element by element to <= 2e-12 MeV (1/2+ T1/2, 1/2+ T3/2, 3/2- T1/2,
  5/2+ T1/2, 11/2- T3/2; matrix elements up to 0.83): this also fixes the factor 3 inside the AS matrix;
- with our own `cfp` (a different orthonormal basis of the same space) the eigenvalues agree to 4e-12 and the trace to 4e-12.
At N3max 8 (`n400_N8_bare` and `_ind`, kept local, not committed) eight channels (1/2+, 1/2-, 3/2+, 5/2+, 7/2+, 11/2+ at T1/2; 1/2+ and
3/2- at T3/2; AS dimensions 31 to 113) agree element by element with NuHamil's cfp to <= 1.5e-12 MeV; our scalar code takes 0.3 to 6 s per channel.
The Lambda -> infinity cE check planned for this step is not needed: the Cartesian check (5.2.1) already derives the cE normalisation from the
physics formula alone at finite Lambda to 1e-10.

# Phase 5.2.5: scaling and the xi mesh at large N

Assembly was restructured for scale (`nas_matrix`): the Racah coefficients of each channel pair stay host scalar code but run in `workers` forked processes
(pure Python and NumPy, no JAX in the children; the fork warning JAX prints is about its own threads); the weighted sum over radial functions is one matrix
product per chunk of 64 pairs; the HO projection is `kernels.nnn.projection.project_pairs` (a `vmap` of `project_pair`), batched on the device. Results
are bit-identical with 1 and 2 workers and unchanged by a larger `x_max` (`tests/test_local3n_assembly.py`).

Agreement with NuHamil at larger N (3NF400, hw 16, NuHamil's cfp, `n400_N16_bare` and `_ind`, kept local): J = 1/2, P = +-, T = 1/2 and 3/2 at N3max 16: max element
difference <= 1.5e-12 MeV (matrix elements up to 2.2).

Timing (CPU only, this 16-core machine, 14 worker processes; `benchmarks/bench_3n_dense.py force`, records `benchmarks/baseline_cpu_3n_force_j1.json` and `_j5.json`;
stage times exclude the Jacobi basis (5.7 s at N3max 40, J=1/2) and the radial tables (0.9 s)):

| N3max | J=1/2+ T=1/2: NAS / AS states | assembly | peak RSS | J=5/2+ T=1/2: NAS / AS | assembly | peak RSS |
|---|---|---|---|---|---|---|
| 8 | 190 / 64 | 0.7 s | 0.5 GB | 340 / 113 | 1.6 s | 0.5 GB |
| 12 | 504 / 169 | 1.1 s | 0.6 GB | 1050 / 350 | 4.9 s | 0.6 GB |
| 16 | 1050 / 351 | 1.9 s | 0.8 GB | 2376 / 792 | 11.0 s | 0.8 GB |
| 24 | 3094 / 1033 | 6.1 s | 1.0 GB | 7644 / 2548 | 39.7 s | 1.4 GB |
| 32 | 6834 / 2280 | 14.6 s | 1.5 GB | 17680 / 5893 | 104.7 s | 3.8 GB |
| 40 | 12782 / 4263 | 30.5 s | 2.7 GB | (34020 / about 11 340: dense NAS matrix 9.3 GB, not run here) | | |

Where the time goes at N3max 40 (J=1/2): projection 19.1 s, radial sums 6.5 s, scatter 3.5 s, antisymmetrisation 0.6 s; the coefficients are hidden behind
them (0.16 s of waiting) because the workers run ahead. A single core needs about 8 ms per channel pair at N3max 40 (J=5/2: 108 811 pairs, about 14 core-minutes), so
the projection and the radial sums (both GEMM-shaped and device-friendly) are the remaining cost on the CPU.

NuHamil on the same machine (16 threads, first block of the log, J=1/2+ T=1/2, `precalculations` plus the five `set`s): N3max 8: 8.8 s; N3max 16: 26.5 s; N3max 24: 91.8 s
(22.3 + 12.4 + 21.3 + 12.4 + 14.0 + 9.3), i.e. our assembly is 12x, 14x and 15x faster at 8, 16 and 24 (NuHamil's time roughly triples with each added 8 in N3max). The N3max 24 and 40 NuHamil
matrices were not generated (only timed, and stopped after the first block).

Memory: the NAS matrix is dense in `nas_matrix`: 9.3 GB for the largest production channel (J = 5/2 at N3max 40), 3.7 GB at N3max 32 (measured peak 3.8 GB). That fits the
GH200 (480 GB host) but not comfortably this machine (19 GB); a block-wise accumulation of `cfp W cfp^T` is the way to shrink it if needed.

xi mesh (`refs/nnn_xi_mesh_study.py`, outputs `refs/nnn_xi_mesh_study_*.txt`; AS matrix against a 200-point mesh to 22 fm; p and cosine meshes converged):

| channel | mesh | HO overlap defect on the mesh | max element difference | max eigenvalue difference |
|---|---|---|---|---|
| J=1/2+, N3max 24, hw 16 | NuHamil (80 points, 15 fm) | 2e-10 | 4e-14 | 1e-12 |
| J=5/2+, N3max 24, hw 16 | NuHamil | 2e-10 | 9e-15 | 3e-13 |
| J=1/2+, N3max 40, hw 16 | NuHamil | 1.2e-2 | 1.2e-9 | 2.7e-11 |
| same | 120 points, 18 fm | 9e-11 | 3e-14 | 2e-12 |
| J=1/2+, N3max 24, hw 8 | NuHamil | 0.26 | 6.3e-9 | 1.5e-9 |
| same | 120 points, 20 fm | 2e-7 | 1e-14 | 5e-13 |

Verdict: NuHamil's mesh is converged to 1e-14 up to N3max 24 at hw 16. Beyond that its 15 fm edge cuts the outer part of the HO states (overlap defect 1% at N3max 40, 26% at hw 8, N3max 24),
but the 3NF is short ranged, so the matrix elements stay within 1.2e-9 MeV (hw 16, N3max 40) and 6e-9 MeV (hw 8, N3max 24): physically irrelevant, but not roundoff. The mesh must
end beyond the outer turning point `sqrt((2N + 3) / nu)`, `nu = 2 m hw / (hbar c)^2` (14.7 fm for N 40 at hw 16, 16.3 fm for N 24 at hw 8); 22% beyond with 7 points per fm gives
1e-14 in all three cases (`nuham.nnn.local_force.radial.xi_mesh_size`). The 80-point mesh is kept as the NuHamil-compatible option; production defaults should use `xi_mesh_size`.
The p and cosine meshes are converged at Lambda 400 to 650 with NuHamil's sizes (5.2.2).

GH200 (DeltaAI job 3308734, node `gh136`, commit `3946614`, 32 CPUs and 30 worker processes, 7 min 56 s wall, about 0.13 GPU-hours; records
`benchmarks/baseline_gh200_3n_force_{j1,j5,j5_n40}.json`). The 48 radial, angular, assembly and Jacobi tests pass on the GPU backend, including the forked worker
pool after CUDA initialisation (JAX's fork warning appears but nothing hangs). Assembly time, the stage times behind it and the peak memory:

| N3max | J=1/2+ T=1/2: assembly (CPU, this machine) | GH200 | J=5/2+ T=1/2: assembly (CPU) | GH200 | GH200 stages for J=5/2+ (basis / sums / projection / scatter / cfp) | peak host RSS |
|---|---|---|---|---|---|---|
| 16 | 1.9 s | 2.0 s | 11.0 s | 5.1 s | 5.1 / 2.1 / 2.4 / 0.1 / 0.05 s | 0.7 GB |
| 24 | 6.1 s | 3.1 s | 39.7 s | 13.6 s | 6.7 / 7.1 / 4.3 / 0.7 / 1.0 s | 1.3 GB |
| 32 | 14.6 s | 6.3 s | 104.7 s | 39.8 s | 21.1 / 17.2 / 10.4 / 3.2 / 8.4 s | 3.9 GB |
| 40 | 30.5 s | 13.4 s | (not run) | 118.6 s | 90.3 / 34.5 / 26.0 / 10.8 / 46.3 s | 11.7 GB |

"Assembly" is the work of `local_3nf_matrix` and excludes the Jacobi basis (J=1/2+: 7.8 s at N3max 40; J=5/2+: 90 s) and the radial tables (2-5 s). The largest production channel (J = 5/2, N3max 40,
34 020 NAS and 11 340 AS states) therefore takes 118.6 s plus 90 s for its basis, with 11.7 GB of host memory (the dense 9.3 GB NAS matrix) and only 0.44 GB of device memory. Totals for the J=1/2
channel at N3max 40: 13.4 s plus 7.8 s of basis. Against NuHamil (91.8 s for the J=1/2+ block at N3max 24 on 16 CPU threads, the same block takes 3.1 s here: 30x).

The GPU runs the projection (4.8 s at N3max 40 for J=1/2) but the radial sums (a host NumPy product), the scatter into the dense matrix and the `cfp W cfp^T` products (46 s for the largest
channel) still run on the host, so the speed-up over the CPU run is only 2.3x for J=1/2 and about 2.6x for J=5/2 at N3max 32. The next steps if more speed is wanted are the radial sums and
the antisymmetrisation as device products (the dense matrix then also lives in the 98 GB of device memory).
