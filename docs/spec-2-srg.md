# Spec 2: SRG evolution of the NN interaction in the HO relative basis

Source root: `/home/soham/Projects/nuclear-physics/NuHamil-public/src` (abbrev. `src/`).
Status tags: **[V]** = verified (code read AND/OR reproduced numerically), **[U]** = unsure / not checked.
Scratch scripts: `/tmp/claude-1000/-home-soham-Projects-nuclear-physics-nuhamil-alt/9bd3c06c-e436-4068-8fa0-49a1c11b409e/scratchpad/{srg.py,stiff.py}`.

## 0. Which files are actually used

- `src/Renormalization.F90` is the compiled one (Makefile:358-359, 469). `src/TwoBody/Renormalization.F90` is a stale copy that only adds `SRGHflow_test` (diff confirmed). The SRG routine used is `SRGHflow` (Renormalization.F90:83-190). [V]
- Call chain for `rank=2, renorm="srg"`: `TwoBodyManagers` (TwoBody/TwoBodyManager.F90:10) forces `pn_form=.true.` (":24-28": "isospin formalism not supported") -> `manage_trans_to_lab` -> `trans_to_lab_op` (":176-232") -> `NNForceHO%SetNNForceHO` (TwoBody/NNForce.F90:112) -> `renorm_ho_space` (NNForce.F90:631) -> `HOSRGChannel` (NNForce.F90:665) -> `SRGSolver%SRGFlow` -> `SRGHflow`. The isospin variant (`NNForceIsospin.F90`, same recipe with `m_red_pn`) is NOT reached for NN. [V]
- Operator evolved: `hamil` (default `Operators='hamil'`, NuHamilInput.F90:241). In `trans_to_lab_op`, `oprel = vnn` (TwoBodyManager.F90:~183) i.e. the evolved *V only*. [V]

## 1. Pipeline for the example run (Input_a.dat: hw=16, lambda=2.0, emax=4, e2max=8, N2max default 100)

1. **Bare relative V in HO basis** (pn formalism, per channel (J,P,S,Z)). Read from file `NNint_A2_rel_<NNInt>_bare_Nmax<N2max>_hw<hw>.gz` if it exists in `path_to_NNrel` (default = path_to_tmp_dir) (NNForce.F90:130-131, 214-218 `readf`); otherwise computed from momentum-space chiral potential (`input_nn_file` binary, NNForce.F90:152-205) and written to that file *before* Coulomb/SRG (NNForce.F90:207-209). File name builder: NNForce.F90:538-548 (note `s%str(hw)` formatting gives `hw16`). [V]
   - File format: header line `ntot N2max Jmax` (here `121551 100 8`); then lines `J P S Z n1 l1 n2 l2 value` format `(8i4,es18.6)`, lower triangle only (`i2<=i1` over channel state index), value in MeV (NNForce.F90:432-488). Reader fills both (i1,i2),(i2,i1), skips entries with 2n+l > N2max, j > Jmax (NNForce.F90:512-533). Z=-1 pp, 0 pn, +1 nn. Parity P = (-1)^L. [V] I parsed it (srg.py `load`).
   - Channel state set per (J,P,S,Z): all (n,l) with 2n+l <= N2max, l allowed by J,S,P (l = J+-1 coupled for S=1, otherwise l=J). For (J=1,P=+,S=1,Z=0): l=0: n=0..50 (51 states), l=2: n=0..49 (50 states): dimension **101**. [V] (counted from file)
   - Tz=0 channel is the isospin-symmetrized pn state (|pn>+(-1)^(l+S)|np>)/sqrt2 (OperatorDefinitions.F90:1334-1335 comment). For (J=1,P=+,S=1) only T=0 exists so the Z=0 block is the pure 3S1-3D1 deuteron channel. [V]
2. **Add Coulomb** (only if `coul=.true.` which is the **default**, NuHamilInput.F90:248, and `evolve_coulomb=.true.` default, :250): before SRG, in channels with Z=-1 only (pp), `V_pp += sum_r w_r R_{n1 l1}(r) R_{n2 l2}(r) (hc/alpha_em)/r` with alpha_em=137.035999 (hc=197.32705), 500-point Gauss-Legendre on [0,40] fm, HO oscillator parameter `a = 0.5*m_proton*hw/hc^2` (NNForce.F90:300-337, 275-283; MyLibrary.F90:10). Called at NNForce.F90:239-242 *before* `renorm_ho_space`. So pp Coulomb IS evolved by default. The Input_a.dat does not set `coul`, so for the example pp channels include Coulomb in the flow. [V code-read; Coulomb ME details belong to another slice] Bare path (renorm='bare'): Coulomb added after (NNForce.F90:249-256), no SRG.
3. **SRG per channel** with `H = T + V` (see section 2). Every channel with n>=1 states, all J <= J2max_NNint (=8 default, NuHamilInput.F90:224; header Jmax=8 in the file) is evolved independently, **in the full N2max=100 relative space (NOT truncated by emax/e2max before the flow)**. [V]
   - In `trans_to_lab_op` the spin space is `relspin%init(hw, params%N2max, params%J2max_NNint)` (TwoBodyManager.F90:~165): N2max=100, J<=8. [V]
4. `V_evolved = H_evolved - T` (HOSRGChannel: `v = sol%h - t`, NNForce.F90:~734-736). The unitary `u = sol%U` is also built (see section 4). [V]
5. **Truncation** (after the flow): `TwoBodyRelOp` over `rel%init(hw, N2max, Jmax2)` is filled from the spin-basis V (`SetTwoBodyScalarChanSpinBreaking`, TwoBodyRelOps.F90:343-386: entries with j > vho Jmax(8) are left at zero), then `oprel%truncate(rel_for_trans)` with `rel_for_trans%init(hw, e2max, Jmax2)` (TwoBodyManager.F90:~201-202; TruncateModelSpace TwoBodyRelOps.F90:664-716): simple sub-block copy keeping states with 2n+l <= e2max (here 8) — i.e. a *plain submatrix cut of the evolved V*, no re-diagonalisation / re-orthonormalisation. Then T-coefficients to lab frame (other slice). [V]
   - Quirk: with e2max=8, relative states up to 2n+l=8 give J up to 9, but V for J=9 is zero because the SRG/NN space only has Jmax=J2max_NNint=8. [V code-read; effect not numerically checked]
6. Output file (`TwBME-HO_NN-only_..._srg2.0_hw16_emax4_e2max8.me2j.gz`) contains the NN interaction V only ("NN int." header, TwoBodyLabOps.F90:2305); **kinetic energy is NOT added back at this stage** — T_rel is subtracted inside HOSRGChannel and never re-added. The generator/flow T is only an internal object. [V]

## 2. Generator, flow equation, s vs lambda

- Flow in matrix form (Renormalization.F90:173-178 inside `h_equation`):
  `eta = G H - H G`,  `dH/ds = eta H - H eta = [[G,H],H]`.
- `G` (= `Gen`) for default `srg_generator="kinetic"` (NuHamilInput.F90:235) is the **relative kinetic energy matrix in the HO basis**, built via `CalcMERel(Gen,...)` -> `get_two_body_me_pn` 'kinetic' (OperatorDefinitions.F90:1130-1134, 1372-1378): diagonal in J, S, Z, **diagonal in l**, tridiagonal in n:
  ```
  T(n,l; n,l)   = (2n + l + 3/2) * hw/2
  T(n,l; n+1,l) = sqrt((n+1)(n+l+3/2)) * hw/2     (symmetric)
  ```
  (two_body_kinetic, OperatorDefinitions.F90:2112-2125). Here hw = the HO frequency in MeV, with *no* mass dependence (this is the HO-basis matrix of p^2/(2 mu) = Trel; exact in the HO basis for any mass because b_rel is tied to hw via mu — the mass enters only through the HO length used when making V). The same expression (same hw) is used for `T` used in H=T+V (NNForce.F90:681-697) so G == T exactly for 'kinetic'. [V]
  - The matrix has the same tridiagonal structure in every channel; for a coupled channel (3S1-3D1) it is block diagonal in l with no S-D coupling. [V]
  - The relative kinetic energy is exactly p^2/(2 mu) with 2n+l+3/2 spectrum (tested: bare spectrum below gives -2.2246 for the deuteron with this T). [V]
  - Other generators (any `OperatorDef` name reaching `CalcMERel`, e.g. HOHamil, hopot, with l/J cutoffs, OperatorDefinitions.F90:1126-1160) exist; only 'kinetic' used/verified here. Note `Gen%InitOpDef(type,.true.)` pn-form; `hw` taken from the module-level `parameters%hw` (OperatorDefinitions.F90:60-70). [U for non-kinetic]
- **s vs lambda** (NNForce.F90:645-650, 644): with `pn_same_mass=.true.` (default, NuHamilInput.F90:251) for ALL channels `m_red = m_red_pn = m_p m_n/(m_p+m_n)` with m_p=938.27231, m_n=939.56563 MeV (MyLibrary.F90:11-14);
  ```
  lc    = lambda * hc / sqrt(2 * m_red)         [lambda in fm^-1, hc=197.32705 MeV fm]  -> units sqrt(MeV)
  alpha = (1/lc)^4  = s_final                   [MeV^-2]
  ```
  This is exactly s = 1/lambda^4 [fm^4] converted to energy units (T_E = (hbar^2/2mu) k^2, so s_E = s_k /(hbar^2/2mu)^2). For lambda=2.0: lc=12.879608692..., **alpha = 3.634037e-5 MeV^-2**. [V]
  - If `pn_same_mass=.false.`: m_red = m_p/2 for Z=-1, m_red_pn for Z=0, m_n/2 for Z=+1 (NNForce.F90:640-645). [V code-read]
  - The ODE variable s is integrated from 0 to alpha in a single `dvode_f90` call, `start_point=0`, `itask=1`, `istate=1` (Renormalization.F90:116-122); only the final state is used, there are no intermediate outputs. [V]
- All channels use the same alpha (when pn_same_mass); no channel subsets: loop `do ich = 1, NumberChannels` (NNForce.F90:640). Channels with n<1 are skipped. No test on J, S, L; all Z evolved. [V]
- `Nmax_srg_edge` (default -1): "Neumann boundary" — rows with 2n+l >= Nmax - Nmax_srg_edge are collected in `rhs_zero_term` (NNForce.F90:709-725). With default -1 the criterion 2n+l < Nmax+1 is always true so `n_zero=0` and the plain flow is used. Even if set >=0: `sol%init(h,'Hflow')` hard-codes mode 'Hflow' so `SRGFlow` dispatches to `SRGHflow` and **ignores rhs_zero_index** (Renormalization.F90:63-81; ModifiedSRGHflow is dead code in this path). [V]

## 3. ODE solver (SRGHflow)

- State vector: **lower triangle packed row-wise**, `vec(i(i-1)/2 + j) = H(i,j), 1<=j<=i<=n`, 1-based (Renormalization.F90:107-115). Length `n_ode = n(n+1)/2` (n=101 -> 5151). In numpy: `np.tril_indices(n)` order (row-major lower triangle) reproduces this ordering. The RHS fills the symmetric H from the packed vector, computes `eta=GH-HG` and `rhs = eta H - H eta` by 4 DGEMMs, and packs the lower triangle of rhs (:156-189). [V]
- Initial condition `H(0) = T + V_bare` (full Hamiltonian, symmetric). `H` evolved, not `V`. [V]
- Solver: `dvode_f90` with `set_opts(method_flag=10, abserr=atol, relerr=rtol, mxstep=50000)` (Renormalization.F90:119), defaults `atol=rtol=1.d-8` (:27-28, not overridden by any caller: `sol%init(h,'Hflow')` NNForce.F90:733). [V]
  - MF=10: Adams-Moulton (non-stiff), **functional (fixed-point) iteration, no Jacobian** (MITER=0) (dvode_f90_m.f90:655-730 doc of MF; code 4685-4780). Max order default 12 for Adams, error test: weighted RMS norm with EWT_i = rtol*|y_i|+atol over the 5151 packed components (scalar tolerances), step-size and order variable, h0 chosen automatically, no hmax/hmin/tcrit set, `mxstep=50000`, MXHNIL default. [V for MF/tolerances/mxstep; U for VODE internal defaults — standard DVODE.]
  - `istate` is **not checked after the call**; if mxstep were exceeded the code silently continues with a partially integrated H. (Not an issue for lambda=2.0: ~hundreds of steps/derivative calls.) [V]
- After the solve: `this%h` is unpacked; `U` is built as `sbare%vec * seff%vec^T` after diagonalising bare and evolved H with sign alignment of each eigenvector so `<bare_i|evolved_i> >= 0` (Renormalization.F90:138-149) (assumes eigenvector ordering is preserved; sorted ascending eigenvalues). U is used only to evolve *other* operators (`oprel%UT(U,U)`), not for the NN-only V. [V]
- Parallelism (omp) only in pack/unpack; no effect on results beyond DGEMM roundoff.

## 4. Storage of the evolved interaction

- Not stored in the SRG routine other than in `this%H` (full evolved H, symmetric, size n x n) and `this%U`. `HOSRGChannel` overwrites `vnn%MatCh(ich)` with `V_evolved = H_evolved - T` and `Trs%MatCh(ich)` with U (NNForce.F90:733-737). [V]
- The *bare* relative V file is the only on-disk relative file in this path. An evolved relative file is written only if `fn` is passed to `SetNNForceHO` (`write_nn_ho`, NNForce.F90:243), which `trans_to_lab_op` does not do. The separate `output_nn_file != "none"` branch (`nn_srg_analysis`, TwoBodyManager.F90:60-63) is a different analysis mode. [V for this path; U for `nn_srg_analysis`]
- Hcm / Lawson: none in the NN SRG step. `HCM` operators are built as separate lab ops (TwoBodyLabOps.F90:1515-1608; `ReducedToNonReduced` for HCM at TwoBodyManager.F90:~114) and, when requested as operator 'HCM', the relative operator is evolved by `oprel%UT(U,U)` (TwoBodyManager.F90:~190). Nothing Lawson-like touches the NN Hamiltonian file here. [V code-read]
- Bare path: `renorm='bare'` skips `renorm_ho_space` (NNForce.F90:234 condition) -> V = file contents (+Coulomb for pp if coul) truncated to e2max. Also `renorm_space2='mom'` gives no HO SRG. [V]

## 5. Numerical validation (3S1-3D1, J=1 P=+ S=1 Z=0, N2max=100, hw=16)

Script `srg.py`: builds T (tridiagonal, hw/2 factors) + V from the bare file, packs the lower triangle, flows to alpha with `scipy.integrate.ode('vode', method='adams')` (rtol=atol=1e-8, nsteps=50000, order 12, no Jacobian = same VODE algorithm as MF=10, tolerance semantic equal; scipy's VODE is the original F77 DVODE) and others.

| quantity | result |
|---|---|
| dimension | 101 (51 S-wave + 50 D-wave) |
| lowest bare eigenvalue (deuteron) | **-2.2245754 MeV** (also -2.2245754 after SRG at lambda=2.0, in a 1e-11 level with tight tol) |
| next eigenvalues | 0.535451, 1.284589, 2.054320 MeV |
| lc, alpha | 12.87960859 sqrt(MeV); 3.6340370e-5 MeV^-2 |
| eigenvalue invariance, VODE(Adams) rtol=atol=1e-8 | max abs change over all 101 eigenvalues = **1.0e-7 MeV** |
| eigenvalue invariance, tight tolerance (DOP853 1e-12, Radau 1e-10, LSODA 1e-10) | 2.5e-12, 2.7e-12, 1.0e-10 MeV |
| difference of H between VODE(1e-8) and the 1e-12 reference | max 2.07e-7 MeV (the solver error of NuHamil's own settings) |
| other channels (DOP853 1e-12): 1S0 pn (J0,Z0), 1S0 pp (J0,Z-1), 3P0.. (J1,P-,S1,Z0), 1D2.. (J2,Z0) | eigenvalues invariant to < 3e-12 MeV |
| evolved V(0,0) (3S1) | -10.7100 MeV vs bare -5.4930 (matrix elements in HO basis grow in low-N as high-momentum decouples) |
| far-off-diagonal decoupling (S-wave, |i-j|>=10) max |H_ij| | 0.120 evolved vs 9.62 bare |
| spectral radius of the RHS Jacobian (power iteration, MeV^2 per unit s) times alpha | **~80** at both start and end |

Deuteron: -2.2246 MeV (N3LO EM500, bare file with hw=16, N=100 truncation) as expected. [V]

Stiffness: rho(J)*alpha ~ 80 (dominant frequency ~ 2 x (E_max)^2 type scale with E_max ~ 1.5 GeV in the truncated basis and alpha tiny). This is mildly stiff only: explicit RK45/DOP853 reach 1e-10 in 350-520 RHS evaluations (0.1 s for n=101), and implicit BDF/Radau (with finite-difference dense Jacobian of 5151^2 entries) are 200-500x SLOWER and not needed. VODE-Adams with functional iteration is therefore adequate at lambda=2.0, and larger N or smaller lambda (alpha rho ~ N^2 scaling; e.g. lambda=1.5 gives alpha x3.2) could eventually stiffen it. [V for lambda=2.0, N=100; U for other cases.]

## 6. Recommendations for the JAX reimplementation

- Per channel dense symmetric ODE `dH/ds = [[T,H],H]` (T tridiagonal-in-n, diagonal-in-l). Integrate over the full symmetric matrix in JAX (batch by channel padding or per-channel loops; matmuls are n^3 with n<=~101-ish per channel, J<=8, 3 Z values -> several hundred small ODEs). The packed lower-triangle form of the Fortran is only an implementation detail; error control using the full matrix norm vs. the packed-triangle RMS differs by a negligible factor (off-diagonals counted twice).
- Use diffrax `Dopri8`/`Tsit5` with `PIDController(rtol=1e-10, atol=1e-10)` (or float64 RK with fixed ~ 200 steps). Matches the VODE (rtol=atol=1e-8) result to ~2e-7 MeV in matrix elements and reproduces eigenvalues to 1e-11 vs. exact. If you need to match *NuHamil's own numbers*, expect NuHamil to be off from the exact flow by up to ~2e-7 MeV in V elements (observed difference vs. converged solution), so a test tolerance of ~1e-6 MeV (abs) on the evolved relative V is appropriate, and 1e-6 relative on lab ME files after transformation likely.
- Enable `jax_enable_x64`. Alternative: a one-step exact solution is not available; but a matrix-exponential Magnus/Omega flow is not what NuHamil does here.
- Do the flow in the full N2max=100 space, THEN cut to e2max=8 (submatrix). Cutting first changes results (SRG induces couplings through high-N states).
- Rely on eigenvalue invariance (to 1e-8 MeV) of each channel as a unit test; and -2.22458 MeV for 3S1-3D1.

## 7. Open questions / unsure items

1. Whether the Input_a reference output was generated with Coulomb in pp (default `coul=.true.`, `evolve_coulomb=.true.`): code says yes, evolved. Needs comparison against the reference me2j for pp ME. [U]
2. J=9 relative channels (L=8,S=1, Z=0) have zero V because J2max_NNint=8 although e2max=8 allows J=9 — probable intentional/no impact; check against reference output. [U]
3. Non-kinetic generators (`srg_generator` other values) not analysed beyond reading the dispatch. [U]
4. Exact DVODE step statistics/internal defaults (initial step, safety factors) not replicated; scipy `vode adams` should be the same algorithm but step sequences may differ by ULPs. [U]
5. No evolved relative file exists to compare numerically against NuHamil: the available reference is the lab-frame me2j; a final check requires Talmi-Moshinsky (other slices). [U]
