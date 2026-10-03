# Spec 5c: the 3N Hamiltonian in the Jacobi basis and its SRG evolution

Source root: `/home/soham/Projects/nuclear-physics/NuHamil-public/src` (abbrev. `src/`). Citations are `file:line` relative to
`src/`. Status tags: **[V]** = verified (code read and/or reproduced from the reference runs), **[U]** = unsure.
Scope: how NuHamil builds H = T + V_NN + V_3N in the antisymmetrised 3N Jacobi HO basis, evolves it with the SRG, and what
it stores. The Jacobi basis itself (states, CFP), the genuine local N2LO force and the Jacobi-to-lab transform are other
slices. Spec 2 (`docs/spec-2-srg.md`) describes the NN SRG that is already implemented; this document reuses its conventions.

Reference runs (both made for this spec with the `nuhamil` image, hw=16, lambda=2.0, N3LO_EM500, e3max=4, N3max=8,
ramp `flat8`, local N2LO 3NF with Lambda=400 for run A):
- A: `<scratch>/nnn/` = `Input_3n_e4.dat`, `log_3n_e4.txt` (24 channels J<=11/2, 385 s wall, 16 threads), plus `ops/NNN-full_*.bin`.
- B (made here): `<scratch>/nnn_ind/`, same input with `genuine_3bf=.false.`, `jmax3=5`, `file_name_3n=nnn_ind_e4...` (12 channels,
  48 s), plus `log_ind.txt` and `ops/NNN-ind_*.bin`. B is the NN-only (induced 3N) oracle; recipe in the last section.
`<scratch>` = `/tmp/claude-1000/-home-soham-Projects-nuclear-physics-nuhamil-alt/b6aad67b-aa74-41b6-a308-3216b5d25052/scratchpad`
(session scratch, not permanent: regenerate with the recipe at the end).

## 0. Big picture (one paragraph)

For each 3N Jacobi channel (J, P, T) NuHamil builds a dense symmetric matrix in the antisymmetrised Jacobi HO basis
(N = 2 n12 + l12 + 2 n3 + l3 <= Nmax_ramp(J)),

    H0 = T_rel3 + 3 * V_NN(bare, embedded) + V_3N(bare, genuine)

evolves it with dH/ds = [[T_rel3, H], H] to s = lambda^-4 (in MeV^-2) with an adaptive Adams integrator (VODE, rtol=atol=1e-8),
and stores

    V3(channel) = H(s) - T_rel3 - 3 * V_NN(evolved in 2-body space, embedded)        ("NNN-full", or "NNN-ind" without genuine 3NF)

i.e. everything that is three-body after subtracting the embedded 2-body-evolved NN force. The Hamiltonian operator of the
`me3j` file is exactly this V3 (cut to N <= e3max, then transformed to the lab frame). The NN part of the Hamiltonian is NOT
in the me3j file; it goes to the me2j file, built by the separate rank=2 run. [V]
`src/ThreeBody/NNNForceHOIsospin.F90:78-168` (`set_nnn_force_ho_isospin`) is the whole recipe; the SRG is `:170-216`.

## 1. Where it is called, channels, ordering

- Entry for the lab file (`trans2lab=.true.`, the default path of `NuHamil_3BME.py`): `TMTransFunctions.inc:12-26` loops over all
  channels with t = 1,3 (T doubled), j = 1,3,..,jmax3 (J doubled), p = +1,-1, in that order (t outermost), `nch = ((jmax3+1)/2)*4`
  (`:14`), and calls `calc_each_channel` (`:70-133`) under `parent_child_procedure` (MPI over channels; OpenMP inside). [V]
- Per channel: `Nmax = GetRampNmax(J, ramp)` (`:71`) fixes the *flow space*; the Jacobi basis is built (or read from the `cfp/`
  cache, `ThreeBodyJacChanIso.F90:374-440`), then `InitNNNForce`, `SetNNNForce` (`NNNForceHOIsospin.F90:25-76`). If both output
  files `ops/NNN-{full,ind}_j..._srg<lambda>_hw<hw>.bin` and `ops/UT_...` exist the channel is skipped (`TMTransFunctions.inc:119-121`;
  `NNNForceHOIsospin.F90:51-62`): caching is by file name only, so changing e.g. `Nmax_srg_edge` or generator without changing
  the name reuses stale files. [V]
- The file name carries: NNN-full if `genuine_3bf` and J <= `J3max_initial_3nf` (default = jmax3), else NNN-ind; Nmax of the ramp,
  regulator, LECs, `_srg2.0`, `_hw16` (`ThreeBodyJacOpsChanIso.F90:1074-1127`). `hw_target` (frequency conversion) is applied only
  when the file is read, never in the flow (`:275-306`, `ThreeBodyJacOpsIso.F90:~395-405`). [V]
- File format of the stored operator: `int32 n`, then `n x n` float64 (Fortran column-major, symmetric) =
  `WriteThreeBodyJacScalarChan` (`ThreeBodyJacOpsChanIso.F90:134-143`). I checked sizes for run B: 32772 bytes = 4 + 64^2*8. [V]
- The `UT` file (`U = sbare.vec * seff.vec^T`, from the SRG solver) is also written; it is only for evolving *other* operators
  and is not needed for the Hamiltonian. [V]

## 2. The Hamiltonian in Jacobi space

### 2.1 Basis and the two spaces: non-antisymmetrised (NAS) and antisymmetrised (AS)

- NAS states `|(n12 l12 s12 j12 t12)(n3 l3 j3); J T>`, restricted to l12+s12+t12 odd (i.e. already antisymmetric in particles 1,2),
  N = 2n12+l12+2n3+l3 <= Nmax, parity (-1)^(l12+l3) = P (`ThreeBodyJacChanIso.F90:498-526`). One shared HO frequency hw for both Jacobi
  coordinates (normalised Jacobi coordinates), so a state is labelled by quantum numbers only. [V]
- AS basis: eigenvectors with eigenvalue 1 of the antisymmetriser A = (1 + P+ + P-)/3 (matrix in the NAS space, computed from
  Racah/Moshinsky algebra, `:546-591`, `:617`). A conserves N, so the CFP matrix is block diagonal in N: `cfp(N)` of shape (n_AS(N), n_NAS(N)).
  `GetCFPMat` (`:791-812`) assembles the full (n_NAS x n_AS) matrix with orthonormal columns (C^T C = 1). Blocks of N with no AS
  state are dropped (my NAS count is one higher than NuHamil's in two channels, J=3/2+ T=1/2: 305 vs 304 and J=1/2+ T=3/2: 95 vs 94).
  Dimensions: NAS ~ 3 x AS (190/64, 304/101, 340/113 in the log). The AS basis within a given N is an arbitrary orthonormal basis of
  the degenerate eigenspace (eigenvector gauge of a dense symmetric solver), so stored matrices are only defined up to an orthogonal
  rotation inside each N block; compare gauge-invariant quantities or rotate with NuHamil's own `cfp/*.bin`. [V code; U for the file layout of `cfp` bins]
- AS index order is N ascending (`set_indices`, `:394-460`). Therefore "cut to N <= N' " is the **leading principal sub-block** of
  the matrix (`ReadThreeBodyJacScalarChan`: `this%m = scalar%m(:n_cut,:n_cut)`, `ThreeBodyJacOpsChanIso.F90:275-306`) with no
  re-orthonormalisation. [V]

### 2.2 Embedding rule (general)

All the scalar two-body operators (NN force, kinetic) are embedded the same way: build the matrix W in the NAS space for the pair (1,2)
only (identity in the spectator quantum numbers n3, l3, j3 and no dependence on J or on T beyond t12), then antisymmetrise by
C^T W C, then multiply by the number of equivalent pairs. `NonAsym_to_Asym` (`ThreeBodyJacOpsChanIso.F90:1031-1072`) does the two
dgemms C^T (W C). No extra normalisation factors appear because the NAS states are already antisymmetric in (12), so
the pair matrix element is the usual antisymmetrised relative ME (no sqrt(2) factors). Selection rules in
`set_three_body_from_two_body_scalar_lim_isospin` (`:371-417`): n3, l3, j3 equal; |j12-j12'| <= rank (0), parity of l12 + l12',
t12 triangle. Matrix elements come from the 2-body isospin relative operator (`get_me12_from_qns`, `:656-691`): stored
elements are used for j12 <= Jmax(rel space) and l12 <= Jmax+1, otherwise zero for interactions (`mode_int=.true.`) and analytic for
kinetic. [V]

### 2.3 Relative kinetic energy of three particles

- Two-body relative kinetic operator in the HO basis at hw: T(n,l;n,l) = (2n+l+3/2) hw/2, T(n,l;n+-1,l) = sqrt(.)*hw/2 (same
  tridiagonal formula as in spec 2; `OperatorDefinitions.F90:1130-1134`, `two_body_kinetic` at `:2112-2125`, and the identical local copy
  `set_three_body_kinetic_isospin` kinetic() at `ThreeBodyJacOpsChanIso.F90:521-531`). No mass enters: hw is the only scale. [V]
- Embedding: `T_jac = 2 * C^T (T_rel2(12) (x) 1_3) C` (factor 2: `:462-467`, selected by `OpName == 'kinetic'`; the 'kinetic' operator is
  not channel-restricted, `OperatorDefinitions.F90:269-275`, so it takes the non-"lim" branch `:419-470` and may fall back to the analytic
  ME for J12 above the stored range). The factor is 2, not 3, because T_rel(3) = (2/3) sum_{i<j} t_ij^rel; on antisymmetric states
  every pair gives the same value. I rederived it: sum_{i<j} (p_i-p_j)^2/(4m) = (3/2) T_rel, so T_rel = (2/3) sum t_ij and 3 * (2/3) = 2. [V]
  Equivalent form (code left unused, `:472-519`): T = t(n12,l12) + t(n3,l3) on NAS then antisymmetrise (same hw/2 formula on both
  Jacobi coordinates; this equality holds because the N-truncation commutes with permutations). T mixes N and N+-2 (not N-diagonal)
  and is block tridiagonal in N, dense inside blocks. [V by derivation, U numerically]
- The same dense T_jac is the SRG generator for the default `srg_generator="kinetic"` (`NNNForceHOIsospin.F90:150-152`). For another
  generator `set_srg_generator` (`:1030-1050`) embeds the 2-body generator with the generic factor 3 and then rescales by 2/3, giving
  an effective factor 2 as for T. [V code, U not run]

### 2.4 NN force in Jacobi space

- Two versions are built at the same hw and N2max (both on the 2-body spaces `relspin%init(hw, N2max, J2max_NNint)` and
  `rel%init(hw, N2max, Jmax2)`, N2max = 100 default, J2max_NNint = 8, Jmax2 = 12; `NNNForceHOIsospin.F90:103-105`,
  `NuHamilInput.F90:221-224`):
  1. `vnn_bare`: `input%renorm = "bare"` (`:109`) -> embedded as `vnn_jac` and enters **H0** (initial condition).
  2. `vnn_sub`: `params` as given (renorm = srg) (`:114`) -> embedded as `vnn_sub_jac`; the evolved 2-body NN force, subtracted at the end. [V]
- Embedding factor: `3 * C^T W C` (scalar-lim branch, `ThreeBodyJacOpsChanIso.F90:361-364`, since 'NNint' is channel-restricted,
  `OperatorDefinitions.F90:257-267`). Three pairs, equal on antisymmetric states. [V]
- How the 2-body isospin NN force is made (`TwoBody/NNForceIsospin.F90:60-222`):
  - Start from the pn-formalism HO matrices (pp, pn, nn per (J,P,S)), `SetNNForceHO` with `renorm="bare"` when `renorm_space2='ho'`
    (default; `:~149-153`, `NNForce.F90:117-250`). The bare relative HO matrix is cached in `NNint_A2_rel_<NNInt>_bare_Nmax<N2max>_hw<hw>.gz`
    (format of spec 2). The file stores 7 significant digits (`es18.6`), spec 2 / CLAUDE.md note ~2.5e-6 difference to the `.bin`. In the
    first channel of a fresh run (file absent) the full-precision matrices are used; every later channel reads the 7-digit file
    (`NNForce.F90:140-205` vs `:214-218`; log of run A prints "Not found ..." only in the first channel). So NuHamil's 3N output is
    not channel-independent at 1e-6 level. [V code]
  - T=0 channel: the pn (Tz=0) matrix. T=1 channel: **equal-weight charge-independent average** (pp + pn + nn)/3, weights cpp=cnn=cpn=1/3
    (`NNForceIsospin.F90:~85-90`, `:178-196`; weights depend on (T, Tz) only if T_total/Tz_total are passed, which is done solely in
    the eigenvalue solver `solve_three_body_hamil_isospin`, `ThreeBodyManager.F90:~729`). The EM500 file has isospin-breaking in 1S0, so this
    averaging matters; the me3j is isospin symmetric by construction. [V code]
  - **Coulomb is not included**: `ThreeBodyManagers` forces `coul=.false.` when `trans2lab` (`ThreeBodyManager.F90:~53-58`, printed
    "Then, 'coul' is turned to false" in both logs). In the non-lab path (eigenvalue solver) `coul` keeps its default `.true.` and pp
    Coulomb would be averaged in (U: untested). [V for lab path]
  - SRG (when `renorm != bare` and `renorm_space2 == 'ho'`, `NNForceIsospin.F90:~207-210`): `renorm_ho_space_isospin` (`:298-328`) -> `HOSRGChannel`
    (`:330-415`) per isospin channel in the **full N2max = 100 relative space**: H = T + V, generator = relative kinetic (`CalcMERel`), same
    `SRGHflow` as spec 2, V_evolved = H_ev - T. alpha uses `m_red_pn` (`:~310-313`): alpha = (lambda hc / sqrt(2 m_red_pn))^-4, with hc =
    197.32705, m_p = 938.27231, m_n = 939.56563 (`MyLibrary.F90:8-14`). Numerically identical to the pn-form NN flow of spec 2 for the
    same channel structure: alpha = 3.634037e-5 MeV^-2 at lambda=2.0. [V]
  - If `renorm_space2 = 'mom'` the bare/evolved pair would instead come from a momentum-space renormalisation; not analysed. [U]
- Number of 2-body isospin channels solved per call: (J,P,S,T) with J<=8: S=0 (9) + S=1 uncoupled L=J (8) + coupled L=J+-1 (9) = 26
  (matches 648 SRG calls = 24 x 27 in run A, 27 = 26 + the 3-body flow; 324 = 12 x 27 in run B). **These 26 SRGs and the NN projection
  are redone for every 3N channel** (the code does not share them across channels). 2-body flows cost 43 s of the 48 s of run B
  (`log_ind.txt` profile: `SRG H flow` 43.6 s, 324 calls); a JAX version should do them once (they depend only on hw, lambda). [V]

### 2.5 The genuine 3N force

- `if(params%genuine_3bf) call set_genuine_three_body_interaction(v3n_jac, params)` (`NNNForceHOIsospin.F90:138-140`), dispatching on
  `NNNInt` (default ChEFT_N2LO, `:220-231`). For N2LO (`set_nnn_interaction_chEFT_n2lo`, `:233-262`): the local-regulator HO partial-wave
  matrix `work` in the NAS space is obtained from `set_nnn_int_chEFT_n2lo_isospin_local` (cached in `ops/N2LO_3NF_HOPW_...bin`, LECs
  applied on reading, `:406-441`) and then V3N = C^T work C (`:259-260`). No factor 3 here: the stored NAS matrix is the
  full V_3N already (the sum over the three cyclic permutations is inside the partial-wave code; the slice on the local N2LO module must
  confirm this normalisation). [U: factor convention belongs to the force slice]
- For the semilocal/LNL regulators an additional non-local regulator is multiplied in HO space (`multiply_non_local_regulator_hospace`, `:264-399`;
  500-point Gauss mesh in momentum, pmax = 25*6/sqrt(hw)). [V code, not needed for the local N2LO target]
- The genuine force is **bare**: it is added to H0 and evolved by the 3-body flow only; it is never evolved in a smaller space. [V]
- J > `J3max_initial_3nf` channels have no genuine force and are written as `NNN-ind` (`ThreeBodyJacOpsChanIso.F90:1099`, `jmax3` default
  = j3max_initial; whether `set_nnn_int..._local` returns zeros above it: U).
- Run A / B: lowest J=1/2+ T=1/2 eigenvalue of H0: -1.349102 MeV with the 3NF400, -1.093825 MeV without. [V]

## 3. The SRG for the 3N system

### 3.1 What is evolved

`h = T_jac + vnn_jac + v3n_jac` (`NNNForceHOIsospin.F90:141`), diagonalised for the printout "Eigen values of original 3-body H" (`:143-148`),
then (only if `renorm == 'srg'`) evolved in place by `three_body_srg_evolution` (`:156`):

- **Everything is evolved in the 3-body space**: bare NN (embedded) + bare 3NF + T. The 2-body-evolved NN is not used as the
  starting point; it is only used afterwards for the subtraction. So V3 contains the induced 3N force plus the evolved
  genuine 3N force, and, in a truncated space, also the effect of the finite ramp on the NN part. [V]
- Order of operations (`:78-168`): (1) NN bare and sub at N2max=100 in 2-body space; (2) kinetic and bare NN embedded into the 3-body
  AS space at Nmax=ramp(J); (3) V3N bare; (4) H0 and eigenvalues; (5) flow; (6) eigenvalues after; (7) `this%DMat = h - T_jac - vnn_sub_jac`
  (`:158`); (8) write V3 and U. [V]

### 3.2 Generator, flow equation, parameter

- Generator G = T_jac (kinetic) = the same dense embedded T as in H (`:150-152`). Flow: `Renormalization.F90:SRGHflow` (shared with NN, same routine):
  eta = G H - H G; dH/ds = eta H - H eta = [[G,H],H] with four DGEMMs on dense matrices (`Renormalization.F90:~156-189`).
  H is symmetric; the ODE vector is the packed lower triangle `n(n+1)/2` (`:107-115`). [V]
- Flow parameter: `lc = lambda hc / sqrt(2 m_p m_n/(m_p+m_n))`, `alpha = lc^-4` (`NNNForceHOIsospin.F90:185-186`): s_final in MeV^-2, lambda in
  fm^-1; the same alpha as for NN (**not** rescaled for three particles; the "T" in the generator already carries the 3-body
  normalisation, as T = t12 + t3 in Jacobi coordinates). alpha(lambda=2.0) = 3.634037e-5 MeV^-2; for lambda=1.8: 5.54e-5 (derived, alpha ~ lambda^-4). [V]
  Single integration call from s=0 to alpha, only the final point is used. [V]
- Integrator: `dvode_f90`, MF=10 (Adams-Moulton, functional iteration, no Jacobian), rtol = atol = 1e-8 (defaults `Renormalization.F90:27-28`;
  the commented alternative `atol=1e-6, rtol=1e-7` at `NNNForceHOIsospin.F90:204` is NOT used), mxstep = 50000, `istate` not checked
  (`Renormalization.F90:116-122`). Same VODE caveats as in spec 2 (matching NuHamil's own numbers is limited to ~1e-7 in matrix
  elements even with a converged integrator). Eigenvalue invariance in the log is at the printed precision (1e-6, see section 5). [V]
- After the solve the code diagonalises H0 and H(s), aligns eigenvector signs and builds U = S_bare S_eff^T (`:138-149`): two more full dense
  `eigh` calls per channel (cost O(n^3)), only for the UT file. [V]
- `Nmax_srg_edge` default -1 makes `n_zero = 0` (all states have EAS < Nmax+1, `:187-201`); even if >= 0, the solver's mode is hard-coded
  'Hflow' so `rhs_zero_term` is ignored (spec 2, section 2). No Neumann boundary is ever active. [V]

### 3.3 Space truncation: N3max, e3max, the ramp

- The flow space of channel (J,P,T) is all AS states with N <= `GetRampNmax(2J, ramp)` (`ThreeBodyJacobiSpaceIso.F90:167-195`):
  - `flatK`: Nmax = K for every J (`:177-180`). `flat40` is the default (`NuHamilInput.F90:266`).
  - `rampN0-j1-N1-j2-N2-...`: Nmax = N0 by default; for each pair (j_i, N_i), if the doubled J exceeds j_i the value becomes N_i; the later pair
    wins (`:182-193`). The J argument is the **doubled** J (code uses `j = 1,3,5,...`). Example from `NuHamil_3BME.py`:
    `ramp40-5-36-7-32-9-28-11-24` gives Nmax = 40 for J = 1/2, 3/2, 5/2; 36 for 7/2; 32 for 9/2; 28 for 11/2; 24 for all J >= 13/2. Other
    script option: `ramp40-7-38-9-36`: 40 up to 7/2, 38 for 9/2 (j=9 > 7), then 36 for J>=11/2 (note 9 > 9 is false, so 9/2 is 38).
    Care: the string is split on "-" and the pairs are read as (j, N): do not confuse the first number "40" with a pair. [V]
  - States above the ramp are simply **absent**: the Hamiltonian is the compression of H0 onto the space N <= ramp (no padding, no
    boundary term, no extrapolation). The NN force used inside is exact for those states (N2max=100 projection), so the
    truncation only limits the induced 3N operators. [V]
- `N3max` does **not** control the flow space in the lab-frame path. It is passed as `Nmax_for_file` and only checked to be >= the
  working space (`ThreeBodyJacobiSpaceIso.F90:~129-136`). It sets the Nmax of the Jacobi space in the eigenvalue solver (`ThreeBodyManager.F90:~729-770`,
  where it also acts as the HO-space size N_ls). In run A the ramp (`flat8`) coincides with N3max=8. [V]
- `e3max` (lab) sets the output cut: `jac%init(hw, jmax3, e3max, ramp, N3max)` (`TMTransFunctions.inc:~134-137`) reads each stored
  channel truncated to N <= e3max (`Nmax = min(e3max, Nmax_file)`, `ThreeBodyJacChanIso.F90:~750`). The stored matrix is therefore the leading block
  of the ramp-sized V3 (section 2.1). Example (run A): flow at N=8, used at N<=4. With e3max = 16 and ramp up to 40 the stored sub-block
  is only ~350-910 AS states while the flow runs at up to 11340 (my counts, section 6). `jmax3 = 2*e3max+3` in the script. [V]
- Consequence for a reimplementation: the physics of the output depends on the ramp (through induced 3N forces), so there is no
  "converged-in-N" reference; match the ramp, not N3max. Convergence in ramp is a separate study (the NN analogue is
  `docs/study-srg-truncation.md`: UV errors fall quickly with margin N_flow - e_cut; for 3N the published ramp is chosen for that purpose). [U numerically]

### 3.4 Induced 3N force definition

V_3N,ind+evolved = H_3(s) - T_rel3 - [3 C^T V_NN(s)^(2b) C]. Here V_NN(s)^(2b) is the 2-body flow (N2max=100) of the isospin-averaged
NN force and T_rel3 is the same T as in H. For lambda -> 0 (alpha = 0) the stored object reduces to the bare genuine 3NF (and
exactly 0 without genuine force, since the 3-body embedding of V_NN(0) and the subtraction cancel). In general the difference
H_3(s) - T - 3 C^T V_NN^(2b)(s) C is nonzero even without 3NF because the 3-body flow in the truncated Jacobi space is not the
embedding of the 2-body flow (the commutators generate 3-body operators); in run B the traces of this "induced" matrix are 164.0
(J=1/2+ T=1/2, n=64) and 125.1 (J=1/2-). Note the size at N<=8 is large because the 3-body ramp is tiny; it falls with the ramp. [V]

## 4. What NuHamil itself checks and prints

- Per channel (`NNNForceHOIsospin.F90:143-148`, `:211-215`): lowest five eigenvalues of H0 and of H(s), format `5f12.6`, to the log.
  No assertion, no tolerance: the user must read them. Also `solve_three_body_hamil_isospin` (`ThreeBodyManager.F90:695-809`, operator
  'hamil' without `trans2lab`) rebuilds H = T + V_NN(evolved at N3max) + V3(stored file) and prints the ground-state energy: that sum equals
  H(s) and so gives the same eigenvalues as "after SRG" (consistency check of the stored V3; U: not run).
- Counts printed: "Number of physical and orthonormal states: n_NAS n_AS" (`ThreeBodyJacChanIso.F90:~417`). An internal consistency error is
  raised if tr(A) is not an integer (`get_dim_as`, `:~537-543`, tolerance 1e-4) or if A has an eigenvalue not near 0 or 1 (`set_cfp`,
  warning only, `:~580-588`).
- The three-body matrix count and the profiler table at the end of the log.

Expected invariants for tests (our implementation): spectrum of H(s) == spectrum of H0 (to the integrator error), tr H and tr H^2
conserved, V3 symmetric (asymmetry in the stored files: 4e-15 to 7e-15), V3 = 0 for lambda -> 0 without 3NF, and independence of the
result of the order in which channels are processed.

## 5. Numerical checks against the oracle

Both runs' logs print H0 and H(s) eigenvalues, lowest five, 6 decimals.

**Invariance**: over all 24 channels of run A the lowest five agree before/after to <= 1e-6 (the printed precision): differences of 1e-6
appear in J=5/2+, 7/2+ (T=1/2) and J=1/2- (T=3/2); all others agree to the printed digits. In run B: -1.093825 vs -1.093826 (J=1/2+) and
13.679500 vs 13.679499 (J=5/2+ T=1/2). The VODE (1e-8) error of the NN flows was 1e-7 on all eigenvalues (spec 2).
Test tolerance for our flow: 1e-8 MeV with a tight Dopri8.

**Run B (NN only, genuine off): eigenvalues of H0 = H(s) (lowest five, MeV)**, from `log_ind.txt`:

| J, P, T | n_NAS, n_AS | eigenvalues (MeV) |
|---|---|---|
| 1/2 + 1/2 | 190, 64 | -1.093825, 9.401649, 13.750714, 15.352855, 19.418842 |
| 1/2 - 1/2 | 140, 47 | 8.351840, 10.231260, 20.391014, 24.630258, 25.215298 |
| 3/2 + 1/2 | 304, 101 | 12.450161, 13.754423, 15.380979, 16.378291, 19.305606 |
| 3/2 - 1/2 | 220, 73 | 8.874411, 9.935286, 20.417136, 23.136964, 25.101567 |
| 5/2 + 1/2 | 340, 113 | 13.679500, 15.427355, 16.182191, 28.631442, 29.008929 |
| 5/2 - 1/2 | 236, 79 | 8.600064, 23.104895, 23.237084, 23.697695, 24.960465 |
| 1/2 + 3/2 | 94, 31 | 14.155033, 19.212224, 30.256808, 32.615271, 33.873176 |
| 1/2 - 3/2 | 70, 23 | 10.232892, 25.685846, 27.410391, 28.611219, 43.828630 |
| 3/2 + 3/2 | 150, 49 | 16.013869, 18.253692, 29.387190, 32.454724, 32.766409 |
| 3/2 - 3/2 | 110, 37 | 10.054364, 25.506970, 26.957200, 27.275257, 28.318240 |
| 5/2 + 3/2 | 170, 57 | 16.007049, 18.499435, 29.124627, 32.653127, 32.749445 |
| 5/2 - 3/2 | 117, 39 | 23.685150, 26.649669, 27.195887, 28.835709, 44.767317 |

Run A (with 3NF400), H0 lowest eigenvalue, selection: J=1/2+ T=1/2 -1.349102, J=1/2- 8.365957, J=3/2+ 12.449326, J=3/2- 8.881818,
J=5/2+ 13.686849 (after: 13.686848), J=1/2+ T=3/2 14.155316 (full table in `log_3n_e4.txt`, 24 channels to J=11/2). These test T + V_NN(bare) + V_3N(bare)
in the CFP basis *before* any flow: pure linear algebra, and a first milestone independent of the SRG.

**Stored V3 matrices as gauge-invariant references** (`ops/NNN-ind_*.bin`, `ops/NNN-full_*.bin`; trace, Frobenius norm and extreme
eigenvalues are invariant under the orthogonal rotation freedom of the AS basis within N blocks, section 2.1):

| channel (n_AS) | NNN-ind: trace, Frobenius, emin, emax | NNN-full (3NF400): trace, Frobenius, emin, emax |
|---|---|---|
| 1/2+ T1/2 (64) | 164.007348, 43.904710, -1.023398, 28.585935 | 155.201973, 39.892640, -1.216865, 22.063988 |
| 1/2- T1/2 (47) | 125.065890, 35.086234, -0.547509, 18.503642 | 125.856210, 35.848369, -0.839748, 19.089868 |
| 3/2+ T1/2 (101) | 178.612698, 37.789276, -1.023043, 12.886912 | 179.231059, 38.116162, -1.046297, 13.469116 |
| 1/2+ T3/2 (31) | 50.609986, 19.209335, -0.494034, 13.416018 | 51.413089, 19.600421, -0.456159, 13.728555 |
| 5/2+ T1/2 (113) | 159.360254, 34.770729, -0.828921, 12.724094 | 159.583169, 35.065715, -0.969860, 13.305138 |

Accuracy of these references: limited by VODE rtol 1e-8 (elements ~1e-7), the 7-digit NN cache (1e-6, section 2.4) and, for run A, the
local 3NF accuracy. The "full" numbers also need the genuine N2LO matrices of the other slice. These are the quantities to compare once the
CFP basis and the 3N force exist. Without the 3NF, run B is the target for this slice alone.

Suggested staged tests (all gauge-invariant):
1. T_jac alone: the two embeddings agree, 2 C^T t12 C == C^T (t12 + t3) C (exact identity, no oracle needed; the diagonal of t12 + t3 is
   (N+3) hw/2 per state), and T_jac is block tridiagonal in N.
2. H0 eigenvalues (first row of the table per channel).
3. After-SRG eigenvalues = H0 eigenvalues (1e-8).
4. Trace, Frobenius norm and spectrum of V3 (table above) at the 1e-6 level, plus comparison with the ops bins after rotating with NuHamil's
   `cfp/cfp_A3_j*_Nmax8.bin` for element-wise tests (convention for the byte layout is U).
5. Lowest eigenvalue of T + V_NN^(2b)(s) + V3 (run `solve_three_body_hamil_isospin`) = H(s).

## 6. Cost structure at production sizes

All numbers below are my own counts from the NAS enumeration (rule in section 2.1, scratch script `<scratch>/s5c/count.py`; AS = NAS/3, which holds to
~1% in the log) for the `NuHamil_3BME.py` settings: ramp `ramp40-5-36-7-32-9-28-11-24`, e3max = 16, jmax3 = 35 (72 channels: J = 1/2..35/2, P = +-, T = 1/2, 3/2).

| J (P=+, T=1/2) | flow Nmax | n_AS (flow) | dense H (GB, float64) | n_AS stored at e3max=16 |
|---|---|---|---|---|
| 1/2 | 40 | 4260 | 0.15 | 350 |
| 3/2 | 40 | 8080 | 0.52 | 619 |
| 5/2 | 40 | 11340 (largest) | 1.03 | 792 |
| 7/2 | 36 | 10252 | 0.84 | 885 |
| 9/2 | 32 | 8186 | 0.54 | 914 |
| 11/2 | 28 | 5793 | 0.27 | 893 |
| 13/2 | 24 | 3556 | 0.10 | 835 |
| 21/2 | 24 | 2968 | 0.07 | 433 |
| 35/2 | 24 | 981 | 0.01 | 5 |

P=- channels are about 7% smaller, T=3/2 channels about half of T=1/2 (n_AS halves: 11340 -> 5670).

- Dense linear algebra: H, T (generator), eta, rhs are dense n x n float64. One RHS evaluation = 4 GEMMs, 8 n^3 flops. Summed over all 72
  channels sum(n^3) = 8.2e12, i.e. **6.6e13 flop per RHS evaluation of the whole set**; the largest channel alone 1.2e13. [estimate]
  The number of RHS calls is unknown for 3N (NN lambda=2.0: ~350-520 for Dopri8 at 1e-10, spec 2; VODE counts not printed). If
  100-1000: 7e15 - 7e16 flop. At a realistic 25-40 TFLOP/s FP64 on an H200 / GH200 about 3-45 minutes for everything on one GPU; on one
  48-core CPU node (~1-2 TFLOP/s) hours to a day. [estimate, U]
- Extra O(n^3) work per channel in NuHamil (not needed for the Hamiltonian): 2 `eigh` for the printouts (H0 and H(s)) and 2 for the U matrix;
  we can drop all four (keep one eigenvalue-only check for debugging), and drop U.
- Memory per channel: dense matrices 1.03 GB each at n = 11340; VODE keeps ~13 history vectors of the packed length n(n+1)/2 = 6.4e7 (0.5 GB each,
  ~7 GB total for Adams order 12) + 4 dense work matrices; NuHamil peaks at several GB per thread. A Dopri8 stage set in JAX is 13 dense stages =
  ~13 GB + work at n = 11340, fine on 80-141 GB devices; a fixed-step or low-storage scheme reduces it. T is block tridiagonal in N (about 3 of 20
  blocks non-zero), so two of the four GEMMs could use block structure (saves ~10-15%); not worth it initially. eta is antisymmetric and rhs symmetric,
  which can halve the work with tricks; also a later optimisation. [estimate]
- Suited to the GPU: every flow stage is four batched/stacked dense products; channels with equal padded size batch with `vmap` exactly as the NN flow
  (`kernels/srg.py`), but at n ~ 1e4 each channel fills a GPU on its own, so loop over channels (descending size) and keep only one or two resident.
  The basis assembly (Racah algebra for A, `anti_op_isospin`, eigendecomposition of A per N block, `cfp` cache), the NN projection and the 3NF HO matrix are
  the host/other-slice work; the embedding itself is just C^T W C (two GEMMs per operator, cheap).
- Measured, run A (24 channels, 16 threads, 385 s): `SetGenuineThreeBodyForce` 277 s (72%, of which ~7.5 s per channel is "precalculations" of the local 3NF),
  `SRG H flow` 98 s (25%, 648 calls: 24 x 26 two-body flows + 24 three-body), `Read from file` 7.8 s, 9j 3.1 s, Jacobi/cfp 0.2 s. Per-call 3-body flows here are
  tiny (n <= 113); the cost of the SRG at production sizes cannot be extrapolated from this log. In run B: 324 calls, 43.6 s of 48 s. [V]
- Redundant in NuHamil, worth removing in ours: the 26 isospin NN SRGs and projection are redone per 3N channel (section 2.4); the bare/evolved
  NN HO matrices depend only on (hw, lambda) and can be computed once and kept (a few hundred MB at N2max=100).

## 7. Open questions / uncertainties

1. Genuine 3NF normalisation and the cutoff of J: V3N = C^T work C without a factor (section 2.5): confirm in the force slice that `work` is the full V_3N
   (all three cyclic terms) and that J > J3max_initial_3nf gives zero (names `NNN-ind`).
2. Number of RHS calls and wall time for the real 3N flow (n ~ 1e4) is unknown; the 24-channel log has only n <= 113. Measure one large channel with our own
   flow before choosing the integrator (Dopri8 vs implicit); a stiffness check like spec 2 (spectral radius x alpha) is needed at n ~ 1e4. Expect T-dominated
   stiffness growing with the ramp (UV cutoff ~ sqrt(Nmax hw)).
3. Stored basis gauge: V3 matrices (`ops/*.bin`) and `cfp/*.bin` are in an arbitrary eigenvector gauge of A. Element-wise comparison requires reading the `cfp` files
   (layout: per N, `nphys, north, N`, NAS quantum numbers, then `north` rows of length `nphys`, `ThreeBodyJacChanIso.F90:720-785`); our code can choose its own
   gauge but must then rotate before comparing. The lab transform (other slice) is gauge-independent.
4. The 7-digit NN cache: NuHamil's first channel (file absent) uses full precision, later ones the 7-digit file (differences ~2.5e-6). Our reference outputs from a
   fresh run mix both; both of my runs started with the cache copied (run B) or generated (run A, first channel only). Plan tolerances for V3 at >= 1e-5.
5. Isospin-averaged NN (1/3 weights) with Coulomb off: the induced 3N force is defined relative to this NN. The user's me2j (pn form, Coulomb on, from the rank=2 run)
   differs from the NN embedded here, so the combination me2j + me3j contains a small inconsistency by construction (charge-dependent and Coulomb parts are not
   consistently evolved in 3N). That is NuHamil's convention (reproduce it, but note it). [U: I did not check how imsrg++ combines the files]
6. `renorm_space2 = 'mom'` and non-kinetic `srg_generator` paths, `hw_target != hw` conversion, and `renorm = 'ls'`, `'vlowk'` are outside this spec. (`hw_conversion` multiplies
   V3 by HO overlap matrices after reading, `ThreeBodyJacOpsChanIso.F90:917-960`.)
7. The log of run A states 24 channels only up to J=11/2 because `jmax3 = 11`; V3 for higher J is needed at e3max = 16 (jmax3 = 35): checks in the log are not available there.
8. Whether NuHamil's VODE error in the 3N flow (1e-8 relative) is visible in eigenvalue invariance at larger n: the log prints at 1e-6 only. To measure it, run a channel with
   tighter tolerances (needs a recompile; `Renormalization.F90:27-28`).

## 8. Reproduction recipe (oracle runs)

```bash
# in an empty directory D containing the NN relative cache or not (it is created on first use, ~1 MB)
cat > Input_ind.dat <<'EOF'
&input
  rank=3, hw=16, renorm="srg", lambda=2.0, emax=4, e2max=8, e3max=4, N3max=8, jmax3=5, ramp="flat8"
  NNInt="N3LO_EM500"
  input_nn_file="/root/NuHamil-public/input_nn_files/N3LO_EM500_kmax8_N100_Jmax8.bin"
  genuine_3bf=.false.
  file_name_3n="nnn_ind_e4.me3j.gz"
  lab_3bme_precision="double"
&end
EOF
podman run --rm -v "$PWD:/work:Z" -w /work nuhamil:latest NuHamil.exe Input_ind.dat > log_ind.txt   # ~50 s
```

(The one-line namelist form is the same as the multi-line one used in the scratch run.) Run A adds `genuine_3bf=.true., Regulator="Local",
RegulatorPower=2, LECs=-0.81,-3.2,5.4,-0.2,0.098, lambda_3nf_local=400` and takes ~6.5 minutes for jmax3=11 (the 3NF precalculation dominates).
Stored V3: `ops/NNN-ind_j<2J><p>t<2T>_Nmax8_srg2_hw16.bin` (`int32 n`, `float64[n,n]`); AS basis: `cfp/cfp_A3_j<2J>p<sign>t<2T>_Nmax8.bin`.
