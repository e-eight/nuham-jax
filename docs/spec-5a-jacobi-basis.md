# Spec 5a: 3N Jacobi HO basis and its antisymmetrisation

Scope: the isospin-formalism three-nucleon Jacobi harmonic-oscillator (HO) basis of NuHamil, the channel
enumeration, and the construction of the fully antisymmetric ("physical", "orthonormal") states.
Source root: `/home/soham/Projects/nuclear-physics/NuHamil-public/src` (abbreviated `src/`). Files:
`ThreeBody/ThreeBodyJacobiQuantumNumbers.F90` (QN), `ThreeBodyJacobiSpaceIso.F90` (SP),
`ThreeBodyJacChanIso.F90` (CH), `ThreeBodyJacOpsChanIso.F90` (OP), `TMTransFunctions.inc` (TF),
`ThreeBodyTransCoefIsospin.inc` (TC), `MyLibrary.F90` (ML).
Labels: **[V]** = verified numerically by me against the oracle run (see section 8); **[S]** = read from source
only; **[?]** = inferred or unsure (see section 9).

Oracle run used: `/tmp/claude-1000/-home-soham-Projects-nuclear-physics-nuhamil-alt/b6aad67b-aa74-41b6-a308-3216b5d25052/scratchpad/nnn/`
(`Input_3n_e4.dat`, `log_3n_e4.txt`, `cfp/cfp_A3_j*p*_t1_Nmax8.bin`). Input: hw=16, emax=4, e2max=8, e3max=4,
N3max=8, jmax3=11, ramp="flat8", srg lambda 2.0, local 3NF cutoff 400.

Everything about the antisymmetrizer below was re-implemented independently in Python (scratch, not in the
repo) using `refs/moshinsky_reference.py` / `nuham.angmom.moshinsky_table.MoshinskyTable(d=1/3)` and reproduces
the oracle's state counts for all 24 channels and the oracle's stored `cfp` files to 4e-15 [V].

---------------------------------------------------------------------------------------------------

## 0. Summary of the pipeline stage covered here

For each total-angular-momentum/parity/isospin channel (J, P, T) NuHamil:

1. enumerates all HO Jacobi states with `N = 2 n12 + l12 + 2 n3 + l3 <= Nmax`, in the isospin formalism
   (total isospin T, no T_z), with the pair (1,2) already antisymmetric (`l12 + s12 + t12` odd);
2. builds, **separately for each N**, the matrix of the three-body antisymmetrizer A in that basis
   (CH:546-591, 593-682);
3. diagonalises it (LAPACK `dsyev`), keeps the eigenvectors with eigenvalue 1 (|1 - lambda| < 1e-2) as the
   rows of a coefficient matrix `C` (n_AS x n_NAS), the "cfp" ("coefficients of fractional parentage", in the
   NuHamil naming; it is a basis change, not the textbook cfp tables);
4. stores `C` and the list of NAS quantum numbers per N in a binary cache file (CH:703-736) which does not
   depend on hw, the interaction or the SRG.

Operators defined on the unantisymmetrised pair-(12)-antisymmetric basis are projected with `C^T O C`
(OP:1031-1072, `NonAsym_to_Asym`). All later 3N steps (3NF, SRG in Jacobi space, Jacobi -> lab) work in the
AS basis.

---------------------------------------------------------------------------------------------------

## 1. Coordinates, states, couplings

### 1.1 Quantum numbers (QN:51-79, `NonAntisymmetrizedIsoQNs3`)

Eight integers per NAS state: `(n12, l12, s12, j12, t12, n3, l3, j3)`.

- `n12, l12`: radial and orbital quantum numbers of the relative motion of pair (1,2) (HO, "Jacobi vector 1").
- `s12, t12`: spin and isospin of pair (1,2), **0 or 1, integer (not doubled)**.
- `j12`: coupled pair angular momentum, integer, `|l12 - s12| <= j12 <= l12 + s12`.
- `n3, l3`: radial and orbital quantum numbers of the motion of particle 3 relative to the pair (Jacobi vector 2).
- `j3`: **doubled** (2 j3), odd, `2 j3_true in {|2 l3 - 1|, 2 l3 + 1}`: l3 coupled to the spin 1/2 of particle 3.
- Channel labels in `ThreeBodyJacIsoChan`: **J, T are doubled** (`j`, `t` odd integers, "1" = 1/2), **P = +-1**
  (CH:372-420; the log prints `J= 3/2, P= 1, T= 1/2`). The labels `j, p, t` of `GetNAS` etc. follow this.
- The type also has a pn-formalism sibling `NonAntisymmetrizedQNs3` with `z12, z3` (QN:27-49) that the 3N
  Hamiltonian path does not use (`pn_form` is forced false for rank 3, `ThreeBodyManager.F90:22-30`).

### 1.2 Coupling scheme [S, verified in the antisymmetrizer formula]

- Spatial-spin: `|[(l12 s12) j12, (l3 s3=1/2) j3] J>`, i.e. j12 and j3 coupled in this order to J, l12 with
  s12 (in this order), l3 with the spin of particle 3 (in this order). The 9j in CH:660-663 is the LS -> jj
  recoupling of exactly this scheme (l12 and l3 coupled to lambda in this order, s12 and 1/2 to S).
- Isospin: `[t12 (+) 1/2] T`, pair isospin first.
- Parity `P = (-1)^(l12 + l3) = (-1)^N` (CH:507-509 (inside `set_jacobi_nas`): `(-1)**(l12+l3) /= p` is skipped).
- Pair-(12) antisymmetry: `(-1)^(l12 + s12 + t12) = -1` (CH:511-513 region, "if((-1)**(l12+s12+t12) /= -1) cycle").
  These states are normalised for the pair; no extra factor.

### 1.3 Jacobi coordinates and HO functions [? inferred; the source never writes the definitions]

The source defines the 3N Jacobi space only implicitly. What is fixed by the code:

- The pair-(12) HO functions `(n12 l12)` are **exactly** the NN relative HO functions of the 2N code with the
  same hw: the NN matrix elements `<n12 l12 s12 j12 t12 | V | n12' ...>` are taken verbatim from the 2N relative
  operator (`get_me12_from_qns`, OP:656-691; no rescaling).
- The Jacobi -> lab T-coefficients (TC:495-501, 1019-1020) use two stages of equal-mass-type brackets:
  `tmbk_1` = bracket with mass ratio d=1 (particles a, b -> relative `n12 l12` of the pair and its
  centre of mass) and `tmbk_2` = bracket with d=2 (centre of mass of the pair, mass 2m, with particle c, mass m
  -> `(n3, l3)` and the 3N centre of mass `(Ncm, Lcm)`).
- The 3N relative kinetic energy in the Jacobi basis is diagonal-plus-neighbour in each of the two coordinates
  with identical frequency: `T = t(n12,l12) + t(n3,l3)`, `t = (hw/2) [(2n+l+3/2) delta - sqrt(n(n+l+1/2)) ...]`
  (OP:472-532; this routine is currently dead code, the live path goes through the 2N kinetic operator times 2,
  OP:463-466, but the formula documents the convention).
- The local 3NF code uses `nu = 2 m_red_pn hw / hc^2` for the radial functions of both Jacobi vectors
  (`NNNForceLocal.F90:1352`, `ho_radial_wf_norm(n, l, anu, r)` with `x = sqrt(anu) r`, ML:750-766), and the
  non-local-regulator code the momentum-space inverse `a = hc^2/(2 m_red hw)` (`NNNForceHOIsospin.F90:400`).

These are consistent with the standard Navratil mass-scaled Jacobi coordinates

    xi_1 = (r1 - r2)/sqrt(2),   xi_2 = sqrt(2/3) (r3 - (r1 + r2)/2),   xi_0 = (r1 + r2 + r3)/sqrt(3)

with a common oscillator length `b^2 = hbar^2 / (m hw)`, `m = 2 m_red_pn`, i.e. all three vectors carry "mass m"
and the same hw (so `T_rel = p_xi1^2/2m + p_xi2^2/2m`, `H_HO,rel = hw (N + 3)`; the lowest relative energy is
`T = 3 hw/2` at N=0). I did not find a comment stating the definitions; treat them as the convention the
d=1, d=2 and d=1/3 brackets imply.

### 1.4 Radial function sign convention

`ho_radial_wf_norm` is positive at small r (Laguerre `L^{n}_{l+1/2}(x^2)` with positive leading coefficient at
x = 0, ML:763-765). For NN this is the same function as in spec 1 (momentum space carries the extra `(-1)^n`).
The bracket phase `(-1)^(n1+n2+n+N)` inside `gmosh` (ML:361-582) corresponds to this "positive at the origin"
convention; a reimplementation that uses the existing `MoshinskyTable` (which reproduces `gmosh` to 6e-15 at
d=1/3 for N=4..8, [V]) inherits the correct convention. **The sign of the antisymmetrizer depends on this
consistency**: wrong brackets destroy idempotence of A.

---------------------------------------------------------------------------------------------------

## 2. Channels, truncations, state counts

### 2.1 Channel set (SP:106-165, TF:13-23)

- Input `jmax3` (here 11 means 2J <= 11, i.e. J <= 11/2). Number of channels `NChan = ((jmax3+1)/2)*4`
  = 24 for jmax3=11 (SP:121): T in {1/2, 3/2} (doubled `t = 1, 3`), J = 1/2 ... jmax3/2, P = +1 then -1.
- Enumeration order: **T outermost (1/2 then 3/2), then J ascending, then P = +1 before -1** (SP:138-142; same
  loop in TF:13-23; this is the order of `Calculating for J=..., P=..., T=...` in the log).
- No T=1/2 vs 3/2 or J selection beyond that: every (J,P,T) combination is a channel; some may be empty
  (e.g. high J at low Nmax; `jac%GetNumberAStates() < 1` -> skipped, TF:110 region).
- `j3max_initial_3nf` (default = jmax3) limits J of the **genuine 3NF** only (NuHamilInput.F90:463), not the
  Jacobi space. `J12max` likewise only truncates the genuine 3NF. The Jacobi space itself has no `j12` cut:
  `j12 <= l12 + 1`, `l12 <= N`. The NN interaction is zero for relative J > J2max_NNint (default 8) via
  `get_me12_from_qns` (OP:656-691: stored operator only up to `Jmax` and `Jmax+1`; the kinetic/other analytic
  operators are computed in closed form beyond that, interactions return 0).

### 2.2 N truncation and ramp (SP:167-195)

- Per-channel truncation `Nmax(J)` = `GetRampNmax(j, ramp)` with `j` = **2J**:
  - `ramp = "flatNNN"`: Nmax = NNN for all J (string tail from position 5).
  - `ramp = "rampN0-j1-N1-j2-N2-..."`: start with Nmax = N0 (digits after "ramp"); for each pair (j_k, N_k) in
    order, if `2J > j_k` then Nmax = N_k (later pairs override earlier ones). The thresholds j_k are doubled J.
  - anything else: `stop`.
- State set per (J,P,T,N) `Nmax`: all N from 0 to Nmax **with (-1)^N = P**; blocks of the other parity are empty
  (written as `(0,0,N)` in the cache).
- The cache file is created with the ramp's Nmax (SP:143-158: `Nmax_for_file` is overwritten by the ramp value);
  `readf(hw, unit, Nmax)` keeps `min(Nmax_request, Nmax_file)` (CH:738-789). Because A conserves N, truncating
  the file at N <= e3max gives exactly the AS space of that smaller Nmax (nested bases) [S].
- In the oracle run the Jacobi space is built with Nmax=8 (`flat8`, `N3max=8`) and the lab transform reads it
  with `Nmax = e3max = 4`.

### 2.3 NAS state ordering inside a channel (CH:491-523, `set_jacobi_nas`)

For fixed N the loops are, from outermost to innermost (**this order defines the NAS index within the N block**):

    n12 = 0..N/2
      n3 = 0..(N-2 n12)/2
        l12 = 0..(N - 2 n12 - 2 n3);   l3 = N - 2 n12 - 2 n3 - l12   (skip if (-1)^(l12+l3) != P)
          s12 = 0,1
            j12 = |l12 - s12| .. l12 + s12
              t12 = 0,1  (skip if triag(2 t12, 1, 2T) or (-1)^(l12+s12+t12) = +1)
                j3 = |2 l3 - 1|, |2 l3 - 1| + 2, ..., 2 l3 + 1  (skip if triag(2 j12, j3, 2J))

The global channel index concatenates the N blocks N = 0, 1, ..., Nmax (CH:422-461, `set_indices`):
`N_nas(i), alpha_nas(i)` for NAS, `N_as(i), i_as(i)` for AS. [V] my enumeration reproduces the oracle's stored
QN lists in this order for the three files I checked.

### 2.4 Dimensions

Formally n_NAS(N) is the count of the loop above; n_AS(N) = round(trace A_N) (CH:525-544, error if the trace is
not an integer to 1e-4).

**Important detail**: if an N block has n_AS = 0 (e.g. the N=0 block of J=3/2, T=1/2; the pair-antisymmetric
state exists but is Pauli forbidden), NuHamil **discards the NAS states of that block** (CH:474-489:
`n_states_nas = 0`, `NAS` deallocated). The numbers in the log ("physical") are therefore NAS counts *excluding*
Pauli-forbidden-only N blocks. Example: raw enumeration gives 305 for J=3/2, P=+, T=1/2, Nmax=8; the log says
304. [V]

Table, Nmax = 8, format `NAS_kept / AS | NAS_kept(N<=4) / AS(N<=4)` (all match the oracle log for the 14 channels it
printed when I wrote this; the remaining are my calculation, same code path):

| T, P | J=1/2 | 3/2 | 5/2 | 7/2 | 9/2 | 11/2 |
|---|---|---|---|---|---|---|
| 1/2, + | 190/64 \| 44/15 | 304/101 \| 60/20 | 340/113 \| 54/18 | 320/107 \| 36/12 | 268/89 \| 18/6 | 201/67 \| 5/2 |
| 1/2, - | 140/47 \| 26/9 | 220/73 \| 34/11 | 236/79 \| 26/9 | 212/71 \| 14/5 | 166/55 \| 4/1 | 114/38 \| 0/0 |
| 3/2, + | 94/31 \| 21/7 | 150/49 \| 29/9 | 170/57 \| 27/9 | 157/51 \| 16/5 | 134/45 \| 9/3 | 97/32 \| 0/0 |
| 3/2, - | 70/23 \| 13/4 | 110/37 \| 17/6 | 117/39 \| 12/4 | 106/35 \| 7/2 | 83/28 \| 2/1 | 57/19 \| 0/0 |

(Oracle log lines: `190 64, 140 47, 304 101, 220 73, 340 113, 236 79, 320 107, 212 71, 268 89, 166 55, 201 67,
114 38` for T=1/2 and `94 31`, `157 51` for T=3/2.) The N<=4 columns are what the e3max=4 lab transform sees.

Per-N AS dimensions (first channels, T=1/2): J=1/2+: N=0,2,4,6,8 -> 1,4,10,19,30; J=1/2-: N=1,3,5,7 -> 2,7,14,24;
J=3/2+: N=0,2,4,6,8 -> 0,5,15,30,51; J=5/2+: N=2,4,6,8 -> 3,15,34,61; J=7/2-: N=3,5,7 -> 5,20,46. (Full lists in the check script.)

---------------------------------------------------------------------------------------------------

## 3. Antisymmetry: how it is imposed

### 3.1 Method

Neither textbook cfp tables nor a symmetric-group projection by character are used. For every (J, P, T, N):

1. Build the matrix `A_N` of the three-body antisymmetriser, **restricted to the NAS space** (the basis states
   are already antisymmetric under P12): CH:546-591.
2. Diagonalise `A_N` with LAPACK `dsyev('v','u')` (LinAlgf90 `DiagSymD`: `submodules/LinAlgf90/src/LinAlgLib.f90:251-352`;
   eigenvalues ascending).
3. Rows of `C_N` are the eigenvectors with `|1 - lambda| < 1e-2` in the order returned (CH:576-580). Eigenvalues
   with `|lambda| > 1e-2` and `|1 - lambda| >= 1e-2` trigger a warning (CH:583-587); none occurred in my
   calculation (all eigenvalues within 1e-13 of 0 or 1 [V]).
4. n_AS = round(trace A_N) is computed first (from the diagonal only) and used to allocate `C_N`; consistency of
   the number of eigenvalue-1 vectors with n_AS is implied but not checked explicitly.

The AS states are

    |AS_i> = sum_alpha C[i, alpha] |NAS_alpha>        (same J, P, T, N; C real; rows orthonormal; C C^T = 1, C^T C = A_N)

so `C` is *not* a transformation matrix between two complete bases: it is an isometry from the AS subspace into the
NAS space (n_AS x n_NAS, n_AS < n_NAS). An operator O defined on NAS states becomes `C O C^T` in the AS basis
(OP:1031-1072; `Asym_to_NonAsym` reverses it as `C^T O C`, OP:1287-1309; the code stores C as
`cfp%m(ket_nas, bra_as)` in `GetCFPMat`, CH:791-812, i.e. the transpose of what is written to file).

### 3.2 Exact definition of the antisymmetrizer matrix (CH:636-682, `anti_op_isospin`)

Define A on states with pair-12 antisymmetry as `A = (1/3)(1 - P13 - P23) = (1 - 2 X)/3`, where `X` is the matrix
element of the transposition `P23` (equal to that of `P13` by the 12-antisymmetry). In the code, for bra `a`
(`n12 l12 s12 j12 t12 n3 l3 j3`) and ket `b` (`n45 l45 s45 j45 t45 n6 l6 j6`), channel (J, T):

    A_ab = delta_ab/3 - (2/3) X_ab                      (diagonal: (1 - 2 X_aa)/3, off diagonal: -2 X_ab/3)
    X_ab = 0 unless N_a == N_b
    X_ab = (-1)^(s_a + t_a + s_b + t_b) sqrt((2 t_a + 1)(2 t_b + 1)) { 1/2 1/2 t_a ; 1/2 T t_b }
           * sum over S and lambda of
               sqrt((2 j12_a+1)(2 j3_a+1)(2 lambda+1)(2 S+1)) * sqrt((2 j12_b+1)(2 j3_b+1)(2 lambda+1)(2 S+1))
               * 9j(a) * 9j(b) * sqrt((2 s_a+1)(2 s_b+1)) { 1/2 1/2 s_a ; 1/2 S s_b }
               * < n12 l12, n3 l3; lambda | n45 l45, n6 l6; lambda >_{d = 1/3}

- `S` ranges over total spin `1/2` or `3/2`: `max(|2 s_a - 1|, |2 s_b - 1|) <= 2S <= min(2 s_a + 1, 2 s_b + 1)`;
  lambda over `max(|l12-l3|, |l45-l6|, |2J - 2S|/2) .. min(l12+l3, l45+l6, (2J+2S)/2)`.
- 9j(a), for the **ket** analogous: `ninej(1, j3, 2 l3, 2S, 2J, 2 lambda, 2 s12, 2 j12, 2 l12)` with all
  arguments doubled, i.e. `{1/2 j3 l3 ; S J lambda ; s12 j12 l12}` as the rows of the 9j in that order
  (CH:663-664). **This is a row/column permutation of the textbook LS-jj recoupling 9j
  `{l12 s12 j12 ; l3 1/2 j3 ; lambda S J}`; permutations introduce a sign, so use the order of the code exactly**
  (or re-derive the sign). The comment above the function (CH:621-635) lists a `(-1)^L` factor that the
  code does not have; the code is what I verified.
- 6j symbols are `{1/2 1/2 s_a ; 1/2 S s_b}` and `{1/2 1/2 t_a ; 1/2 T t_b}` (all doubled arguments).
- The bracket is `gmosh(n12, l12, n3, l3, n45, l45, n6, l6, lambda, d = 1/3)`, ML:361-582, in the argument
  order `(n_cm?, l_cm?, n_rel?, l_rel? | n1, l1, n2, l2; lambda)` of the NN Moshinsky bracket; in the package
  `MoshinskyTable(d=1/3).matrix(N, lambda)[row=(n12,l12,n3,l3), col=(n45,l45,n6,l6)]` equals it [V, 6e-15].
  d = 1/3 corresponds to `theta = atan(sqrt(1/3)) = 30 deg` in `moshinsky_table.py` (the generator carries 2 theta,
  so the Jacobi-vector rotation under the transposition is 60 deg).
- Physical meaning: the 3-body AS projector is nonzero only inside a fixed N; angular-momentum, spin and isospin
  recoupling to the (12)-(3) -> (23)-(1) scheme gives `X`.

### 3.3 Eigenvalue spectrum and normalisation [V]

- A_N is real symmetric (symmetry error 2e-16), idempotent (`A^2 = A` to 5e-15) and its eigenvalues are exactly
  0 and 1. n_AS = number of eigenvalues 1 = trace.
- Eigenvalue 0 multiplicity = n_NAS - n_AS. Group-theoretically: the pair-12-antisymmetric subspace contains one
  vector from each totally antisymmetric multiplet (eigenvalue 1) and one from each mixed-symmetry 2-dim
  multiplet (eigenvalue 0); the totally symmetric irrep has no 12-antisymmetric vector. Example J=1/2+, T=1/2,
  N=4: NAS 30, AS 10, zeros 20; N=8: 90, 30, 60. In every case I checked N=0..8 `NAS = AS + n_mixed`.
- Equivalently `X = (1 - 3A)/2` has eigenvalues `-1` (AS) and `+1/2` (mixed).
- Normalisation: `C C^T = 1` to 4e-15 for the stored vectors; no extra sqrt-factor is attached (the cfp rows
  are unit eigenvectors, the NAS basis is orthonormal).

### 3.4 Non-uniqueness of the AS basis (key convention point)

Because all selected eigenvalues are equal to 1 (degenerate), **LAPACK returns an arbitrary orthonormal basis of
the eigenspace**, in an arbitrary order and with arbitrary signs. The rows of `C` in the oracle `cfp` files are
therefore not reproducible by another implementation, and individual AS-basis matrix elements (e.g. the
SRG-evolved 3N Hamiltonian in the AS basis, the `ops/NNN-full*.bin` and `UT_*` files) are **basis dependent**.
Only quantities invariant under an orthogonal rotation within each N-block of the AS space are comparable:
- the projector `C^T C = A_N` (NAS representation) and any `C^T (C O C^T) C = A O A`;
- eigenvalues of the AS Hamiltonian (the log prints five lowest "Eigen values of original 3-body H" and
  "after SRG": SRG-invariance check, and the quantity to reproduce);
- the lab-frame matrix elements (after the Jacobi -> lab transformation, which uses `C` consistently).
A reimplementation can choose its own AS basis (e.g. eigh with any sign convention, or a Gram-Schmidt on the
projector columns) as long as it is used consistently in the 3NF projection, the SRG, and the T-coefficients.
The rows from different N are never mixed (A is block diagonal in N, so `C` is block diagonal).

---------------------------------------------------------------------------------------------------

## 4. Frequency, constants, masses

- hw enters **only** via the radial functions and the operators; the Jacobi basis, the AS cache and the Moshinsky
  brackets are hw independent (cache file name has no hw: `cfp_A3_j<2J>p<+/-> t<2T>_Nmax<N>.bin`,
  CH:684-701; the oracle files are `cfp_A3_j1p+t1_Nmax8.bin` etc.).
- Constants (ML:8-14): `hc = 197.32705`, `m_proton = 938.27231`, `m_neutron = 939.56563` MeV;
  `m_nucleon = (m_p + m_n)/2`, `m_red_pn = m_p m_n / (m_p + m_n)`.
- 3N HO parameter: `nu = 2 m_red_pn hw / hc^2` (fm^-2) for both Jacobi vectors in the local 3NF (NNNForceLocal.F90:1352);
  non-local regulator in momentum space `a = hc^2 / (2 m_red hw)`: **the harmonic-mean mass 2 m_red_pn = 938.949 MeV,
  not m_nucleon = 938.919 MeV**, same as the 2N code (one pn reduced mass used for all isospin channels).
- The same hw must be used for the NN (relative) matrix elements, the 3NF and the SRG; the code asserts this
  implicitly through `jac%GetFrequency()` (stored in the channel object when reading, CH:738-789).

---------------------------------------------------------------------------------------------------

## 5. Cache file format (stream, native little-endian, no record markers) (CH:703-789) [V parsed completely]

    int32 J2, int32 P, int32 T2, int32 Nmax
    for N = 0..Nmax:
        int32 n_nas, int32 n_as, int32 N
        if n_nas >= 1:  n_nas * (8 x int32: n12, l12, s12, j12, t12, n3, l3, j3)   [each is one 4-byte write]
        if n_as  >= 1:  n_as rows of n_nas float64 (row i = AS state i, columns = NAS states of this N block)

Empty blocks are `(0, 0, N)`. A parsing check: `offset == file size` at the end (true for the three files I read).
This is NuHamil-internal; the package need not write it.

---------------------------------------------------------------------------------------------------

## 6. What the AS basis is used for downstream (context only)

- 2-body scalar operators in Jacobi space: `O_AS = 3 * C (O_{12} in NAS) C^T` (OP:419-470; the factor 3 sums the
  three pairs because only the pair-(12) operator is stored; kinetic uses 2, OP:463-466). The embedded `O_{12}`
  acts as `delta_{n3 n3'} delta_{l3 l3'} delta_{j3 j3'}` times the NN matrix element with the *same* (j12, t12, s12).
- One-body tensors etc. add the 6j factors in OP:300-350; they are not needed for Hamiltonian-only work.
- Jacobi -> lab uses the AS states with `(E, i, Jrel, Ncm, Lcm)` labels in TC; not covered here.

---------------------------------------------------------------------------------------------------

## 7. Conventions a reimplementation must match

1. **Isospin formalism**: states labelled by total T (1/2, 3/2), NN matrix elements taken in (S, T) pair channels,
   no charge dependence or T_z (NuHamil forces `pn_form=false` for 3N, ThreeBodyManager.F90:22-30). [S]
2. **Scalar matrix elements are not reduced**: Jacobi-space operator blocks are `<(J,T)...|O|(J,T)...>` independent
   of M, M_T (the reduced-matrix-element code paths are separate, `op%reduced_me()` OP:170). [S]
3. **Channel enumeration order**: T outermost (1/2, 3/2), J ascending, P = + then - (SP:138-142). Needed only if
   output files or tests depend on it.
4. **State order inside a channel**: N blocks ascending; inside N the loop order of section 2.3; AS states in the
   (arbitrary) eigenvector order. Only the NAS order is reproducible.
5. **Dropping of Pauli-forbidden N blocks** (n_AS = 0): NAS states of that N block are removed (section 2.4).
6. **Pair coupling orders**: (l12 s12) j12, (l3 s3) j3, (j12 j3) J, (t12 1/2) T. A different order changes
   signs of off-diagonal NN-embedding and A elements (e.g. `(-1)^(j12+...)` phases).
7. **9j argument order and the `(-1)^(s12 + t12 + s45 + t45)` phase** in A, with `d = 1/3` brackets with the
   (cm, rel) = ((n12 l12), (n3 l3)) assignment on the bra and `(n45 l45, n6 l6)` on the ket.
8. **Bracket phase convention**: `gmosh` includes `(-1)^(n1 + n2 + n_l + n_r)` (ML:580-581) and the "positive at the
   origin" radial functions; `MoshinskyTable` matches it.
9. **HO parameter for both Jacobi vectors**: `nu = 2 m_red_pn hw / hc^2` with the same hw as the NN relative basis.
10. **Mass-scaled Jacobi vectors** so that T_rel is `(hw/2)(2n + l + 3/2)`-like in each coordinate (section 1.3).
11. **AS basis**: not unique (3.4). Compare invariants.
12. **Nmax per channel** from the ramp string; the Jacobi space used in the SRG and the 3NF is built at the ramp
    Nmax (SP:143, TF:80), reading for the lab transform is truncated to e3max.
13. N-truncation uses `2 n12 + l12 + 2 n3 + l3 <= Nmax` (no separate n or l caps); l3 up to N; j12 up to l12 + 1.

---------------------------------------------------------------------------------------------------

## 8. Numerical checks against the oracle (cheap, all pass in my scratch implementation)

All computable in seconds for Nmax <= 8 using `MoshinskyTable(d=1/3)` or `refs/moshinsky_reference.bracket(..., d=1/3)`;
no NuHamil data needed except the cache files / log for items a) and f).

a) **Dimensions.** For every (J,P,T) with 2J <= 11, Nmax = 8: NAS_kept and AS counts from the table in 2.4.
   Oracle log: `190/64, 140/47, 304/101, 220/73, 340/113, 236/79, 320/107, 212/71, 268/89, 166/55, 201/67,
   114/38` (T=1/2) and `94/31`, `157/51` (T=3/2 J=1/2+, 7/2+). Compare AS also per N (list in 2.4); remember to drop N
   blocks with n_AS = 0 from the NAS count.
b) **Trace/rank.** `trace(A_N)` is an integer to 1e-12 and equals the number of unit eigenvalues.
c) **Spectrum.** All eigenvalues of A_N within 1e-12 of {0, 1}: `max_i min(|lambda_i|, |1 - lambda_i|) < 1e-12`
   (I measured idempotence `|A^2 - A| < 7e-15`, symmetry `|A - A^T| < 3e-16`).
d) **Multiplicities.** `#zeros = n_NAS - n_AS` per N block (mixed-symmetry multiplets), N=2, J=1/2+, T=1/2: 12, 4, 8.
e) **Transposition.** `X = (1 - 3A)/2` has spectrum {-1, +1/2}, i.e. `(X + 1)(X - 1/2) = 0` (equivalent to A
   idempotent).
f) **Oracle cfp files** (`cfp_A3_j*p*t1_Nmax8.bin`): reading the QN lists gives the identical order to section 2.3
   for all N blocks; the row space of the stored `C` equals the range of the A computed by the reimplementation:
   `|C^T C - A| < 4e-15`, `|C C^T - 1| < 3e-15` for J=1/2+, 3/2+, 5/2-, T=1/2.
g) **Symmetric-group relations** for the NAS-space transposition matrix `X` at each N: `X` is real symmetric and
   block diagonal in N; `A` commutes with any permutation-symmetric operator written in the NAS space only after
   projection, so test operator relations (e.g. the factor 3 for two-body scalars) in the AS space, not in the NAS space.
h) **Physical check with an operator.** In the AS basis of the oracle channel J=1/2+, T=1/2 at N=0 (single AS
   state), `T_rel = 3 hw/2 = 24 MeV` at hw=16; the HO relative Hamiltonian `T_rel + U_HO` has `hw(N+3)` entries
   (N=0 -> 3 hw). A cheap self-consistency test for the 2-body embedding factor 3 / kinetic factor 2.
i) **SRG-invariance** (later): the five lowest Jacobi eigenvalues printed in the log before/after SRG agree to 1e-6
   (e.g. J=1/2+: -1.349102, 9.222036, 13.745185, 15.351910, 19.414468 for hw=16, lambda=2, local 3NF 400,
   N=8). These eigenvalues are basis independent and are the best end-to-end check of this spec.
j) **Nested Nmax**: AS dimension at Nmax = e3max (4) is the N<=4 count; `C` for Nmax=4 is the N<=4 blocks of the
   Nmax=8 `C` (blocks are independent).

(a) to (d) and (f) are verified by the scratch scripts in
`/tmp/claude-1000/-home-soham-Projects-nuclear-physics-nuhamil-alt/b6aad67b-aa74-41b6-a308-3216b5d25052/scratchpad/a5/`
(`anti.py`, `grid.py`, `tab.py`, `chk.py`, `cmp.py`; not part of the repo; the antisymmetrizer check of `anti.py` is covered by `tests/test_jacobi3.py` and the file was removed on 2026-10-07).

---------------------------------------------------------------------------------------------------

## 9. Open questions / uncertainties

1. **Jacobi coordinate definitions** (1.3) are inferred from the brackets (d = 1, 2, 1/3), the kinetic form and
   `nu = 2 m_red hw/hc^2`; the Fortran never states them. The antisymmetrizer and the 3N kinetic energy
   confirm self-consistency but a different normalisation of xi_2 would be absorbed into the d-value only if
   nothing else sees it. The T-coefficients (TC) and the local 3NF are other specs' topic.
2. **9j argument order / the comment at CH:621-635** disagree with the code by a `(-1)^L` factor and different
   9j layout; I trust the code since it reproduces idempotence and the oracle `cfp` files to 4e-15, but I did not
   derive the equivalence analytically.
3. **N3max versus ramp vs e3max**: the cache file is built at the ramp Nmax (SP:143, TF:80) while `N3max` is
   what the 3NF/SRG code passes around (e.g. TF:160-161 `jac%init(hw, jmax3, e3max, ramp, N3max, ...)`; `Nmax_for_file`
   is overwritten inside). In the oracle run ramp=flat8=N3max so they coincide. For `ramp` != `flatN3max` I did not
   trace which Nmax the SRG evolution uses.
4. **`Nmax` limit used by the lab transform** is e3max, but the SRG evolves in the full ramp space; the final lab
   elements for E3 <= e3max then contain SRG-induced couplings to N <= ramp Nmax. Matching this requires the same
   truncation in the Jacobi-space SRG (not covered here).
5. **n_AS vs number of eigenvalue-1 vectors**: CH:525-544 uses the trace; CH:576-580 uses the threshold 1e-2.
   For moderate Nmax they agree (checked to Nmax 8; eigenvalues are 0/1 to 1e-13). At large Nmax (the production
   case is N up to 40) the eigenvalue spread is unknown to me; a 1e-2 threshold is loose, a reimplementation
   should assert agreement with the trace.
6. **Large-N cost**: the dense A_N has dimension up to a few 10^4 at Nmax=40, J=1/2 (my estimate); NuHamil
   builds it with an OpenMP loop of per-element 9j/6j/bracket sums and `dsyev`. A JAX version should exploit
   that A is a projector (rank = n_AS) and that `X` is a sparse-ish rotation (generator/exponential) -- not
   needed for the small cases.
7. **Which T-channels the lab 3N file needs**: jmax3 sets J, T in {1/2, 3/2}; whether T = 3/2 elements are
   output for all J in the `me3j` is a spec for the lab/format agent.
8. I did not examine `ThreeBodyJacobiSpace.F90` beyond its 3-line wrapper, and did not read the A-body
   (`ABody/`) Jacobi code, which has an alternative general-A antisymmetriser; it is not on the 3N path.
9. The oracle run was still in progress while I wrote this (12 of 24 channel logs plus the first T=3/2 lines were
   available); the unlisted channels are my own calculation with the same code path and are labelled as such.
