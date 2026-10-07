# Spec 5d: Jacobi -> laboratory three-body transformation and the me3j file (Phase 5.0)

Scope: the last stage of NuHamil's 3N pipeline for a scalar operator ("hamil", isospin formalism, `trans2lab`): from the
antisymmetrised Jacobi-HO matrix elements of V_3N (plus the evolved NN part, already inside the Jacobi operator) to
laboratory-frame three-body matrix elements of the harmonic-oscillator single-particle basis, and the file imsrg++ reads.
The Jacobi operator itself (SRG evolution, local N2LO force, cfp) is the subject of the other 5.0 specs; here it is an input:
a dense symmetric matrix per Jacobi channel (J_rel, P, T) in the antisymmetric (AS) Jacobi basis.

Paths: NuHamil `src/...` is `/home/soham/Projects/nuclear-physics/NuHamil-public/src`; imsrg++ `src/...` is
`/home/soham/Projects/nuclear-physics/imsrg/src`. `.inc` files are included once per precision (Half/Single/Double) by the
`ThreeBodyLabOpsIso{Half,Single,Double}.F90` wrappers, so line numbers below are valid for all three variants.

Status tags as in spec-3: **[V]** verified numerically (against the reference run or by an independent count),
**[S]** read from source, **[?]** unsure. Throwaway scripts used for the checks are `refs/count3.py`, `refs/dims3.py` and
`refs/nas3.py` (run with `EMAX E2MAX E3MAX`; committed 2026-10-07 and rerun: `count3.py 6 12 6 file` gives 456 320 values, `dims3.py 4 8 8` the logged 113 and 79).

Reference run (read-only inputs): `nnn/Input_3n_e4.dat`, `log_3n_e4.txt`, `nnn_3NF400_e4.me3j.gz` in the session scratchpad:
3NF400 local N2LO, SRG lambda 2.0, hw 16, emax 4, e2max 8, e3max 4, N3max 8, jmax3 11, `ramp=flat8`, double precision.

---------------------------------------------------------------------------------------------------------------

## 1. Where this stage sits

`ThreeBody/TMTransformManager.F90:31-46` `manage_trans_to_lab` -> `trans_to_lab_basis_isospin_{half,single,double}`
(`ThreeBody/TMTransFunctions.inc:4-226`, precision from `lab_3bme_precision`, default "single", `NuHamilInput.F90:~275`).
For `Operators="hamil"` the per-operator driver `trans_to_lab_op` (`TMTransFunctions.inc:135-225`) does:

1. `sps%init(emax, lmax)` and `lab%init(sps, e2max, e3max, hw_target)` (L:142-143): lab single-particle orbits and lab channels.
   `lmax` defaults to `emax` (`NuHamilInput.F90:459`).
2. `jac%init(hw, jmax3, e3max, ramp, N3max, tmpdir)` (L:160): Jacobi space, **read back truncated to E <= e3max**
   (`ThreeBodyJacobiSpaceIso.F90:~143-146`, `readf(hw, unit, Nmax=e3max)`), although the channel files are written with the
   ramp Nmax (`GetRampNmax`, L:~159-190 of that file). So the lab transform only sees Jacobi quanta E <= e3max.
3. `jac2lab%init(jac, lab, e3max, jmax3, nblock)` (L:161): all T-coefficients for **all** lab channels, kept in memory.
4. `opjac%set(params)` (L:183-188): the Jacobi operator `V` in the AS basis (including NN pieces, SRG-evolved); for
   "hamil" it is the stored `NNNint` Jacobi file. Dimensions `m` per Jacobi channel (J_rel, P_rel, T): all AS states with E <= e3max.
5. `oplab%TMtrans(jac2lab, opjac)` (L:217) -> `TMTransScalar` (`ThreeBodyLabOpsIso.inc:855-1083`).
6. `oplab%writef(filename)` (L:219) -> writer selected by extension (section 6).

Frequency: `hw` is the HO frequency of the Jacobi basis; `hw_target` (default = `hw`; if different, `hw_conversion=.true.`,
`NuHamilInput.F90:470-476`) is the frequency of the lab basis. The conversion is a change-of-frequency overlap transformation
applied **to the Jacobi operator** (`FrequencyConversion`, `ThreeBodyJacOpsChanIso.F90:242-273`, called inside the operator
read path L:301 etc.) before the lab transform. The T-coefficients themselves contain no frequency: Moshinsky brackets
depend only on mass ratios, so the same T-coefficients serve every hw. Lab space gets `hw_target` only as metadata
(`ThreeBodyLabSpaceIso.F90:~118`) and for file names. **[S]** The me3j file does not store hw; imsrg++ takes it from its own input.

---------------------------------------------------------------------------------------------------------------

## 2. The laboratory three-body basis

### 2.1 Orbits

Isospin-less orbits `(e=2n+l, n, l, j)` (j stored doubled), enumerated e = 0..emax, l = e%2, e%2+2, ..., min(e, lmax),
j = l-1/2 (omitted for l = 0) before l+1/2, i.e. the same list as the `me2j` "remap" list in spec-3 section 4.2
(`SingleParticleState.F90:InitOrbitsIsospin`, L:~119-155). Index is 1-based in Fortran (`i1 .. norbs`), 0-based in imsrg++.
Counts: 15 orbits at emax 4, 120 at emax 14, 153 at emax 16 (lmax = emax). **[V]** (reproduced in `count3.py`)

### 2.2 Coupled states

`|a b c; J12 T12; J T>` with `|[(a b) J12 T12, c] J T>`: orbits a, b coupled to J12 (integer) and isospin T12 (0 or 1), then c
coupled to total J (half-integer, stored doubled) and total T (1/2 or 3/2, stored doubled as 1 or 3). The state is a plain
product in the pair-coupled scheme: **not** antisymmetrised in the three particles, **not** symmetrised in (a,b) either.
Isospin formalism: no tz labels; for a scalar operator the matrix element does not depend on M_J or M_T, and the stored value is
the non-reduced one (no (2J+1) or (2T+1) factors). Real symmetric matrix per (J, P, T). **[S]**

### 2.3 Which states exist (`ThreeBodyLabChanIso.F90:145-268`)

A lab channel is `(J, P, T)`, `jtot = 1,3,..,2*e3max+3` (doubled), `ptot = +1,-1`, `ttot = 1,3` (doubled)
(`ThreeBodyLabSpaceIso.F90:129-152`). Within a channel the states are enumerated by three nested loops

```
for i1 in 1..norbs:  for i2 in 1..i1:  for i3 in 1..i2:                    # orbit ordering a >= b >= c (index)
    keep triple iff  e1+e2 <= e2max, e1+e3 <= e2max, e2+e3 <= e2max, e1+e2+e3 <= e3max,
                      (-1)^(l1+l2+l3) == P                                  # L:230-237
    for j12 in |j1-j2|/2 .. (j1+j2)/2:   keep iff triangle(2*j12, j3, J)
        for t12 in 0,1:                   keep iff triangle(2*t12, 1, T)       # i.e. T=1/2: t12=0,1 ; T=3/2: t12=1
            skip iff i1 == i2 and (j12+t12) is even                            # L:167, 247, 259
            state index += 1
```

Notes **[S]**:
* Permutation symmetry is used only in the orbit ordering a >= b >= c (index order, which is non-decreasing e). The other
  orderings are never stored. All (J12, T12) pairs are kept for a > b; for a == b only j12 + t12 odd survives (the pair is already
  antisymmetric). The Pauli-forbidden three-body cases (e.g. a=b=c=0s1/2 with T=3/2, or any state that is annihilated by the
  antisymmetriser) are **not removed from the basis**; their matrix elements come out zero (see 5.3, 6.3).
* e-conditions: single-orbit e <= emax is implied by the orbit list; pair sums <= e2max; triple sum <= e3max. Because
  e3max <= e2max in all practical runs the pair conditions are redundant; if e3max > e2max they bite and the imsrg++ reader
  (which only tests e_a+e_b <= E2max and the triple sum) no longer counts the same number of values as the writer
  (section 7.3). Keep e2max >= e3max.
* The oscillator quantum number of a lab state is `N = e1+e2+e3` (`ThreeBodyLabQNs%Nmax`, `ThreeBodyTransCoefIsospin.inc:1004`).
  The transformation is exactly block diagonal in N (energy conservation of HO states, centre of mass included).

### 2.4 Sizes of the lab space **[V]** (channel dimensions equal `count_memory` output, see section 8)

| emax/e2max/e3max | lab channels (J,P,T) | total lab states | largest channel | AS+CM dimension (sum over channels) |
|---|---|---|---|---|
| 4/8/4 | 22 | 715 | 99 (J=3/2, +, T=1/2) | 527 |
| 8/16/8 | 38 | 14278 | 1199 | 12606 |
| 10/20/12 | 54 | 124587 | 7534 | 116842 |
| 14/28/14 | 62 | 307305 | 16075 | 290732 |
| 14/28/16 (production) | 70 | 687352 | 32259 (J=9/2, +, T=1/2) | 657655 |

(The lab space is slightly larger than the antisymmetric subspace; the lab matrices have rank equal to the AS+CM dimension of
the block, so they are rank-deficient by construction.)

---------------------------------------------------------------------------------------------------------------

## 3. Jacobi-side ingredients (interfaces only)

Single frequency, standard 3-body Jacobi coordinates (pair relative, third-particle relative to the pair, total CM), all with the
same HO frequency; this is the choice consistent with the mass ratios d = 1 and d = 2 used in the brackets below. **[S]**

* NAS (non-antisymmetrised, pair-antisymmetric) Jacobi basis, `ThreeBodyJacChanIso.F90:491-523` (`set_jacobi_nas`):
  `|n12 l12 S12 j12 T12 ; n3 l3 j3 ; Jrel, T>` with the pair antisymmetry condition `(-1)^(l12+S12+T12) = -1`,
  `j3 = l3 +- 1/2`, `|j12 - j3| <= Jrel <= j12 + j3`, parity `(-1)^(l12+l3)`, total isospin from `T12 x 1/2`. Quanta
  `E = 2 n12 + l12 + 2 n3 + l3`.
* AS basis: the eigenvectors with eigenvalue 1 of the three-body antisymmetriser A in the NAS basis
  (`set_cfp`, L:546-591; dimension from the trace of A, `get_dim_as`, L:525-544). `cfp(E)%m(i_AS, alpha_NAS)` is
  the (rows = AS states, columns = NAS states) orthonormal matrix. The AS basis is **gauge dependent** (arbitrary
  orthonormal basis of the eigenvalue-1 space from a dense symmetric eigensolver); only the projector `cfp^T cfp` is
  physical. Everything downstream is gauge independent provided the Jacobi V and the T-coefficients use the same cfp.
  Example sizes from the log: J=5/2, +, T=1/2, Nmax 8: 340 NAS, 113 AS; J=5/2, -, T=1/2: 236 NAS, 79 AS.
* "AS+CM" state of the transformation (`ASJacobiCMQNs`, `ThreeBodyTransCoefIsospin.inc:53-58`): `(E, i, Jrel, Ncm, Lcm)`,
  with total quanta `N = E + 2 Ncm + Lcm`, coupling `Jrel (x) Lcm -> J`, parity `P = (-1)^N`.
* The AS dimension per (E, Jrel, T) is checkable independently: it equals the number of antisymmetric three-nucleon
  m-scheme multiplets after dividing out the centre of mass (my `dims3.py` reproduces the logged 113 and 79). **[V]**

---------------------------------------------------------------------------------------------------------------

## 4. The T-coefficients

### 4.1 Definition

For each lab channel (J, P, T) and each N block (N = 0..e3max) NuHamil builds the overlap matrix

    T_N[ (E,i,Jrel,Ncm,Lcm) ; (a,b,c,J12,T12) ]  =  < AS Jacobi (E,i,Jrel;T) (x) CM(Ncm,Lcm) ; J T | a b c; J12 T12; J T >,

rows = AS+CM states with that N (`mat(i_as, i_lab)`), columns = lab states with `e1+e2+e3 = N`
(`TCoefs%mat`, `ThreeBodyTransCoefIsospin.inc:77-110`, `InitTCoefs` L:398-430). It is built in two steps:

    T_AS  =  cfp(E)[i_AS, alpha]   .   T_NAS[alpha, lab]       (one dgemm per (channel, N), `set_as_tcoefs`, L:649-710)

where `T_NAS[(n12 l12 S12 j12 T12 n3 l3 j3 Jrel Ncm Lcm), lab]` is the overlap of the NAS Jacobi+CM state with the lab state.
The cfp factor is block diagonal in (Ncm, Lcm, Jrel, N): only pairs with equal `Ncm, Lcm, N, Jrel` are filled
(L:683-686), and `ChJac = jac channel (Jrel, (-1)^(l12+l3), T)`.

### 4.2 The NAS coefficient (`set_nas_tcoefs`, L:546-644; the active branch, `old_tcoefs` is the older 9j variant L:433-543)

Notation: all angular momenta true values here (the code carries doubled j, J, Jrel, j3, Lambda); hats `^x = sqrt(2x+1)`;
`{..}` Wigner 6j; `(..;..;..)` 9j in the order (la 1/2 ja; lb 1/2 jb; lab S12 Jab).
Moshinsky brackets `<N L, n l ; lambda | n1 l1, n2 l2 ; lambda>_d` with mass ratio d (`gmosh`, MyLibrary.F90:361-582, Kamuntavicius;
stores `tmbk_1` with d = 1 and `tmbk_2` with d = 2, `ThreeBodyTransCoefIsospin.inc:1019-1020`). The lab state is
(a b)Jab,T12 then c; set `ia=(na,la,ja)`, etc.

    T_NAS = delta(T12 = Tab)                                   # isospin: only the pair isospin must match (L:581-582)
          * phase * ^ja ^jb ^jc ^Jab ^S12 ^j12 ^j3 ^Jrel
          * SUM_{lab=|Jab-S12| .. min(Jab+S12, la+lb)}  (2 lab+1)  (la 1/2 ja; lb 1/2 jb; lab S12 Jab)            # LS->jj for the pair
          * SUM_{lit}  < nit lit, n12 l12 ; lab | na la, nb lb ; lab >_{d=1}  { lit l12 lab ; S12 Jab j12 }
          * SUM_{Lambda}  (-1)^(Lambda - Jrel) (2 Lambda+1)  { j12 lit Jab ; jc J Lambda } { Lcm j3 Lambda ; j12 J Jrel }
          * SUM_{lam}  (2 lam+1) { lit lc lam ; 1/2 Lambda jc } { Lcm l3 lam ; 1/2 Lambda j3 }
                        * < Ncm Lcm, n3 l3 ; lam | nit lit, nc lc ; lam >_{d=2}

with overall phase `(-1)^(S12 + l12 + 1 + j12 + lc + l3 + (jc + J)/2)` (doubled jc, J; `(jc+J)/2` is an integer).
Summation limits and quantum-number bookkeeping (L:595-631):
* `lab` from `max(|Jab-S12|, |la-lb|)` to `min(Jab+S12, la+lb)`;
* `lit` (orbital angular momentum of the pair centre-of-mass motion "it") from `max(|lab-l12|, |Jab-j12|)` to
  `min(lab+l12, Jab+j12, 2na+la+2nb+lb-2n12-l12, 2Ncm+Lcm+2n3+l3-2nc-lc)`; `nit` is not summed: it is fixed by pair energy
  conservation, `nit = (2na+la+2nb+lb-2n12-l12-lit)/2`;
* `Lambda` (half-integer) from `max(|j12-J|, |lit-jc|, |Lcm-j3|)` to `min(j12+J, lit+jc, Lcm+j3)`; geometrically Lambda = lit (x) jc = Lcm (x) j3 and J = j12 (x) Lambda;
* `lam` from `max(|lit-lc|, |Lcm-l3|, |1/2-Lambda|)` to `min(lit+lc, Lcm+l3, 1/2+Lambda)`;
* terms below 1e-16 are skipped. Everything is double precision inside the loops; the stored `mat` is cast to the
  working precision only at the end.

Meaning of the pieces: first bracket (d = 1) takes the two lab orbits a, b to pair-relative (n12 l12) and pair-CM (nit lit);
second bracket (d = 2, the pair has twice the mass of nucleon c) takes (pair-CM, c) to (Jacobi coordinate 3: n3 l3) and (total CM: Ncm Lcm);
the 6js/9j recouple the LS/jj coupling schemes: [(l12 S12) j12 ; (l3 1/2) j3] Jrel with [Lcm] to J on the Jacobi side, versus [(la 1/2)ja,(lb 1/2)jb] Jab, jc, J on
the lab side. No sqrt(2) pair-normalisation factors appear (see 5.2). The d = 1 brackets are what `nuham` already has in
`MoshinskyTable`; **d = 2 is new** (rotation angle of the matrix-exponential construction changes; `gmosh` takes d as a
real argument so it is the same Kamuntavicius formula with a different mass ratio). **[S]**, not isolated numerically (no T-coefficient dump
exists in NuHamil).

### 4.3 Sizes and cost **[V counts]**, flop figures are my estimates

Per (channel, N) the T matrix is `n_AS+CM(N)` x `n_lab(N)` (dense). Totals over all channels (`dims3.py`):

| emax/e2max/e3max | T_AS elements (double GiB) | NAS+CM states | (NAS,lab) pairs with T12=Tab | cfp dgemm flops | V-contraction flops (5.1) |
|---|---|---|---|---|---|
| 4/8/4 | 2.3e4 (0.0002) | 1628 | 4.1e4 | 6e6 | 4e6 |
| 8/16/8 | 4.8e6 (0.04) | 38160 | 8.6e6 | 1.4e10 | 1.4e10 |
| 10/20/12 | 2.2e8 (1.65) | 351904 | 4.0e8 | 3.4e12 | 4.9e12 |
| 14/28/14 | 1.08e9 (8.0) | 874634 | - | 3.3e13 | 5.7e13 |
| 14/28/16 | 4.45e9 (33.2) | 1977028 | 8.0e9 | 2.5e14 | 5.1e14 |

NAS+CM counts ignore the `jmax3` cut (not binding when jmax3 >= 2*e3max+3). Cost split: building `T_NAS` is the irregular
part (nested sums with ~4 levels, ~8e9 coefficients at production, each a handful to tens of terms; this is host-side in NuHamil,
OpenMP over lab states, `n_threads_tcoef`); the cfp product and the V contraction are dense GEMMs. NuHamil keeps all T_AS for all
channels resident at once (`InitTransJac2LabSpace`, L:939-964), 33 GiB double at production (16.6 single), but each channel is
independent so a per-channel (or per-(channel, N)) schedule is possible in a new code. The T_NAS stage (the large sparse-ish
coefficient set, ~2e6 x 7e5 entries over blocks, mostly structural zeros from the T12=Tab and parity selections) is the natural
candidate for JAX/GPU; the 6j/9j/bracket tables are small and host-cacheable.

---------------------------------------------------------------------------------------------------------------

## 5. The contraction

### 5.1 Formula and loop structure (`TMTransScalar`, `ThreeBodyLabOpsIso.inc:855-1083`)

For each lab channel (J, P, T) (parallel over channels, `parent_child_procedure`, L:918) with `nlab` states:

    V_lab(J,P,T)  =  6 * SUM_{(Jrel, Prel, Ncm, Lcm)}  T^T  V_jac(Jrel, Prel, T)  T          (nlab x nlab, real symmetric)

where the sum runs over `jrel = 1,3,.. <= min(2*e3max+3, Jmax_jac)` (doubled), `prel = +-1`, `ncm = 0..e3max/2`, `lcm = 0..e3max`
(list `jpnl`, L:893-905, blocks of `nblock` consecutive entries are processed together, default `nblock=400`,
`NuHamilInput.F90:271`: purely a memory-batching parameter for the temporary `tc` arrays, no effect on results), and for each term

* `T` = `tc(jrel,prel,ncm,lcm)%v`, an `m x nlab` array (m = AS Jacobi dimension of the channel (Jrel,Prel,T) over **all** E <= e3max),
  assembled from the per-N T_AS blocks: `tc(vbra, vket) = N3max(nabc)%mat(tcbra, tcket)` with `nabc = E + 2 ncm + lcm`
  (L:1008-1028). Rows whose `E + 2ncm + lcm > e3max` stay zero; columns (lab states) whose `N` differs from `E+2ncm+lcm` stay
  zero. So each term is a tall dense-with-structured-zeros matrix; per (Ncm, Lcm) it couples row block E to column block
  N = E + 2Ncm + Lcm only (diagonal-in-N structure: a block-banded layout, one dense block per E).
* `dgemm( V_jac . T )` then `dgemm( T^T . (V_jac T) )` (L:1058-1059); both are plain dense GEMMs of sizes m x m x nlab and nlab x m x nlab,
  accumulating in `v3` with the factor `fac = 6` folded into the second call.
* The Jacobi operator is symmetric and the spectator (CM) quanta are untouched by V: **V_jac depends on (Jrel, Prel, T) only, not on Ncm, Lcm**.
  This is what makes the structure "dense GEMM per (Ncm, Lcm, Jrel) block": the same V_jac is reused for all CM states.

Dense parts: `V_jac T` and `T^T (V_jac T)`, plus the cfp product. Irregular parts: building T_NAS (nested sums over lab, lit,
Lambda, lam with energy-conservation limits and 6j/9j/bracket lookups), the index bookkeeping that scatters the
`N3max(nabc)%mat` blocks into `tc`, and the final scatter of the lab matrix into the file order (section 6).
Rough contraction cost with the zero blocks exploited (m restricted to E <= e3max - 2Ncm - Lcm and nonzero columns only):
5e14 flop at production (my estimate in the table of 4.3), i.e. tens of seconds to minutes on an H200 in fp64 if well batched (peak fp64 about 60 TFlop/s); NuHamil uses the full m
(all E <= e3max) for every block, so it does more work than this bound.

### 5.2 Normalisation convention of the result **[S, consistent with the check in 5.3]**

Let `|abc>` denote the coupled product state of 2.2 and `A = (1/6) sum_P sgn(P) P` the three-particle antisymmetriser. Since the
AS Jacobi states span the image of A and V_jac is the matrix of the (permutation-symmetric) V_3N in that space,

    V_lab(abc,def) = 6 <abc| A V A |def> = <abc| V (sum_P sgn P) |def>

i.e. the **fully antisymmetrised, un-normalised** matrix element, the same convention as the `me2j` Darmstadt element
(`<ab|V(1-P12)|cd>`, spec-3 section 4.4). T_NAS contains no sqrt(2) or 1/sqrt(6) normalisation factors: for three
distinct orbits the value equals the matrix element between normalised antisymmetric states; for coinciding orbits it is the
un-normalised value (no division by sqrt(n!) of the occupation), exactly what imsrg++ expects (it applies no normalisation to 3N
elements, section 7). The factor 6 = 3! is hard coded (`ThreeBodyLabOpsIso.inc:1046-1048`; the NO2B and monopole variants use the
same 6, `ThreeBodyNO2BIsospin.inc:308-317`, `ThreeBodyVMonIso.inc:185-194`). The Jacobi V already includes all three cyclic pair terms of V_3N
(e.g. `3.d0 * lec * vv` in `NNNForceHOIsospin.F90:612,675,765,868,1005`), so the lab element is that of the total three-body operator. **[S]**

### 5.3 Structural checks visible in the reference file **[V]**

* Entries that must vanish do vanish: of 4748 tuples with `a==b, J12+T12` even, or `d==e, J45+T45` even, or `a==b==c` with T=3/2 and
  `j < 3/2` (and the ket analogues), **0** are non-zero (threshold 1e-9) in the reference file (24890 values).
* The (0s1/2)^3 diagonal block (a=b=c=d=e=f=0s1/2, J=1/2, T=1/2) is rank one as it must be (a single AS state): the four
  allowed entries are +21.46781035 (J12=0,T12=1 / J45=0,T45=1), -21.46781035 (J12=0 / J45=1), -21.46781035 (J12=1 / J45=0), +21.46781035 (J12=1 / J45=1); the
  T=3/2 entry (J12=J45=1) is 0.0. Under the antisymmetrised convention above this means <AS|V|AS> = 21.4678/3 = 7.156 MeV
  for the 3H-like (0s)^3 state (overlap^2 of each product state with the AS state is 1/2). I did not verify the absolute
  factor 6 independently (open question 3). Hermiticity/symmetry is automatic (the matrices are symmetric by construction).

---------------------------------------------------------------------------------------------------------------

## 6. Output files

### 6.1 Name and dispatch

`write_scalar_operator_isospin` (`ThreeBodyLabOpsIso.inc:788-827`) chooses the writer by substring of the name, in this order:
`.me3j.gz` (gzip text), `.me3j` (plain text), `.stream.bin` (raw binary), `.bin` (Fortran sequential unformatted, one record per value),
`.readable.txt` (human readable with labels, drops |v|<1e-8), `.txt`, else plain me3j text with a warning. `file_name_3n` is used verbatim for
"hamil"; with `file_name_3n="default"` the name is `ThBME[_renorm_lambda_NNint]_ChEFT_N2LO_cD..cE.._Local2_IS_hw16_ms{emax}_{e2max}_{e3max}.me3j.gz`
(`GetFileNameThreeBodyLabOp`, L:1391-1442). Also: if the file already exists the transformation is skipped (TMTransFunctions.inc:152).

### 6.2 Tuple loop (identical in all writers, L:1673-1745 text, 1803-1878 gz, 1936-2008 stream, 2055-2128 bin) **[V]**

```
header line (text/gz only)
for i1 in 1..norbs:                                      a
  for i2 in 1..i1:                                       b <= a
    skip if e1+e2 > e2max
    for i3 in 1..i2:                                     c <= b
      skip if e1+e3 > e2max or e2+e3 > e2max or e1+e2+e3 > e3max
      p123 = (-1)^(l1+l2+l3)
      for i4 in 1..i1:                                   d <= a
        i5max = i2 if i4 == i1 else i4
        for i5 in 1..i5max:                              e <= d  (and e <= b if d == a)
          skip if e4+e5 > e2max
          i6max = i3 if (i4 == i1 and i5 == i2) else i5
          for i6 in 1..i6max:                            f <= e  (and f <= c if d,e == a,b)
            skip if e4+e6 > e2max or e5+e6 > e2max or e4+e5+e6 > e3max
            skip if p123 != (-1)^(l4+l5+l6)
            for j12 in |j1-j2|/2 .. (j1+j2)/2:
              for j45 in |j4-j5|/2 .. (j4+j5)/2:
                for J2 in max(|2 j12 - j3|, |2 j45 - j6|) .. min(2 j12 + j3, 2 j45 + j6) step 2:     # doubled total J
                  for t12 in 0,1:
                    for t45 in 0,1:
                      for T2 in 1 .. min(2 t12 + 1, 2 t45 + 1) step 2:                               # doubled total T
                        emit  GetThBME(i1,i2,i3,j12,t12, i4,i5,i6,j45,t45, J2,T2)
```

The bra triple (a,b,c) is >= the ket triple (d,e,f) lexicographically; the J2 loop sits inside the (j12, j45) loops, with the
5 isospin combinations innermost in the fixed order `(T12,T45,2T) = (0,0,1), (0,1,1), (1,0,1), (1,1,1), (1,1,3)`.
One value per tuple. **Every** (j12,j45,J2,isospin) combination allowed by triangle rules is written, including those forced to
zero by the Pauli-type conditions of 2.3 (written as 0.0; `GetThBME` returns 0 when the state is not in the basis,
`ThreeBodyLabOpsIso.inc:243-281`). The numbers of 3-body channels and tuples are consistent: the writer does not enumerate channels
at all, it queries the lab matrix of channel (J2, P, T2) at (bra index, ket index); the (bra, ket) pair is looked up via
`ABC2Index`/`SubIndex` (`ThreeBodyLabChanIso.F90:~100-135`) and the full square channel matrix is read (so bra>=ket in the *file* order is
a triple-wise statement; the lab matrix is symmetric anyway).

### 6.3 Text/gzip layout **[V]**

* Header: one line `NNN int. calculated by NuHamil (Tokyo code), main:<git sha>` (`L:1665`, `L:1794`). The git sha (when NuHamil was built with `VERSION`) is the only metadata.
  No emax/e2max/e3max/hw/lmax/counts: the reader must be told them (section 7).
* Body: values in Fortran `10f16.8` (ten per line, 16 characters each, eight decimals, right aligned, no separators other
  than padding, 160 characters + newline per line). Written through a buffer of `numbf = 10000000` values (a multiple of 10 so
  line breaks never depend on chunking); the last partial line has `n` fields (`(nf16.8)`, n < 10; only if the total is not a multiple of 10).
  Values with |x| >= 1e7 would overflow the field. Precision loss: 8 decimals = 5e-9 MeV absolute; `real(4)` runs print the float value to 8 decimals
  (noise digits beyond ~7 significant), `half` runs print the binary16-rounded value converted to real(4) (relative error ~5e-4).
* gz: libz via `gzip_open(.., "wt")`, `gzip_writeline` per 160-char line (so gunzip gives the plain text). Reference run: 400820 bytes of text
  (2490 lines = header + 2489 lines of 10), **105307 bytes gzip (3.8x)**; first 10 values `0, 0, 0, 21.46781035, 0, 0, 0, -21.46781035, 0, 0`. 24890 values total, of which 19088 non-zero.
* Total number of values (also printed at the end of the run, "Number of three-body matrix elements is N"): **24890** at 4/8/4 (log line 337 = my count = file) **[V]**.

### 6.4 Binary variants

* `.stream.bin` (`write_scalar_operator_binary_stream_isospin`, L:1899-2016): `open(form='unformatted', access='stream')`, **no header, no
  record markers**; the values in the same order, native byte order, element type by precision: `double` -> 8-byte IEEE, `single` -> 4-byte IEEE,
  `half` -> 2-byte IEEE binary16 bit pattern (`my_real16%i16`, `half_precision_floating_points.f90`, round-to-nearest conversion from real(4) via lookup tables). File size = 8/4/2 bytes x N.
* `.bin` (L:2018-2138): sequential unformatted, one `write` per value, i.e. gfortran default 4-byte record markers around each element
  (this is what imsrg++ calls "navratil" format, `ReadWrite.cc:1466-1477`: `uint32 delimiter; float v; uint32 delimiter`); element type real(4)/real(8) per precision.
  Not recommended.
* `.readable.txt` / `.txt`: labelled text, not for imsrg++.

### 6.5 Sizes **[V counts, analytic]**

Number of values N from the loop above (`count_scalar_3bme`, L:2697-2786, identical formula; checked against the reference file
and against a Python re-implementation of the loops): 

| emax/e2max/e3max | N values | text (16.1 B/value) | gz (3.8x, reference ratio) | double stream | single | half |
|---|---|---|---|---|---|---|
| 4/8/4 | 24,890 | 0.40 MB | 0.105 MB (measured) | 0.2 MB | 0.1 MB | 0.05 MB |
| 6/12/6 | 456,320 | 7.3 MB | ~2 MB | 3.7 MB | 1.8 MB | 0.9 MB |
| 8/16/8 | 5,199,790 | 84 MB | ~22 MB | 42 MB | 21 MB | 10 MB |
| 10/20/12 | 273,181,225 | 4.4 GB | ~1.2 GB | 2.2 GB | 1.1 GB | 0.55 GB |
| 12/24/12 | 275,560,660 | 4.4 GB | ~1.2 GB | 2.2 GB | 1.1 GB | 0.55 GB |
| 14/28/14 | 1,437,247,460 | 23 GB | ~6 GB | 11.5 GB | 5.7 GB | 2.9 GB |
| 14/28/16 (production) | 6,367,911,165 | 103 GB | ~27 GB (est.) | 51 GB (47.4 GiB) | 25 GB (23.7 GiB) | 12.7 GB (11.9 GiB) |

(The gz estimate scales the 3.8x seen at e4; production files have a smaller fraction of exact zeros, so 3-3.8x, i.e. 27-34 GB.)
Note the e3max dependence is steep (e3max 12 -> 16 multiplies N by 23) and almost independent of emax once emax >= e3max
(10/20/12 vs 12/24/12: 273 vs 276 million).

In-memory lab matrices (NuHamil, `InitThreeBodyLabOp` L:199-241 allocates the **full square** block `m(nb,nk)` for every channel):
sum of n^2 over lab channels = 1.22e10 elements at production = **90.9 GiB double**, 45.5 single, 22.7 half (largest channel 32259: 7.8 GiB double);
14/28/14: 20.3 GiB double; 10/20/12: 3.8 GiB double. NuHamil's `count_memory=.true.` mode prints only `sum n(n+1)/2` x bytes,
i.e. the lower triangle: 45.458067 GB (double) at 14/28/16, 1.898222 GB at 10/20/12, 0.034253 GB at 8/16/8, 0.002820 at 6/6, 0.000146 at 4/8/4,
**matching my channel-dimension enumeration** (`count3.py`) to all printed digits **[V]**. (Run: `NuHamil.exe` on a namelist with `rank=3, count_memory=.true., lab_3bme_precision=...`; it only builds the lab space and takes < 1 s.)

---------------------------------------------------------------------------------------------------------------

## 7. What imsrg++ reads

### 7.1 Entry points

`imsrg++.cc:473-492`: parameter `3bme_type` (default "full"), `3bme` file, `file3e1max` (default 12), `file3e2max` (default -1 = "2 x e1max"),
`file3e3max` (default 12), model-space `emax`, `e3max` (default 12), `lmax3` (default -1), `no2b_precision`. 

* `full`: `ReadWrite::Read_Darmstadt_3body(file, Hbare, file3e1max, file3e2max, file3e3max)` (`ReadWrite.cc:730-819`): dispatch on
  the **last** extension: `.me3j` plain text; `.gz` through a boost gzip filter (the stream then goes through text extraction `operator>>`);
  `.bin` = skip a **255-byte header** (`HEADERSIZE`, L:30,779-786) then `float32` array; `.h5` HDF5; anything else assumed text. **A NuHamil
  `.stream.bin` has no 255-byte header and may be double/half, so it cannot be read by the `full` path**; use `.me3j.gz` (what the production recipe uses) or
  convert. **[S]**
* `no2b`, `mono`: `ThreeBodyStorage_{no2b,mono}::ReadFile` (`ThreeBodyME/ThreeBodyStorage_no2b.cc:552-810`, `_mono.cc:722-1011`); modes by
  name: `*stream.bin*` -> headerless binary of `StoreType` (float32, or binary16 for `no2b_precision=half`, which is exactly NuHamil's single/half stream
  files), `*.gz*` gzip text, else text. These read a **different file layout** (written by NuHamil's NO2B / monopole writers,
  `ThreeBodyNO2BIsospin.inc:630-762`: loops over i1>=i2, any i3, i4<=i1, i5<=i4, i6 with the same (l,j) as i3, a single J = J12 = J45 and no Jab/Jde/J123 coupling).
  They are out of scope for the first 3N milestone (they cannot be derived from the "full" file without redoing the normal ordering). **[S]**

### 7.2 Layout assumptions of the `full` reader (`ReadWrite.cc:1206-1395` count, `1398-1485` read, `1492-1731` store)

* Text parse: skip first line (`getline`, buffer `LINESIZE 496`), then `infile >> float` for `nread` values, whitespace separated (so any line layout works,
  as long as the order/count is right). Values are stored in `std::vector<float>` (float32: values are rounded to single precision; entries with |V| <= 1e-6
  are not stored). **[S]**
* Orbit list `orbits_remap`: e = 0..min(E1max, model emax), l = e%2 .. min(e, lmax) step 2 with `lmax = E1max` ("lmax truncation for 3body not yet implemented",
  L:1257), j = l-1/2 first. This must be the same list as NuHamil's, which holds if the file was written with `lmax = emax` (default).
  Orbits with e > model emax are dropped from the list; since `a` is the outermost loop and e is non-decreasing in the index, the values for e_a <= emax
  form a **prefix of the file**, so an emax 14 file can be read by an emax <= 14 model space (only the prefix is consumed; everything beyond is ignored).
* The nested loop is exactly the writer's loop (L:1284-1385 vs the loop in 6.2), with the checks: `e_a + e_b <= E2max` (break), `e_a+e_b+e_c <= E3max` (break),
  `e_d+e_e+e_f <= E3max`, parity, the J overlap test, and for each (Jab, Jde) the J2 range `max(|2Jab-jc|,|2Jde-jf|) .. min(2Jab+jc, 2Jde+jf)`
  with `iso_dim = 5` values per J2 (`ISOSPIN_BLOCK_DIMENSION = 5` for a scalar operator, `ThreeBodyME.cc:36`). It does **not** test `e_d+e_e <= E2max`,
  `e_a+e_c`, ... separately: valid only when `E3max <= E2max` (then the omitted pair tests are implied by the triple test since all e >= 0). I verified
  with a Python port that the reader's count equals the writer's count (24890 and 456320 for 4/8/4 and 6/12/6, also with `file3e2max` defaulted to 2 x e1max). **[V]**
* Isospin order inside a J2 block: `tab=0..1`, `ttab=0..1`, `twoT=1..2tab+1`, `twoTT=1..2ttab+1` with the scalar constraint `twoT == twoTT`
  (`|2T - 2T'| <= 2 TRank`): (0,0,1), (0,1,1), (1,0,1), (1,1,1), (1,1,3), same as the writer; J2 is the outer index (L:1621-1647). The (T12, T45, T) triple is the coupled-isospin `(tab, ttab, twoT)` of `SetME_iso`.
* Per value: zero if any orbit has `l > lmax3` or `l > lmax` (model-space cuts). Auto-zero (and flagged "AAAHHHH" with `goodstate=false` if the file has a non-zero there):
  `a==b and (tab+Jab) even`, `d==e and (ttab+JJab) even`, `a==b==c and twoT==3 and j2 < 3`, `d==e==f and twoTT==3 and j2 < 3` (L:1665-1668). The writer (6.2) satisfies these (**[V]** 0 violations at e4).
  Stored only if `e_i <= model emax` for all six orbits and `e_a+e_b+e_c <= model e3max`, `e_d+e_e+e_f <= model e3max` (L:1671-1672); calls
  `Hbare.ThreeBody.SetME_iso(Jab, JJab, twoJC, tab, ttab, twoT, twoTT, a, b, c, d, e, f, V)` with the six file orbit indices mapped to model-space (proton) indices.
* No normalisation or sign correction is applied: the value is used as `V_{abc,def}^{Jab Jde J; tab tde T}`. Hence the file must hold the fully antisymmetrised, un-normalised element
  of 5.2, in the pair-coupled orbit order a>=b>=c, d>=e>=f. imsrg++ stores precisely the same set (`ThreeBodyStorage_iso.cc:80-196`: a>=b>=c, d>=e>=f, a>=d, ties broken
  lexicographically, 5 isospin entries per (Jab, Jde, J)); other orderings are produced on the fly by recoupling with sign/6j factors (`SortOrbits`, `AccessME`),
  which is correct only if the stored element is the fully antisymmetrised one. The set uses `SetME_iso`, which overwrites (not adds). **[S]**
* Hermiticity: only a>=d triples are stored; the bra<->ket flip is done by imsrg++ (`herm_flip`).

### 7.3 Parameters the user must give imsrg++ for a file written with (emax, e2max, e3max)

`file3e1max = emax`, `file3e3max = e3max` and `file3e2max = e2max` (**required**: with the default -1 the imsrg++ build used here stores zero 3N values without a message, checked 2026-10-04, `docs/imsrg-check.md`), `emax <= file emax`, `e3max <= file e3max`
(model-space values), `3bme_type=full`, and `hw`/target mass for Trel as for the 2N part. Wrong `file3e*max` silently misaligns the stream (there is no consistency
check except reading past the end). If the file has fewer values than `nread` (wrong parameters), `infile >> float` fails silently (zeros).
`lmax` in the file must equal emax. Production: a 14/28/16 file read with `emax=14 e3max=16` stores about as many values as the file holds (the file is already in imsrg++'s storage order), i.e. tens of GB (imsrg++ side; not measured).
Storing/reading memory: the reader first allocates `std::vector<float>(nread)` = 4 bytes x 6.37e9 = 25.5 GB for production, plus the isospin storage.

---------------------------------------------------------------------------------------------------------------

## 8. Counts and sizes, summary and derivation

Value count (file/reader): `N = 5 * sum_{(bra, ket): bra>=ket, same parity} nJ(bra, ket)` where `nJ` is the number of
`(J12, J45, J)` combinations (6.2). My `file_tuples_fast` groups triples by `(j1, j2, j3, parity)` and reproduces 24890 / 456320 exactly (the second one against the reader port only,
not against a NuHamil file). Channel dimensions: counting per (J, P, T) with the 2.3 rules (matches `count_memory` for 5 sizes).
Table of the dev and production sizes in 2.4, 4.3, 6.5. The `jtot` loop runs to (2*e3max+3)/2 = 35/2 at e3max 16, but only channels that contain states are kept (70 channels at production, 22 at e4); `jmax3`
(default 51, doubled) should be >= 2*e3max+3 = 35, and the Jacobi operator must be defined for all `Jrel <= jmax3`.

---------------------------------------------------------------------------------------------------------------

## 9. Implications for the JAX implementation

1. A lab block (J, P, T) is independent of the others, but the file order (section 6.2) interleaves all channels (every tuple needs
   its (J, T) block): the writer needs random access to all channel matrices. NuHamil holds all of them (91 GiB double at production). Options: single precision (45 GiB, the
   file only has 8 decimals anyway), half, or a two-pass design (compute channel -> store in a memory-mapped array/npz per channel -> writer). The file is `a`-major, so a second pass
   could process `a` blocks reading only the rows of each channel that belong to the current `a` (channels sorted by state index, which is in the same triple order as the file); this
   needs the lab states of each channel enumerated in the writer's triple order (2.3 order is identical: i1>=i2>=i3 ascending), which it is.
2. T-coefficients per (channel, N): AS+CM x lab dense matrices (33 GiB total at production but per channel at most a few GiB); recompute per channel rather than keep.
3. Everything outside T_NAS is dense GEMM; per (Ncm, Lcm) use only E <= e3max-2Ncm-Lcm rows (NuHamil multiplies the full m, with zero rows included).
4. Bracket tables with d = 2 are needed in addition to d = 1 (hw independent). Mind that the `lit` upper limit uses both bracket energy constraints, so `nit` from the first bracket must satisfy the second.
5. The writer-side conventions to reproduce: orbit order, `a>=b>=c` ordering, Pauli-forbidden entries written as 0.0, 5-fold isospin loop order, `10f16.8` text, header line, and (for imsrg++ with `full`)
   the `.me3j.gz` name.

---------------------------------------------------------------------------------------------------------------

## 10. Numerical checks against the oracle

Available now (reference run at 4/8/4, double): `nnn_3NF400_e4.me3j.gz` (24890 values), `log_3n_e4.txt`.

Done in this spec:
* Value count and structure of the file: 24890 values = 2490 lines, header + 2489 full lines of ten 16-character fields, 400820 bytes uncompressed (2489 x 161 + 91 for the header line), 105307 bytes gzip. **[V]**
* My re-implementation of the writer loop reproduces N = 24890 and positions the zeros exactly: 4748 forbidden tuples, 0 non-zero; the (0s1/2)^3 block rank-1 pattern (5.3). **[V]**
* imsrg++'s reader count equals the writer count (4/8/4, 6/12/6). **[V]**
* Lab channel dimensions equal `count_memory` at 4/8/4, 6/12/6, 8/16/8, 10/20/12, 14/28/16 (to the printed 6 decimals). **[V]**
* AS Jacobi dimensions from the m-scheme count agree with the log (J=5/2: 113 (+), 79 (-)). **[V]**

Proposed tests for the package (to be built; none depends on NuHamil data beyond the one reference file and no `refs/`):
1. **Reference file at 4/8/4, hw 16**: compare the full value stream element-wise (tolerance 1e-6 MeV absolute after the print rounding, 5e-9 for the transformation itself if
   the Jacobi input is exact). Needs the Jacobi AS operator; the `ops/NNN-full_*.bin` files in the scratchpad (and `cfp/cfp_A3_*.bin`) are NuHamil's own intermediate Jacobi V and cfp and give an
   **isolated test of the lab transformation** (use them as input; per-channel sizes are small). Their binary layout is described in the Jacobi spec; the file `UT_...` is the SRG transformation (not needed here).
2. **Identity operator** (no oracle needed): set V_jac = identity on the AS space; the transform must return `6 <abc|A|def> = sum_P sgn(P) <abc|P|def>` for every file tuple, which can be computed
   independently from one-body 6j/9j recoupling in the lab basis (and gives exactly 3 for the rank-one (0s1/2)^3 diagonal element in the pair-coupled basis, a quick check of the factor 6).
   Also N-block structure: with V_jac = identity the lab matrix must be a projector times 6 (idempotency test: (V/6)^2 = V/6 restricted to each N block).
3. **Frequency/CM independence**: with V = a pure function of Jacobi quantum numbers independent of Ncm, Lcm the lab result must not change under a different split between (Ncm, Lcm) blocks
   (i.e. completeness: sum over all (Ncm, Lcm) of T^T T on the AS+CM space is the identity per N block); verifies the d=1 and d=2 brackets and the 9j/6j phases separately from V.
4. **Pair-only operator** (independent of the 3N machinery): if V_jac is the NN Hamiltonian-like operator built from a two-body relative operator via `sum over 3 pairs`, the lab 3N element must
   equal the 3-body lab tensor assembled from the NN `me2j`-type two-body elements (the standard "NN as a 3N operator" test; NuHamil provides `NNNfrom2N` operators for exactly this, `TMTransFunctions.inc:165`). That test would
   also pin down the factor 6 and orbit-ordering conventions. (I did not run it.)
5. imsrg++: read the file (`3bme_type=full`, `file3e1max=4`, `file3e3max=4`, `emax 4 e3max 4`) and compare e.g. 3H/4He energies in the existing container recipe (as done for the NN part, docs of the 4He check).

---------------------------------------------------------------------------------------------------------------

## 11. Open questions / uncertainties

1. **[?]** Absolute factor 6 and the "fully antisymmetrised, un-normalised" interpretation (5.2) are derived from the structure of the code (`fac = 6`, T built from overlaps with plain product states,
   Darmstadt convention) and the consistency of imsrg++'s reader, but not isolated numerically yet (identity-operator test of section 10 settles it). The rank-one check (5.3) shows internal consistency only.
2. **[?]** Exact expression of the T_NAS phase and the orientation of the 6j arguments were read from the active (non-`old_tcoefs`) branch; I did not re-derive them by an independent recoupling calculation. A
   new implementation should be validated by tests 2 and 3 of section 10 (and against the `old_tcoefs` formula, which has a different decomposition: nine 6j/9j sums over `lab, lit, lam, ltot, stot, lreltot`).
3. **[?]** Jacobi conventions that fix the sign of cross terms: the choice of Jacobi coordinates (which pair is "12", the particle-3 coordinate normalisation sqrt(2/3)) and the sign of the (l12, S12) pair coupling are the other
   half of the T-coefficient definition and live in the Jacobi spec; the `Lls/e3o3/e2o4` 6j definitions here assume standard Wigner 6j with doubled arguments (`SixJsStore`, same convention as the NN spec).
4. **[?]** `jmax3`: in the reference run jmax3 = 11 equals 2*e3max+3; production needs jmax3 >= 35, and J_rel above `j3max_initial_3nf` carries only the NN part. How NuHamil treats V_jac for large Jrel (zero for the 3N part) belongs to the Jacobi spec.
5. **[?]** `lmax < emax` is not supported by the imsrg++ 3N reader (assumes l <= emax); NuHamil's `lmax` keyword exists but the writer then enumerates a different orbit list. Only lmax = emax is covered.
6. **[?]** Reading files with `e3max > e2max`: writer and reader counts differ (section 2.3, 7.2); never produce such a file.
7. **[?]** gzip ratio at production (3-3.8x) is extrapolated from the e4 file; real compression of dense non-zero 8-decimal text may be nearer 3x (about 34 GB).
8. **[?]** `lab_3bme_precision="single"` (the NuHamil default) accumulates the GEMMs in float32 (`sgemm`); `half` also uses `sgemm` and rounds the result. The double reference run is the right oracle for a float64 JAX implementation; single/half
   differences of order 1e-6 relative are expected to be noise from NuHamil's own precision, not from the new code.
9. **[?]** imsrg++ memory for production (the `std::vector<float>(nread)` plus hash-indexed storage) was not measured; the `no2b`/`mono` modes avoid that but need NuHamil's NO2B files (different layout, section 7.1).
