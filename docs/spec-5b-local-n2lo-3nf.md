# Spec 5b: local chiral N2LO three-nucleon force as NuHamil computes it

Scope: Phase 5.0, topic "local N2LO 3NF" (coordinate-space kernel -> HO Jacobi isospin basis, before the
Jacobi SRG and before the Jacobi -> lab transform). Source root: `/home/soham/Projects/nuclear-physics/NuHamil-public/src`
(abbreviated `src/`). Labels: **[S]** = read from source only; **[D]** = derived by me on paper from the source (not run);
**[?]** = unsure. Nothing here was run numerically except the reference log timings (section 5). The reference run is
`scratchpad/nnn/{Input_3n_e4.dat,log_3n_e4.txt}` (3NF400: `Regulator="Local"`, `RegulatorPower=2`, `lambda_3nf_local=400`,
`LECs=-0.81,-3.2,5.4,-0.2,0.098`, hw=16, N3max=8, jmax3=11).

Files: `src/ThreeBody/NNNForceLocal.F90` (module `NNNForceLocal`, "NFL" below, ~1640 lines) builds the kernel and
projects it; `src/ThreeBody/NNNForceHOIsospin.F90` ("NFH") calls it per (J,P,T), multiplies the LECs, applies the
CFP (antisymmetrisation) and, for `LNL`, the non-local regulator. Both cite P. Navratil, Few-Body Syst. 41, 117 (2007)
("Nav07") for the formulas; the code is a partial-wave/Racah transcription of that paper's coordinate-space method.

---------------------------------------------------------------------------------------------------

## 0. Pipeline for one (J, P, T) block, in order

1. `set_nnn_interaction_chEFT_n2lo` (NFH:238-268) takes `LECs = [c1, c3, c4, cD, cE]`, builds the **non-antisymmetrised**
   (NAS) HO matrix `work` (nphys x nphys) with `set_nnn_int_chEFT_n2lo_isospin` -> `..._isospin_local` (NFH:415-460, 682-772),
   then forms the matrix on the antisymmetrised, orthonormal states `V_A = CFP^T * work * CFP` (NFH:260). [S]
2. If `Regulator` is `LNL` (local+non-local), `V_A` is further sandwiched: `V_A <- F^T V_A F` with `F = CFP^T R CFP`
   and `R` the HO matrix of a non-local Gaussian regulator (NFH:263-267, 296-413; section 6). [S]
3. The result is the bare 3N Jacobi matrix, stored in the file `NNN-full_j<J>p<+/->t<T>_Nmax..._srg..bin` after the 3-body SRG
   (other specs). Units: MeV. [S]

Only `Regulator` in {`Local`, `LNL`} enters the local code path (NFH:429-451); `NonLocal`/`SemiLocal` read Hebeler's HDF5 files
(a different code path, not part of this spec). N3LO local is "not implemented" (NFH:496-499).

---------------------------------------------------------------------------------------------------

## 1. Physics: what is implemented

### 1.1 Constants and units (src/MyLibrary.F90:8-27) [S]

| symbol | value | note |
|---|---|---|
| hbar c | 197.32705 MeV fm | same constant as the NN code |
| f_pi | 92.2 MeV | |
| g_A | **1.29** (`g_A_GT`) | not 1.27; "corrected Goldberger-Treiman" |
| m_pi | **138.038 MeV** | one pion mass, used for all pions |
| Lambda_chi | 700 MeV | breakdown scale in cD and cE terms |
| nucleon mass in Jacobi HO | `m_red_pn = m_p m_n/(m_p+m_n)` | `nu = 2 m_red hw / hc^2` (NFL:1352); m_p=938.27231, m_n=939.56563 |

LEC units (NFH input, `NuHamil_3BME.py:223-290`): c1, c3, c4 in **GeV^-1** (the `1.d-3` in the prefactors converts to MeV^-1),
cD and cE dimensionless. Named sets: `3NF400` = (-0.81, -3.2, 5.4, -0.2, 0.098), Lambda_local=400, n=2;
`3NF500-BetaDecay-Local` = (-0.81, -3.2, 5.4, 0.83, -0.052), Lambda_local=500; `N3LOEMlnl` (and N4LO variant) = LNL with
Lambda_local=650, Lambda_nonlocal=500, cD=0.7, cE=-0.06, n=2. Defaults (NuHamilInput:259-265): Lambda_local=400,
Lambda_nonlocal=500, `Regulator='Local'`, `RegulatorPower=2`, `Ngrid_3nf=120`, `J12max=-1`, `j3max_initial_3nf=-1` (-> `jmax3`).
The log prints c1/c3/c4 as "GeV-1".

### 1.2 Operator content (standard N2LO form; code coefficients below are consistent with it) [D]

With nucleon labels i,j,k all distinct, momentum transfers q_i, pion propagator D(q)=1/(q^2+m_pi^2):

- two-pion exchange (TPE):
  `V = 1/2 (g_A/2f)^2 sum_{i!=j!=k} (sigma_i.q_i)(sigma_j.q_j) D(q_i) D(q_j) F^{ab}_{ijk} tau_i^a tau_j^b`,
  `F^{ab} = delta^{ab} [ -4 c1 m_pi^2/f^2 + 2 c3/f^2 q_i.q_j ] + c4/f^2 eps^{abg} tau_k^g sigma_k.(q_i x q_j)`;
- one-pion-exchange-contact: `V_D = -(g_A/8f^2)(cD/(f^2 Lambda_chi)) sum_{i!=j!=k} (sigma_j.q_j) D(q_j) (sigma_i.q_j)(tau_i.tau_j)`;
- contact: `V_E = cE/(f^4 Lambda_chi) * 1/2 sum_{j!=k} tau_j.tau_k`.

Each term is **local**: it depends only on q_i (two independent momentum transfers q_i, q_j, third fixed by q_i+q_j+q_k=0), so in
coordinate space it is a multiplicative function of the three positions, times spin-isospin operators. NuHamil evaluates
**one representative topology** with fixed nucleon roles in the Jacobi triple (1,2,3) and obtains the full sum by antisymmetry
(section 3). The roles inferred from the code (isospin/spin/radial structure) are [D]:

| term | central nucleon (vertex with c_i or contact) | operators in code | radial functions |
|---|---|---|---|
| c1, c3, c4 | nucleon **1** | tau_2.tau_3 (c4: tau_1.(tau_2 x tau_3)); sigma_2, sigma_3 (c4: sigma_1 too) | pion line 1-2 -> `fk(k1)(r12)`, pion line 1-3 -> `fkx(k2,x)(r1,r2)` |
| cD | nucleon 1 is the contact partner; pion from 2 | tau_2.tau_3, sigma_2, sigma_3 | pion line `fk(k)(r12)`, contact delta `zx(x)` between 1 and 3 |
| cE | symmetric | tau_2.tau_3 (stands for any pair, x3 by symmetry) | `z0(r12)` * `zx(x)` : regulated delta(r12) delta(r13) |

Jacobi geometry (NFL:286-287, 420-421, 1449-1450, 1524-1525; `x1`,`x2` = grid values of xi1, xi2 in fm) [D]:
`xi1 = (r1 - r2)/sqrt2`, `xi2 = sqrt(2/3) (r3 - (r1+r2)/2)` (mass-weighted Jacobi, same `nu` for both). The code uses
- `s = x1/sqrt2 = |r12|/2` (distance of nucleon 1 from the pair centre; called `r1` or `xi1` in the code),
- `a = x2*sqrt(3/2) = |r3 - R_cm12|`,
- `r12 = sqrt2 * x1` (used in `z0`, `fk`),
- `|r13| = sqrt(s^2 + a^2 - 2 s a cos th)`, which is why `fkx(1,x)`, `fkx(2,x)` integrate over `cos th` with Legendre `P_x`.

### 1.3 Regulator (src/NNNForceLocal.F90:1592-1598) [S]

`local_regulator(q, Lambda, n) = exp( -(q/Lambda)^(2n) )`, q in MeV (`p*hc`), n = `RegulatorPower` (default 2: `exp(-(q/Lambda)^4)`).
It multiplies **each pion/delta Fourier transform separately**, i.e. each independent momentum transfer carries one factor
(no regulator on the third momentum, which is determined by the other two). It sits inside the radial Fourier-Bessel integrals:
`z0, zx, fk, fkx, f1_func, f2_func` all contain it; **no separate short-range regulator** exists for cD/cE besides the same
Gaussian (cE: two regulated deltas; cD: one regulated delta times one regulated Yukawa-tensor).
`Lambda` is `lambda_3nf_local` (field `NNNIntLocal%lambda`, set in `InitNNNIntLocal` NFL:118). The type field
`lec` (NFL:52) is set to 1 (or the optional argument) in `SetNNNIntLocal` (NFL:169-170) and is the multiplier in each prefactor;
**NFH always calls `set(term)` with no `lec` argument, so `lec=1` and the true LEC is applied later** by NFH:765
(`that%m = this%m + 3 * v3 * LEC`). The reference run's `save_3nf_before_lec` option writes the LEC-free pieces.
Careful with `regulator_power`: `InitNNNIntLocal` has a local with an initialiser (implicit SAVE, NFL:115); harmless.

### 1.4 Radial/coordinate functions (NFL:1402-1636) [S]

All integrals use the **p-mesh**: 100-point Gauss-Legendre on `p in [0, 8] fm^-1` (`NMesh_p`, `pmin`, `pmax`), and the
**cos-mesh**: 80-point Gauss-Legendre on `[-1, 1]` (`NMesh_cos`). `mpi = m_pi/hc` (fm^-1). Let `F(p) = local_regulator(p hc, Lambda, n)`.

- `z0(i1) = (1/2pi^2) sum_p w p^2 j0(p r) F(p)`, `r = sqrt2 * x1`. A regulated 3D delta function delta_L(r12) (scalar).
- `zx(x)(i) = (1/2pi^2) sum_p w p^2 j_x(p s) j_x(p a) F(p)`, x = 0..N3max+2. These are the Legendre coefficients of a regulated
  delta between the two vectors: `delta_L(a - s) = sum_x (2x+1) zx(x) P_x(cos)` (up to the sign `(-1)^x` convention absorbed in the
  angular coefficients). [D]
- `fk(k)(i1) = (1/2pi^2) sum_p w p^kk j_k(p r) F(p) / (p^2 + mpi^2)`, `r = sqrt2 x1`, `kk = 4` for k=0,2 and `kk = 3` for k=1. These are
  the radial parts of the pion propagator and its first/second tensor derivatives (k=0: `q^2 D(q)`; k=1: `q D(q)`; k=2: tensor `q_a q_b D(q)`
  rank-2 part). The extra `p^2` of the measure is included in `kk`.
- `fkx(0,x)(i) = (1/2pi^2) sum_p w p^4 j_x(p s) j_x(p a) F(p)/(p^2+mpi^2)`: pion line between 1 and 3 expanded in the partial wave x, k=0 (scalar).
- `fkx(1,x)(i) = 1/2 sum_cos w P_x(cos) f1(r)/r`, `fkx(2,x)(i) = 1/2 sum_cos w P_x(cos) f2(r)/r^2`, `r = |r13|` as above,
  `f1(r) = (1/2pi^2) sum_p w p^3 j_1(p r) F(p)/(p^2+mpi^2)` (NFL:1600-1617), `f2` analogous with `p^4 j_2` (NFL:1619-1636).
  k=1,2 are the vector/tensor-valued Yukawa derivatives expanded in P_x; the vector/tensor direction is later recoupled with the
  binomial factor below.

Binomial expansion of the vector/tensor argument (solid-harmonic addition theorem): the radial factors `xi1^k3 xi2^k4` (with `xi1 = x1/sqrt2 = s`,
`xi2 = x2 sqrt(3/2) = a`) and the weight `fk3k4 = sqrt((2k3+1)(2k4+1)) sqrt( (2k+1)! / ((2k3+1)! (2k4+1)!) )` (NFL:245-247, 372-374, 511-513)
come from splitting `(a - s)` into its two parts with `k3 + k4 = k` (c1: k=1; c3, c4: k2 in {0,2}).

---------------------------------------------------------------------------------------------------

## 2. Term by term: prefactors, spin-isospin factors, angular sums (all [S])

Common definitions. `Jtot`, `j3`, `j6` are stored as **2J, 2j**; `j12, j45, l, s, t` are plain integers. Bra labels `(l12 s12 j12 t12 l3 j3)`, ket
labels `(l45 s45 j45 t45 l6 j6)`. `CGs(a,b,c) = <a0 b0|c0>` (Clebsch-Gordan with zero projections, store initialised NFL:1316).
`sjs` = 6j, `snj` = 9j; `lsj12, lsj3, jjx, kkxy` = 6j stores, `ls12, ls3, spin12` = 9j stores (NFL:1320-1347). The spin/isospin reduced factors
and sqrt(dimension) weights:

```
isospin(t12,t45,T):  tau2_dot_tau3 = sqrt((2t12+1)(2t45+1)) {t12 t45 1;1/2 1/2 1/2} {t12 t45 1;1/2 1/2 T} (-1)^(t12+t45+(2T+1)/2)   (NFL:1129-1137)
                     tau_cross     = sqrt((2t12+1)(2t45+1)) {t12 t45 1;1/2 1/2 T} {1/2 1/2 t45;1/2 1/2 t12;1 1 1} (-1)^(t45+(2T+1)/2) (NFL:1139-1147)
spin(s12,s45):       sqrt((2s12+1)(2s45+1)) {s12 s45 1;1/2 1/2 1/2}                                                                  (NFL:226)
```

(The `6j{1/2 1/2 ...}` pieces make the "tau.tau" factors carry the reduced matrix element `<1/2||tau||1/2>^2 = 6`; that 6 is **not** in
these functions but in the explicit constants 36 = 6x6 (c1), 6 (cE), etc. below; check: for T=3/2, t=1, `6*tau2_dot_tau3 = 6*3*(1/6)*(1/3) = 1 = <tau2.tau3>` for
fully aligned isospins [D].) `isospin_factor` and `spin_factor` are zero => term skipped. The c4 isospin factor vanishes for T=3/2 (log: "set c4: 0.0001 s" for T=3/2).

### 2.1 c1 (NFL:196-317; "Nav07 Eq. 42")

`prefact = lec * m_pi^2 g_A^2 hc^4 1e-3 / f_pi^4 * sqrt((2j12+1)(2j45+1)(2j3+1)(2j6+1)(2l45+1)(2l6+1))` (`dble(j3+1)` = 2j3+1 with doubled j3).
Phase `(-1)^((Jtot-j3)/2 + s12 + j45)`. Radial function on the 80x80 grid:

`V_c1(i) = -36 * prefact * spin * isospin * phase * sum_{k3=0,1; k4=1-k3} fk3k4 sum_{x=0..N3max+2} (2x+1) A_x xi1^k3 xi2^k4 fk(1)(i1) fkx(1,x)(i)`

with `A_x = sum_{r,v,y} [ (-1)^y sqrt((2y+1)(2v+1)(2r+1)) CG(v l45 l12) CG(r l6 l3) CG(x k4 r) CG(y 1 v) * 9j_L(v,1,y; l12 s12 j12; l45 s45 j45) * 9j_L3(r,1,y; l3 1/2 j3; l6 1/2 j6) * 6j(y j12 j45; Jtot j6 j3) * CG(x k3 y) * 6j(k4 1 k3; y x r) ]`.
Bounds: `v in |l12-l45|..l12+l45, r in |l3-l6|..l3+l6`, parity `l12+l45+v` even, `r+l3+l6` even, `x+k4+r` even, `x+k3+y` even. The sum over x is
truncated at `x = N3max + 2` (the Legendre expansion is truncated: since `l12 + l3 <= N3max` the angular momenta that can couple are bounded).

### 2.2 c3 (NFL:319-457; "Eq. 47")

`prefact = lec g_A^2 hc^6 1e-3 / f_pi^4 * sqrt(same dims as c1)`, same phase and spin/isospin factors; overall coefficient `+18`.
Rank decomposition: `k1, k2 in {0,2}` (tensor rank of the two pion-propagator derivative products; `fk1k2 = (-1)^((k1+k2)/2) sqrt((2k1+1)(2k2+1)) CG(1 1 k1) CG(1 1 k2)`,
which vanishes for k=1), `k3+k4 = k2`. Radial: `fk(k1)(i1) fkx(k2,x)(i) xi1^k3 xi2^k4`. Angular sum
`A_x = sum_{r,v,y,z} (-1)^(v+r) sqrt((2v+1)(2r+1)) CG(v l45 l12) CG(r l6 l3) CG(x k4 r) sqrt(2y+1) CG(y k1 v) CG(x k3 y) 6j(k3 k4 k2; r y x) * (-1)^z (2z+1) 9j_L(v,1,z..) 9j_L3(r,1,z..) 6j(z j12 j45; Jtot j6 j3) 6j(1 1 k1; v y z) 6j(1 1 k2; r y z)`.
(`z` is the total rank of the (sigma_i sigma_j) coupled spin tensor combined with orbital.) The `fk3k4` weight is `sqrt((2k3+1)(2k4+1)) sqrt((2k2+1)!/((2k3+1)!(2k4+1)!))`.

### 2.3 c4 (NFL:459-608; "Eq. 49")

Same `k1,k2,k3,k4` structure as c3, isospin factor `tau_cross` (no spin_factor), `prefact` contains `sqrt((2s12+1)(2s45+1))` instead,
phase `(-1)^((Jtot+j3)/2 + j45)`, coefficient **-324** (= -6^2 * 9 [D]). The spin part is not a plain 6j: it involves a rank-`k5` recoupling
(`k5` from `max(|s12-s45|,|v-z|,|k1-1|)` to `min(s12+s45, v+z, k1+1, 2)`, using `spin12` 9j with two spin-1/2 pairs, `ls12(v,k5,z)` and the 6j's `kkxy(k5,1,k1; y v z)`, `kkxy(1 1 1 1; k5 k1)`), because
`sigma_k . (q_i x q_j)` is a rank-1 spin-tensor built from the cross-product coupling of two rank-1 gradients. `z` runs over `|v-1|,|j12-j45|,|r-1|,|j3-j6|/2,|y-1|` .. min.

### 2.4 cD (NFL:610-721; "Eq. 38")

`prefact = lec * 9 g_A / (f_pi^4 Lambda_chi) hc^6 * sqrt(dims as c1)`, phase `(-1)^((Jtot-j3)/2+s12+j45)`, coefficient **-1**:

`V_cD(i) = - prefact * spin * isospin * phase * sum_{k=0,2} sqrt(2k+1) (-1)^(k/2) CG(1 1 k) sum_x (2x+1) CG(x l6 l3) [ sum_{v,z} (-1)^v sqrt(2v+1) CG(v l45 l12) CG(x k v) (2z+1) 9j_L(v,1,z..) 9j_L3(x,1,z..) 6j(z j12 j45;..) 6j(1 1 k; v x z) ] fk(k)(i1) zx(x)(i)`

(`fk` here with `k` = 0 or 2: the tensor part of `(sigma_i.q)(sigma_j.q)` ; no radial polynomial because the vector is `r12` only.)
`set_one_pion_exchange_cd_sigma1` (NFL:723-815) is a variant with `sigma_1` only; **unused** (not called by `SetNNNIntLocal`). `set_contact_ce_old` is a
duplicate of `set_contact_ce`, also unused.

### 2.5 cE (NFL:903-986; "Eq. 15")

`prefact = lec / (f_pi^4 Lambda_chi) hc^6` [units: MeV fm^6]. Returns only if `s12 = s45` (spin-independent contact), isospin factor `tau2_dot_tau3(t12,t45,T)`:

`V_cE(i) = 6 * prefact * isospin * (-1)^((Jtot-1+j6-j3)/2 + l12 + l3 + s12) sqrt((2j12+1)(2j45+1)(2j3+1)(2j6+1)(2l45+1)(2l6+1)) * z0(i1) * sum_x (-1)^x (2x+1) 6j(x l45 l12; s12 j12 j45) 6j(x j12 j45; Jtot j6 j3) 6j(x l6 l3; 1/2 j3 j6) CG(l45 x l12) CG(l6 x l3) zx(x)(i)`

with x from `max(|l12-l45|,|j12-j45|,|j3-j6|/2,|l3-l6|)` to the min of the sums, parity `l12+l45+x` and `l3+l6+x` even. This is a scalar
(rank-0 in spin) potential; the Legendre index x couples orbital angular momenta of both Jacobi vectors. OpenMP over the grid loop `i` (cD, cE only).

### 2.6 What is NOT in the code

No relativistic corrections, no c_i^2/ (1/M) terms, no D (E) with a *different* regulator (cD and cE share the same Gaussian). No
nucleon mass differences or isospin breaking in the 3NF (only `T`, not `Tz`, appears; the HO uses `m_red_pn`). No `J12max` or `Ngrid_3nf` use in the local path (section 4).

---------------------------------------------------------------------------------------------------

## 3. Antisymmetrisation, symmetry, normalisation, phases

- `NNNIntLocal%get` returns the matrix element of the **single representative term** in the NAS Jacobi basis (non-antisymmetric
  states `|n12 l12 s12 j12 t12; n3 l3 j3; J T>`, pair (1,2) antisymmetric by the parity condition `(-1)^(l12+s12+t12) = -1`, NFL:1229). [S]
- `set_inside` (NFH:744-771) assigns `v(nelm) = V_term` and `that%m(i,j) = this%m(i,j) + 3 * V_term * LEC`: the factor **3** replaces the
  sum over the 3 cyclic choices of the roles (i,j,k) (the other 3 odd permutations are equal after the pair antisymmetry and are absorbed in the
  prefactors: c1, c3: factor 1/2 of Nav07 compensates; cD, cE are accounted in the numeric constants 9, 6) [D, **[?] needs the oracle check, see section 7**].
  This is exactly valid only after sandwiching between *fully antisymmetric* states, which is what `CFP^T W CFP` does (NFH:260). The `3 *`
  is applied **before** the CFP transform, i.e. the NAS matrix `work` is not itself symmetrised/hermitian-in-permutation; only `V_A` has the symmetry. [S]
- Hermiticity: `SetNNNIntLocal` computes `chket <= chbra` only, and fills `MatCh(ket,bra) = MatCh(bra,ket)^T` (NFL:176-190, 306-309). Within `set_inside` only `iket <= ibra` is evaluated and mirrored: `that%m(iket,ibra) = this%m(ibra,iket)`, so the NAS matrix is **exactly symmetric** by construction. All elements are real (no complex phases anywhere: Gaunt-type CG with zero projections, 6j, 9j). [S]
- Phases: in the HO radial functions `ho_radial_wf_norm` (src/MyLibrary.F90:750-766) is the **positive-at-origin** convention (`u = r R`, leading coefficient positive, `L_n^{l+1/2}(x^2)`, `x = sqrt(nu) r`), i.e. the standard NuHamil/Moshinsky phase `(-1)^n` for momentum space is not used here. Overall phases `(-1)^((Jtot-j3)/2+s12+j45)` etc. are part of the Racah recoupling of the transposition between coupling orders. [S]
- Normalisation: radial functions are `u = r R` normalised `int u^2 dr = 1` (`prefact = sqrt(2 nu_)` with `nu_ = sqrt(anu)`, `exp_component` includes `l+1` power), so `int dxi1 dxi2 u u V u u` is the radial matrix element of the local kernel `V(xi1, xi2)` **in MeV, no extra Jacobian**. Because
  `V` is local, the delta function in coordinate space has been replaced by the diagonal multiplication; the weights `rw_mesh(i1) rw_mesh(i2)` contain no `r^2`. [S, D]
- Units: matrix elements in MeV. cE kernel: `cE hc^6/(f^4 Lambda_chi)` (MeV fm^6) times regulated delta^2 (fm^-6) -> MeV. [D]
- Integer conventions: `Ttot` is 2T (1 or 3) (NFL:1230 `triag(2*t12,1,Ttot)`), `Jtot` is 2J, `Ptot = +-1`; `j12`, `l`, `s`, `t12` are plain integers. [S]

---------------------------------------------------------------------------------------------------

## 4. Model space, grids, truncations

### 4.1 Channel set (`InitModelSpace`, NFL:1203-1288) [S]

For given `(Jtot=2J, Ptot, Ttot=2T, Nmax=N3max, hw)`: loop `j12 = 0..N3max+1`, `s12 = 0,1` (skip `j12 = N3max+1, s12=0`), `l12 = |j12-s12|..j12+s12`, `t12 = 0,1` with
`(-1)^(l12+s12+t12) = -1` (pair antisymmetry) and `t12` coupling `1/2` to `T`, `l3` from `|abs(Jtot-2 j12)-1|/2` to `(Jtot+2 j12+1)/2`, `l12 + l3 <= N3max`,
parity `(-1)^(l12+l3) = Ptot`, `j3 = |2 l3 - 1|, 2 l3 + 1` (doubled) with the triangle `(j3, j12, J)`. Each channel `alpha` holds `Nho = sum_{N=0}^{(N3max-l12-l3)/2} (N+1)` HO states `(n12, n3)`
with `2 (n12+n3) + l12 + l3 <= N3max`, index map `ns2i(n12,n3)`. The tables `(n12,n3)` for all channels are the **same radial pairs** for every (j12,s12,t12,j3): only `(l12,l3)` matter for the radial states. Lookup in `GetMEFromQuantumNumbers` returns 0 with a printed warning if the channel or `(n12,n3)` is absent.

There is **no truncation in J12 inside the local code** other than `j12 <= N3max+1` (implied by `l12 <= N3max`). The input
parameter `J12max` (NuHamilInput:35, default -1) is passed to `set_nnn_int_chEFT_n2lo_isospin` but is **ignored** for `Local`/`LNL` (only the Hebeler-file reader uses it, NNNFFromFile:226, 581-593). `Ngrid_3nf` (default 120) is likewise ignored in the local path: `NMesh_in` is **not passed** to `InitNNNIntLocal` in NFH:707-709 (so `NMesh_r = 80`, module default, NFL:78).

### 4.2 What each knob does (local path) [S]

| knob | effect |
|---|---|
| `N3max` (`jac%GetNmax()`) | truncation `2(n12+n3)+l12+l3 <= N3max` of the NAS HO basis; sets `l12+l3`, `j12` ranges; the largest Legendre index `x <= N3max+2` and `radial_ho_wf` size `(80, 0:N3max/2, 0:N3max)` |
| `jmax3` (`jac` construction, `Input: jmax3`) | largest `2J` of 3N Jacobi blocks produced; in the reference `jmax3=11` -> J = 1/2..11/2, 24 (J,P,T) blocks |
| `j3max_initial_3nf` (default = `jmax3`) | `if (jac%GetJ() > J3max_initial_3nf) return` (NFH:699): blocks with larger 2J are **zero** 3NF (cut on the *total* 2J in the Jacobi basis, not on j3) |
| `J12max`, `Ngrid_3nf` | not used by the local force |
| `NMesh_r = 80`, `rmin=0`, `rmax = 15 fm` (`xmax_in=15.d0` hard-coded NFH:708) | Gauss-Legendre mesh for **both** xi1 and xi2: `Nxis = 80^2 = 6400` points (`xis(i)`, `i1 = (i-1)/80 + 1` slow, `i2` fast) |
| `NMesh_p = 100`, `[0, 8 fm^-1]`, `NMesh_cos = 80` | quadrature for the Fourier-Bessel and Legendre integrals in the radial functions |
| `lambda_3nf_local`, `RegulatorPower` | regulator width and order |

### 4.3 Projection onto the HO basis (NFL:988-1120) [S]

`transform_xis_to_ho(Nxis, xis, chbra, chket, v)` computes `M = O_bra^T diag(w1 w2 v) O_ket` with
`O_ch(i, (n12,n3)) = u_{n12,l12}(x1_i1) u_{n3,l3}(x2_i2)`, `u` = `ho_radial_wf_norm(n, l, nu, r)` stored in `radial_ho_wf`, `nu = 2 m_red hw / hc^2` (fm^-2) **with the same hw as the Jacobi basis (`jac%GetFrequency()`)**. I.e. a plain 2D Gauss quadrature of the product of four HO functions and the local kernel; the HO matrix of the local operator is *not* the exact matrix element if the quadrature is under-resolved (no exactness check in the code; the `eye` norm check after `return` at NFL:1095 is dead code). The kernel `V(i)` is the **spin-isospin-angular reduced radial kernel for the fixed pair of channels**, so the projection is done per channel pair. `transform_xis_to_ho_element` and `_old` are unused alternatives (scalar loop and explicit diagonal matrix).

Mesh adequacy: the 80-point mesh is not convergence-tested in the code. Estimate for hw=16: `b = hc/sqrt(m hw) ~ 1.6 fm`, the outer turning point for `N = 2n+l` is `~ b sqrt(2N+3)`; at N3max=28 this is ~12.4 fm (inside 15 fm), at N3max=40 ~14.7 fm (at the boundary). Quadrature order for the product of four oscillating functions of degree ~(N3max) is the real constraint. [D, **[?]** must be checked in the oracle, section 7]

---------------------------------------------------------------------------------------------------

## 5. Cost structure (from `log_3n_e4.txt`, 16 OpenMP threads, N3max=8) [S]

Per (J,P,T) block the log prints seconds for `precalculations` and for each `set`:

| block | #NAS states | precalc | c1 | c3 | c4 | cD | cE |
|---|---|---|---|---|---|---|---|
| 1/2+, T=1/2 | 190 | 7.47 | 0.20 | 0.61 | 0.38 | 0.16 | 0.11 |
| 5/2+, T=1/2 | 340 | 7.70 | 1.07 | 3.24 | 1.84 | 0.75 | 0.44 |
| 7/2+, T=1/2 | 320 | 7.46 | 1.11 | 4.49 | 2.62 | 1.07 | 0.66 |
| 9/2+, T=1/2 | 268 | 7.65 | 1.17 | 4.66 | 2.39 | 0.82 | 0.46 |
| 11/2-, T=1/2 | 114 | 7.98 | 0.85 | 2.20 | 1.08 | 0.30 | 0.19 |
| 3/2+, T=3/2 | 150 | 7.99 | 0.30 | 1.02 | ~0 | 0.34 | 0.19 |

Totals for the 24 blocks: precalculations ~24 x 7.7 = **~185 s**; sum of the five `set` terms over the 12 T=1/2 blocks ~ 8.8 + 30.3 + 16.7 + 6.4 + 3.6 = **~66 s**, T=3/2 blocks ~15-20 s more. So at this size **precalculations are ~70%** of the local-force time, even though the
tables inside it do **not depend on J, P, T, LECs** (only on `N3max`, `Lambda`, `n`, and for `radial_ho_wf` on hw); `precalculations` is called once per (J,P,T) from `InitNNNIntLocal` (NFL:122) and recomputed from scratch each time (`release_arrays` deallocates it, NFL:1371-1400). [S]

What dominates `precalculations` (NFL:1307-1369): `init_fkx_function` k=1,2 (NFL:1542-1588). For each `x in 0..N3max+2` (11 values), each of the `Nxis=6400` grid points and each of the 80 `cos` nodes, it calls `f1_func`/`f2_func`, which each re-evaluate a 100-point p-integral with a spherical Bessel
function and the regulator: ~ 11 x 6400 x 80 x 100 x 2 ~ 1.1e9 Bessel evaluations, **recomputed for every x although `f1(r), f2(r)` do not depend on x**. This is the 7.5 s. Next are `zx`, `fkx(0,x)` (`11 x 6400 x 100` double Bessel products, small) and the Racah stores (`CGs`, 6j/9j; sizes grow with `2*N3max+4`) which are cheap here but grow with N3max.

What dominates `set c*` [D]: three nested layers per channel pair:
1. **Angular coefficient sum** (host scalar, serial, table lookups): loops `k3,k1,k2,x,r,v,y,z,(k5)` per pair of channels, ~`(N3max+3) x (2l)^3` terms; produces ~`(k1,k2,k3,x)` coefficients per pair. c3/c4 are the most expensive (`c3 ~ 3 x c1`, `c4 ~ 2 x c1`).
2. **Grid vector assembly** `vi%v(i) = xi1^k3 xi2^k4 fk(k1)(i1) fkx(k2,x)(i)`: an O(Nxis=6400) vector rebuilt and accumulated (`vx = vx + vi*fx`) for **every (x,k..) for every channel pair** (not hoisted: the `vi` basis functions are the same for all channel pairs).
3. **Projection** `transform_xis_to_ho`: builds two `Nxis x Nho` overlap matrices (`get_overlap_xis_ho`, OpenMP) and does `(Nho_bra x 6400) * (6400 x Nho_ket)` DGEMM, per channel pair `(chbra, chket)`; cost `2 Nho_bra Nho_ket 6400` flops plus the overlap rebuild. In the reference, `Nho <= 15`; with larger N3max it grows ~`N3max^2/8` and the GEMMs and the number of channel pairs (`Nalpha^2/2`) take over.

Memory per block: `MatCh(Nalpha, Nalpha)` of dense `Nho_a x Nho_b` blocks (only nonzero if non-skipped); plus tables `fkx (3 x (N3max+3) x 6400)` doubles (small). The NAS matrix `work` is `nphys^2` (e.g. 340^2 at N3max=8; at emax 14 and N3max=16 it is a few thousand, at N3max 28-40 tens of thousands: `nphys^2 x 8` bytes ~ GB), plus CFP (nphys x north).

### 5.1 GPU/linear-algebra suitability and a restructuring that follows from the formulas [D]

Each term has the form `V(alpha, alpha'; i) = sum_a C_a(alpha, alpha') g_a(i)`, where `g_a(i)` are *universal* grid functions indexed by `a = (k1,k2,k3,x)` (c1,c3,c4: ~`2*2*(k2+1)*(N3max+3)` of them; cD: `(k,x)`; cE: `x`) that depend only on Lambda, n, m_pi (not on channel, J, P, T, LECs, hw), and `C_a` is a purely algebraic Racah/CG coefficient (spin, isospin, J). The HO projection then factorises through
`B_a^{(l12 l3 l45 l6)}[(n12 n3),(n45 n6)] = sum_i O_{l12 l3}(i,.) w(i) g_a(i) O_{l45 l6}(i,.)`, which depends only on the four orbital l's and on hw, **not on s, j, t, J, T, P**. So (i) all `g_a` can be tabulated once (and `f1, f2` need only a 1D table in r with interpolation, replacing the 1e9 Bessel calls), (ii) the `B_a` are batched GEMMs over `(l12,l3,l45,l6,a)`, ideal for GPU (one-time, hw-dependent), (iii) the `C_a(alpha,alpha')` are 6j/9j contractions (host NumPy or small JAX gathers), and the final assembly `V = sum_a C_a B_a` is a batched contraction. The CFP sandwich and the non-local regulator sandwich are dense GEMMs (`nphys x nphys` by `nphys x north`) also GPU-suitable. This is a design note; the spec for 5.0 is to reproduce NuHamil's numbers, not necessarily its loop order.

---------------------------------------------------------------------------------------------------

## 6. The `LNL` (local + non-local) variant (NFH:263-267, 296-413) [S]

Same local kernel with `lambda_3nf_local` (e.g. 650), then in the **HO NAS basis** a non-local Gaussian regulator on the *Jacobi* momenta acts on both sides:
`F(p, k) = exp( -( (p^2 + k^2)/2 * hc^2 / Lambda_nl^2 )^n )`, `p` = momentum conjugate to xi1, `k` conjugate to xi2 (equal to `(p^2 + 3q^2/4)` of Hebeler/Nav07 with physical `p,q`),
`n = RegulatorPower` ("limitation: power of local regulator and non-local regulator have to be same", NFH:681). Its HO matrix `R` (diagonal in `l12,s12,j12,t12,l3,j3` and in nothing else) is computed with a 500-point Gauss-Legendre p-mesh on `[0, pmax]`, `pmax = 25*6/sqrt(hw)` fm^-1 (hw in MeV; NFH:318, comment "x^2 = p^2 m w/hbar ~ 25^2"), using the **momentum-space** HO functions `ho_radial_wf_norm(n,l,a,p) (-1)^n` with `a = hc^2/(2 m_red hw)` (inverse of the position-space `nu`, NFH:400-405). Then `V <- (CFP^T R CFP)^T V (CFP^T R CFP)` (NFH:326-337): the regulator is applied after antisymmetrisation as a full matrix product `F^T V F`, with `F` symmetric. For pure `Local` this step is skipped. The 500-point p-mesh integrals cost `O(nphys^2 x 500^2)` scalar work (OpenMP) before the GEMMs.

---------------------------------------------------------------------------------------------------

## 7. Numerical checks against the oracle

NuHamil oracle files for the reference run: `scratchpad/nnn/ops/NNN-full_j<J>p<P>t1_Nmax8_Local2_400_c1_-0.81_c3_-3.2_c4_5.4_cD_-0.2_cE_0.098_srg2_hw16.bin` (24 blocks, after SRG) and `UT_...` (SRG unitary). To obtain
the **pre-SRG** local matrix (what this spec describes) run with `save_3nf_before_lec=.true.` (NFH:711-737): file `ops/N2LO_3NF_HOPW_J<J>P<+/->T<T>_Nmax8_Local2_400_hw16.bin` per block (name from NFH:509-537), containing a `(j,p,t,Nmax,n)` header then five records of the NAS lower triangle
(`n(n+1)/2` doubles each, order c1, c3, c4, cD, cE), each holding the bare representative-term element `V_term` (LEC=1, **without** the factor 3: `set_inside` stores `vtmp = v3` and puts `3*v3*lec` only into the matrix, NFH:764-765) [S]. These files are NAS-basis, pre-CFP, pre-SRG, and are the best direct oracle for sections 2-4.
Independent checks, in order of cost:

1. **Isospin factors by hand.** `tau2_dot_tau3(t12,t45,T)*6` should reproduce the 8x8 (3 isospin-1/2) matrix of `tau_2.tau_3` in the `(t12, 1/2) T` basis: e.g. T=3/2 gives +1; for T=1/2 the 2x2 matrix in the (t12=0,1) basis has off-diagonal elements (compute with explicit 2x2x2 Kronecker products and project on T). c4: `tau_cross` must vanish for T=3/2 (log: "set c4: ~1e-4 s"); test `tau_1.(tau_2 x tau_3)` similarly.
2. **cE, unregulated limit, analytic (cheapest physics check).** Take `Lambda -> infinity` (practically Lambda >> hc/b; needs a finer `NMesh_r`/larger `pmax` in the oracle, since the 80-point 15 fm mesh cannot resolve a delta of width `hc/Lambda`; use the oracle numpy chain instead of NuHamil for this). Then `delta^3(r12) delta^3(r13) = delta^3(xi1) delta^3(xi2) / (3 sqrt3)` (Jacobian of `(xi1,xi2) -> (r12, r13)` is `(sqrt2 * sqrt(3/2))^3 = 3 sqrt3`), and the representative term in antisymmetric states gives
   `<V_E> = 3 * cE hc^6 /(f^4 Lambda_chi) * <tau2.tau3> * 1/(3 sqrt3) * <delta^3(xi1) delta^3(xi2)>`,
   with `<n12 0 n3 0| delta^3(xi1) delta^3(xi2) |n45 0 n6 0> = R_{n12}(0) R_{n3}(0) R_{n45}(0) R_{n6}(0)/(4 pi)^2`, `R_{n0}(0) = sqrt(2) nu_^(3/2) (2/sqrt(pi)) sqrt(Gamma(n+3/2)/n!)`, `nu_ = sqrt(nu)`, spin part `delta_{s12 s45}`, orbital s-waves only (`l12=l3=0`). The result is a **rank-1 outer product in (n12,n3)**, nonzero only for `l12=l3=0`. If the coefficient 6 and the factor 3 are right, this fixes both against a 1e-6 relative target (limited by quadrature in the finite-Lambda case).
3. **cE, finite Lambda, independent 3D quadrature.** For `l12=l3=0` s-wave states compute `int xi1^2 dxi1 xi2^2 dxi2 R R R R delta_L(sqrt2 xi1) <delta_L(|a-s|)>_cos` directly (regulated deltas as `delta_L(r) = (1/2pi^2) int p^2 j0(pr) exp(-(p hc/Lambda)^4) dp`, avg over cos by 80+ point Gauss-Legendre); this does not use any 6j/9j and validates `z0, zx`, the (x) Legendre expansion, the prefactor and the HO projection simultaneously. A second check with `l12=2` or `l3=1` states validates the `(-1)^x` and 6j parts of the cE angular sum (compare against brute-force angular integration over `xi1`-hat, `xi2`-hat).
4. **cD, finite Lambda.** The only non-trivial 1D function is `fk(0/2)`. A brute-force 6D numerical integral for the lowest `J=1/2` channels (spin-isospin by explicit matrices, Cartesian `q` derivatives of the Yukawa potential done analytically) validates the relative sign/factor 9 and the factor-3 vs 6 counting; the overall sign of cD must follow `V_D = -(g_A/8f^2)(cD/f^2 Lambda_chi)...` above (check with `cD > 0` repulsive/attractive test in the lowest channel against the NuHamil value).
5. **Symmetries of each matrix.** `V_A` symmetric to ~1e-12 (time reversal; `set_inside` mirrors exactly); NAS matrix identical in the `chbra<->chket` transpose; `V_A` has the permutation symmetry of the CFP states: replacing the CFP by any other orthonormal basis of the antisymmetric subspace leaves `CFP^T work CFP` eigenvalues invariant. The T=3/2 block of c4 is zero.
6. **Lambda dependence.** At `Lambda -> infinity` the cE and cD matrices become Lambda-independent (cE finite as a delta; cD: Yukawa tail times delta). For 3NF400 (400 MeV) vs 500 MeV the TPE pieces (c1, c3, c4) differ at short distance only: their `T=1/2, J=1/2+` diagonal matrix elements are expected to change smoothly and monotonically; a sign-flip would point to an expansion error. `RegulatorPower` n: the matrix must be continuous in n; `n -> infinity` gives a sharp cutoff.
7. **Cross-check vs the Hebeler non-local files** (if available): the local TPE with `n=2`, Lambda=500 is a different regularisation (cannot match); skip.
8. **End-to-end:** the 3-body Jacobi eigenvalues printed in the log (`Eigen values of original 3-body H`, e.g. J=1/2+, T=1/2: -1.349102, 9.222036, ...) are the first scalar targets for an oracle that builds `T_rel + V_NN + V_3N` in the Jacobi NAS basis (they are invariant under SRG to 1e-6, so no SRG is needed for this check).
9. **Quadrature convergence of the oracle itself** (precondition for trusting 1e-6): vary `NMesh_r` (80, 120, 160), `rmax` (15, 20), `NMesh_p` (100, 200), `pmax` (8, 12) and verify the HO matrix elements at N3max=8 change by <1e-8 relative; this tells how much of the NuHamil result is mesh error rather than physics.

---------------------------------------------------------------------------------------------------

## 8. Open questions / uncertainties

1. **Role of nucleon labels and overall constants.** I inferred "central nucleon = 1" and the 3-fold representative-term counting from the radial dependence and the 6j structure; I did **not** re-derive coefficients `-36, 18, -324, -1 (with 9), 6` from Nav07. The factor of 3 in `set_inside` vs the six permutations of `sum_{i!=j!=k}` for cD (no explicit 1/2) should be verified numerically (check 4 or the oracle). The constants 36 = 6x6 and 6 are consistent with `<1/2||tau||1/2>^2 = <1/2||sigma||1/2>^2 = 6`; 18 (c3) and 324 = 36*9 (c4) are not obviously decomposed.
2. **Exact meaning of `x` expansion sign** in `zx` and `fkx`: the Legendre argument is `cos(angle between a and s)` with a sign convention (`|a-s|` vs `|a+s|`) absorbed into `(-1)^x`/`(-1)^v` phases in the Fortran; reproducing it needs the oracle.
3. **`fkx(1,x)` and `fkx(2,x)` at `r -> 0`** (when `s = a` and `cos -> 1`): `f1(r)/r`, `f2(r)/r^2` evaluated at tiny r (Gauss nodes never hit `cos = 1` exactly; `j1(pr)/r ~ p/3` is finite, `j2(pr)/r^2 ~ p^2/15` also finite in the limit but computed as ratios). Possible round-off; the `spherical_bessel` implementation for small argument was not inspected. Same for `zx`'s guard `x > 42 .and. r*p < 1e-4` (irrelevant for N3max <= 40).
4. **Ignored parameters**: `Ngrid_3nf`, `J12max` do nothing for `Local`/`LNL`. `rmax = 15 fm`, `NMesh_r = 80`, `NMesh_p = 100`, `pmax = 8 fm^-1`, `NMesh_cos = 80` are **hard-coded** and may limit accuracy at large N3max and at large `Lambda` (e.g. for `Lambda = 1000 MeV`, `F(p=8 fm^-1) = exp(-6) ~ 2e-3` is not negligible: the p integral is truncated). Any reproduction that uses different meshes will differ by the mesh error, not only by roundoff; matching NuHamil to 1e-6 may require using **the same meshes** (same Gauss-Legendre nodes), or a convergence study (check 9).
5. **`LNL` momentum-space HO phase** `(-1)^n` and the `pmax = 25*6/sqrt(hw)` heuristic (NFH:318) are copied as-is; I did not check that `R` is converged in `Nmesh = 500` for large N3max, nor the sign convention relative to the position-space `Radial`.
6. **Relation to physical `hw` conversion:** the local force is computed at the Jacobi `hw` of the block (`jac%GetFrequency()`) directly by quadrature; if NuHamil's `hw_target`/frequency conversion (`hw_target` input) is used the 3N file is converted in the HO basis later (not part of this spec).
7. **CFP source.** `jac%GetCFPMat()` (the `cfp/` directory in the reference run) defines the antisymmetric subspace and its normalisation; its phase conventions are the topic of another spec (5a); the sign of the whole `V_A` block depends on them only through `CFP^T ... CFP` (invariant under an orthogonal change of the antisymmetric basis, but not under inconsistent NAS phase conventions between `GetNAS` ordering and the local-force index `ns2i`).
8. **Dead code:** `set_one_pion_exchange_cd_sigma1`, `set_contact_ce_old`, `transform_xis_to_ho_element/_old`, the norm check in `get_overlap_xis_ho`, the unused factor `a` in `init_zx_function` (NFL:1444). They are not on the executed path.
9. **Per-J recomputation:** the 3NF kernel tables are rebuilt per (J,P,T) block (NFL:122); any new implementation can share them across blocks and across hw (except the HO overlaps), but cannot be compared one-to-one with NuHamil's timings per block.
