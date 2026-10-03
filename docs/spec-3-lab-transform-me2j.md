# Spec 3: relative/CM -> lab (HO single-particle) transformation and the me2j file

Scope: stage 3 of NuHamil's NN pipeline for the "hamil" operator, example run
`refs/work/Input_a.dat` (emax=4, e2max=8, hw=16, SRG lambda=2.0, N3LO_EM500).
Fortran paths are relative to `/home/soham/Projects/nuclear-physics/NuHamil-public/src`.
imsrg++ paths are relative to a shallow clone of https://github.com/ragnarstroberg/imsrg (`src/...`).

Status tags: **[V]** = verified numerically against the example output; **[S]** = read from source, consistent with
the numerical check but not isolated by it; **[?]** = unsure/open.

Numerical verification summary (throwaway python, scratchpad `verify.py`, `run1.py`, `run2.py`, `me2j_read.py`):
I re-implemented the transformation described below from the Fortran formulas (Moshinsky bracket transcribed from
`gmosh`, sympy 6j/9j), fed it the bare relative matrix elements `refs/work/NNint_A2_rel_N3LO_EM500_bare_Nmax100_hw16.gz`
(plus the Coulomb term, see 2.3) and compared with `TwBME-HO_NN-only_N3LO_EM500_bare2.0_hw16_emax4_e2max8.me2j.gz`.
170 entries (60 random, 50 with a==b or c==d, 30 with J>=7, 30 with e_a+e_b=e_c+e_d=8), all four columns each:
max |difference| = 5.0e-9 (the file's print rounding, 8 decimals). So the spec below reproduces the file to print precision. **[V]**
(The SRG-evolved file `..._srg2.0_...` has the same format; its relative input needs the stage-2 SRG evolution, not reproduced here.)

---------------------------------------------------------------------------------------------------------------

## 1. Pipeline and flow of control

`TwoBody/TwoBodyManager.F90:77-226` `manage_trans_to_lab`:

1. `sps%init(emax, lmax)`; `lab%init(hw, sps, e2max, snt_mass)` builds lab two-body channels (J,P,Tz) (L:97-98).
   lmax defaults to emax (`NuHamilInput.F90:459`).
2. `rel2lab%init(lab, e2max, Jmax2)` builds the transformation coefficients (L:99). `Jmax2` default 12
   (`NuHamilInput.F90:223`); effective relative Jmax = min(Jmax2, e2max+1) (`TwoBodyTransCoef.F90:569`) = 9 here.
3. `trans_to_lab_op` (L:148-226): build the relative operator in the pn basis (stage 1/2 output, `vnn`), for `hamil`
   simply `oprel = vnn` (L:181-183); truncate to relative `2n+l <= e2max` (L:201-202); then
   `op_lab%TMtrans(rel2lab, oprel)` (L:223) -> `TMTransScalar` (`TwoBody/TwoBodyLabOps.F90:634-803`).
4. `oplab%writef(filename)` (L:136) -> `write_scalar_operator`; for a file name ending `.me2j.gz` and op name `hamil`/`NNint`
   this calls `write_scalar_operator_gzip_me2j` (`TwoBodyLabOps.F90:1869-1877`, writer at 2489-2604).

`pn_form` is forced true; the isospin-formalism path (`TwoBodyTransCoefIso.F90`, `TwoBodyRelCMIsoOps.F90`, ...) is
not used for NN ("NN calculations with isospin formalism is not supported", `TwoBodyManager.F90:25`, log line in `refs/work/log_bare16.txt`). **[V]**
`TwoBodyRelCMOps*.F90` / `trans_to_lab_relcm_op` are only for operators like L, Tel, Tmag, M1_2BC (L:63-66 of the selection
block), not for `hamil`.

## 2. The transformation

### 2.1 Basis and channels

* Lab single-particle HO orbits: `(e=2n+l, n, l, j, tz)`, proton tz=-1, neutron tz=+1 (`SingleParticleState.F90:195-218`,
  `iso2pn` uses z=-1 for proton at L:319-333). Same HO frequency hw for lab, relative and CM (Moshinsky mass ratio d=1). **[V]**
* Lab two-body pn channels (J, P, Tz=(tz1+tz2)/2), `TwoBody/TwoBodyLabChanPN.F90:79-125`:
  * Tz=0 (pn): ordered pair (i1 = proton, i2 = neutron) only (L:109 skips tz1=+1,tz2=-1); all orbit pairs, no antisymmetry restriction,
    including same (n,l,j) for p and n. Product states `|a_p b_n; J>`, NOT symmetrised.
  * Tz=+-1 (pp, nn): i1 >= i2 only (L:111); same orbit (i1==i2) excluded for odd J (L:112). States are normalised
    antisymmetric `|ab;J>_NAS`.
  * Constraints: `e_a+e_b <= e2max`, triangle(ja,jb,J), parity.
  * Swap phase stored for the "other" ordering: `iphase(ii2,ii1) = -(-1)^{(ja+jb)/2 - J}` (with doubled j; i.e. `-(-1)^{ja+jb-J}` with true j) (L:116-117).
    NOTE the extra minus sign even for the pn channel: `<a_n b_p;J| ... >` is defined as `-(-1)^{ja+jb-J}` times `<b_p a_n;J| ...>`
    (fermionic creation-operator ordering). **[V]** (the me_00/me_10 assembly below depends on it and reproduces the file)

### 2.2 Transformation coefficients (T-coefficients)

`TwoBody/TwoBodyTransCoef.F90:334-410` (`InitTCoefs`), mirrored by explicit `explicit_me` at L:688-718.
The relative+CM state is `|(n_rel l_rel, S) j_rel; (N_cm L_cm); J>` with `S` the two-nucleon spin (0 or 1),
`l_rel` coupled with S to `j_rel`, then `j_rel` coupled with `L_cm` to J. For the lab state `|a b; J>` (a=(n_a,l_a,j_a), b likewise),

```
T[(nrel,lrel,S,jrel,Ncm,Lcm) ; (a,b,J)] =
   ft * delab * (-1)^{Lcm + lrel + S + J}
   * sum_{lam = max(|Lcm-lrel|, |J-S|, |la-lb|)}^{min(Lcm+lrel, J+S, la+lb)}
        sqrt((2lam+1)(2S+1)(2ja+1)(2jb+1)) * { la  lb  lam ; 1/2 1/2 S ; ja jb J }          (9j, LS -> jj)
      * sqrt((2lam+1)(2jrel+1)) * { Lcm lrel lam ; S J jrel }                               (6j)
      * <Ncm Lcm, nrel lrel ; lam | na la, nb lb ; lam>_{d=1}                                (Moshinsky bracket)
```

(9j/6j are the standard Wigner symbols; Fortran `snj`/`sjs` take doubled arguments and `ninej` is Talmi's formula = standard.
In the code: 9j = `ninejs%get(2la,2lb,2lam,1,1,2S,ja,jb,2J)` with `ja`,`jb` already stored doubled, 6j = `sixjs%get(2Lcm,2lrel,2lam,2S,2J,2jrel)`.)

* `delab = 1/sqrt(2)` if a==b (same orbit index, hence same tz), else 1 (L:379-380).
* `ft`: for Tz=0, `ft = 1`; for |Tz|=1, `ft = (1 + (-1)^{la+lb+S-Lcm}) / sqrt(2)` (L:391-392).
  Parity conservation requires la+lb+lrel+Lcm even and for |Tz|=1 lrel+S is even, so the bracket is always +2, i.e. **ft = sqrt(2)**
  for pp and nn (the relative states are the normalised antisymmetric T=1 states, lab states |ab;J>_NAS). Combined with
  delab: ft*delab = sqrt(2) for a!=b and 1 for a==b, i.e. normalised-antisymmetrised lab states. **[V]**
* Allowed relative/CM states per lab channel (`set_rel_cm`, L:509-546): N=2nrel+lrel+2Ncm+Lcm <= e2max, jrel in 0..Jmax(9), S in {0,1},
  lrel in |jrel-S|..jrel+S, triangle(Lcm, jrel, J), parity (-1)^{lrel+Lcm}=P, and for |Tz|=1 additionally lrel+S even (T=1 only).
  For Tz=0 both T=0 and T=1 relative states (lrel+S odd and even) are present.
* The HO Moshinsky bracket: `TMbks%init(e2max, 1.d0)` (L:572), `TwoBody` store `StoreCouplings.F90:1184-1256`, computed with
  `MyLibrary.F90:361-582` `gmosh(nl,ll,nr,lr,n1,l1,n2,l2,lm,d)` (Kamuntavicius et al. NPA 695, 191 (2001) formula).
  Call is `tmbks%get(ncm,lcm,nrel,lrel,na,la,nb,lb,lam)` = `gmosh(nl=ncm, ll=lcm, nr=nrel, lr=lrel, n1=na, l1=la, n2=nb, l2=lb, lm=lam, d=1)`,
  i.e. the bracket `<Ncm Lcm, nrel lrel; lam | na la, nb lb; lam>` for the transformation `|n1 l1, n2 l2> = sum |N L, n l> <..|..>`
  of two equal-mass particles with relative coordinate (r1-r2)/sqrt(2), CM coordinate (r1+r2)/sqrt(2), all with the same oscillator length.
  Formula (d=1; `ee=2nl+ll, er=2nr+lr, e1=2n1+l1, e2=2n2+l2`, zero unless ee+er=e1+e2, triangles (ll,lr,lm), (l1,l2,lm)):
  ```
  <..|..> = (-1)^{n1+n2+nr+nl} * 2^{-(e1+e2)/2} * sum_{ed=0}^{min(er,e2)} (-1)^{ed}
            sum_{ld=ed,ed-2,..>=0} sum_{lb=eb,eb-2,..} sum_{lc=ec,..} sum_{la=ea,..}
            [triangle(ld,lb,lr), (ld,lc,l2), (la,lb,l1), (la,ll,lc) must hold]
            9j{la lb l1; lc ld l2; ll lr lm} * g(e1,l1,ea,la,eb,lb) g(e2,l2,ec,lc,ed,ld) g(ee,ll,ea,la,ec,lc) g(er,lr,eb,lb,ed,ld)
     with eb=er-ed, ec=e2-ed, ea=e1-er+ed,
     g(E,L,Ea,La,Eb,Lb) = <La 0 Lb 0|L 0> sqrt((2La+1)(2Lb+1) * (E-L)!!/((Ea-La)!!(Eb-Lb)!!) * (E+L+1)!!/((Ea+La+1)!!(Eb+Lb+1)!!))
  ```
  General d: prefactor `sqrt(d^{e1-er}/(1+d)^{e1+e2})`, inner sign `(-d)^{ed}`. The python transcription of exactly this reproduces the file. **[V]**
  Sign check hint: bracket for (nl=0,ll=0,nr=0,lr=0 | n1=0 l1=0, n2=0 l2=0) = +1; phase convention is the above `(-1)^{n1+n2+nr+nl}`, so no extra `(-1)^{lam}`-type factor. **[S]**

### 2.3 Assembling the lab matrix element

`TMTransScalar` (`TwoBodyLabOps.F90:634-803`), per lab channel (J,P,Tz): for every relative channel group `(jrel, prel, Ncm, Lcm)`
the T-coefficient block `T(nrel-state(lrel,S), lab-pair)` (rows = relative states of that relative channel (jrel,prel,Tz) with
`N_ab = 2nrel+lrel+2Ncm+Lcm` <= e2max, columns = lab pairs with `e_a+e_b = N_ab`) is built (L:723-760) and

```
V_lab(J,P,Tz)[bra, ket] = sum_{jrel,prel,Ncm,Lcm}  T^T  V_rel(jrel,prel,Tz)  T            (dgemm pair, L:763-778)
```

with, explicitly, for lab pairs (a,b) and (c,d):

```
<ab;J|V|cd;J> = sum_{Ncm,Lcm} sum_{S} sum_{jrel: tri(Lcm,jrel,J)}  sum_{(nrel,lrel),(nrel',lrel')}
                   T[(nrel',lrel',S,jrel,Ncm,Lcm);(a,b,J)] * <nrel' lrel' S jrel Tz | V | nrel lrel S jrel Tz> * T[(nrel,lrel,S,jrel,Ncm,Lcm);(c,d,J)]
```

Important points (all **[V]**):
* V is diagonal in (jrel, S, Ncm, Lcm) and independent of J; there is NO separate "CM conservation" factor: the CM quantum numbers
  (Ncm, Lcm) are conserved by V, but `N_rel` is not, so `e_a+e_b` and `e_c+e_d` may differ (bra and ket have independent N_ab, N_cd, both <= e2max).
  The sum over Ncm, Lcm is the sum over all CM states compatible with both.
* The relative operator is block-diagonal in S (`SetTwoBodyScalarChanSpinBreaking`, `TwoBodyRelOps.F90:343-386`, skip when spin
  channels differ, L:380), also for pn. `j_rel > J2max_NNint` (=8 here) is exactly zero (L:357 early return). The relative HO
  rows used are `2n+l <= e2max`, which suffices because lab states have e_a+e_b<=e2max (no further truncation error).
* Antisymmetrisation/normalisation: pp, nn: the T-coefficients already contain the antisymmetriser (ft) and normalisation (delab), so `V_lab`
  is `<ab;J|V|cd;J>_NAS` with the standard 1/sqrt(1+delta) normalisation. pn: lab states are unsymmetrised pn products; no factors.
* No 1/(1+delta) outside the above; no explicit "(1 +/- P)" sums.
* Hermiticity: each channel matrix is symmetric by construction (`T^T V T` with symmetric V). Only `chket <= chbra` channels are stored (`GetTBME`, L:560-601).
* The relative matrix elements in the pn z=0 channel are the plain pn-basis `<n' l' S j|V|n l S j>` (T=0 and T=1 mixed, charge-dependence retained).
  The z=-1 (pp) channel contains the Coulomb force (see 3).

### 2.4 Parameters that matter

* hw only enters through the relative HO matrix elements (stage 1/2) and the Coulomb term; the transformation itself is hw-independent
  (dimensionless brackets). The same hw labels lab, relative and CM oscillators. The lab length is `b = hbar c / sqrt(m_N hbar omega)` (m_N = 938.9 MeV)
  only matters when interpreting the file (imsrg++ is told hw separately, header does not contain it).
* e2max truncation: `e_a+e_b <= e2max` and `e_c+e_d <= e2max`; `e_a, e_b <= emax`, `l <= lmax` (lmax=emax by default).

---------------------------------------------------------------------------------------------------------------

## 3. Which operator is written

* The operator is `hamil` = **potential V only** (stage-1/2 relative interaction, including SRG-induced evolution as `V_srg = H_srg - T_rel`
  with `T_rel` = relative HO kinetic energy; `NNForce.F90:737-742`), plus the Coulomb potential in the pp channel. **No kinetic energy,
  no Hcm, no one-body part is in the file.** **[V]**: the python reproduction used only V(+Coulomb) with no T.
* Coulomb: `coul` defaults to `.true.` (`NuHamilInput.F90:248`) and the example input does not override it, so the file is "with Coulomb":
  `add_coulomb_force` (`TwoBody/NNForce.F90:286-350`) adds, to the relative **pp** channels only (`two_ch%GetZ() /= -1` skipped, L:325),
  `<n l|(hbar c/alpha)/r|n' l>` between relative HO radial functions (diagonal in l, S, J) with oscillator parameter
  `nu^2 = 1/b_rel^2 = 0.5*m_proton*hw/hbarc^2`, evaluated by 500-point Gauss-Legendre on [0, 40] fm, alpha = 137.035999
  (`hc/alpha`: `factor = hc/alpha`, L:306). For a "bare" run it is added after reading the bare file (L:236-243); for SRG
  with `evolve_coulomb=.true.` (default, `NuHamilInput.F90:250`) it is added before the SRG flow so it is evolved (L:224-228).
  The bare-file reproduction confirms: pp(0s1/2,0s1/2;J=0 col 4) = -1.6853 with Coulomb vs -2.1609 without; file = -1.68531158. **[V]**
  If a file name contains `_wocoul`, Coulomb was off (only in the default name scheme, irrelevant here since `file_name_nn` is given).
* Trel handling is entirely on the imsrg++ side. imsrg++ (`src/imsrg++.cc:516-520`): after `ReadBareTBME_Darmstadt`, does
  `Hbare += imsrg_util::Trel_Op(modelspace)` (unless fmt2=nushellx, atomic, or hw_trap>0), with
  `Trel_Op = KineticEnergy_Op - TCM_Op` (`src/imsrg_util/imsrg_util.cc:743-748`):
  * `KineticEnergy_Op`: one-body, `<n l|T|n l> = (hw/2)(2n+l+3/2)`, `<n l|T|n+1 l> = (hw/2) sqrt((n+1)(n+l+3/2))` with `hw = modelspace.GetHbarOmega()` (L:717-737).
  * `TCM_Op` (L:821-): one-body `T/A`; two-body `p1.p2/(m A)` via `Calculate_p1p2` (Moshinsky-free HO formula), `A = modelspace.GetTargetMass()`.
    So Trel = (1-1/A) sum_i T_i - (1/(mA)) sum_{i<j} p_i.p_j, with A the target-nucleus mass number (not `snt_mass` of NuHamil, which is only for snt files). **[S]**
  The user must give imsrg++ the same hw (not stored in the file) and the file's emax/e2max (`file2e1max`, `file2e2max`, `file2lmax`; defaults
  in the reader: emax = model-space Emax, E2max = 2*emax, lmax = emax; `ReadWrite.cc:841-850`). **[S]**
  The "A-dependence" therefore enters only through imsrg++'s Trel; NuHamil's SRG in the 2-body relative space uses `T_rel` of the A=2 problem
  (m_red_pn, `lc = lambda*hc/sqrt(2 m_red)`, alpha_srg = lc^-4, `NNForce.F90:644-656`, pn_same_mass default true) and does not know A.
  (There is a separate `write_scalar_operator_ascii_mfd` writer that also writes Trel and H_HO_rel scaled by hw; not used for me2j.)

---------------------------------------------------------------------------------------------------------------

## 4. me2j file format

Writer: `TwoBody/TwoBodyLabOps.F90:2489-2604` (`.me2j.gz`); ASCII twin `write_scalar_operator_ascii_me2j` (L:2281-2376, same layout,
uncompressed, no line-length check); binary twin `.me2j.bin` (L:2606-2702, stream-unformatted floats, no header here). Reader in imsrg++:
`src/ReadWrite.cc:664-722` (dispatch), `:834-963` (`ReadBareTBME_Darmstadt_from_stream`). All **[V]** against the example gz unless noted.

### 4.1 Container

* gzip-compressed text, written through libz via `gzip_open(f,"wt")` (`MyLibrary`; `gzip_writeline` per line). Compressed 131162 bytes, uncompressed 440200
  bytes (`gzip -l`). Plain `gzip -dc` works; the extension `.gz` selects gzip in both the NuHamil writer and imsrg++ (`filename.substr(...)==".gz"`).
* Line 1: free-text header, no counts, no emax/hw. Example:
  `NN int. calculated by NuHamil (Tokyo code), main:4d310c14970d9b47fa6f40dfde78d03b61afb526`
  (`'NN int.' // ' calculated by NuHamil (Tokyo code)' // ', ' // VERSION`, L:2523-2528). Readers just skip the line
  (`infile.getline`, `ReadWrite.cc:880`).
* Lines 2..: floats in Fortran `10f16.8` (10 per line, each field 16 chars wide, 8 decimals, right-aligned, line length 160, no separators other than
  padding; values with |x| >= 1e7 would overflow the field into `*`). The data are one continuous stream; the last line holds the remainder
  (`(Nf16.8)`, N<=9 or 10; here 6 values, total values = 27336 = 4*6834 -> 2733 full lines + 1 line of 6; file has 2735 lines incl. header and a final newline). The number of values is always a multiple of 4.
* Units: MeV. No hbar-omega or A scaling (the "unscaled" potential).

### 4.2 Orbit ordering (the "remap" orbit list; isospin-less (n,l,j))

`SingleParticleState.F90:InitOrbitsIsospin` (L:119-155) and imsrg++ reader L:855-872: loop `e = 0..emax`, then `l = 0..min(e,lmax)` with `(e-l)` even
(`n=(e-l)/2`), then `j = l-1/2` (skipped for l=0) before `j = l+1/2`. Orbit index (0-based): 0:0s1/2, 1:0p1/2, 2:0p3/2, 3:1s1/2, 4:0d3/2, 5:0d5/2, 6:1p1/2, 7:1p3/2,
8:0f5/2, 9:0f7/2, 10:2s1/2, 11:1d3/2, 12:1d5/2, 13:0g7/2, 14:0g9/2 for emax=4 (15 orbits). Note e=2 lists l=0 (n=1) before l=2 (n=0).

### 4.3 Loop order (exact; writer L:2528-2578 = imsrg++ reader L:884-956, verified identical)

```
for a in 0..norb-1:
  for b in 0..a:                         if e_a+e_b > e2max: skip
    for c in 0..a:
      dmax = (c==a ? b : c)
      for d in 0..dmax:                  if e_c+e_d > e2max: skip
        if (l_a+l_b+l_c+l_d) odd: skip   # (-1)^(la+lb) /= (-1)^(lc+ld)
        Jmin = max(|ja-jb|,|jc-jd|)/2 ; Jmax = min(ja+jb, jc+jd)/2   (j doubled, integer division, i.e. true J)
        for J in Jmin..Jmax:             if Jmin>Jmax: skip all
          emit 4 numbers: [ V(T=0,Tz=0), V_nn(T=1,Tz=+1), V_pn(T=1,Tz=0), V_pp(T=1,Tz=-1) ]
```

Ordering of the 4 values in the stream: **(0,0) , (1,+1)=nn , (1,0) , (1,-1)=pp** (writer v_buffer(1..4) = me_00, me_nn, me_10, me_pp, L:2593-2596 region;
reader comment `ReadWrite.cc:896`). The unrestricted-by-J loop covers J parity both ways: no J,T selection rules are applied (entries that vanish
by symmetry are written as 0.00000000, e.g. a==b with T=0 and J even, or a==b with T=1 and J odd). Checked: 0 of 865 forbidden T=0 entries and 0 of 1053 forbidden T=1 entries are non-zero. **[V]**
Entry count with emax=4, e2max=8: 6834 (a,b,c,d,J) tuples (reproduced by my loop in `me2j_read.py`), x4 = 27336 values = file size. **[V]**
(Reading code in imsrg++ truncates by `emax`, `E2max`, `lmax` of the model space, but still consumes the whole stream in the file's loop order, `ReadWrite.cc:921-944`.)

### 4.4 What the four numbers are (conventions)

Let `ap, an` be proton/neutron versions of orbit a. `GetTBME(...)` returns the lab matrix element of 2.3 with the swap phase of 2.1
for out-of-order index pairs. Then (L:2540-2578):

```
norm_fact = (a==b ? sqrt2 : 1) * (c==d ? sqrt2 : 1)
me_pp = GetTBME(ap,bp,cp,dp,J) * norm_fact
me_nn = GetTBME(an,bn,cn,dn,J) * norm_fact
me_00 = 0.5*( TBME(ap,bn,cp,dn) + TBME(an,bp,cn,dp) - TBME(ap,bn,cn,dp) - TBME(an,bp,cp,dn) )
me_10 = 0.5*( TBME(ap,bn,cp,dn) + TBME(an,bp,cn,dp) + TBME(ap,bn,cn,dp) + TBME(an,bp,cp,dn) )
```
(no norm_fact on me_00, me_10; here `TBME(an,bp,...)` etc. use the stored-order swap phase `-(-1)^{ja+jb-J}` of 2.1, and J (all four) is the same J.)
This is exactly the "un-normalised" Darmstadt convention: a state with identical orbits (a==b) is NOT divided by sqrt2 relative to the
distinguishable convention, i.e. `<ab;JT|V|cd;JT>_unnorm = sqrt(1+delta_ab) sqrt(1+delta_cd) <ab;JT|V|cd;JT>_NAS`.
imsrg++ undoes this on reading (`ReadWrite.cc:928-944`): `norm_factor = (a==b ? 1/sqrt2 : 1)*(c==d ? 1/sqrt2 : 1)`, then
`SetTBME(J,parity,-1,a,b,c,d, pp*norm)`, `SetTBME(J,parity,+1,a+1,...,nn*norm)`, `Set_pn_TBME_from_iso(J,T=1,0,...,10*norm)` (only if `norm>0.9 or J even`),
`Set_pn_TBME_from_iso(J,T=0,0,...,00*norm)` (only if `norm>0.9 or J odd`). The pn columns carry the sqrt2 automatically for identical orbits
(for a==b the four terms are equal, so `0.5*(sum of 4)` = 2*X vs NAS value X), which is why NuHamil does not multiply them explicitly. **[V]** (verified that
my pp/nn with explicit norm_fact and pn with the 0.5*(..) combos match the file for a==b / c==d / mixed cases.)
imsrg++ stores into float (`float tbme_pp,...`, `ReadWrite.cc:874`).
imsrg++'s own writer `Write_me2j` (`ReadWrite.cc:3123-3375`) uses the same convention (`// me2j format is unnormalized`).

Sanity: `<0s1/2 0s1/2; J=0|V|0s1/2 0s1/2; J=0>` in the SRG file: T=0: 0.0, nn: -15.12895093, pn(T=1): -15.52660206, pp: -13.58314035
(normalised NAS nn value = -15.129/2 = -7.564 MeV). Charge-independence-breaking rms: rms(nn-pn)=0.025 MeV, rms(pp-pn)=0.129 MeV (Coulomb) over all entries, rms(pn)=0.65 MeV.

### 4.5 Isospin vs pn naming

The file stores isospin-coupled T=0 (Tz=0) and T=1 (Tz=+1,0,-1) elements, not "explicit pp, pn, nn". The T=1 Tz=0 and T=0 pieces are
the symmetrised/antisymmetrised combos of the pn-basis lab matrix elements above, so charge-independence breaking within pn (T=0<->T=1 mixing, e.g. 1P1-3P1 type
S-mixing) is dropped because the relative operator is S-diagonal and because only the diagonal T blocks are written. (me_00 and me_10 only use the
(T,T) blocks of the pn matrix; T-mixing terms are discarded by construction. **[S]**; and S-mixing is dropped already in the relative ME, see 2.3.)

---------------------------------------------------------------------------------------------------------------

## 5. Constants and conventions

NuHamil, `MyLibrary.F90:8-14`: `hc = 197.32705` MeV fm; `m_proton = 938.27231`; `m_neutron = 939.56563`; `m_nucleon = (mp+mn)/2 = 938.918970`;
`m_red_pn = mp*mn/(mp+mn)`; `alpha = 137.035999` (inverse fine-structure constant despite the name).
imsrg++ `src/PhysicalConstants.hh:13-17`: HBARC 197.3269718, M_PROTON 938.2720813, M_NEUTRON 939.5654133 (slightly different, irrelevant to the file).

HO length / frequency:
* hw is in MeV, given in the input (`hw=16`). Oscillator parameter of a particle of mass m: `b^2 = hbar^2/(m hbar omega) = hc^2/(m hw)`.
  Relative coordinate (r1-r2) with reduced mass: `b_rel^2 = hc^2/(m_red hw)` (momentum-space HO set up `NNForce.F90:416`, `a = hc**2/(m_red*hw)`; with
  pn_same_mass=true, m_red = m_red_pn for all of pp, pn, nn, `NNForce.F90:409-411`); in the Coulomb code the same relative orbital uses `m_proton/2` (L:311). Both are in the relative-space
  part (stages 1, 2), not the lab transform. Radial functions: `R_nl(r)=sqrt(2 nu) sqrt(n!/Gamma(n+l+3/2)) x^{l+1} e^{-x^2/2} L_n^{l+1/2}(x^2)`, `x=nu r`, `nu^2=anu`,
  positive at the origin (Laguerre positive; `MyLibrary.F90:750-766`).
* The lab transformation is mass-independent (d=1: equal masses). The proton-neutron mass difference is not represented in the lab frame at all.
* hw dependence of the me2j: via the relative ME only (V depends on hw because it is the HO matrix element of a fixed momentum-space potential; SRG adds the
  hw-dependent T_rel in the HO basis). There is no explicit hw in the transformation or in the file.

---------------------------------------------------------------------------------------------------------------

## 6. Reimplementation checklist (for a JAX package)

1. Build lab orbits and loop exactly as 4.2/4.3; for each (a,b,c,d,J) compute the six pn lab elements of 2.3 for (pp, nn, pn x4 permutations).
2. T-coefficients per 2.2 with `d=1` Moshinsky brackets (precompute brackets per N<=e2max, 9j with (1/2,1/2) spin row, 6j), `ft`, `delab` exactly as given.
3. Contract with relative V (S-diagonal, J<=8, N_rel<=e2max, pn basis; pp includes Coulomb): `T^T V T` per (jrel, Ncm, Lcm, S).
4. Apply me2j assembly (4.4) with sqrt2 factors, write `10f16.8` text and gzip.
5. For validation, the bare pair `NNint_A2_rel_..._bare_Nmax100_hw16.gz` -> `TwBME-..._bare2.0_hw16_emax4_e2max8.me2j.gz` is a complete test of this stage alone
   (max abs deviation 5e-9; the file string "bare2.0" is just the filename, lambda is ignored for renorm=bare).

## 7. Open questions / unsure

* **[?]** The relative file's exact bare/SRG content relative to the nominal NN interaction (cutoffs, `Nmax=100`, mesh) is stage 1; here I only confirmed
  the transformation. The SRG-evolved me2j (`..._srg2.0_...`) was not reproduced end-to-end (needs stage-2 SRG evolution).
* **[?]** Whether other defaults (e.g. a `Jmax2`, `emax_convert`) alter the `.me2j.gz` produced when `lmax < emax` (not exercised).
* **[?]** Sign convention of the pn exchange phase `-(-1)^{ja+jb-J}` is empirical-consistent with the reader (verified via reproduction), but the
  conceptual meaning (creation-operator ordering vs isospin formalism) is my interpretation.
* **[?]** Values |x|>=1e7 would corrupt the `f16.8` field; not an issue for NN-sized values, but the writer has no guard.
* **[?]** imsrg++'s `Calculate_p1p2` was not numerically cross-checked against NuHamil's T_rel (out of scope; the file has no kinetic part).
