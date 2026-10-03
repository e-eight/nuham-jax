# Spec 1: NN momentum-space input file -> bare NN matrix elements in the HO relative basis

Scope: stage 1 of the NN pipeline. Source root: `/home/soham/Projects/nuclear-physics/NuHamil-public/src`
(abbreviated `src/`). Labels: **[V]** = verified numerically in Python by me; **[S]** = read from source only; **[?]** = unsure.

Everything below was cross-checked by re-implementing the projection in numpy and reproducing
`refs/work/NNint_A2_rel_N3LO_EM500_bare_Nmax100_hw16.gz` to the printed precision (see section 8).

---------------------------------------------------------------------------------------------------

## 0. Pipeline summary (for the example run `refs/work/Input_a.dat`)

Input: rank=2, hw=16, renorm="srg", lambda=2.0, emax=4, e2max=8, NNInt="N3LO_EM500", input_nn_file=...bin.
Defaults that matter (src/NuHamilInput.F90:215-251): N2max=100, J2max_NNint=8, jmax2=12, nmesh2=100, pmax2=8.0,
renorm_space2='ho', pn_same_mass=.true., coul=.true., evolve_coulomb=.true., pn_form=.false. (forced true).

1. `TwoBodyManagers` (src/TwoBody/TwoBodyManager.F90:10-26) forces `pn_form=.true.` ("NN calculations with isospin formalism is not supported"). So the NN code path is the proton-neutron (Tz) formalism.
2. `trans_to_lab_op` (TwoBodyManager.F90:235-360) builds `relspin` = TwoBodyRelSpaceSpinHOBasis(hw, N2max=100, J2max_NNint=8), calls `vnnspin%setNNForceHO(Uspin, input)` with `input%N_ls = e2max` (N_ls only matters for Lee-Suzuki).
3. `SetNNForceHO` (src/TwoBody/NNForce.F90:117-244):
   - cache file name = `<path_to_NNrel>/NNint_A2_rel_<NNInt>_bare_Nmax<N2max>_hw<hw>.gz` (NNForce.F90:538-548). It is keyed on **N2max (=100), not e2max**, and on hw; NOT on lambda/renorm.
   - if the file does not exist (or renorm_space2=='mom'): read momentum file, project to HO (this spec), write the cache (if renorm_space2=='ho'; NNForce.F90:195-197). The cache is written BEFORE any SRG and BEFORE Coulomb is added, so it is pure bare strong NN, no Coulomb. [V: deuteron test, section 8]
   - else read the cache (`ReadNNForceHORelative`).
   - then (other slices) SRG in HO space (`renorm_ho_space`, NNForce.F90:624) and `add_coulomb_force` (NNForce.F90:286) where applicable.
4. `SetTwoBodyScalarSpinBreaking` (src/TwoBody/TwoBodyRelOps.F90:302-386) copies the (n,l,S,J,P,Tz) spin-basis matrices into the TwoBodyRelOp channel basis (a pure index map; entries with S1 != S2 are zero, because the NN potential is block diagonal in (J,P,S,Tz) channels, and J > J2max_NNint=8 channels are left zero).

---------------------------------------------------------------------------------------------------

## 1. Momentum-space file `input_nn_files/N3LO_EM500_kmax8_N100_Jmax8.bin`

Reader: `read_nn_mom` (src/TwoBody/NNForce.F90:1676-1712). Writer: `write_nn_mom` (NNForce.F90:1620-1674, the `.bin` branch at 1649-1673).
Python readers exist in the repo: `input_nn_files/plot_nn_file.py`, `plot_local_projection.py` (consistent with what follows).

### 1.1 Container
- Fortran `open(..., form='unformatted', access='stream')`. This is a raw **stream** file: **NO Fortran record markers** at all (no 4-byte length prefixes). [V: file size = 12 + 16*100 + 54*20 + 8*sum(ndim^2) exactly = 8,162,692 bytes, parsed to EOF with no remainder]
- Native little-endian (x86). int32 = Fortran default `integer`, float64 = `real(8)`.

### 1.2 Layout (byte-exact)
```
offset 0   : int32 NMesh                (=100)
offset 4   : int32 Jmax                 (=8)
offset 8   : int32 NChan                (=54)
offset 12  : float64 k[NMesh]           mesh points, fm^-1, ascending
           : float64 w[NMesh]           mesh weights, fm^-1
then NChan times:
           : int32 J, int32 P, int32 S, int32 Tz, int32 ndim     (20 bytes)
           : float64 V[ndim*ndim]       Fortran column-major dump of the (ndim x ndim) matrix
```
Python: `np.frombuffer(b,'<f8',count,offset)`, then `.reshape(ndim,ndim)`. The matrices are symmetric (max |V - V^T| = 4e-14 [V]) so row/column-major ordering is immaterial in practice. Reading with `reshape(nd,nd)` (C order) gives V^T = V.

### 1.3 Values seen [V]
- NMesh=100, Jmax=8, NChan=54.
- k[0]=0.00114509, k[1]=0.0060322, ..., k[99]=7.99885491 fm^-1. w[0]=0.00293854, w[1]=0.00683757, ..., symmetric; sum(w)=8.000000 (to 1e-15).
- The mesh is exactly **Gauss-Legendre with 100 points on [0, 8] fm^-1** (GSL `gsl_integration_glfixed` mapped to [x1,x2], MyLibrary.F90:866-891). Reproduced with `numpy.polynomial.legendre.leggauss(100)`: x=4*(x+1), w=4*w; max |dk| = 4e-16, max |dw| = 2e-14. So the file mesh is the quadrature, and w are plain GL weights for the measure dk (NOT multiplied by k^2, NOT sqrt).
- The mesh in the file overrides whatever mesh the code created from `pmax2/nmesh2`: `read_nn_mom` calls `two%SetMeshWeight(pmesh,wmesh)` (NNForce.F90:1699-1700) and the NNForceMom holds a pointer to that space, so the HO radial table uses the file mesh. `NMesh` must equal `params%nmesh2` (else stop; NNForce.F90:1685-1689). `pmax2` is not checked against the file. If J > the code's Jmax, or channel not in the code's list, the channel is skipped.
- Number of channels with Tz=-1: 14, Tz=0: 26, Tz=+1: 14.

### 1.4 What the matrix elements are (units, normalization, factors)
- **Units: MeV fm^3** (the writer header says "ME (MeV fm-3)", NNForce.F90:1637; `plot_nn_file.py` colorbar label is the same).
- **V is the bare, UNWEIGHTED partial-wave matrix element V^{J}_{L'L,S,Tz}(k',k)**: no factor of k, no sqrt(w) is stored. `NNForceMom%weighted=.false.` after read (`InitNNForceMom`, NNForce.F90:852-868); `WeightNormalize` (NNForce.F90:870, multiplies by k1*k2*sqrt(w1*w2)) is NOT called in the HO-projection path [S; the projection path SetNNForceHO never calls it].
- Normalization convention implied by the projection (section 3): completeness is `integral k^2 dk |k><k| = 1` with no 2/pi factor in the measure. Evidence [V]: deuteron from T_rel + V in this HO basis gives E=-2.2246 MeV at Nmax=100 (3S1-3D1, hw=16).
- `P` in the header is parity (+1 / -1) = (-1)^L; `Tz` = -1 pp, 0 pn, +1 nn (section 2).
- Sample values [V]: 3S1 (J=1,P=+,S=1,Tz=0) V[0,0] = -36.9496 MeV fm^3, V[0,1] = -36.9480; 3S1 matrix max |V|=36.95; 1S0 pn V max |V| = 47.67; all channels symmetric. S-D off-diagonal block `m[:100,100:]` equals `m[100:,:100]^T` to 4e-14; corner element `m[0,100]` = -1.357e-4 (small at low k, as expected for tensor force).
- The file has no Coulomb and no regularization beyond the chiral cutoff already in the potential (N3LO EM500: Entem-Machleidt, Lambda=500 MeV). [?: details of the regulator are inside the generator, not in this repo.]

---------------------------------------------------------------------------------------------------

## 2. Channel definition and ordering

Source: channel builders `InitTwoBodyRelSpaceSpinMBasis` (src/TwoBody/TwoBodyRelSpaceSpinMeshBasis.F90:77-144) for the file/mesh space and `InitTwoBodyRelSpaceSpinHOBasis` (src/TwoBody/TwoBodyRelSpaceSpinHO.F90:72-142) for HO. Both use identical logic. The file writer iterates `two%jpsz(ich)` in this same order, so the file order = this order [V: checked against the file].

### 2.1 Quantum numbers
Channel label (J, P, S, Tz). Basis states inside: orbital L (relative), mesh point i (or HO n).
- J = total relative ang. mom. 0..Jmax (file Jmax=8).
- S = 0 or 1, total two-nucleon spin.
- P = (-1)^L (+1 or -1).
- Tz = -1 (pp), 0 (pn), +1 (nn). Evidence: `add_coulomb_force` only acts on `GetZ() == -1` (NNForce.F90:326); reduced mass for z=-1 is m_proton/2, z=+1 m_neutron/2 (NNForce.F90:416-418); `plot_nn_file.py` docstring. [V: pp(1S0) differs from nn(1S0) by 0.31 MeV fm^3 and from pn by 1.17, so the file is charge-dependent (CIB at the level of the potential).]
- **No isospin T quantum number**: this is the pn (Tz) formalism. In the pn channel (Tz=0) both T=0 and T=1 np states are present (the pn channel has all (L,S) allowed by J and P). For Tz=+-1 (identical particles), antisymmetry (L+S+T odd with T=1) requires `(-1)^(L+S) = +1` ; combinations with `(-1)^(L+S) = -1` are excluded (TwoBodyRelSpaceSpinMeshBasis.F90:112, TwoBodyRelChanSpinHO.F90:126). No T-mixing term is stored: the pn channel is a plain (L,S,J) channel.
- A channel exists only if it has at least one allowed L.

### 2.2 Channel loop order
```
for Tz in (-1, 0, +1):          # outer
  for J in 0..Jmax:
    for P in (+1, -1):          # +1 first
      for S in (0, 1):
        L in abs(J-S) .. J+S with (-1)^L == P and (|Tz|==0 or (-1)^(L+S)==+1)
        if at least one L: emit channel
```
First channels in the file [V]: (J,P,S,Tz) = (0,+,0,-1), (0,-,1,-1), (1,-,1,-1), (2,+,0,-1), (2,-,1,-1) [ndim 200, coupled 3P2-3F2], (3,-,1,-1), ... ; last: (7,-,1,+1), (8,+,0,+1), (8,-,1,+1) [ndim 200].
For Tz=0 and J=0: (0,+,0,0) = 1S0, (0,-,1,0) = 3P0. For Tz=0, J=1: (1,+,1,0) coupled 3S1-3D1 (ndim 200), (1,-,0,0) 1P1, (1,-,1,0) 3P1 (ndim 100 each).

### 2.3 State ordering within a channel (momentum)
`InitTwoBodyRelChanSpinMBasis` (TwoBodyRelChanSpinMeshBasis.F90:146-192): loop `L` ascending over the allowed values, inner loop mesh index i=1..NMesh. State index = (position of L in the ascending list)*NMesh + i. So for coupled J>0,S=1 triplet channels (ndim = 200) the matrix is
```
V = [[ <L=J-1|V|L=J-1>  <L=J-1|V|L=J+1> ],
     [ <L=J+1|V|L=J-1>  <L=J+1|V|L=J+1> ]]      each block NMesh x NMesh, row index = k' (bra), col = k (ket).
```
(`plot_nn_file.py` uses the same slicing.) Uncoupled channels: single NMesh x NMesh block. J=0,S=1 is only 3P0 (L=1). Note that for J=0 (any P) with S=0 only L=0 (P=+); with S=1 only L=1 (P=-), so P and S are both determined.
Also note pn-channel (Tz=0) at J>0 includes 1P1(S=0,P=-) etc. as separate channels.

### 2.4 State ordering within a channel (HO)
`InitTwoBodyRelChanSpinHOBasis` (TwoBodyRelChanSpinHO.F90:109-165): loop L ascending over allowed values (same parity/Tz filter), inner n = 0..n12max with n12max=(Nmax-|J-S|)/2 and the constraint `2n + L <= Nmax`. Index = running count (L-major, n-minor). Nmax=N2max=100.
Example [V]: 1S0 has 51 states (n=0..50), 3S1-3D1 has 51+50 = 101 states (L=0: n 0..50; L=2: n 0..49).

---------------------------------------------------------------------------------------------------

## 3. Projection formula momentum -> HO relative

Source: `trans_mom_to_ho` (src/TwoBody/NNForce.F90:352-391) and `set_ho_radial` (NNForce.F90:393-430); radial function `ho_radial_wf_norm` (src/MyLibrary.F90:750-766).

### 3.1 Formula [V: reproduces the cache file]
For each channel (J,P,S,Tz) and HO bra |n' L'>, ket |n L> (L', L in the allowed set, 2n+L <= N2max):

```
<n' L' S J Tz | V | n L S J Tz>
   = sum_{i,j=1..NMesh}  [ k_i w_i  R_{n'L'}(k_i) (-1)^{n'} ]  V^{J}_{L'L S Tz}(k_i,k_j)  [ k_j w_j  R_{nL}(k_j) (-1)^{n} ]
```
In matrix form per channel: `V_HO = O^T V_mom O` with `O[(L,i),(n,L)] = k_i w_i R_{nL}(k_i) (-1)^n` if the L labels match, else 0 (NNForce.F90:372-383). I.e. O is block-diagonal in L. There is no sqrt(w), no extra k factor beyond the single `k*w` (see below) and **no 2/pi, no i^L phase**.

Equivalent statement: define f_{nL}(k) = R_{nL}(k)/k (the standard radial function with `int_0^inf k^2 f^2 dk = 1`). Then the formula is `sum_ij k_i^2 w_i f_{n'L'}(k_i) V(k_i,k_j) k_j^2 w_j f_{nL}(k_j)` with phase (-1)^{n'+n} = the usual quadrature of `int k'^2 dk' int k^2 dk f V f`.

### 3.2 Radial function (src/MyLibrary.F90:750-766)
`ho_radial_wf_norm(n,l,anu,r)`: with `nu = sqrt(anu)` (called "nu" in the code but it is a LENGTH, see 3.3), `x = nu * r`:
```
R_{nl}(r) = sqrt(2 nu) * sqrt( Gamma(n+1) / Gamma(n+l+3/2) ) * x^{l+1} * exp(-x^2/2) * L_n^{(l+1/2)}(x^2)
```
- `L_n^{(a)}` = generalized (associated) Laguerre polynomial (GSL `gsl_sf_laguerre_n`, standard convention, L_0 = 1, positive leading... i.e. `scipy.special.eval_genlaguerre`).
- Evaluated via logs of Gamma for stability.
- It includes the factor r (the "reduced radial function", x^{l+1}). Orthonormality: `sum_i R_{nl}(k_i) R_{n'l}(k_i) w_i = delta_{nn'}` [V: 1.0000, 1.0, 6e-16].
- Used for momentum with r -> k: `x = nu*k`, nu has units of fm. (`set_ho_radial`, NNForce.F90:393-430 passes `a = hc^2/(m_red*hw)` and the mesh momentum.)
- **Phase**: with the explicit `(-1)^n` in the projection, the net HO momentum-space basis function is `(-1)^n R_{nL}(k)` (no i^L). This corresponds to the "standard" positive-at-origin convention for the *coordinate-space* wave function up to the i^L that cancels between bra and ket [V: with this convention T_rel has POSITIVE off-diagonals: `<n+1 l|T|n l> = +(hbar w/2) sqrt((n+1)(n+l+3/2))` and `<n l|T|n l> = (hbar w/2)(2n+l+3/2)`; the deuteron binds at -2.2246 MeV; with the opposite sign -61 MeV (wrong)]. The i^{L'-L} phase for the S-D off-diagonal block is NOT applied (momentum file convention already assumes it; verified indirectly by the deuteron energy and by reproducing the file).

### 3.3 Oscillator length / mass / hbar*omega (NNForce.F90:393-430)
```
b^2 = (hc)^2 / (m_red * hw)      [fm^2]        (the variable "a" passed as anu)
```
- `hw` = `two%GetFrequency()` = input hw = 16 MeV (the oscillator parameter hbar*omega; no 1/2 factors).
- `m_red` depends on `pn_same_mass` (default .true.; the example input does not set it):
  - `.true.` (default): `m_red = m_p*m_n/(m_p+m_n)` for ALL Tz (pp, pn, nn) = 469.45... MeV exactly `938.27231*939.56563/(938.27231+939.56563)`.
  - `.false.`: z=-1 -> `m_proton/2`; z=0 -> `m_p*m_n/(m_p+m_n)`; z=+1 -> `m_neutron/2`.
- Relative coordinate convention: r = r1 - r2, p = (p1-p2)/2, so reduced mass ~ m/2, and "hw" is the same hbar*omega as for the single-particle HO of mass m. [S/V consistent with T_rel diag (hbar w/2)(2n+l+3/2) reproducing -2.2246 MeV.]
- Number of radial functions tabulated: n = 0..N2max/2, l = 0..min(Jmax+1, N2max), per z in {-1,0,1}, on the mesh.

### 3.4 Quadrature and truncation
- Quadrature: the file's GL mesh (100 points, [0,8] fm^-1), integrals `int_0^8 k^2 dk` done as `sum_i w_i k_i^2 (...)` as above. No extrapolation beyond k_max = 8 fm^-1.
- Truncation: HO basis `2n + L <= N2max` (=100 in the example, independent of e2max; later truncation to e2max=8 happens in `trans_to_lab_op`, `oprel%truncate`; other slice). Also `J <= J2max_NNint` (=8).
- **Mesh limit warning**: with hw=16, b = 2.28 fm; high-n HO functions (n up to 50) extend beyond k=8 fm^-1 (classical turning point ~ sqrt(4N+6)/b ~ 8.8 fm^-1 at N=100); the 100-point mesh also limits how accurately R_n for n>~30 is integrated. The reference code just uses what it gets; to reproduce the cache, replicate the same quadrature, do NOT "improve" it. (Orthonormality check with this mesh was fine for the low n tested, 1e-16.)
- Hermiticity: the result is symmetric by construction; the cache stores only lower triangle (i2 <= i1) and read-back symmetrizes.

### 3.5 Antisymmetry / parity constraints
Already enforced by the channel lists (2.1): L parity = P; for Tz=+-1 only (-1)^(L+S)=+1; no further constraints in the projection. For Tz=0 no antisymmetry restriction in the pn channel (T=0 and T=1 both). The HO basis does not carry T.

---------------------------------------------------------------------------------------------------

## 4. Physical constants (src/MyLibrary.F90:6-30)
```
hc        = 197.32705      MeV fm      (NOT 197.3269804)
m_proton  = 938.27231      MeV
m_neutron = 939.56563      MeV
m_nucleon = 0.5*(m_p+m_n)  = 938.91897   (NOT used in this stage)
m_red_pn  = m_p*m_n/(m_p+m_n)           (parameter exists; set_ho_radial recomputes it inline; same value)
alpha     = 137.035999     (this is 1/alpha, used as hc/alpha in Coulomb: e^2 = hc/alpha)
pi        = acos(-1)
```
Used in this stage: `hc`, `m_proton`, `m_neutron` only (set_ho_radial). Coulomb (not in the bare file) uses `hc/alpha / r` with `a = 0.5*m_proton*hw/hc^2` (NOTE: this is nu^2 = m_p*hw/(2 hc^2) in fm^-2 while set_ho_radial's `a` is the inverse, a length^2... see open questions) and GL r-mesh with 500 points on [0,40] fm (NNForce.F90:286-350). [?]

---------------------------------------------------------------------------------------------------

## 5. Relative-space cache file `NNint_A2_rel_*.gz`

Writer: `WriteNNForceHORelative` (src/TwoBody/NNForce.F90:432-488). Reader: `ReadNNForceHORelative` (NNForce.F90:490-536). Name: `GetFileNameBareNNForceRelative` (NNForce.F90:538-548): `<dir>/NNint_A2_rel_<NNInt>_bare_Nmax<N2max>_hw<hw>.gz` (hw formatted by `s%str(real)`; "16" for 16.0, behaviour for non-integer hw not checked [?]).

### 5.1 Format (gzip text, one record per line) [V]
- Line 1: `ntot N2max jmax`, list-directed: `121551 100 8`. ntot = number of data lines.
- Then ntot lines, format `(8i4, es18.6)`: `J P S Tz n1 l1 n2 l2 value`.
  Example: `0 1 0 -1 0 0 0 0 -6.177961E+00`.
- `value` = `<n1 l1 S J Tz | V | n2 l2 S J Tz>` in **MeV**, bare NN (no Coulomb, no SRG, no kinetic energy). Precision: 7 significant digits (es18.6). Rows: bra=(n1,l1), ket=(n2,l2).
- Record order: channels in the order of section 2.2 (Tz -1,0,+1; J; P=+1,-1; S=0,1); within a channel `for i1 in 1..nstates: for i2 in 1..i1` with state index in the HO order of section 2.4 (L-major, n-minor). So **lower triangle including diagonal**, row-major by row i1. Matches: 1S0 has 1326 = 51*52/2 lines [V]; 3S1-3D1 has 5151 = 101*102/2 [V]. Total lines = 121551 [V].
- Reading: entries with 2n1+l1 > N2max or 2n2+l2 > N2max or J > Jmax(code) or missing channel are skipped; the file's `jmax_r` must be >= code Jmax (else stop). Value is stored at both (i1,i2) and (i2,i1) (NNForce.F90:526-527). N2max in the file may be larger than the code's N2max (truncation on read), but not smaller [S: no explicit check for N2max_r < N2max; it simply wouldn't find the rows -> zeros; ?].
- In the 3S1-3D1 channel the (n1,l1=0 | n2,l2=2) rows have l1 < l2 ordering as per state index (l=0 states first), so the stored off-diagonal rows are bra=D-wave, ket=S-wave (i1 > i2), value = `<n1 L=2|V|n2 L=0>` (= its transpose by symmetry) [V].
- An alternative text dump `write_nn_ho` (NNForce.F90:1714-1747; used only for `file_name_nn`-style output of SRG-evolved interactions to a plain text file; full square, no gzip) is not what the example cache uses.

### 5.2 The second file in refs/work
`TwBME-HO_NN-only_N3LO_EM500_srg2.0_hw16_emax4_e2max8.me2j.gz` is the final lab-frame me2j output (other slice, not described here).

---------------------------------------------------------------------------------------------------

## 6. Not used / out of scope in this stage
- `NNForceIsospin.F90` (isospin-formalism version `SetNNForceHOIsospin`): it builds T=0/T=1 channels from the pn-formalism `NNForceHO` result (it calls `vpn%SetNNForceHO` internally, NNForceIsospin.F90:60-200) with weights cpp/cnn/cpn=1/3 (or depending on T_total, Tz_total) - used only in EDM calculations (TwoBodyManager.F90:821-860); NOT part of the NN-only pipeline. Details of the T=1 averaging rule were only skimmed.
- `NNForceMom` renormalizations (`renorm_mom_space`: SRG/Vlowk in momentum space) are only used when `renorm_space2='mom'`; default is 'ho' so the example does SRG in HO space after the projection.
- `PartialWaveDecomposition.F90` and `NNForceCartesian.F90`: not used for this input path (they generate the .bin from a Cartesian/momentum-vector potential file `file_nn_cartesian`). [S: not read in detail; `input_nn_file` goes through `read_nn_mom` directly.]

---------------------------------------------------------------------------------------------------

## 7. Reimplementation recipe (no code, just steps)
1. Parse the .bin (section 1.2). Use mesh k, w from the file.
2. For each of the channels (J<=8, same list as the file), build allowed L list (2.1), HO states (n,L) L-major (2.4).
3. Build `O` of shape (nL_blocks*NMesh, nstates): column (n,L) has, in block L, entries `k_i w_i R_{nL}(k_i) (-1)^n`; `R` from 3.2 with `b^2 = hc^2/(m_red*hw)`, `m_red = m_p m_n/(m_p+m_n)` if pn_same_mass (default).
4. `V_HO = O^T V O`.
5. (Optional) emit in the cache format of section 5.

---------------------------------------------------------------------------------------------------

## 8. Verification done (python/numpy, script in session scratchpad)
- Parsed .bin to the last byte; GL mesh identity; symmetry; channel list and ndim.
- Projection (3.1) with scipy `eval_genlaguerre` and `gammaln` reproduces the cache file channel by channel for (J,P,S,Tz) = (0,+,0,-1), (0,+,0,0), (1,+,1,0) [coupled], (1,-,1,0), (2,-,1,0) [coupled 3P2-3F2 at Tz=0], (8,-,1,+1) [coupled]: max abs difference over all lower-triangle entries 5e-6 (pp 1S0), 5e-6 (pn 1S0), 5e-6 (3S1-3D1), 5e-7, 3e-6, 5e-9; consistent with the 7-digit rounding in the file (max |value| 10-13 MeV, rounding 5e-7 relative -> up to ~5e-6 abs). First diagonal elements: pp 1S0 -6.177961 (file) vs -6.177961365 (mine).
- Physics check: <3S1-3D1| T_rel + V |> in N=100 HO basis (hw 16, T offdiag sign +): lowest eigenvalue -2.22458 MeV (deuteron, 2.2246 vs exp 2.2246); N=40: -2.2236; N=20: -1.78. With opposite T sign: -61.7 (wrong), establishing the (-1)^n phase convention.

## 9. Open questions / unverified
1. Coulomb `a` in `add_coulomb_force` (NNForce.F90:316) is `0.5*m_proton*hw/hc^2` = nu^2 (fm^-2), whereas `set_ho_radial` uses `hc^2/(m_red hw)` = b^2 (fm^2); `ho_radial_wf_norm(n,l,anu,r)` does `x = sqrt(anu)*r`, so the Coulomb call uses coordinate-space nu=sqrt(m hw/(2 hc^2)) with r (correct for coordinate space with reduced mass m_p/2... ) while the momentum call uses b*k. Both look right for their space, but the Coulomb part belongs to a later slice; confirm there.
2. How `s%str(real(8))` formats hw in the cache filename for non-integer hw (e.g. 16.5, 20.0) was not checked (ClassSys.f90).
3. pp/nn Tz sign convention is inferred (Coulomb acts on Tz=-1 -> pp; mass assignment) and not explicitly commented in the .bin writer; it is consistent with the plot script's docstring.
4. The i^{L'-L} phase convention for the off-diagonal (S-D) block is not documented; I rely on the equality "matrix as stored + (-1)^n R_nl projection reproduces the cache and gives E_d=-2.2246". A full phase-shift check was not done.
5. Whether the source generator of the .bin applies a regulator beyond what the chiral potential has is not inferable from this repo.
6. When N2max_r in a cache file is smaller than the code's N2max, `ReadNNForceHORelative` does not stop (only jmax is checked); behaviour is silent zeros. Not tested.
7. `pn_same_mass=.false.` path (separate masses for pp, nn) was not exercised; only read from source.
