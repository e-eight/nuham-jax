# Spec 4: end-to-end numpy check of specs 1-3 against the SRG reference me2j files

Script: `refs/oracle_numpy.py` (slow throwaway oracle; plain numpy/scipy/sympy-CG; NOT the package).
Run: `python refs/oracle_numpy.py --n 0` (all tuples, ~25 s on 15 cores incl. the 85676-tuple emax=6 file);
`ORACLE_SOLVER=vode` emulates NuHamil's integrator. Cache of evolved relative V: `$ORACLE_CACHE` (default /tmp/oracle_cache).

## 1. What was run

Chain, exactly as specs 1-3 describe (no deviations needed in the physics):

1. Bare relative V per channel (J<=8,P,S,Z), full N2max=100 space. Two sources:
   - `bin`: projection of `N3LO_EM500_kmax8_N100_Jmax8.bin` per spec 1 (full double precision);
   - `gz`: the `NNint_A2_rel_*_bare_Nmax100_hw{16,20}.gz` cache file (values rounded to 7 digits, `es18.6`).
2. pp channels (Z=-1): `V += <n l|(hc/alpha)/r|n' l>`, 500-pt Gauss-Legendre on [0,40] fm, `nu^2 = 0.5 m_p hw/hc^2`, alpha=137.035999,
   added BEFORE the flow (evolve_coulomb=.true. default).
3. SRG per channel in the full 100-space: `H = T+V`, `G = T` (tridiagonal in n, diagonal in l, hw/2 factors), `dH/ds=[[T,H],H]`,
   `s = lc^-4`, `lc = lambda*hc/sqrt(2 m_red_pn)` for ALL Z (pn_same_mass). All 54 channels (J<=8) are evolved (~2.5 s total with DOP853 1e-10).
   Packed lower triangle state vector as in Fortran. Two integrators: scipy DOP853 rtol=atol=1e-10, and scipy `vode` (Adams, order 12,
   rtol=atol=1e-8, nsteps=50000), which is the same DVODE algorithm family NuHamil calls (MF=10).
4. `V = H_srg - T`; the lab transform uses only states with 2n+l <= e2max (plain sub-block, J_rel>8 identically zero).
5. Relative -> lab per spec 3 (T-coefficients with `ft`, `delab`, d=1 Moshinsky brackets transcribed from `gmosh`, 9j/6j), then the four me2j columns
   `[T=0, nn, pn T=1, pp]` with the spec-3 4.4 assembly (norm_fact sqrt2 for a==b / c==d on pp,nn; the swap phase `-(-1)^{ja+jb-J}` for pn).
   6j/9j are my own Racah-formula implementation (fast), checked against sympy to 5e-16.
6. Compared against the three SRG reference files over ALL tuples (6834, 6834, 85676 tuples; x4 values).

## 2. Agreement numbers (max |oracle - reference|, MeV, all tuples)

Reference-faithful setup (source = what the reference run actually used, see 3.1), integrator emulating NuHamil (scipy vode adams 1e-8):

| file | tuples | T=0 | nn | pn T=1 | pp |
|---|---|---|---|---|---|
| srg2.0_hw16_emax4_e2max8 (source bin) | 6834 | 5.3e-9 | 5.0e-9 | 5.1e-9 | 5.1e-9 |
| srg2.0_hw20_emax4_e2max8 (source bin) | 6834 | 1.2e-8 | 7.0e-9 | 7.9e-9 | 8.3e-9 |
| srg1.8_hw16_emax6_e2max12 (source gz) | 85676 | 1.3e-8 | 7.4e-9 | 9.1e-9 | 5.1e-9 |

These are at the print precision of the files (`f16.8`, +-5e-9) up to a couple of 1e-9. Largest |reference value| is 22-26 MeV, so relative agreement < 1e-9.

Same, but with a converged integrator (DOP853 rtol=atol=1e-10, i.e. what the JAX code should do):

| file | T=0 | nn | pn T=1 | pp |
|---|---|---|---|---|
| srg2.0_hw16 | 2.4e-7 | 5.5e-8 | 6.2e-8 | 1.5e-8 |
| srg2.0_hw20 | 1.6e-7 | 4.8e-8 | 5.9e-8 | 1.6e-8 |
| srg1.8_hw16_e6 | 3.2e-7 | 6.0e-8 | 6.2e-8 | 1.7e-8 |

Where the largest differences come from (DOP853 case): T=0 entries involving the 0s1/2 0s1/2 / 3S1-3D1 (deuteron) channel at J=1
(e.g. <0s 0s;J=1|V|0s 0s;J=1> = -21.42 MeV, |diff| 2.4e-7), then nn/pn T=1 (1S0, 3P) ones. These are NOT errors of the oracle: they are the
integration error of NuHamil's own DVODE at rtol=atol=1e-8 (the stiffest/largest channel is 3S1-3D1). Evidence: switching the oracle to scipy's
DVODE-Adams at 1e-8 makes the difference vanish to print precision (table above), and DOP853 1e-10 vs 1e-13 differ by only 1.1e-8 in the evolved relative V.
The pp column is the most accurate (1.5e-8) because 1S0/3P pp are smoother than the deuteron channel; Coulomb handling is confirmed correct (see 3.3).

## 3. Findings and discrepancies relative to specs 1-3

### 3.1 (IMPORTANT) The 7-digit rounded cache file is not what two of the three reference runs used
- `log_srg20_hw20.txt` says "Not found .//NNint_A2_rel_..._hw20.gz ... Calculating NN interaction in relative coordinate", i.e. NuHamil projected the .bin in memory
  at full precision and wrote the cache afterwards (the cache write precedes use but in-memory matrices are unrounded). The same holds for the srg2.0_hw16 run
  (Input_a; it created the hw16 cache). `log_srg18_hw16_e6.txt` shows "Read from file": that run read the already-existing, rounded hw16 cache.
- Consequence: reproducing srg2.0_hw16 / srg2.0_hw20 from the `.gz` gives max diff 2.5e-6 / 2.4e-6 (T=0; nn/pp/pn 7e-7..1.3e-6) even with the emulated integrator,
  vs 5e-9 when the .bin is projected at full precision. Reproducing srg1.8 from the .bin gives 2.2e-6 vs 1.3e-8 from the .gz.
  The reason is that the 7-digit rounding (relative 5e-7 per element, i.e. up to a few 1e-6 MeV in the larger elements) is amplified by the flow and lab sums to ~2e-6.
- Spec 1 section 8 quoted 5e-6 for projected-vs-cache differences; that is this same rounding. Spec 2 section 6/Spec 3 header: the "1e-6 MeV" tolerances are of this order and are mostly
  rounding of the cache, not the SRG.
- Recommendation: validate stage 1 against the .gz only to 1e-5 (abs, MeV) and validate the full chain against the me2j files using the matching source (see 5).

### 3.2 Integrator error of NuHamil (spec 2 open item 4)
Spec 2 estimated NuHamil's own error at ~2e-7 MeV in V; confirmed: a converged integrator differs from the reference by up to 3.2e-7 in the lab ME (T=0), 6e-8 in nn/pn, 2e-8 in pp.
scipy `vode` adams 1e-8 reproduces the Fortran step behaviour to the print precision, so the Fortran dvode_f90 MF=10 and scipy VODE are equivalent in practice.
A JAX implementation cannot (and should not) reproduce this 1e-7 solver error; it must be absorbed in the tolerance (section 5).
Note: in scipy DOP853 a few RuntimeWarnings (overflow in matmul) occur for trial steps in one hw=20 channel; the step is rejected and the result is fine, but in JAX use an
error-controlled integrator that rejects non-finite steps (or start with a small initial step).

### 3.3 Coulomb in pp (spec 2 open item 1, spec 3 sec 3, spec 1 open item 1): CONFIRMED
Default `coul=.true.` and `evolve_coulomb=.true.`: Coulomb added to pp (Z=-1) relative V before SRG and evolved. Ablations on srg2.0_hw16 (vode):
- no Coulomb: pp column off by 1.38 MeV (e.g. <0s 0s;0|V|0s 0s;0> pp -14.967 vs ref -13.583); T=0/nn/pn unaffected.
- Coulomb added after the flow (not evolved): pp off by 6.7e-2 MeV.
- Coulomb as specified (a = 0.5 m_p hw/hc^2 as coordinate-space nu^2, 500-pt GL [0,40], hc/alpha, diagonal in l, applied for all n up to the full space, n<=50, l<=9): pp matches to 5e-9.
So spec 1 open item 1 (a = nu^2 vs b^2) is resolved: coordinate-space nu^2 = m_p hw/(2 hc^2) is correct for Coulomb, b^2 = hc^2/(m_red hw) for the momentum projection.
The mesh (500 pts on [0,40]) is adequate for n<=50 at hw=16 and 20; replicate it anyway (do not "improve" the quadrature).

### 3.4 J=9 relative channels are zero (spec 2 open item 2): CONFIRMED
`relME` returns 0 for jrel>8 (V does not exist for J>J2max_NNint=8). With e2max=8 the only lab entry sensitive to J_rel=9 (l=8,S=1,Tz=0, e.g. involving 0g9/2 pairs) agrees with the reference to 5e-9 with J=9 set to zero;
a test perturbation (fake V=10 MeV in that channel) moves exactly 1 of 6834 tuples (by 5.5 MeV), so the test is sensitive and the zero is confirmed. For e2max=12 (J_rel up to 12 allowed by Jmax2=12) the same zeroing for J>8 gives 1.3e-8 agreement over all 85676 tuples.

### 3.5 nn/pp mass handling: CONFIRMED (pn_same_mass=true)
A single m_red_pn is used for the HO relative length (projection), for s = lc^-4 and for T in all Z channels. Only the Coulomb term uses m_p/2-based nu^2.

### 3.6 Antisymmetrisation/normalisation and pn conventions (spec 3): CONFIRMED
Spec 3 sections 2.2-2.3 and 4.4 reproduce the SRG files over all 6834 + 6834 + 85676 tuples, including a==b / c==d, odd/even J, pn swap phase.
No corrections needed. Spec 3's loop `jrel in 0..12` can be limited to `min(J+Lcm, 8)` since V=0 above 8.

### 3.7 Other points
- Truncation after the flow (plain sub-block to 2n+l<=e2max, from the full N=100 space) is confirmed. (Flowing in a truncated space was not tried; spec 2 already says it changes results.)
- Spec 1: the (-1)^n phase, `k*w*R` projection, state ordering and constants are confirmed by agreement at 5e-9 (bin source).
- No discrepancy in spec 2's generator/T/s=1/lambda^4 conversion; T uses hw directly (no mass).

## 4. Practical numbers for the JAX build
- Oracle cost: SRG of all 54 channels ~2.5 s (DOP853 1e-10, numpy); lab transform 85676 entries ~20 s on 15 procs after caching 6j/9j/Moshinsky.
- Evolved V magnitudes: <0s 0s;J=1|V|..> T=0 -21.4 (hw16, lam 2.0), -26.2 (hw20), -22.6 (hw16, lam 1.8).

## 5. Recommended test tolerances (absolute, MeV)
Stage-level (unit) tests:
- Projection from .bin vs `.gz` cache: 1e-5 (cache has 7 digits). Against the oracle with the .bin: 1e-9 (double).
- Coulomb matrix elements vs oracle quadrature: 1e-10 (same quadrature).
- SRG: eigenvalue invariance per channel 1e-8 (rtol=atol=1e-10 integrator), deuteron -2.22458 MeV to 1e-5.
- Lab transform (relative V given, e.g. bare file with Coulomb, spec 3 check): 1e-8 (print precision; 5e-9 achieved).
End-to-end vs the reference me2j (JAX with converged integrator, projecting from the .bin; for the lambda=1.8 file project from the 7-digit .gz bare file instead, or accept 3e-6):
- Column pp: 5e-8; nn, pn T=1: 2e-7; T=0: 1e-6 (observed 2.4e-7 .. 3.2e-7; the deuteron channel carries NuHamil's own VODE error).
  A single uniform tolerance of 1e-6 absolute (and 1e-7 relative to max |value|) passes all three files with a margin of >3x on every column.
- If the JAX integrator is run with rtol=atol=1e-8 Adams-type, do not expect to beat ~1e-7 versus the reference; do not tighten below 5e-7 for T=0.
- If the JAX code projects from the .bin but the comparison file was made from the rounded cache (srg1.8), the expected difference is 2.2e-6: use 5e-6 there, or provide a `.gz`-read path.
