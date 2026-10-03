# NN physics spec (consolidated) — Phase 0 result

Details and source citations are in specs 1-4. This page is the short version
plus the decisions that follow from it. If this page and a numbered spec
disagree, the numbered spec (and spec 4's numerical check) wins.

Status: the chain below is verified end to end. `refs/oracle_numpy.py` (slow numpy,
not the package) reproduces every tuple of three SRG `me2j` references to ~1e-8
(print precision), with a DVODE-like integrator.

## Pipeline

1. **Input.** `N3LO_EM500_kmax8_N100_Jmax8.bin`: raw little-endian stream, no record
   markers. 100-point Gauss-Legendre mesh on [0, 8] fm^-1; 54 channels (J, parity, S, Tz);
   unweighted V(k',k) in MeV fm^3. pn formalism (Tz=0 mixes T=0/1). Spec 1.
2. **Projection to HO relative basis.** `V[n'L',nL] = sum_ij f_n'L'(k_i) V(k_i,k_j) f_nL(k_j)`,
   `f = k w R_nL(k) (-1)^n`; no 2/pi, no i^L. b^2 = hc^2 / (m_red hw), m_red = pn reduced mass
   for all Tz. Truncation 2n+L <= 100, J <= 8. Spec 1.
3. **Coulomb.** Added to pp channels BEFORE the flow (default on) and evolved with it.
   nu^2 = 0.5 m_p hw / hc^2. Without it pp is off by 1.4 MeV. Spec 4.
4. **SRG.** H = T_rel + V, dH/ds = [[T_rel, H], H], s = 1/lambda^4 in fm^4, converted to
   energy units as alpha = 1/lc^4 with lc = lambda hc / sqrt(2 m_red) (lambda=2 gives
   alpha = 3.634e-5 MeV^-2), run in the full N=100 space per channel,
   all channels with J <= 8. Stored V = H - T_rel. Spec 2.
5. **Cut and transform.** Cut V to 2n+L <= e2max (plain sub-block), relative J > 8 is
   zero. Transform with the equal-mass Moshinsky bracket (Kamuntavicius `gmosh`) and
   9j/6j recoupling: `V_lab = sum_{jrel,S,Ncm,Lcm} T^T V_rel T`. N_rel is not conserved;
   e_a+e_b and e_c+e_d are each <= e2max. Spec 3.
6. **Write `me2j`.** gzip text, one free-text header line, Fortran `10f16.8` (10 values/line, MeV).
   Loops a = 0..N, b <= a, c <= a, d <= (c==a ? b : c), then J = Jmin..Jmax; 4 values per tuple in order
   T=0, nn, pn T=1, pp; Darmstadt un-normalised convention (pp/nn times sqrt2 per a==b, c==d);
   forbidden entries are 0.0. Matches the imsrg++ reader loop. Contains V only (no T, Hcm).
   imsrg++ builds its own Trel. Spec 3.

Constants: hc = 197.32705, m_p = 938.27231, m_n = 939.56563 (MyLibrary.F90:8-13).

## Verified and not verified

Verified (spec 4): full chain for hw=16 and 20, lambda=2.0 and 1.8, emax=4 and 6; Coulomb
default and evolution; J=9 relative channels zero; single m_red for pn/pp/nn;
antisymmetrization and pn swap phase. Independently re-run for this page: srg2.0_hw16,
300 random tuples, max |diff| 5e-9.

Not verified (out of scope or untested): `pn_same_mass=false`; non-kinetic SRG generators;
isospin formalism and other operators; larger emax (>6) and e2max > 2*emax; high ħω;
behaviour when lmax < emax; flowing in a truncated space.

## Test tolerances (absolute, MeV)

| Test | Tolerance |
|---|---|
| Full chain vs `me2j` files | 1e-6 (per column: pp 5e-8, nn/pn T=1 2e-7, T=0 1e-6) |
| Same, srg1.8 reference if projecting from the .bin | 5e-6 (reference used the 7-digit gz cache) |
| Projection vs relative `.gz` cache | 1e-5 |
| Lab transform given relative V | 1e-8 |
| SRG eigenvalue invariance per channel | 1e-8 |

Why 1e-6 and not tighter: NuHamil's own DVODE error (rtol = atol = 1e-8) dominates for
the 3S1-3D1 channel; a converged integrator differs from it by up to ~3e-7. Also, which
bare source NuHamil used (gz cache vs .bin) changes results by ~2.5e-6.

## Consequences for the JAX design

- The SRG flow is the dominant cost in the Fortran run (9.6 s of 11 s at emax=4) and it
  is **independent of emax**: it always runs in the Nmax=100 space. It is only mildly
  stiff (Jacobian spectral radius x alpha ~ 80), so an explicit high-order solver
  (Diffrax Dopri8, rtol=atol=1e-10) is enough and is cheap to batch. Reject non-finite
  trial steps (scipy DOP853 overflowed in rejected steps once).
- Channels are independent: batch with `vmap` over padded channel dimensions (about 100-200 states).
  Whether Nmax=100 is really needed is a Phase 2 experiment (changes results; must be a
  documented deviation, off by default).
- The Moshinsky/9j/6j coefficient build is host-side Python in the oracle (slow at high emax).
  Phase 1 must decide: host NumPy with caching, or JAX kernels. Needed at emax=14, e2max=28.
- Output ordering and the 4-values-per-tuple convention are fixed by the imsrg++ reader.
- Keep `refs/oracle_numpy.py` as a regression oracle; it is not part of the package.

## Reference set (refs/work)

`Input_*.dat`, `log_*.txt`, `NNint_A2_rel_*bare*.gz`, and four `me2j` files:
bare hw16 emax4; srg2.0 hw16 emax4; srg2.0 hw20 emax4; srg1.8 hw16 emax6 e2max12.
Produced with the `nuhamil` podman image (built from `NuHamil-public/Containerfile`).
