# Study: Jacobi-to-lab T-coefficients of three nucleons (step 5.1b)

Code: `src/nuham/tcoef3.py` (NumPy, host). Tests: `tests/test_tcoef3.py`, helpers `tests/ref3n.py`.
Throwaway oracles: `refs/tcoef3_reference.py` (element-wise transcription of the nested sums),
`refs/tcoef3_gemm_prototype.py` (the sums as dense products). Spec: `docs/spec-5d`, sections 4-5.

## Result in one paragraph

The coefficients of spec 5d 4.2 are correct as written. With NuHamil's own Jacobi operator and `cfp`
as input, `6 sum T^T V_jac T` reproduces **all 24 890 values** of the 4/8/4 `me3j` reference to the
print rounding (max difference 5.0e-9, file has 8 decimals), including the file order, the factor 6
and the phases. With our own antisymmetriser basis (`nuham.jacobi3`), `6 T_AS^T T_AS` equals the
lab-basis matrix of `sum_P sgn(P) P` computed independently in the m-scheme (explicit Clebsch-Gordan
coefficients, tensor permutations) to 1e-14 for every block checked. The d = 2 Moshinsky bracket
needs no new code. The host NumPy build costs about 1-3 microseconds per matrix element of `T_NAS`,
which is 4-11 core-hours at production size: feasible on the host, and the structure maps to dense
products for a GPU (not measured here).

## 1. The d = 2 bracket

`MoshinskyTable(d=2.0)` already exists (it was added for the NN lab transform tests). Checked for
`N <= 6`, all `lam`:

- equal to `refs/moshinsky_reference.bracket(..., d=2.0)` (the explicit Kamuntavicius sum, i.e. the
  `gmosh` that NuHamil's `tmbk_2` stores with `d = 2`) to **7.9e-15**;
- every block orthogonal to **1.3e-14**;
- (`d = 0.5` also matches the reference, 6.8e-15; it is the same rotation with the roles of the two
  particles exchanged and is not used.)

`tmbk_2%get(ncm, lcm, n3, l3, nit, lit, nc, lc, lam)` is `table.matrix(2 nit + lit + 2 nc + lc, lam)`
with row `(ncm, lcm, n3, l3)` (cm, rel) and column `(nit, lit, nc, lc)`: pair centre of mass first,
then `c`. The `d = 1` table likewise takes `(nit, lit, n12, l12 | na, la, nb, lb)`. Both go through
`MoshinskyTable.gather` with no changes.

## 2. What was built

For a lab channel `(J, P, T)` and block `N`:

- `lab_states`: lab states in the order of spec 5d 2.3 (reproduces the table of 2.4: 22 channels,
  715 states, largest 99 at 4/8/4; 70 channels, 687 352 states, largest 32 259 at 14/28/16).
- `jacobi_cm_rows`: NAS Jacobi x centre-of-mass rows `(n12 l12 s12 j12 t12 n3 l3 j3 jrel ncm lcm)`
  built from `jacobi3.nas_states`.
- `tcoef_nas`: the dense `T_NAS`. The sum over `(lab, lit, Lam, lam)` is split into a left factor
  `L[(a b jab), (n12 l12 s12 j12); lit]` (sum over `lab`), a right factor
  `R[c, (n3 l3 j3 ncm lcm jrel); lit, Lam]` (sum over `lam`) and the two recoupling factors
  `a1[jab, jc, j12; lit, Lam]` and `g[lcm, j3, j12, jrel; Lam]`. The terms of a block are combined as
  outer products of the non-zero left and right lists and scattered into the matrix.
- `tcoef_as`: `cfp` rows times the NAS rows of each `(jrel, E)` group (block diagonal in
  `jrel, ncm, lcm, E`).
- `lab_scalar`: `6 sum T^T V_jac T` of a scalar Jacobi operator in one lab channel (blocks of
  different `N` couple through `V`; this is the 5.4 contraction in its simplest dense form).

`nit` is not a free sum: for one `N` block `Eab - E12 = Eq - Ec`, so the left and right `nit`
agree for every pair of columns and rows that belong to the block.

## 3. Validation

| Check | Result |
|---|---|
| Vectorised `T_NAS` vs element-wise transcription of spec 4.2 (3 blocks) | 4e-15 |
| `6 T_AS^T T_AS` vs m-scheme `sum_P sgn P` (own `cfp`), blocks (J,P,T,N): (1/2,+,1/2,0), (1/2,-,1/2,1), (3/2,-,1/2,1), (3/2,+,3/2,2), (1/2,+,1/2,2), (3/2,-,1/2,3), full | <= 2e-14 |
| same, random 16-column subsets of (3/2,+,1/2,4), (5/2,+,1/2,4), (7/2,-,3/2,3), (5/2,+,3/2,4) | <= 4e-14 |
| `(0s1/2)^3`, J = 1/2, T = 1/2 block of `6 T^T T` | entries +-3.0000000000 (so factor 6 is right; with `V = 21.4678/3`, spec 5d 5.3 follows) |
| all 24 890 values of `nnn_3NF400_e4.me3j.gz` from NuHamil's `ops/NNN-full_*.bin` + `cfp_*.bin` | max diff 5.0e-9, 0 entries above 1e-6; 22 channels, 2.8-12 s |
| lab channel counts and sizes (spec 5d 2.4, 4/8/4 and 14/28/16) | equal |
| `MoshinskyTable(d=2)` vs reference sum, orthogonality | 8e-15, 1.3e-14 |

Conventions that turned out different from the spec text:

1. **Orthonormality is not what the spec's proposed test 3 says.** `T_AS T_AS^T` is not the identity:
   the lab basis holds only `a >= b >= c`, and the pair `(a b)` product states are not antisymmetrised,
   so `T_AS` has rows that overlap only a fraction of each antisymmetric state. The right invariant
   is `6 T_AS^T T_AS = <abc| sum_P sgn P |def>` over the stored lab states (tested above). Do not test
   `T T^T = 1`.
2. The `nit` sum is a bookkeeping partition, not an independent sum (section 2).
3. Nothing else differed: phase `(-1)^(S12+l12+1+j12+lc+l3+(jc+J)/2)`, the 6j argument orders,
   factor 6 and the file order all agree with the source and with the reference output.

Details of NuHamil files, now parsed in `tests/ref3n.py`:

- `ops/NNN-full_*.bin`: `int32 n`, then `n x n` float64 (Fortran order, symmetric), no other header.
  It is the full Jacobi `V` (3N plus the evolved NN part) at `Nmax = 8` in NuHamil's `cfp` gauge.
  The lab file uses only the leading `E <= e3max` blocks of it: taking that sub-block reproduces the
  file, so the SRG flow runs at the ramp `Nmax` and the lab transform truncates afterwards (this
  settles open question 4 of spec 5a).
- `cfp_A3_*.bin`: layout of spec 5a section 5 parses with `offset == file size` for all 24 files.

## 4. Cost

Sizes are `n_rows x n_cols` of `T_NAS` per `(channel, N)`, summed over the channels and blocks. They
come from `lab_states` and `jacobi_cm_rows` (no coefficients computed).

| e3max (emax) | channels | lab states | `T_NAS` elements | `T_AS` elements (about 1/3) | largest channel | its `T_NAS` (double) |
|---|---|---|---|---|---|---|
| 8 (8) | 38 | 14 278 | 1.4e7 | 4.8e6 | J=5/2+, T=1/2, 1 199 states | 17 MB |
| 12 (12) | 54 | 125 295 | 6.7e8 | 2.2e8 | J=7/2+, 7 566 | 0.55 GB |
| 14 (14) | 62 | 307 305 | 3.2e9 | 1.1e9 | J=9/2+, 16 075 | 2.3 GB |
| 16 (14) | 70 | 687 352 | 1.34e10 | 4.5e9 | J=9/2+, 32 259 | 8.4 GB (largest block 6.1 GB) |

The `T_AS` column uses the measured ratio AS rows / NAS rows = 0.333 and agrees with spec 5d 4.3
(8.0e9 NAS-lab pairs with `T12 = Tab`; 4.45e9 `T_AS` elements; 33 GiB double).

Measured, host NumPy, one core, `OMP_NUM_THREADS=1`, J=3/2+, T=1/2 (the machine was shared with other
jobs, so times carry up to +-50%):

| e3max | block N | rows x cols | non-zeros | `tcoef_nas` time | peak RSS |
|---|---|---|---|---|---|
| 8 | 8 | 1 887 x 682 | 5.5e5 | 1.5 s warm (3.3 s with cold brackets) | 0.36 GB |
| 12 | 10 | 4 263 x 1 528 | 2.8e6 | 18-24 s | 0.60 GB |
| 12 | 12 | 8 540 x 2 995 | 1.1e7 | 47 s | 0.80 GB |
| 12 | all (N = 0..12) | - | - | 85 s, `cfp` excluded | 0.80 GB |

That is **1-3 microseconds per `T_NAS` element** (non-zeros are 22-44% of the elements). Production
(1.34e10 elements) is then 1.3e4-4e4 core-seconds, **4-11 core-hours** for all 70 channels; channels
are independent, so 16 processes give 15-40 minutes. About half of the time is in the 6j/9j
arrays (`sixj_array` is called per `(lit, Lam)`; caching them per block and hoisting the
`lit`-independent 9j would save much of it) and half in the outer-product scatter. `tcoef_as` costs
under 10% (0.2 s for the largest block of e3max 8).

Dense-product form (`refs/tcoef3_gemm_prototype.py`): for fixed `(j12, jc)` the sum is
`A[(pair, p), k] @ B[k, (c, q)]` with `k = (lit, Lam)`. It reproduces `tcoef_nas` to 6e-17.
Flops counted for the block of J=3/2+: N=8 0.93 GFlop (720 flop per `T_NAS` element), N=10 10.4 GFlop
(1 600 per element); the factor per step of 2 in `N` is about 2.2, so an extrapolation to N = 16 is
about 1.7e4 flop per element, **about 2e14 flop for all production channels** (dense, no sparsity
used). On the CPU the NumPy form is **2.5x slower** than the term form (47 s vs 18 s at N=10);
nothing was run on a GPU.

## 5. Recommendation for 5.4

1. **Keep the term-wise NumPy `tcoef_nas` as the validated reference and for development sizes**
   (e3max <= 12: 1-3 core-minutes per channel, about 22 core-minutes in total at 12).
2. **Do not store `T`; recompute per channel.** Peak memory is set by the largest `T_NAS` block
   (6.1 GB double at production). Build it in row groups `(jrel, ncm, lcm)` and multiply by `cfp`
   at once, as `tcoef_as` already does per group. What has to stay resident for a lab channel is
   `T_AS` for all blocks: about 2.8 GB double (1.4 GB single) for the largest channel, because
   blocks of different `N` couple through `V_jac` (`lab_scalar`).
3. **For production run the same stages on the device.** Left and right factors and the 6j tables
   are small (megabytes) and host cacheable; the `(lit, Lam)` product and the contraction with
   `V_jac` (about 5e14 flop, spec 5d 5.1) are dense; the scatter into `(row, col)` is the irregular
   part and is done with precomputed index arrays on the device. The CPU numbers above do not prove
   this is faster; the **gate before committing is to time the product form on one H200/GH200 for
   the N = 12 block of the e3max 12 channel above** (11 M non-zeros, 47 s on one CPU core), and to
   exploit the energy structure (`2 nit + lit = Eab - E12`) so that fewer `k` are used per pair.
4. Optimise the host path only if a CPU-only production run is wanted (cache 6j arrays, process
   pool over channels); no algorithmic risk remains.

## 6. What was not done

No JAX/GPU measurement (no accelerator on this machine). The e3max 14 and 16 cost is extrapolated
from sizes and from the per-element time at e3max 8 and 12. The own-`cfp` check against the m-scheme
covers `N <= 4`; the NuHamil-gauge check covers 4/8/4 only (the only reference file); a larger
reference (e.g. 6/12/6) would test blocks up to `N = 6` against NuHamil.
