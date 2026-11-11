# nuham: plan

A from-scratch JAX package that produces the same physics output as NuHamil
(https://github.com/Takayuki-Miyagi/NuHamil-public), for use with imsrg++.
It is not a translation of the Fortran code. NuHamil is used as a numerical oracle.

This file records what was done, what was decided and what is open. Finished items get one line (outcome and a link);
details live in `docs/`. How to update it: `CLAUDE.md`, section "Status and PLAN.md". Design rules: `CLAUDE.md`.

## Scope

- Done: EM500 NN, Hamiltonian only, `me2j` output (Phases 0-4); the local N2LO 3N force, the LNL regulator and the
  frequency conversion, with `me3j` output and the production files (Phase 5).
- Planned, not started: consistently evolved operators (gap L, below).
- Later: the non-local 3N regulator (`NonLocal`, needs external HDF5 matrix elements), other output formats (HDF5/Zarr must
  stay easy to add), scans in 3N.
- Kokkos: no work now. Keep kernels pure so a port stays easy.
- Develop on this machine (CPU-only JAX, float64, small cases); large runs on DeltaAI (GPU). Nothing may assume a device.

## Layout

```
nuhamil-alt/
  src/nuham/
    constants.py runtime.py timing.py fetch.py   shared infrastructure (NuHamil constants, writer
                                                 options, stage timers, input-file download)
    angmom/       6j/9j (scalar and array), Moshinsky bracket tables (shared, host-built)
    basis/        relative-state and lab-orbit bookkeeping (shared)
    kernels/      pure JAX, shared: HO radial functions, rotation (matrix exponential), SRG flow,
                  host transfer
      nn/           projection, Coulomb, T_rel, lab contraction, me2j assembly and formatter
      nnn/          local-force radial transforms and HO projection, antisymmetrisation, lab contraction,
                    regulator, frequency overlaps
      compile.py    per-kernel compiler options (no GEMM autotuning for the large 3N products)
    nn/           NN pipeline: input_file (.bin reader), relative (SRG driver), lab_transform,
                  me2j (writer), generate (+CLI), scan (+CLI)
    nnn/          3N pipeline: jacobi (basis, antisymmetrizer), tcoef (Jacobi -> lab T-coefficients),
                  local_force/{radial, angular, assembly}, regulator, hamiltonian, srg3, lab_transform, packed
                  (packed lower triangle), me3j (lookup and writer), generate (+CLI)
  tests/          NuHamil references + independent identity checks (flat; helpers in the same folder)
  benchmarks/     stage-timed runners and recorded baselines
  scripts/deltaai/  Slurm job scripts of the verification and production runs on DeltaAI
  refs/           throwaway oracle scripts and small NuHamil outputs
  docs/           specs, studies, run records
```
Regrouped after Phase 5.2.4: `relative_nn` -> `nn/relative`, `interactions/nn_file` -> `nn/input_file`, `jacobi3` -> `nnn/jacobi`,
`tcoef3` -> `nnn/tcoef`, `local3n*` -> `nnn/local_force/*`, `kernels/nnn_*` -> `kernels/nnn/*`. Older docs keep the names of their time.

## Decisions

- Benchmark runner lives outside the package (`benchmarks/`).
- Tests are not tied to a NuHamil checkout: `nuham.fetch` finds or downloads the momentum-space files (pinned commit, checksums),
  and a synthetic charge-independent interaction tests the whole NN pipeline with exact identities.
- Provenance: code transcribed from NuHamil's Fortran lives in `refs/`, not `src/`. The package has only the rotation-based
  Moshinsky tables; the 3N angular algebra in `src/` comes from Navratil 2007 (Few-Body Syst. 41, 117), checked against the
  NuHamil paper (arXiv:2302.07962), with NuHamil's formulas (spec 5b) in `refs/` as a cross-check.
- Moshinsky brackets and 6j/9j: host NumPy with caching; JAX only if profiling shows a bottleneck.
- SRG flow: default N_flow = 100 (matches NuHamil to ~1e-7); `n_flow` is an optional, documented knob (`docs/study-srg-truncation.md`).
- HO radial functions: the normalized three-term recurrence in n with closed-form coefficients as a `lax.scan`, momentum form
  sqrt(b) g(b k) (-1)^n; checked against mpmath (`refs/ho_radial_check.py`).
- 3N force: start with the **local** N2LO force (NuHamil `Regulator="Local"`, sets `3NF400` and `3NF500-BetaDecay-Local`), then
  `LNL` (2026-10-03). Each term is stored LEC-free as `sum_a C_a B_a` (angular coefficients times HO projections of universal radial
  tables), so LEC and cutoff scans reuse everything. Radial meshes converged by default; NuHamil's hard-coded meshes as an option.
- 3N oracle strategy: the antisymmetric Jacobi basis is a gauge choice per N block, so compare gauge-invariant quantities
  (A = C^T C, eigenvalues before and after the flow, traces) and the final `me3j` values.
- Production file uses the genuine 3N force at all relative J (NuHamil's default); `--j3max-3nf` stays an option (user, 2026-10-05).
- No checksum or second copy of the production files (user, 2026-10-06); regenerating takes about 15-30 min of GH200.
- Gaps of the production file: close the cheap ones first, not all (user, 2026-10-06).
- ICRN baselines parked (user, 2026-10-07).

## Phases

### Phase 0: spec and oracle — DONE
`docs/physics-spec.md` and `docs/spec-1..4`; references in `refs/work`; `refs/oracle_numpy.py` reproduces all references to ~1e-8.

### Phase 1: building blocks — DONE
HO functions (vs mpmath 1e-14), projection (vs NuHamil cache 5e-6, the cache has 7 digits), 6j/9j (vs sympy 1e-13), Moshinsky
brackets (orthogonal, equal to 480 oracle values), deuteron -2.2246 MeV.

### Phase 2: relative-space NN — DONE
Coulomb and SRG flow (Diffrax Dopri8): matches the numpy oracle to 1.6e-8 MeV on 16 golden channels, spectrum invariant in all
54 channels, ~2 s for the full flow on the CPU. Not done: unequal-mass pn (`pn_same_mass=false`), non-kinetic generators.

### Phase 3: lab-frame me2j — DONE
Lab transform and writer: srg2.0 hw16/hw20 emax 4 to 2.4e-7/1.6e-7; references made from the 7-digit cache to 3.3e-6.
imsrg++ reproduces the NuHamil-based 4He energies to ~1e-6 MeV (`docs/imsrg-check.md`).

### Phase 4: GPU readiness and speed — DONE (ICRN baselines parked)
Rotation-based Moshinsky tables (`docs/study-moshinsky-rotation.md`), device contraction, vectorized 6j/9j, compile cache,
threaded writer. Runs: `docs/benchmarks.md`.

| Step | Outcome |
|---|---|
| 4a. Compact bracket cache, pinned-host copy | DONE: cache 1.55 -> 0.147 GB, load 38.3 -> 0.5 s; pinned copy correct, no end-to-end gain (`docs/benchmarks.md`) |
| 4b. Stage timers, `benchmarks/` runner, baselines | DONE: local CPU (NuHamil's own 21 s, 1.39 GB as reference) and GH200 (`docs/benchmarks.md`) |
| 4c. Writer | DONE: vectorized byte-identical formatter, `isal`, device assembly and formatter, threaded compression; emax 14 on the GH200 54 -> 8.1 s (`docs/benchmarks.md`) |
| 4d. Parameter scans (`nuham.nn.scan`) | DONE: emax 14 point 4.6-4.8 s on the GH200 at batch 6-12; host memory ~6 GB per extra point (`docs/benchmarks.md`) |

### Phase 5: 3N — DONE (production files made, validation as far as possible)
Specs: `docs/spec-5a-jacobi-basis.md`, `5b-local-n2lo-3nf`, `5b2-lnl-regulator`, `5c-3n-hamiltonian-srg`, `5d-lab-transform-and-me3j`,
`5e-hw-conversion`. Production size: emax 14, e2max 28, e3max 16, 6 367 911 165 values, about 31 GB gzipped, all 70 lab channels
in memory at once (23 GiB as packed lower triangles).

| Step | Outcome |
|---|---|
| 5.0 Specs and NuHamil references | DONE: specs above; reference run 3NF400 4/4/N3max 8 (386 s, 1.1 GB in NuHamil). The numpy oracle was built piecewise in `refs/` as steps needed it |
| 5.1 Jacobi basis and antisymmetrizer (`nnn.jacobi`) | DONE: all 24 channel counts and `C^T C` equal NuHamil's; dense per-N solve affordable up to Nmax 40 (`docs/spec-5a-jacobi-basis.md`) |
| 5.1b T-coefficients (`nnn.tcoef`) | DONE: all 24 890 reference values to 5e-9 with NuHamil's Jacobi V; factor 6 and phases confirmed; host build, device contraction (`docs/study-tcoef3.md`) |
| 5.2 Local N2LO force (5.2.0-5.2.5) | DONE: Cartesian check (factor 3, NAS gauge), radial and angular layers, assembly equal to NuHamil to 2e-12; J=1/2+ at N3max 40 in 30 s on the CPU, NuHamil's mesh converged (`docs/study-local-3nf.md`) |
| 5.3 Jacobi Hamiltonian and SRG (`nnn.hamiltonian`, `nnn.srg3`) | DONE: eigenvalues and V3 of all 24 channels equal NuHamil to 2e-6; the flow is not stiff (15-16 Dopri8 steps at Nmax 40) (`docs/study-3n-flow.md`) |
| 5.4 Lab transform and `me3j` writer (`nnn.generate`) | DONE: NuHamil 4/8/4 files to 2e-6, value counts as in spec 5d; production file made 2026-10-06 (`docs/study-me3j.md`) |
| 5.5 imsrg++ validation | DONE 2026-10-07 as far as possible: our 6/12/12 file equals NuHamil's over 144 345 500 values (5.65e-6); imsrg++ reads the production file, 16O at emax 4 and 6 (`docs/imsrg-check.md`). Open: gaps B, E, I, J below |

**Production files** (14/28/16, ramp `40-5-36-7-32-9-28-11-24`, 3NF400, lambda 2.0, all relative J; on DeltaAI under
`/projects/bhhk/soham/nuham/out`; details in `docs/study-me3j.md` and `docs/imsrg-check.md`):
- `N3LO_3NF400_hw16_e14_e2_28_e3_16.me3j.gz`: hw 16 directly, job 3324394, 2026-10-06, 30 min on one GH200.
- `N3LO_3NF400_hw30to16_e14_e2_28_e3_16.me3j.gz`: flowed at hw 30 and converted to 16 (gap K), job 3350092, 2026-10-10. Use this
  one against NuHamil-based results.

**Validation gaps of the production file** (decided with the user 2026-10-06 and 2026-10-07; results in `docs/imsrg-check.md`):

| | Gap | Status |
|---|---|---|
| A | GPU path against the CPU path | DONE 2026-10-07: GPU and CPU 6/12/12 files differ by at most 1.0e-8 |
| B | Production settings (N3max 24-40) have no NuHamil reference (NuHamil needs ~2 days, ~200 GB) | not now; only self-consistency checks possible |
| C | imsrg++ IMSRG(2) flow does not settle for 16O at emax 6 | DONE 2026-10-07: `white` generator with smax 30 settles |
| D | No independent 16O NN+3N reference (NuHamil's README table is for `N3LOEMlnl`) | D0-D4 DONE: LNL regulator built, equal to NuHamil to 4.1e-6; imsrg++ within keV of the README for 4He. D5 open (below) |
| E | Most of the file has not been read by imsrg++ (only the emax 4 and 6 prefixes) | not now; needs imsrg++ on a large-memory machine |
| F | Non-flat ramp against NuHamil | DONE 2026-10-07: 1.23e-6 |
| G | Other hw and lambda in the 3N chain | DONE 2026-10-07: hw 20 3.34e-6, lambda 1.8 2.03e-6 |
| H | 3NF500 end to end | DONE 2026-10-07: 1.55e-6 |
| I | NN `me2j` element by element at emax 14 (compared up to emax 12) | open, low priority |
| J | `--j3max-3nf` against NuHamil's `J3max_initial_3nf` | open, low priority |
| K | Frequency conversion (`hw_target`), the cause of the 16O offset to the README | DONE 2026-10-10: K4 equal to NuHamil to 7.4e-6, 16O offset to the README from +0.26..0.45 MeV down to a few keV; production file regenerated (`docs/spec-5e-hw-conversion.md`) |

**Gap D, open step D5:** record the LNL results as closed and decide with the user whether to make an LNL production file
(one GH200 job). Spec: `docs/spec-5b2-lnl-regulator.md`; results: `docs/imsrg-check.md`, sections Gap D3 and D4.

**Still open across Phase 5:** non-kinetic SRG generators (not analysed); the coarser padding ladder of the lab contraction
could be reverted (low value, `docs/study-me3j.md`); compile counts of a whole run and the 72-CPU and BLAS-thread variants
were not measured.

## Plan, not started: consistently evolved operators (gap L, drafted 2026-10-11)

Goal: operators other than H (radius, then electromagnetic and weak operators) with the same SRG transformation as H, written
in the operator `me2j` format that imsrg++ reads (`ReadOperator2b_Miyagi`). Do not start before the user asks. What NuHamil does,
the file format and the consequences for each step: `docs/spec-6-operators.md` (section 5 has the details behind each row).

| Step | Work | Gate |
|---|---|---|
| L0 | Spec | DONE 2026-10-11: `docs/spec-6-operators.md`; NuHamil R2 oracle in `refs/work/ops/r2_e4` |
| L1 | `U` from the flow: integrated `U` against NuHamil's eigenvector pairing (measure with a numpy replica in `refs/` first) | `U` unitary to 1e-10, `U H0 U^T` equals the evolved `H`, `U` equals NuHamil's `UT` output on a small channel set (~3e-7) |
| L2 | Relative-space scalar operators (`R2`) and their evolution | evolved relative `R2` against NuHamil's relative cache, 1e-5 |
| L3 | Lab transform of a scalar operator, one-body part, operator `me2j` writer | file equals NuHamil's at emax 4 (and 6/12), 1e-6 MeV |
| L4 | imsrg++ reads our operator file: 4He and 16O radius | agreement to the tolerance of L3 |
| L5 | Tensor operators (`E2`, `M1`, `E1`, `GamowTeller`; ranks 1 and 2), parity-odd and isospin-changing operators | one operator at a time against NuHamil |

Not covered: evolved three-body parts of operators; lepton-number-violation and parity-violating operators. Risks: the
centre-of-mass treatment of `R2`, the cost of carrying `U` (about twice the flow), the operator file ordering (`docs/spec-6-operators.md`).

## Possible future work (not planned; the user asked for it to be recorded, 2026-10-11)

**Scaling the 3N generator beyond one GPU.** Assessed from the code layout and recorded stage times, not measured; no multi-GPU run exists.

- Why it should be easy: the work is hundreds of independent pieces, not one array program. The Jacobi side is 72 channels (2J, parity, 2T), each basis, force, regulator, flow, projection and cache write, with the V3 cache already per channel on disk; the lab side is one block per lab channel, a sum over Jacobi channels of `T^T V T` read from that cache. Kernels are pure functions, and `generate.py` already runs a fork-based process pool. No `pmap`, `shard_map` or device mesh appears in `src/nuham`, and none should be needed: this is job-level parallelism.
- Jacobi side (about 22 of the 23 minutes of the production run): one process per GPU, each with a subset of the channels, sharing the cache on `/projects`. Channel cost varies a lot with J and N3max, so a cost-sorted work queue beats a static split.
- Lab side and file: the output is one gzip stream in a fixed tuple order; a concatenation of gzip members is a valid gzip stream, so ranks could write contiguous segments (boundaries on the enumeration order, value counts computable beforehand) that are concatenated afterwards. This also splits the 75.6 GB host memory of the production run, at the price of every node reading the V3 files it needs. Across nodes a Slurm job array plus one concatenation step would do; no MPI.
- Limits: small cases are dominated by compile and per-channel overhead; host-side formatting, gzip and coefficient builds were already the main remaining cost at the largest sizes, so CPU and I/O limit first; N3max well beyond about 40 needs a distributed dense solver for the largest single channels (the one case that needs real multi-GPU JAX work).
- Unchecked: per-channel cost balance, whether any shared state (for example the Moshinsky table cache) is written during a run, and whether imsrg++'s reader accepts a concatenated gzip file. Settle these before building anything.

## Risks

- Numerical stability of Moshinsky brackets/recursion at high emax; SRG stiffness.
- Padding vs recompilation trade-off for irregular channel sizes.
- `me2j` format details (ordering, truncation, units) must match imsrg++ exactly.
- NuHamil may have quirks. Where independent checks disagree with it, flag it.
