# nuhamil-alt: plan

A from-scratch JAX package that produces the same physics output as NuHamil
(https://github.com/Takayuki-Miyagi/NuHamil-public), for use with imsrg++.
It is not a translation of the Fortran code. NuHamil is used as a numerical oracle.

## Scope

- Done: EM500 NN, Hamiltonian only, `me2j` output readable by imsrg++ (Phases 0-4), and the local N2LO 3N force with SRG-evolved Jacobi operators and `me3j` output (Phase 5.1-5.4; the production file exists). Now: imsrg++ validation of the production file (5.5).
- Develop on this machine: CPU-only JAX in float64, small cases (deuteron, emax <= 4-6).
- Larger runs later on a cluster (Delta, GPU). Nothing may assume a device.
- Later: the non-local and LNL 3N regulators, other output formats, scans in 3N.
- Other output formats (HDF5/Zarr) must stay easy to add.
- Kokkos: no work now. Keep kernels pure so a port stays easy.

## Design principles

- Kernels are pure functions on explicit arrays: no classes, no I/O, no global state.
- Drivers handle channel loops, chunking and I/O.
- Static shapes: bucket and pad channel dimensions so `jit` does not recompile per channel.
- Angular-momentum coefficients and quantum-number tables are built on the host (NumPy).
- The interaction is a plugin: "give partial-wave momentum-space matrix elements
  for channel (J, parity, T, ...)". EM500-from-file is the first implementation.
- Output is built channel by channel and streamed to disk, never held whole in RAM.

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
      nnn/          local-force radial transforms and HO projection, antisymmetrisation, lab contraction
      compile.py    per-kernel compiler options (no GEMM autotuning for the large 3N products)
    nn/           NN pipeline: input_file (.bin reader), relative (SRG driver), lab_transform,
                  me2j (writer), generate (+CLI), scan (+CLI)
    nnn/          3N pipeline: jacobi (basis, antisymmetrizer), tcoef (Jacobi -> lab T-coefficients),
                  local_force/{radial, angular, assembly}, hamiltonian, srg3, lab_transform, packed
                  (packed lower triangle), me3j (lookup and writer), generate (+CLI)
  tests/          NuHamil references + independent identity checks (flat; helpers in the same folder)
  benchmarks/     stage-timed runners and recorded baselines
  scripts/deltaai/  Slurm job scripts of the verification and production runs on DeltaAI
  refs/           throwaway oracle scripts and small NuHamil outputs
  docs/           specs, studies, run records
```
Regrouped after Phase 5.2.4 (the 3N modules had been added next to the NN ones): `relative_nn` -> `nn/relative`,
`interactions/nn_file` -> `nn/input_file`, `jacobi3` -> `nnn/jacobi`, `tcoef3` -> `nnn/tcoef`,
`local3n*` -> `nnn/local_force/*`, `kernels/nnn_*` -> `kernels/nnn/*`. Phase notes below keep the module names of their time.

## Phases

### Phase 0: spec and oracle (no JAX) — DONE
Result: `docs/physics-spec.md` (summary) and `docs/spec-1..4`; reference outputs in
`refs/work`; slow numpy oracle `refs/oracle_numpy.py` reproducing all references to ~1e-8.
Original description:
- Read NuHamil's NN path (TwoBody/NNForce, TwoBodyRelOps, Renormalization,
  TwoBodyTransCoef, TwoBodyLabOps, me2j writer). Write `docs/physics-spec.md`
  with every convention: normalization, phases, antisymmetrization, isospin / pn-pp-nn.
- Build NuHamil locally. Generate small references: deuteron, emax=4 and 6 `me2j`,
  a few hw and SRG lambda values.
- Check the imsrg++ `me2j` format against both NuHamil's writer and imsrg++'s reader.

### Decisions made after Phase 0
- Benchmark runner lives outside the package (`benchmarks/`). Optional binary/HDF5 output waits until after the writer work.
- Tests are not tied to a NuHamil checkout: the momentum-space files are found or downloaded by `nuham.fetch` (pinned commit and checksums for all 12 NuHamil files; one `interaction_bin` fixture turns a missing file into a skip) and a synthetic charge-independent interaction tests the whole pipeline with exact identities. Without any NuHamil data 549 tests pass and 17 skip; without `refs/` as well, 545 pass and 19 skip.
- Provenance: code that transcribes NuHamil's Fortran (the explicit Moshinsky bracket sum, the oracle) lives in `refs/`, not `src/`. The package only has the rotation-based Moshinsky tables, and its tests pass without `refs/` (reference-dependent tests skip).
- Moshinsky brackets and 6j/9j coefficients: host NumPy with caching. Move to JAX only if
  profiling shows a bottleneck.
- SRG flow: default N_flow = 100 (matches NuHamil to ~1e-7). `n_flow` is an optional,
  documented speed/accuracy knob; see `docs/study-srg-truncation.md`.
- HO radial functions: adopt the normalized three-term recurrence in n from
  `pre-quantum-projects/csm/csm/basis/harmonic_oscillator.py`, re-derived with closed-form
  coefficients (no sympy) as a `lax.scan` in JAX; use NuHamil's constants and a
  momentum-space form sqrt(b) g(b k) (-1)^n. Checked against mpmath in `refs/ho_radial_check.py`.
- Phase 1 is done (see below). Phase 2 has not started.

### Phase 1: building blocks, each tested alone — DONE
Implemented in `src/nuham`: `kernels/ho.py` (recurrence as `lax.scan`, momentum form), `kernels/projection.py`,
`kernels/relative.py` (T_rel), `interactions/nn_file.py` (.bin reader), `basis/relative.py`,
`angmom/racah.py` (6j, 9j), `angmom/moshinsky.py` (explicit bracket sum; later moved to `refs/moshinsky_reference.py`, see Decisions). 510 tests pass. Results: HO vs mpmath 1e-14;
6j/9j/CG vs sympy 1e-13; bracket matrices orthogonal (d=1 and 2) and equal to oracle values (480 golden);
projection vs NuHamil cache 5e-6 MeV (cache has 7 digits); deuteron -2.2246 MeV.
Open: bracket cost is ~1 ms uncached at E=28; decide in Phase 3 whether caching is enough.
Original description:
HO radial functions; Gauss-Legendre momentum -> HO projection; Talmi-Moshinsky and
Moshinsky brackets (orthonormality). Test against NuHamil intermediates and analytic identities.

### Phase 2: relative-space NN — DONE
`kernels/coulomb.py`, `kernels/srg.py` (banded commutator, one matmul per RHS, Diffrax Dopri8, max-norm
error control, hw-scaled units), `relative_nn.py` (driver, padded batching). Results: matches the numpy
oracle to <= 1.6e-8 MeV on 16 golden channels (hw 16/24, lambda 2.0/1.8); spectrum invariant in all 54
channels (1e-6 test bound); deuteron -2.2245754 MeV before and after; full flow for 54 channels in
~2 s (scipy: ~40 CPU-s), peak RSS ~670 MB; rtol=atol 1e-10 kept (1e-7 would be only ~25% faster).
Not done: unequal-mass pn (`pn_same_mass=false`), non-kinetic generators, explicit `n_flow` option.
Original description:
Read EM500 momentum-space input (`N3LO_EM500_kmax8_N100_Jmax8.bin`), project to HO,
SRG flow (stiff; implicit solver or exact-unitary scheme; compare with VODE).
Check deuteron energy and SRG eigenvalue invariance.

### Phase 3: lab-frame me2j — DONE (imsrg++ check passed for 4He, docs/imsrg-check.md)
`basis/lab.py`, `angmom/moshinsky_table.py` (cached (N, lam) matrices, optional disk cache),
`lab_transform.py` (T-coefficients vectorized over rows/pairs, per-block contraction), `me2j.py`
(tuple enumeration identical to the imsrg++ reader, 4-value assembly, writer/reader), `generate.py` (+CLI).
Results vs NuHamil `me2j` (all values, same count and order): srg2.0 hw16 emax4 2.4e-7; srg2.0 hw20 1.6e-7;
bare hw16 3.3e-6 and srg1.8 emax6 2.2e-6 (those references used the 7-digit relative cache, limit ~5e-6).
Forbidden entries are exactly zero. 538 tests pass.
Cost: emax 4: 2.5 s; emax 6: 5 s; emax 8 (e2max 16): ~60-90 s cold, of which ~65% is bracket evaluation
and ~25% gzip. Brackets needed (entries incl. zeros): e2max 16: 4e5; 20: 1.8e6; 24: 6.2e6; 28: 1.8e7, at
50-500 us each uncached -> a few CPU-hours at e2max 28 (one-time: hw-independent, disk-cacheable).
Sizes at emax=14/e2max=28: 59 lab channels, largest pair basis 4234, 4.2e7 tuples (1.3 GB float64).
Done later: imsrg++ reads our files and reproduces the NuHamil-based 4He energies to ~1e-6 MeV at emax 4 (full flow) and emax 8 (HF/MBPT), see docs/imsrg-check.md. Not done: bracket generation at e2max > 16 was done in Phase 4.
Original description:
Relative -> lab transform plus Trel/Hcm. Write `me2j`. Compare elementwise with NuHamil.
End-to-end: imsrg++ on 4He with our file and with NuHamil's.

### Phase 4: GPU readiness and speed — DONE except the ICRN baselines, which are parked (user, 2026-10-07)
Done (see `docs/study-moshinsky-rotation.md`, `docs/icrn-run.md`, `docs/deltaai-run.md`):
- Moshinsky brackets from a rotation generator (all tables for e2max 28 in ~15 s, was hours); vectorized 6j/9j; device-resident
  contraction (padded block stacks, one GEMM, result copied back once); threaded writer (gzip level 1) and JAX compile cache;
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` by default on import.
- Runs: ICRN shared H200 (warm emax 14 in 75-80 s) and DeltaAI GH200 (warm emax 14 in ~54 s; emax 10: 14.5 s, emax 12: 25.3 s).
  Local CPU, emax 14: ~174 s (profiled). Largest emax-14 channel build: 0.24 s on GH200.
- Finding: at NN size the GPU-friendly work (contraction, SRG flow) is a small share; file formatting and gzip are ~45% of a warm
  run, and the bracket disk cache (1.52 GB, 38 s to load from /projects) costs more than regenerating.

Agreed order (each phase needs a correctness gate and a benchmark gate; nothing is done until both pass):

| Phase | What gets built | Correctness gate | Benchmark gate (where) |
|---|---|---|---|
| 4a. Quick fixes — DONE (docs/deltaai-run.md): cache 1.55 -> 0.147 GB, load 38.3 -> 0.5 s; pinned copy 28% faster on the largest channel, no end-to-end gain | Compact bracket cache (valid blocks only, ~0.15 GB, or skip the disk cache when regeneration is cheaper). Pinned-host result copy on GPU backends, no fresh host buffer per result. | me2j unchanged; all tests pass. | Cache load time vs the 38 s measured; device-to-host copy vs the 9 GB/s pageable path (DeltaAI, ICRN). |
| 4b. Harness — DONE (docs/benchmarks.md): baselines on the local CPU (`benchmarks/baseline_cpu_emax8.json`, with NuHamil's own 21 s and 1.39 GB on this machine as the reference) and on DeltaAI (`benchmarks/baseline_gh200_*.json`); the ICRN baseline is PARKED (user, 2026-10-07; `docs/icrn-run.md` holds only early run records, no harness baseline) | Stage timers built into `generate_me2j`. A benchmark runner **outside the package** (`benchmarks/`) writing JSON with device, versions, stage times, peak memory. A compute-only mode that skips file writing. | Timers do not change results. | Baselines on local CPU, ICRN, DeltaAI; NuHamil's own NN time and memory on this machine as a reference. |
| 4c. Writer — DONE: vectorized host formatter (5.6x, byte-identical), optional `isal` compressor, device assembly and device formatter (JAX), threaded igzip; emax 14 single run 54 s -> 8.1 s on the GH200 (docs/benchmarks.md); NumPy CPU contraction tried and dropped | Parallel text formatting (gzip stays in its thread); NumPy contraction path for CPU runs. Binary/HDF5 output is deferred. | me2j byte-identical; imsrg++-readable. | Share of run time spent writing (now ~45%); compute-only vs full-run times. |
| 4d. Parameter scans — CPU part DONE (docs/benchmarks.md); GPU measured on DeltaAI (job 3300961): 38% less time per point at batch 6; GPU prefers flow_sets > 1, CPU does not; emax 12-14 measured (job 3301003): compute-only 2.6-2.8x faster per point, but a full emax 14 point is writer-bound (gzip/assemble) and host memory is ~6 GB per extra point (matrices 3.9 GB each); isal (now an optional extra) cuts the write stage 3x on the GH200; batched assemble is bit-identical and 1.5-2x faster locally but gave no gain on the GH200, assemble is now the largest stage of a full scan point (see docs/benchmarks.md); a full emax 14 scan point is now 4.6-4.8 s on the GH200 (batch 6-12); ICRN baseline parked (2026-10-07) | Many (hw, lambda) sets sharing brackets and coefficients; batched SRG flows and batched contraction. | Each set matches a single run. | Time per set vs batch size, CPU vs GPU: the clearest test of GPU advantage at NN size. |

### Phase 5: 3N — 5.0 specs done, same gates per step
Decided with the user (2026-10-03): start with the **local** N2LO 3N force (NuHamil `Regulator="Local"`: sets `3NF400` and `3NF500-BetaDecay-Local`; the `NonLocal` sets need external
HDF5 matrix elements we cannot regenerate; `LNL` adds a non-local regulator and comes after). Output: validate with single-precision `.me3j.gz` (the only format imsrg++'s `full` reader takes; NuHamil's
default), production later. Sizes: small cases for development (emax/e3max 4-8), emax 14, e2max 28, e3max 16 for production (6.37e9 values, about 103 GB of text, 27-34 GB gzipped, all 70 lab channels
needed at once: about 46 GiB single precision as full squares, about 23 GiB as the packed lower triangle that is stored now). Specs: `docs/spec-5a-jacobi-basis.md`, `5b-local-n2lo-3nf`, `5c-3n-hamiltonian-srg`, `5d-lab-transform-and-me3j`.

Oracle strategy: the antisymmetric Jacobi basis is an arbitrary gauge per N block, so compare gauge-invariant quantities (the antisymmetrizer A = C^T C, eigenvalues before and after the SRG that NuHamil prints per channel,
traces and norms) and, at the end, the lab `me3j` values. Reference run: `3NF400`, emax 4, e3max 4, N3max 8, hw 16, lambda 2.0 (386 s, 1.1 GB, 24 890 values).

| Step | What gets built | Correctness gate | Benchmark gate |
|---|---|---|---|
| 5.0 | Specs (done), small NuHamil references (first one done), numpy oracle of the whole chain (not started). | Oracle reproduces the references. | NuHamil 3N time and memory as the baseline (386 s, 1.1 GB for the reference case). |
| 5.1 — DONE (`nuham.nnn.jacobi`, `tests/test_jacobi3.py`): all 24 channel counts and `C^T C` equal NuHamil's; dense per-N solve is affordable: Nmax 40 channels 6.8 s (J=1/2+) to 530 s (J=17/2+, largest N block 10 370, 6.7 GB RSS) on shared host CPU, J=5/2+ AS total 11 340; the 1e-2 threshold is replaced by an idempotence assert (1e-9) | Jacobi basis: channel enumeration (T, J, P) with the ramp, NAS state lists in NuHamil's order, the three-body antisymmetrizer per (J, P, T, N) from 9j/6j and d = 1/3 Moshinsky brackets (existing `MoshinskyTable(d=...)`), eigen-decomposition into the AS basis. | State counts of all 24 channels (table in 5a), A idempotent with eigenvalues 0/1, A equal to the stored `cfp` projector to ~1e-14, trace assertion at large N. | Time and memory of the antisymmetrizer per channel, up to Nmax 40 (is a dense eigensolve per block affordable? the 1e-2 threshold is unchecked there). |
| 5.1b — DONE (`nuham.nnn.tcoef`, `docs/study-tcoef3.md`): d=2 bracket already in `MoshinskyTable`; all 24 890 reference values reproduced to 5e-9 with NuHamil's Jacobi V and cfp; factor 6 and phases confirmed against an m-scheme calculation; energy-blocked build (`tcoef_nas_blocked`): 13x faster than term-wise, whole channel 32 s at e3max 16, about 16 core-min for all 70 channels (scaled), so T_NAS stays on the host and the device does the contraction; recompute per channel, do not store T; a 6/12/6 reference would check N > 4 against NuHamil | (parallel, early) | **T-coefficient feasibility prototype**: the Jacobi-to-lab coefficients are the largest and riskiest piece (about 8e9 irregular coefficients at production, a d = 2 Moshinsky bracket the package does not have). First check the d = 2 bracket (`MoshinskyTable(d=2)` exists for the NN lab transform; verify it is the one the 3N chain needs), then a small T_AS against the reference file, then cost scaling. | d = 2 bracket orthogonality and equality with the oracle brackets; T_AS reproduces the reference lab elements at emax 4; the factor 6 and the phases confirmed numerically. | Cost and memory of building T per lab channel at e3max 8, 12, 16, host vs device: decides the 5.4 design before anything else is built on it. |
| 5.2 | Local N2LO force term by term (c1, c3, c4, cD, cE): coordinate-space kernels, partial-wave decomposition, HO projection as batched dense products; the redundant `precalculations` of NuHamil computed once. | Per-term pre-SRG matrix elements against NuHamil run with `save_3nf_before_lec=.true.`; cE analytic check (Lambda to infinity); mesh-convergence study (NuHamil's meshes are hard-coded and unchecked); settle the open phase and factor-3 questions. | Time per term vs NuHamil; GPU vs CPU. |
| 5.3 — DONE (`nuham.nnn.hamiltonian`, `nuham.nnn.srg3`, `kernels/srg.srg_flow_dense`, `tests/test_nnn_hamiltonian.py`, `docs/study-3n-flow.md`): H0 eigenvalues and the stored V3 of all 24 NNN-full channels (3NF400, flat8) match NuHamil to 2e-6 (spectrum 1.5e-6, trace 4e-6); the subtracted NN is the flow of the charge-averaged bare potential (averaging flowed pp, pn, nn is 1e-4 off); the flow is not stiff (15-16 Dopri8 steps at Nmax 40, rtol 1e-8 gives V3 to 2e-9); GH200: n = 8078 in 11 s, n = 11 340 in 28 s (22.8 GB device memory) | Jacobi-space Hamiltonian (T + 3 V_NN + V_3N), SRG flow per channel with the ramp (reuse the NN flow kernel), NN SRGs computed once and shared, stored V3 = H(s) - T - 3 V_NN(evolved). | Eigenvalues before and after the flow per channel (NuHamil prints them), gauge-invariant traces (e.g. trace of V3 = 164.007348 for J=1/2+, T=1/2 in the NN-only reference run); NN-only V3 reference files. | Dense flow cost: largest production channel about 11 340 states (1.03 GB), about 6.6e13 flop per right-hand side over all channels; number of right-hand-side evaluations unknown, measure at small Nmax. |
| 5.4 — DONE, production file made (job 3324394, 2026-10-06, see **Production run** below; `nuham.nnn.me3j`, `lab_transform`, `generate`, `packed`, `kernels/nnn/lab.py`, `kernels/nnn/antisym.py`, `tests/test_me3j.py`, `test_lab_transform3.py`, `test_generate3.py`, `test_antisym_device.py`, `test_v3_reference.py`; `docs/study-me3j.md`): NuHamil's 4/8/4 files (NN only and 3NF400, 24 890 values) reproduced to 2e-6 end to end; counts 456 320 (6/12/6), 5 199 790 (8/16/8) and 6 367 911 165 (14/28/16; a whole-file test with a flat16 ramp wrote 32.2 GB in 14 min, the real ramp made the production file) as in spec 5d; imsrg++ gives the same 4He energies with our files (5e-6; `file3e2max` must be passed). CLI `python -m nuham.nnn.generate` (worker count defaults to all CPUs and is printed first, stop points, per-channel stage log, cProfile, resumable V3 and Jacobi caches, `--j3max-3nf`). Angular coefficients of the force vectorised (6-27x per channel, identical output, `docs/study-local-3nf.md`). | Jacobi-to-lab transformation with e3max truncation (two dense GEMMs per block) and the `me3j.gz` writer (same loop order as NuHamil, un-normalised, 10 values per line), device-resident lab matrices, reusing the device assembly, formatter and threaded compression of Phase 4. | All 24 890 values of the reference file; then the dev sizes (emax 6-8); then imsrg++ reads the file (`file3e1max=emax`, `file3e3max=e3max` must be passed, wrong values misalign the stream silently). | Stage times at dev sizes, then a production-size estimate before any big job (both done; the estimate of 26 min was met: 20 min of work). |
| 5.5 — DONE 2026-10-07 (wrapped with gaps B, D and E left open, see the gap table below; what was done is in `docs/imsrg-check.md`) | imsrg++ validation of the production file on 4He and 16O. Done: NN for 4He (emax 4, 8, 12; Boost decoding of emax 14 streams, `docs/imsrg-check.md`), 4He with our 4/8/4 3NF400 file (5e-6), our 6/12/6 file equal to NuHamil's over all 456 320 values (4.1e-6). **Step 1 as planned (16O, emax 4, e3max 4, NuHamil against ours) does not work**: HF does not converge for 16O with any 3N file whose N3max is small, NuHamil's own included (table in `docs/imsrg-check.md`); 16O with NN only works. Steps now: **(a)** DONE: NuHamil's 6/12/12 file (N3max 12, 2 h 08 min for its 3N step on this CPU, shared machine) equals ours, generated locally on the CPU and not the DeltaAI file, over all 144 345 500 values, maximum difference 5.65e-6, none above 1e-5 (checks N up to 12 and all channels); **(b)** imsrg++ on a prefix of the production file (N3max 24-40; the file is also in `out/` locally, untracked): model emax 4 needs the first 13.1 million values, about 70 MB of gzip, emax 6 about 3.4 GB, emax 8 about 17 GB, so the first runs can be local. DONE for emax 4: 16O, e3max 12, E(IMSRG(2)) = -119.824798 MeV, HF converged, exactly 13 131 655 values read (`docs/imsrg-check.md`); emax 6: HF converges (EHF -94.80 MeV), and the flow settles with the `white` generator (smax 30): -126.741 MeV (the default imaginary-time generator chatters at that size, cause and settings in `docs/imsrg-check.md`); **(c)** the emax convergence of 16O against the pattern in NuHamil's README (steps of 7.9, 2.8, 0.75, 0.15 MeV from emax 4 to 12), qualitatively only: emax 4 to 6 gives 6.8 MeV with `white` (7.9 in the README for another 3N force); emax 8 (a 17 GB prefix, about 14 GB of imsrg++ memory) is the next point and is borderline on the local machine; **(d)** where to run imsrg++ for emax above 8 (local machine 19 GB; imsrg++ is built here in podman on x86, not on DeltaAI whose nodes are aarch64). | (a): our file equals NuHamil's to about 1e-5. (b), (c): no NuHamil reference of 3NF400 exists at that size; the README table is for another 3N force (`N3LOEMlnl`) and is context only. Sanity: HF converges, e3hf of a few to tens of MeV, E(IMSRG(2)) of the order of the README's -115 MeV at emax 4. | none |

**5.2 plan (agreed with the user, 2026-10-03).** Each term factorises as `V(alpha, alpha') = sum_a C_a(alpha, alpha') B_a[(n12 n3), (n45 n6)]`: `g_a(r)` are universal radial
tables (depend on Lambda, regulator power, m_pi only), `B_a` are their HO projections (depend on hw and the four orbital l only; batched GEMMs), `C_a` are Racah/CG coefficients
(hw and LEC independent). Store the five LEC-free terms and combine with the LECs afterwards (LEC and Lambda scans reuse everything, an hw scan redoes `B_a`); compute the tables
once, not per block as NuHamil does. Decisions: (1) the angular algebra in `src/` comes from the published Navratil 2007 paper (Few-Body Syst. 41, 117, arXiv:0707.4680), also checked
against the NuHamil paper (arXiv:2302.07962); NuHamil's Fortran-derived formulas (spec 5b) go in `refs/` as a slow cross-check; (2) radial meshes: converged by default, NuHamil's hard-coded
meshes (80 points to 15 fm, 100 p-points to 8 fm^-1) available as an option to verify agreement at 1e-8; (3) independent brute-force Cartesian check for cE, cD and c1 only (c3, c4 by the oracle files).

| Sub-step | Work | Gate |
|---|---|---|
| 5.2.0 — DONE (`refs/work/nnn3f`, `docs/study-local-3nf.md`; the AS-matrix gate is deferred to 5.2.4) | Oracle data: NuHamil with `save_3nf_before_lec` for 3NF400 at N3max 6 and 8, and the 500 MeV set (cD 0.83, cE -0.052) at N3max 6; commit the N3max 6 set (about 10 MB). | Files parse; NuHamil's `3 V LEC` and cfp reproduce its stored AS matrix. |
| 5.2.1 — DONE (`refs/nnn_cartesian.py`: factor 3 confirmed, NAS gauge `(-1)^l12`) | Brute-force Cartesian check (`refs/`) of cE, cD, c1 at the lowest l: settles factor 3 vs 6, signs, `(-1)^x`, NuHamil's constants. | cE and cD agree for the lowest J=1/2 states; c1 in one channel. |
| 5.2.2 — DONE (`nuham.kernels.nnn.radial`, `nuham.nnn.local_force.radial`, `tests/test_nnn_radial.py`, `docs/study-local-3nf.md`; the cE Lambda -> infinity check moves to 5.2.4 with the projection) | Radial layer in JAX: `z0, zx, fk, fkx, f1, f2` (f1/f2 as a 1D table in r). | Lambda -> infinity cE limit (spec 5b 7.2), Yukawa closed forms, mesh-convergence study. |
| 5.2.3 — DONE (`nuham.nnn.local_force.angular`, `tests/test_local3n_angular.py`, `docs/study-local-3nf.md`; all 24 blocks and all five terms match NuHamil to 5e-13 at N3max 6 and 8; scalar loops, vectorisation is 5.2.4/5.2.5) | Angular layer `C_a` per term (host NumPy on `racah_array`), from Nav07; NuHamil's formulas in `refs/`. | Each term equals NuHamil's NAS lower triangle to 1e-8 relative, N3max 6 then 8. |
| 5.2.4 — DONE for correctness (`nuham.nnn.local_force.assembly`, `kernels/nnn/projection.py`, `tests/test_local3n_assembly.py`; AS matrix equals NuHamil's stored genuine V3 to 2e-12 at N3max 6 with its cfp, spectrum equal with ours; coefficients are still the scalar 5.2.3 loops, grouping and batching are 5.2.5) | Projection and assembly: batched `B_a` GEMMs, blocks by (N, N'), then `C V C^T` with the 5.1 cfp. | `C^T V_AS C` equals NuHamil's `cfp^T work cfp` (factor 3 included). |
| 5.2.5 — DONE on CPU (`docs/study-local-3nf.md`, `benchmarks/bench_3n_dense.py force`, `refs/nnn_xi_mesh_study.py`): J=1/2+ at N3max 40 assembles in 30 s (12-15x faster than NuHamil at N3max 8-24), agrees with NuHamil to 1.5e-12 at N3max 16; NuHamil's mesh is converged to 1e-14 up to N3max 24 and within 1e-9 MeV at 40, `xi_mesh_size` gives 1e-14 everywhere; GH200 (job 3308734, 8 min): J=5/2+ at N3max 40 assembles in 119 s (+90 s basis, 11.7 GB host, 0.44 GB device), J=1/2+ in 13 s (later 6.6 s for J=5/2+ in the whole chain, `docs/study-me3j.md`) | Scaling to N = 16, 24, 40 (production ramp is 40 for J <= 5/2); time and memory on the GH200 against NuHamil. | Mesh-convergence verdict at large N (NuHamil's 15 fm edge sits at the N = 40 turning point). |

5.2.0, 5.2.1 and 5.2.2 are independent and can run in parallel; 5.2.3 needs 5.2.0; 5.2.4 needs 5.2.3 and 5.1.

**Production run (14/28/16, ramp 40-5-36-7-32-9-28-11-24, 3NF400, hw 16, lambda 2.0, genuine force at all relative J) — DONE 2026-10-06.**
Result: job 3324394 on one GH200 (32 CPUs, 120 GB): 30 min wall (about 20 min of work, then 10 min of `gzip -t` in the script), 6 367 911 165 values, 31.5 GB of gzip, peak host memory 77.1 GB; 24 of the 72 Jacobi
channels were read from the V3 cache of the verification jobs. Estimate before the run: 26 min (20-40). Stage times: `docs/study-me3j.md`, `benchmarks/gh200_stages_production_e14.jsonl`. The file is
`/projects/bhhk/soham/nuham/out/N3LO_3NF400_hw16_e14_e2_28_e3_16.me3j.gz` on DeltaAI, with no copy elsewhere and no checksum (by decision). Only the value count, `gzip -t` and the format at the start were checked.

The four host-side items that were done first (each tested on the CPU, then checked on the GH200 in jobs 3317204, 3317317, 3317423 and 3317876; details in `docs/study-me3j.md`):
1. Device-resident antisymmetrisation, NAS scatter and embeddings (`kernels/nnn/antisym.py`, `scatter_pair_blocks`): scatter 11 to 1.1 s, three embeddings 31 to 2.9 s, `cfp W cfp^T` 46 to 1.5-3.9 s. It is two dense GEMMs
   with a dense block-diagonal `C`, because a loop over the blocks recompiled for every channel. The force, embeddings and flow share one device chain with a single host copy of V3.
2. Fewer compiled kernels in the lab contraction: not the lever. The coarser padding ladder (92 to 53 kernels on the CPU, 42 on the GH200) did not shorten the cold lab stage. The cost of the first call was XLA's
   GEMM autotuning (a warm compile cache does not avoid it), which `kernels/compile.py` now switches off per kernel for the antisymmetrisation and the lab products: cold lab products of the largest channel 45.9 to 14.5 s, the
   flow keeps its tuned products (a global flag made it 2x slower). The ladder change stays in; reverting it is an open, low-value choice.
3. Packed lower triangle of the lab matrices (`nnn/packed.py`): the largest matrix is 2.08 GB instead of 4.16 GB, and the whole-run host peak was 77.1 GB (97.7 GB before).
4. Device QR for the Jacobi basis: measured and not done (1.4x faster than the host QR; the pivots of real projectors tie in norm, so the gauge depends on the platform with either QR).

Cache safety added on the way: V3 files are keyed on the exact basis (`basis_digest`), Jacobi files carry `BASIS_VERSION`, `CACHE_VERSION` is 2 with the NN flow space and relative J cut in the key, and
`tests/test_v3_reference.py` fails when a small V3 moves (then bump `CACHE_VERSION` and regenerate its reference). Build the Jacobi and V3 caches on one machine: the gauge is platform-dependent.

Jobs (`scripts/deltaai/`: `common.sh`, `verify_a..d.sbatch`, `production.sbatch`): 32 CPUs, `--mem=120g` (billing counts requested memory: 180g cost 1623 against 1082 per hour, and Slurm capped the allocation at
113 GB anyway), `OMP_NUM_THREADS=1`, 3 h limit, a check that aborts the job if the worker count differs from the allocated CPUs, V3 cache resumes a run that stops. 72 CPUs and `OMP_NUM_THREADS=8` were not tried.
Ask before every job. Decision (2026-10-05, the user): the production run uses the genuine force at all relative J (NuHamil's default). The cut stays an option to explore later: `--j3max-3nf 9` (above doubled J = 9,
i.e. J > 9/2, only the induced force is kept) changes values by up to 0.036 MeV at 8/16/8 (rms 4.4e-4 MeV) and removes most of the force time.
Not measured: the compile counts of a whole run, and the 72-CPU and BLAS-thread variants. Decision (2026-10-06, the user): no checksum or second copy of the file for now (regenerating takes about 15 min of GPU with the caches, 27 without, plus the queue wait). Next: 5.5 (4He and 16O, NN+3N;
imsrg++ needs about 25-30 GB for emax 14, so a smaller model space or the cluster; a smaller-emax file with the same e2max and e3max is a prefix of the production file, so a prefix suffices for small model spaces).

**Validation coverage and gaps of the production file (2026-10-06).** Checked against NuHamil, all with small files generated on the local CPU (not the production file): 4/8/4 (24 890 values, 1.2e-6), 6/12/6 (456 320 values,
N <= 6, 4.1e-6) and 6/12/12 (144 345 500 values, N <= 12, all 56 channels, 5.65e-6); imsrg++ on 4He with our 4/8/4 file (5e-6). Checked on the production file itself (made on the GH200): value count, `gzip -t`, exactly
the expected number of values read by imsrg++ at emax 4 (13 131 655) and emax 6 (693 209 575), HF converges and 16O at emax 4 gives -119.8248 MeV (plausible, no reference). Gaps, decided with the user (2026-10-06):
we will not close all of them now and will target the cheaper ones first (A and C, both done). F to J were added on 2026-10-07 from a check of what the tests and docs compare with NuHamil (they are other comparisons with NuHamil, not gaps of the production file as such); F, G and H are the cheap candidates, I and J are low priority. F, G and H were run on 2026-10-07 and agree (below); I and J have not been run.

| | Gap | Why it matters | How to close | Cost | Status |
|---|---|---|---|---|---|
| A | GH200 path against the CPU path: the NuHamil agreement was only shown on the CPU | the production run used the GPU backend, the no-autotune kernels and device-resident matrices | generate the same 6/12/12 (`flat12`) case on the GH200 and on the node's CPU, compare | one short job, 7 min 10 s, about 0.12 GPU-hours | DONE 2026-10-07 (job 3327601, and the direct comparison with NuHamil): GPU and CPU files, 144 345 500 values each, differ by at most 1.0e-8 (last printed digit), none above 1e-5; the GPU file, fetched, equals NuHamil's 6/12/12 file to 5.65e-6 (none above 1e-5) and the local x86 CPU file to 1.2e-7 (`docs/imsrg-check.md`) |
| B | Production settings have no NuHamil reference: Jacobi flow space N3max 24-40, N from 13 to 16, e3max 16 | exactly what the production file uses | NuHamil cannot run it here (its README: about 2 days with 32 threads, about 200 GB); only self-consistency and convergence checks (for example values from different ramps) | no NuHamil reference possible | not now |
| C | imsrg++'s IMSRG(2) flow does not settle at emax 6 with default step control (also for NN only) | no 16O energy beyond emax 4 can be quoted | tried step control and generators on 16O at emax 6 | about 10 min per try on the local CPU | DONE 2026-10-07: the imaginary-time generator chatters (`domega` caps the step, sign function); `white` with smax 30 or imaginary-time with `domega` 0.02 settle; 16O NN+3N: -119.946 MeV (emax 4) and -126.741 MeV (emax 6) with `white`, shift of the 3N force +30.9 and +33.6 MeV (`docs/imsrg-check.md`) |
| D | No independent reference for 16O NN+3N: NuHamil's README table is for another 3N force (`N3LOEMlnl`) | no external number to match | plausibility and convergence checks only, unless the LNL regulator is built | LNL is a large task | planned, D0 done (below); the regulator is a new step before the flow, not a new pipeline |
| E | Most of the file has not been read by imsrg++ (only the first 13 and 693 million of 6.4 billion values) | the rest is checked only by the count and the gzip stream | imsrg++ at larger emax on a machine with enough RAM (a cluster; imsrg++ is not built on DeltaAI) | needs an imsrg++ build on a large-memory machine | not now |
| F | A non-flat ramp has never been compared with NuHamil: every file comparison used a flat ramp; `test_ramp_and_channel_order` only checks that our parser gives the numbers we expect (for example `ramp4-3-6-7-8` gives 4, 4, 6, 6, 8), an expectation read from NuHamil's source | the production file uses a J-dependent flow space (`ramp40-5-36-7-32-9-28-11-24`); another reading of the boundaries would give wrong values in the channels in between | one NuHamil 4/8/4 run with a small ramp (N3max at least e3max) against our file; first check how NuHamil's ramp syntax maps to `ramp_nmax` | about 6 min of NuHamil (its 4/8/4 runs took that long) plus ours; local | DONE 2026-10-07: NuHamil `ramp8-5-6-9-4` (N3max 8, e3max 4, J = 1/2..5/2 at N 8, 7/2 and 9/2 at 6, 11/2 at 4) against ours, 24 890 values, max diff 1.23e-6, none above 1e-5; NuHamil's ramp parser (`GetRampNmax`, `j > j_read`, doubled J) read and equal to `ramp_nmax`; the ramp matters (the NuHamil flat8 file differs from the ramp one by up to 0.60) (`docs/imsrg-check.md`) |
| G | Other hw and lambda in the 3N chain: all twelve 3N references are at hw 16 and the SRG ones at lambda 2.0 (NN files were compared at hw 16 and 20 and lambda 1.8 and 2.0, 3N never) | oscillator-length scaling of the force projection, the flow in units of hw and the lambda handling; not used by the fixed production file, but by anyone who varies the parameters | NuHamil 4/8/4 runs at hw 20 and at lambda 1.8, each against our file | about 6 min of NuHamil each; local | DONE 2026-10-07: hw 20 (max diff 3.34e-6) and lambda 1.8 (2.03e-6), 4/8/4 each, 24 890 values, none above 1e-5 (`docs/imsrg-check.md`) |
| H | 3NF500 (cutoff 500 MeV, cD 0.83, cE -0.052) end to end: NuHamil's reference `refs/work/nnn3f/n500_N6` is used only by `tests/test_local3n_angular.py`, at the angular-layer level; no final-file comparison | the cutoff and LEC options of the CLI | generate ours with the matching settings and compare with `n500_N6/nnn.me3j.gz`; first check that the two cases match (N3max 6, e3max 4) | free: the NuHamil file exists; about a minute for ours | DONE 2026-10-07: 4/8/4 case of `n500_N6` (N3max 6, `flat6`), cutoff 500, cD 0.83, cE -0.052, 24 890 values, max diff 1.55e-6, none above 1e-5 (`docs/imsrg-check.md`) |
| I | NN me2j element by element at emax 14: the comparison stops at emax 12 (all 50 715 864 values agree), emax 14 has only the stream decode | small, since emax 8 and 12 agree | NuHamil at emax 14, e2max 28 against our file | NuHamil's README quotes about 15 min with 16 threads | open, low priority |
| J | `--j3max-3nf` against NuHamil's `J3max_initial_3nf`: our cut was measured against the full force (0.036 MeV at 8/16/8), never against NuHamil's option | not used by the production file (all relative J) | one NuHamil run with the option set | about 6 min of NuHamil | open, low priority |
| K | Frequency conversion (`hw_target`): the README example of NuHamil (`exe/NuHamil_3BME.py`, which the README says made its benchmark table) builds the 3N Jacobi Hamiltonian at `hw = 30` with the same ramp as ours and converts it to `hw_target = 16`; our production file is made at hw 16 directly | a truncated operator converted in frequency is not the operator computed directly at the target frequency; this is the best candidate for the 16O offset to the README (+0.26, +0.36, +0.42, +0.45 MeV at emax 4, 6, 8, 10; 4He agrees to a few keV; none of the imsrg++ settings tried closes it, `docs/imsrg-check.md`, settings scan) | plan below | K1-K3 about a day of work, K4 about 20 min of NuHamil, K5-K6 about 3 GPU-hours | K0-K3 in progress (2026-10-09) |

**Gap K plan: frequency conversion (2026-10-09, approved by the user).** What NuHamil does (sources in `docs/spec-5e-hw-conversion.md`): after the SRG flow, `V3 = H(s) - T - V_NN(s)` of each Jacobi channel is replaced by `U^T V3 U` in the antisymmetrised ramp space `N <= Nmax(J)`, with `U` the antisymmetrised product of the radial HO overlaps `<n(hw_target)|n'(hw)>` of the two Jacobi coordinates (same `l`, same `n` range), and then cut to `N <= e3max`. The lab transform and the file format do not change (the me3j file has no `hw`; imsrg++ takes it from its input). The README example uses `hw = 30`, `hw_target = 16`, `emax 14`, `e2max 28`, `e3max 16`; the NN file of the README is made at hw 16 directly.

| Step | Work | Gate |
|---|---|---|
| K0 | Spec: what is converted, where it sits in the chain, the truncation, the quadrature (`docs/spec-5e-hw-conversion.md`) | sources and checks written down |
| K1 | Kernel and host driver: radial overlap table by quadrature (`kernels/nnn/frequency.py`, reusing the HO functions of `kernels/ho.py`), block-sparse assembly shared with the regulator (`nnn/regulator.py`), antisymmetrisation with the existing `antisymmetrize` | overlap tests against closed forms |
| K2 | `hw_target` in `three_body_operator`, `jacobi_operators`, `generate_me3j` and the CLI; the cache key changes only when `hw_target` is set, so existing keys and files are unchanged | existing tests unchanged |
| K3 | Tests: identity at `hw_target = hw`, closed-form overlaps, orthogonality of the low block, round trip, cache keys | all pass |
| K4 | NuHamil at 4/8/4 and 6/12/6 with `hw 30`, `hw_target 16` (and one with `hw_target 20`) against our file | max difference below 1e-5 |
| K5 | Emax 6 LNL file at `hw 30 -> 16` (cold V3 cache at hw 30: about 22 min of GPU for the Jacobi side plus the lab side), imsrg++ for 16O at emax 4 and 6 (`white`, smax 30; HF and oscillator bases) and 4He, against the README | offsets clearly below +0.26 and +0.36 MeV; 4He still within a few keV |
| K6 | Emax 8 and 10 the same way (about 2 GPU-hours), then record in `docs/imsrg-check.md`; decide with the user whether the production file is regenerated with the conversion | the drift with emax is gone or explained |

Risks: the README table may not come from this script's `main()` (the git history has one import commit for both); the generator and basis spread of imsrg++ (about 0.2 MeV at emax 4) limits what K5 can resolve; at hw 30 the quadrature of NuHamil (500 points on [0, 25] fm) may be coarser than ours, which K4 measures.

**Gap D plan: the `LNL` regulator (2026-10-07).** `LNL` is the local force at Lambda_local 650 plus a non-local Gaussian regulator on the Jacobi momenta, applied as `V_A <- Fa^T V_A Fa` before the SRG flow. Preset `N3LOEMlnl`: EM500, n = 2, LECs -0.81, -3.2, 5.4, 0.7, -0.06, Lambda_local 650, Lambda_nonlocal 500. Spec: `docs/spec-5b2-lnl-regulator.md`.

| Step | Work | Gate |
|---|---|---|
| D0 | DONE 2026-10-07: spec from NuHamil's source (`NNNForceHOIsospin.F90:296-413`), mesh, phase and scale settled, existing `momentum_radial` and `antisymmetrize` identified for reuse | `docs/spec-5b2-lnl-regulator.md` |
| D1 | DONE 2026-10-07: kernel `kernels/nnn/regulator.py` (`regulator_table`, per `(l12, l3)`) and host driver `nnn/regulator.py` (`regulator_matrix` on a channel's NAS states, dense on the device); `tests/test_nnn_regulator.py` (11 tests): orthonormal momentum functions, R -> 1, n = 1 factorises (independent mesh), R symmetric with eigenvalues in (0, 1), R commutes with the antisymmetriser `C^T C` (three channels at N 8), `Fa` eigenvalues in (0, 1], the `(-1)^n` phase equals the Fourier transform of the position-space functions; mesh study (`refs/regulator_mesh_study.py`): NuHamil's 500-point mesh agrees with 250, 1000 and 1500 points to twice pmax to 5e-13 for N <= 40, hw 16 and 20 | all checks pass |
| D2 | DONE 2026-10-07: `LocalForce.nonlocal_cutoff` (left out of `repr`, added to the cache key only when set, so local runs keep their keys); the regulator is applied in `jacobi_operators` after `local_3nf_matrix` as `Fa V Fa` (`Fa = antisymmetrize(R)`, stage `regulator`); `R` is built as a sparse host matrix because a dense one with one device scatter per block took 14 s at N = 24 (n_nas 7644), sparse 0.15 s plus 1.8 s of antisymmetrisation; CLI `--preset {3NF400,3NF500,N3LOEMlnl}`, `--cutoff-nl`, `--cutoff`, `--power` and the LECs override the preset; `tests/test_lnl_force.py` (infinite cutoff = local force, finite cutoff changes it, repr and keys unchanged, presets); full suite 757 passed before the sparse change, regulator and LNL tests rerun after it | local results bit-identical, cache keys unchanged |
| D3 | DONE 2026-10-07: NuHamil `LNL` (preset `N3LOEMlnl`: c1, c3, c4, cD, cE = -0.81, -3.2, 5.4, 0.7, -0.06, Lambda_local 650, Lambda_nonlocal 500, n = 2, hw 16, lambda 2.0) against ours: 4/8/4 (`flat8`, 24 890 values) max difference 1.98e-6; 6/12/6 (`flat6`, 456 320 values, N <= 6) 4.13e-6 (the local 3NF400 force gave 4.1e-6 at the same size); none above 1e-5 (`docs/imsrg-check.md`); 6/12/12 not needed (no N-dependent growth between N 8 and 6 cases) | max difference below 1e-5 |
| D4 | DONE 2026-10-08 (16O emax 8 is at the loose end of 'a few tenths') (`docs/imsrg-check.md`, section Gap D4): LNL files for emax 6 (job 3330416, 11 min) and emax 8 (job 3330462, 9 min, 3.4e9 values, 16.9 GB); imsrg++ also built for aarch64 in Apptainer on DeltaAI (81 s, `scripts/deltaai/imsrg.def`, `imsrg_build.sbatch`, `imsrg_run.sbatch`); 4He emax 4 and 6: -27.2247 and -28.2185 MeV against the README's -27.221 and -28.221 (3.7 and 2.5 keV), x86 and aarch64 identical in all digits; 16O emax 4 and 6 (white, smax 30): -115.283 and -123.090 against -115.540 and -123.445 (0.26 and 0.36 MeV above; the generator moves emax 4 by 0.18 MeV, the README's imsrg++ settings are not recorded anywhere); step emax 4 to 6 7.81 MeV (README 7.90). emax 8 (DONE 2026-10-08, jobs 3335164 and 3335165, white, smax 30, flows converged): 4He -28.5764 against -28.569 (7 keV), 16O -125.8512 against -126.270 (+0.419 MeV; offsets 0.257, 0.355, 0.419 at emax 4, 6, 8, a smooth drift attributed to unrecorded README settings and not tested); step emax 6 to 8 2.761 MeV (README 2.825). Open: the 16O emax 6 container cross-check (3330502). Approved by the user for emax up to 8 only. Extension 2026-10-09 (one run approved): 16O emax 10 -126.569920 MeV (step from emax 8 -0.719, README step about 0.75), 33 min 37 s, 45 GB; the serial 3N text parse is 73-83% of the imsrg++ time (`docs/imsrg-check.md`, section D4 extension), forecast emax 12 about 50 min, emax 14 about 1.7 h | 4He within about 0.05 MeV: met; 16O within a few tenths: met (0.26, 0.36); emax 4 to 6 step of the right size: met |
| D5 | Record in `docs/imsrg-check.md`, close D here and in `CLAUDE.md`; decide on an LNL production file (one GH200 job) | |

Risks: the truncated sandwich and NuHamil's fixed 500-point mesh may differ from a converged one at high N (D1 and D3 measure it); N3max above 12 has no NuHamil reference (as gap B); the README numbers may come from other imsrg++ settings, so D4 is a plausibility test.

Order: 5.1 and 5.1b first and in parallel (5.1b decides the riskiest design question early), then 5.2 and 5.3 (independent of each other, both need 5.1), then 5.4, then 5.5. 5.0's numpy oracle is built incrementally as each step needs
it, in `refs/` (provenance rule: nothing transcribed from NuHamil in `src/`). Open items from the specs that are closed: the sign phases `(-1)^x` and factor 3 for cD (5.2.1), the lab factor 6 and the T_NAS phases (5.1b), the antisymmetrizer threshold at Nmax 40 (5.1: replaced by an
idempotence assert), the number of flow steps at 3N size (5.3: 15-16 Dopri8 steps). Still open: non-kinetic generators (not analysed); the `hw_target` conversion is gap K.

Why this order: fix the known confounders (4a) before taking baselines (4b); measure writer gains with the same harness (4c); a scan
multiplies output, so the writer goes before scans (4d); benchmark at every feature step so the GPU question is answered where it matters.
DeltaAI budget: account `bhhk-dtai-gh` only; 113 h of 618 left on 2026-10-06 (the account's fair-share factor was 0.14, jobs waited 2-3 h in the queue). Phase 4 used about 2 GPU-hours; the verification and production jobs of 5-6 October
about 1.2 (billing 1.08 per hour of use); ask before every job.

## Risks

- Numerical stability of Moshinsky brackets/recursion at high emax; SRG stiffness.
- Padding vs recompilation trade-off for irregular channel sizes.
- `me2j` format details (ordering, truncation, units) must match imsrg++ exactly.
- NuHamil may have quirks. Where independent checks disagree with it, flag it.
