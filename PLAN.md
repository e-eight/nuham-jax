# nuhamil-alt: plan

A from-scratch JAX package that produces the same physics output as NuHamil
(https://github.com/Takayuki-Miyagi/NuHamil-public), for use with imsrg++.
It is not a translation of the Fortran code. NuHamil is used as a numerical oracle.

## Scope

- Now: EM500 NN, Hamiltonian only, `me2j` output readable by imsrg++.
- Develop on this machine: CPU-only JAX in float64, small cases (deuteron, emax <= 4-6).
- Larger runs later on a cluster (Delta, GPU). Nothing may assume a device.
- Later: local N2LO 3N force, Jacobi SRG, Jacobi -> lab `me3j` with a streaming writer.
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
      nnn/          local-force radial transforms and HO projection
    nn/           NN pipeline: input_file (.bin reader), relative (SRG driver), lab_transform,
                  me2j (writer), generate (+CLI), scan (+CLI)
    nnn/          3N pipeline: jacobi (basis, antisymmetrizer), tcoef (Jacobi -> lab T-coefficients),
                  local_force/{radial, angular, assembly}; later SRG and me3j
  tests/          NuHamil references + independent identity checks (flat; helpers in the same folder)
  benchmarks/     stage-timed runners and recorded baselines
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

### Phase 4: GPU readiness and speed — in progress
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
| 4b. Harness — DONE on CPU (docs/benchmarks.md); ICRN and DeltaAI baselines still to record | Stage timers built into `generate_me2j`. A benchmark runner **outside the package** (`benchmarks/`) writing JSON with device, versions, stage times, peak memory. A compute-only mode that skips file writing. | Timers do not change results. | Baselines on local CPU, ICRN, DeltaAI; NuHamil's own NN time and memory on this machine as a reference. |
| 4c. Writer — DONE: vectorized host formatter (5.6x, byte-identical), optional `isal` compressor, device assembly and device formatter (JAX), threaded igzip; emax 14 single run 54 s -> 8.1 s on the GH200 (docs/benchmarks.md); NumPy CPU contraction tried and dropped | Parallel text formatting (gzip stays in its thread); NumPy contraction path for CPU runs. Binary/HDF5 output is deferred. | me2j byte-identical; imsrg++-readable. | Share of run time spent writing (now ~45%); compute-only vs full-run times. |
| 4d. Parameter scans — CPU part DONE (docs/benchmarks.md); GPU measured on DeltaAI (job 3300961): 38% less time per point at batch 6; GPU prefers flow_sets > 1, CPU does not; emax 12-14 measured (job 3301003): compute-only 2.6-2.8x faster per point, but a full emax 14 point is writer-bound (gzip/assemble) and host memory is ~6 GB per extra point (matrices 3.9 GB each); isal (now an optional extra) cuts the write stage 3x on the GH200; batched assemble is bit-identical and 1.5-2x faster locally but gave no gain on the GH200, assemble is now the largest stage of a full scan point (see docs/benchmarks.md); a full emax 14 scan point is now 4.6-4.8 s on the GH200 (batch 6-12); ICRN pending | Many (hw, lambda) sets sharing brackets and coefficients; batched SRG flows and batched contraction. | Each set matches a single run. | Time per set vs batch size, CPU vs GPU: the clearest test of GPU advantage at NN size. |

### Phase 5: 3N — 5.0 specs done, same gates per step
Decided with the user (2026-10-03): start with the **local** N2LO 3N force (NuHamil `Regulator="Local"`: sets `3NF400` and `3NF500-BetaDecay-Local`; the `NonLocal` sets need external
HDF5 matrix elements we cannot regenerate; `LNL` adds a non-local regulator and comes after). Output: validate with single-precision `.me3j.gz` (the only format imsrg++'s `full` reader takes; NuHamil's
default), production later. Sizes: small cases for development (emax/e3max 4-8), emax 14, e2max 28, e3max 16 for production (6.37e9 values, about 103 GB of text, 27-34 GB gzipped, all 70 lab channels
needed at once: about 46 GiB single precision, which fits in GPU memory). Specs: `docs/spec-5a-jacobi-basis.md`, `5b-local-n2lo-3nf`, `5c-3n-hamiltonian-srg`, `5d-lab-transform-and-me3j`.

Oracle strategy: the antisymmetric Jacobi basis is an arbitrary gauge per N block, so compare gauge-invariant quantities (the antisymmetrizer A = C^T C, eigenvalues before and after the SRG that NuHamil prints per channel,
traces and norms) and, at the end, the lab `me3j` values. Reference run: `3NF400`, emax 4, e3max 4, N3max 8, hw 16, lambda 2.0 (386 s, 1.1 GB, 24 890 values).

| Step | What gets built | Correctness gate | Benchmark gate |
|---|---|---|---|
| 5.0 | Specs (done), small NuHamil references (first one done), numpy oracle of the whole chain (not started). | Oracle reproduces the references. | NuHamil 3N time and memory as the baseline (386 s, 1.1 GB for the reference case). |
| 5.1 — DONE (`nuham.nnn.jacobi`, `tests/test_jacobi3.py`): all 24 channel counts and `C^T C` equal NuHamil's; dense per-N solve is affordable: Nmax 40 channels 6.8 s (J=1/2+) to 530 s (J=17/2+, largest N block 10 370, 6.7 GB RSS) on shared host CPU, J=5/2+ AS total 11 340; the 1e-2 threshold is replaced by an idempotence assert (1e-9) | Jacobi basis: channel enumeration (T, J, P) with the ramp, NAS state lists in NuHamil's order, the three-body antisymmetrizer per (J, P, T, N) from 9j/6j and d = 1/3 Moshinsky brackets (existing `MoshinskyTable(d=...)`), eigen-decomposition into the AS basis. | State counts of all 24 channels (table in 5a), A idempotent with eigenvalues 0/1, A equal to the stored `cfp` projector to ~1e-14, trace assertion at large N. | Time and memory of the antisymmetrizer per channel, up to Nmax 40 (is a dense eigensolve per block affordable? the 1e-2 threshold is unchecked there). |
| 5.1b — DONE (`nuham.nnn.tcoef`, `docs/study-tcoef3.md`): d=2 bracket already in `MoshinskyTable`; all 24 890 reference values reproduced to 5e-9 with NuHamil's Jacobi V and cfp; factor 6 and phases confirmed against an m-scheme calculation; energy-blocked build (`tcoef_nas_blocked`): 13x faster than term-wise, whole channel 32 s at e3max 16, about 16 core-min for all 70 channels (scaled), so T_NAS stays on the host and the device does the contraction; recompute per channel, do not store T; a 6/12/6 reference would check N > 4 against NuHamil | (parallel, early) | **T-coefficient feasibility prototype**: the Jacobi-to-lab coefficients are the largest and riskiest piece (about 8e9 irregular coefficients at production, a d = 2 Moshinsky bracket the package does not have). First check the d = 2 bracket (`MoshinskyTable(d=2)` exists for the NN lab transform; verify it is the one the 3N chain needs), then a small T_AS against the reference file, then cost scaling. | d = 2 bracket orthogonality and equality with the oracle brackets; T_AS reproduces the reference lab elements at emax 4; the factor 6 and the phases confirmed numerically. | Cost and memory of building T per lab channel at e3max 8, 12, 16, host vs device: decides the 5.4 design before anything else is built on it. |
| 5.2 | Local N2LO force term by term (c1, c3, c4, cD, cE): coordinate-space kernels, partial-wave decomposition, HO projection as batched dense products; the redundant `precalculations` of NuHamil computed once. | Per-term pre-SRG matrix elements against NuHamil run with `save_3nf_before_lec=.true.`; cE analytic check (Lambda to infinity); mesh-convergence study (NuHamil's meshes are hard-coded and unchecked); settle the open phase and factor-3 questions. | Time per term vs NuHamil; GPU vs CPU. |
| 5.3 | Jacobi-space Hamiltonian (T + 3 V_NN + V_3N), SRG flow per channel with the ramp (reuse the NN flow kernel), NN SRGs computed once and shared, stored V3 = H(s) - T - 3 V_NN(evolved). | Eigenvalues before and after the flow per channel (NuHamil prints them), gauge-invariant traces (e.g. trace of V3 = 164.007348 for J=1/2+, T=1/2 in the NN-only reference run); NN-only V3 reference files. | Dense flow cost: largest production channel about 11 340 states (1.03 GB), about 6.6e13 flop per right-hand side over all channels; number of right-hand-side evaluations unknown, measure at small Nmax. |
| 5.4 | Jacobi-to-lab transformation with e3max truncation (two dense GEMMs per block) and the `me3j.gz` writer (same loop order as NuHamil, un-normalised, 10 values per line), device-resident lab matrices, reusing the device assembly, formatter and threaded compression of Phase 4. | All 24 890 values of the reference file; then the dev sizes (emax 6-8); then imsrg++ reads the file (`file3e1max=emax`, `file3e3max=e3max` must be passed, wrong values misalign the stream silently). | Stage times at dev sizes, then a production-size estimate before any big job. |
| 5.5 | imsrg++ validation on 4He and 16O: NN done for 4He (emax 4, 8 and 12, plus Boost decoding of emax 14 streams, docs/imsrg-check.md), 16O and converged larger emax still open; then NN+3N with the local force. | Energies match the NuHamil-based run. | none |

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
| 5.2.5 — DONE on CPU (`docs/study-local-3nf.md`, `benchmarks/bench_3n_dense.py force`, `refs/nnn_xi_mesh_study.py`): J=1/2+ at N3max 40 assembles in 30 s (12-15x faster than NuHamil at N3max 8-24), agrees with NuHamil to 1.5e-12 at N3max 16; NuHamil's mesh is converged to 1e-14 up to N3max 24 and within 1e-9 MeV at 40, `xi_mesh_size` gives 1e-14 everywhere; GH200 (job 3308734, 8 min): J=5/2+ at N3max 40 assembles in 119 s (+90 s basis, 11.7 GB host, 0.44 GB device), J=1/2+ in 13 s | Scaling to N = 16, 24, 40 (production ramp is 40 for J <= 5/2); time and memory on the GH200 against NuHamil. | Mesh-convergence verdict at large N (NuHamil's 15 fm edge sits at the N = 40 turning point). |

5.2.0, 5.2.1 and 5.2.2 are independent and can run in parallel; 5.2.3 needs 5.2.0; 5.2.4 needs 5.2.3 and 5.1.

**Optional, after 5.4 (not scheduled):** move the radial sums, the scatter and `cfp W cfp^T` of the local force to the device. On the GH200 they take 35 s, 11 s and 46 s of the
119 s assembly of J=5/2 at N3max 40, while only the projection (26 s) runs there; the dense NAS matrix (9.3 GB) would then live in the 98 GB of device memory. Block-wise
accumulation of the AS matrix would also cut host memory (11.7 GB peak). Details and numbers: `docs/study-local-3nf.md`, section 5.2.5.

Order: 5.1 and 5.1b first and in parallel (5.1b decides the riskiest design question early), then 5.2 and 5.3 (independent of each other, both need 5.1), then 5.4, then 5.5. 5.0's numpy oracle is built incrementally as each step needs
it, in `refs/` (provenance rule: nothing transcribed from NuHamil in `src/`). Known open items from the specs: sign phases `(-1)^x` and factor 3 vs six permutations for cD (5b), the lab factor 6 and the T_NAS phases (5d), the antisymmetrizer
threshold at Nmax 40 (5a), the number of flow steps at 3N size (5c), the `hw_target` conversion and non-kinetic generators (not analysed).

Why this order: fix the known confounders (4a) before taking baselines (4b); measure writer gains with the same harness (4c); a scan
multiplies output, so the writer goes before scans (4d); benchmark at every feature step so the GPU question is answered where it matters.
DeltaAI budget: account `bhhk-dtai-gh` only (about 115 h left); Phase 4 used about 2 GPU-hours; ask before 3N-scale jobs.

## Risks

- Numerical stability of Moshinsky brackets/recursion at high emax; SRG stiffness.
- Padding vs recompilation trade-off for irregular channel sizes.
- `me2j` format details (ordering, truncation, units) must match imsrg++ exactly.
- NuHamil may have quirks. Where independent checks disagree with it, flag it.
