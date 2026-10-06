# Study: the 3N lab transform and the me3j writer (Phase 5.4)

Chain (`nuham.nnn.generate`, CLI `uv run python -m nuham.nnn.generate --help`): isospin-averaged NN (once) -> per Jacobi channel: basis (cached), kinetic and NN embedding, local 3NF,
SRG flow, `V3 = H(s) - T - 3 V_NN(s)` cut to `N <= e3max` -> per lab channel: host T coefficients, device contraction `6 sum T^T V3 T` -> file order lookup -> `10f16.8` text, gzip.

## Correctness

| check | result |
|---|---|
| NuHamil's 4/8/4 file, 3NF400, all 24 890 values (`tests/test_me3j.py`) | max difference 2.2e-6 (float64 lab matrices), 1.2e-6 through the CLI with float32 storage |
| NuHamil's NN-only 4/8/4 file (made with `jmax3 = 5`) | 1.7e-6 |
| value counts | 24 890 (4/8/4), 456 320 (6/12/6), 5 199 790 (8/16/8): spec 5d |
| tuple order | equal to the nested loops of NuHamil tuple by tuple (`file_tuples`) |
| device contraction against the host `lab_scalar`, random operators, 5 channels | 1e-9 relative |
| imsrg++, 4He, 4/8/4, full flow | EHF 5.8490859 against 5.8490810, E(IMSRG(2)) 1.761308 against 1.761305 (`docs/imsrg-check.md`) |

imsrg++ needs `file3e2max` (see the same file). The lab contraction is exact for any number of workers (`test_parallel_values_equal_serial`).

## Measured on this CPU (16 cores)

Stage times of `generate_me3j` with the 3NF400 (ramp flat`e3max`):

| emax/e2max/e3max | values | NN | basis | force | flow | lab (T coefficients + contraction) | write |
|---|---|---|---|---|---|---|---|
| 4/8/4 | 24 890 | 6 s | 1 s | 76 s | 16 s | 8 s | 0.2 s |
| 6/12/6 | 456 320 | 11 s | 1 s | 25 s | 15 s | 7 s | 0.2 s |
| 8/16/8 | 5 199 790 | 10 s | 2 s | 90 s | 20 s | 52 s | 1.2 s |

(4/8/4 runs 24 channels to jmax3 = 11 with their own radial tables; 6 and 8 use the default `jmax3 = 2 e3max + 3`.) Lab channel J=9/2+ T=1/2 at e3max 12 (7 221 lab states):
T coefficients 8.7 s, contraction 5.7 s on the CPU, 2.3 GB (`benchmarks/baseline_cpu_me3j_lab_e12.json`). The writer (`benchmarks/bench_me3j.py write`, emax 8, e2max 16, e3max 12,
2.5e8 values): lookup and ordering 4.1 M values/s on one process (40.7 s), 12 M/s with 8 worker processes (20.8 s); with formatting and ISA-L compression the whole write is
5.1 M values/s (48 s) on one process and 8.8 M/s with 8 workers (28 s); 1.5 GB of gzip.

## Production size (14/28/16, 6.37e9 values), counted and extrapolated

- **Contraction.** 70 lab channels, 2.7e14 flop in all with the dense per-pair formulation (largest channel J=9/2+ T=1/2: 32 259 lab states, 3.2e13 flop, T coefficients 2.8 GB in float64,
  the dense `V_bd` of one block pair 2 GB). About 7 s at the 40 TFLOP/s measured for FP64 products on the GH200, so the contraction is not a cost; the lab matrices of all channels need 45.5 GiB in float32.
- **T coefficients** (host, hw independent): about 16 core-minutes for all channels (`docs/study-tcoef3.md`), plus the `cfp` products (2.5e14 flop on the host BLAS).
- **Writer.** 6.4e9 values at 8.8 M/s on 8 workers is 12 minutes on this machine; the formatter (22 M values/s) and the compressor (ISA-L 370 MB/s here, 103 GB of text) run in the main thread and
  a background thread. A device-side lookup and formatting is possible (the me2j device assembler is the template) but not needed before the GH200 measurement.
- **Jacobi basis, 3NF and flow** dominate: basis plus force of the largest channel (J=5/2, N = 40) are 90 s + 119 s on the GH200 (`docs/study-local-3nf.md`), the flow about 40 s
  (`docs/study-3n-flow.md`). Summed over the 72 channels of the production ramp I estimate 35-40 minutes for basis plus force and 3-5 minutes for the flows; the estimate is a scaling of the measured
  channels, not a measurement.
- **Memory.** Lab matrices 45.5 GiB (float32) on the host for the lookup, the Jacobi `V3` operators cut to e3max (a few hundred MB in all), and per channel up to about 15 GB on the device.

## The optional device-side move of the 3NF assembly

Move the radial sums (35 s), the scatter (11 s) and `cfp W cfp^T` (46 s) of J=5/2, N = 40 to the device (GH200 times, `docs/study-local-3nf.md`): the assembly of that channel would fall from 119 s to
about 30 s, and the dense NAS matrix (9.3 GB) would live on the device. Over the whole production ramp that saves roughly 15-20 minutes of the 35-40 minutes of basis and force. It is not the
first thing to do:

1. The **Jacobi basis** is as large as the assembly (90 s against 119 s for J=5/2) and depends only on `(J, P, T, N)`. `load_or_build_channel` now caches it (one `.npz` per channel,
   hw and lambda independent), which removes that half for every run after the first, in particular for hw and lambda scans.
2. A scan over hw redoes the radial projection (device) but not the angular coefficients; a scan over lambda or the LECs redoes only the flow and the combination (the five LEC-free terms can be
   stored, as planned in 5.2).
3. The device-side assembly adds device memory pressure (the dense NAS matrix) next to the flow and the lab matrices, and two more kernels to maintain.

Recommendation: measure the production chain on the GH200 as it is (basis cached, 8-16 writer workers), then decide on the device-side assembly from the measured stage shares.

## GH200 measurements (DeltaAI jobs 3309244 and 3309469, 2026-10-04; records `benchmarks/baseline_gh200_me3j_*.json`)

One GH200, 32 CPUs, 30 worker processes, `OMP_NUM_THREADS=1`.

**Lab transform of the largest channel** (J=9/2+ T=1/2, 14/28/16, 32 259 lab states, random Jacobi operators): T coefficients on the host 44 s, upload 0.5 s, device products and copies 4.2 s when every
kernel is compiled, but 46 s in a fresh process: compiling about 90 shape-specific kernels (0.4-0.5 s each) was 90% of the "contraction". The small channel J=1/2+ T=3/2 shows the same (7.8 s cold, 0.1 s warm).
The padding ladder (`lab_transform._rung`) did not cut it for a single channel (41 s before, 46 s after) because the block pairs of one channel have too many different sizes; the compiled shapes are shared between
channels of one process and, with the persistent compile cache (`--compile-cache`, on by default in the CLI), between runs. Peak device memory 14.6 GB, host 20 GB. The contraction itself is not a cost
(4.2 s for the largest channel against the 0.8 s flop estimate, the rest being copies and padding).

**Writer**, 6/28/16 (a prefix of the production file, 6.93e8 values, 4.1 GB of gzip):

| formatting | compression threads | workers | lookup | whole write | values/s |
|---|---|---|---|---|---|
| main process | 1 | 8 / 16 / 30 | 15.2 / 9.2 / 6.6 s | 43.7 / 43.6 / 44.1 s | 15.8 M |
| workers | 4 | 30 | 6.6 s | 14.5 s | 48 M |
| workers | 8 | 30 | 6.6 s | 9.8 s | 71 M |

With the formatting in the workers and 8 compression threads the production file (6.37e9 values) takes about 90 s.

**End to end**, 10/20/12 (2.73e8 values, 1.41 GB, ramp flat12, 3NF400, float32 lab matrices, 30 workers, Jacobi basis cached from the first run): 16 min 42 s in job 3309244, with the force on one process (449 s)
and the writer formatting in the main process; 5 min 13 s in job 3309469 with the force on 30 workers (72 s), the flows 18 s, the lab stage 186 s (host T coefficients dominate), the write 6.5 s, NN projection
16 s, peak host memory 8.9 GB. The two files have the same size, 1 413 800 052 bytes.

## Production estimate after the GH200 measurements (14/28/16, ramp 40-5-36-7-32-9-28-11-24)

| stage | estimate | basis |
|---|---|---|
| NN projection and flows of the 2-body channels | 25 s | measured |
| per channel: Jacobi basis, embeddings, force (30 workers), flow | up to 95 + 31 + 119 + 28 = 273 s (J=5/2); about 35-50 min over all 72 channels | measured for the largest, scaled |
| lab transform | T coefficients about 17 min serial (44 s for the largest channel; 70 channels), contraction a few minutes plus first-run compilation | measured for two channels, scaled |
| writer | 90 s | measured on a prefix |
| total | about 1 hour on one GH200 | |

Open points that would shorten it: the T coefficients are built one channel at a time on one core (the 32 cores are idle in that stage; channels are independent, so a pool of workers or multithreaded BLAS applies);
the Jacobi basis is cached after the first run; the first run compiles the lab kernels once (the persistent compile cache then holds them).

## Smaller ramps and the full-size lab stage (DeltaAI job 3310316, 2026-10-05, node `gh016`, commit `e1c7cb8`, 32 CPUs)

Per-channel stage logs and cProfile dumps of the main process: `benchmarks/gh200_stages_*.jsonl`, `benchmarks/gh200_cprofile_*.prof`. All runs: 14/28/16, hw 16, lambda 2.0, 3NF400, float32 lab matrices,
30 worker processes (every run printed `"workers": 32` first; the job script aborts otherwise).

| run | Jacobi basis | force | embed | flow | lab | write | whole run |
|---|---|---|---|---|---|---|---|
| ramp24-5-22-7-20-9-18-11-16, Jacobi only | 31 s | 521 s | 8 s | 169 s | - | - | about 12 min |
| ramp32-5-28-7-24-9-20-11-16, Jacobi only | 102 s | 757 s | 34 s | 118 s | - | - | about 17 min |
| flat16, whole file | 7 s (cached) | 387 s | 3 s | 84 s | 286 s | 77 s | about 14 min |

The flat16 run wrote the whole production-size file: 6 367 911 165 values (the count of spec 5d), 32.2 GB of gzip (`gzip -t` passed), at 82 M values/s; lab stage 286 s (T coefficients waited for 76 s,
upload 7 s, device products with compilation 201 s); peak host memory 97.7 GB of the 113 GB Slurm gave (the 45 GiB of float32 lab matrices plus in-flight coefficient blocks).

**Compiled shapes of the lab contraction (item 2 of the production-run list, CPU, e3max 12, 8 channels, cold in-process cache).** The ladder of `lab_transform._rung` (1.5x steps from 256) left 92 compiled
kernels (54 products, 38 operator builders), 5-7 new ones per channel. On the CPU a compile costs little, so the test is the count and the product time: doubling everywhere gave 49 kernels but doubled the product time
(29 to 59 s); doubling only below 4096 and for the index lengths gave 53 kernels for 14% more product time (29.5 to 33.5 s), which is what the code now does. At the GH200 rate (0.4-0.5 s per compile) that is
expected to save of the order of 1-2 minutes of the 162 s, not more; the persistent compile cache removes the rest on any repeat run. Not measured on the GH200 yet.

**Packed lab matrices (item 3).** The lab matrices are symmetric, so only the lower triangle is kept, packed by rows (`nuham.nnn.packed`; element (i, j), i >= j, at i(i+1)/2 + j). `contract_blocks` writes each
block straight into the packed array, so the full square never exists on the host, and the writer's lookup orders its two row indices (the rows are in N-major order, not in the order of the file's keys, so the
triangle is a property of the stored row order and not of the lookup). Check on this CPU: the emax 6, e2max 12, e3max 6 file (456 320 values, float32 matrices) is the same text as before the change
(equal md5 of the decompressed stream), and the J=9/2+ T=1/2 matrix at e3max 12 (7221 states) takes 0.104 GB instead of 0.209 GB. The production estimate of 45 GiB to 23 GiB of float32 and the effect on the
97.7 GB host peak are not measured on the GH200 yet.

**Device QR for the Jacobi basis (item 4): measured, not implemented.** (a) On the GH200 (`benchmarks/baseline_gh200_3n_qr.json`) the device pivoted QR matches SciPy's pivots on a random projector but is
only 1.4x faster (11.2 s against 15.8 s at n = 10 400, 1.07 against 1.61 s at 5000); the device `eigh` is 15x faster (1.0 s) but returns an arbitrary basis of the unit eigenspace. (b) Time split on this
CPU for J=5/2 at N = 24: building the antisymmetrizers 3.4 s, projector check 0.2 s, pivoted QR 1.3 s; the QR grows with n^3 and the build with n^2, so the QR dominates at the N = 40 blocks. (c) The
pivots are not robust: in real antisymmetrizer blocks (J = 1/2, 5/2, 9/2 at N = 10-22) the pivot columns tie in norm (relative gaps 0 to 1e-16 at 1 to 9 pivots per block), and a pivoted Cholesky
replica chose other pivots than LAPACK in two of five blocks. So the gauge is already a property of the platform, whichever QR is used, which is why cached operators are keyed on the exact `cfp`.
The basis is cached after the first run and the saving is of the order of 20-25 s for the largest channel, a few minutes over the whole production run at most, so the item was not done.

## Verification A on the GH200 (DeltaAI job 3317204, 2026-10-05, node `gh005`, commit `3a4e8a1`, 32 CPUs, 13 min 7 s, max RSS 22.6 GB)

Records: `benchmarks/gh200_stages_verify_a_j5.jsonl`, `benchmarks/baseline_gh200_me3j_lab_j9_v3.json`. 63 tests passed on the GPU backend (3 min 56 s) and the V3 reference test passed, elements included
(the aarch64 LAPACK gave the same gauge as this machine for those two small channels; nothing is shown for the large blocks).

**Jacobi part of 2J = 5 at N = 40** (production ramp, cold compile cache, Jacobi caches rebuilt because the old files had no version), seconds per channel:

| channel | AS / NAS | basis | force | of it: projection / scatter / antisymmetrize | embed | flow |
|---|---|---|---|---|---|---|
| 5/2+ T=1/2 | 11 340 / 34 020 | 77.6 | 29.0 | 2.8 / 1.1 / 20.9 | 3.1 | 32.7 |
| 5/2- T=1/2 | 10 527 / 31 580 | 60.4 | 21.2 | 2.2 / 1.0 / 16.8 | 2.7 | 30.2 |
| 5/2+ T=3/2 | 5 670 / 17 010 | 9.3 | 11.5 | 0.5 / 0.6 / 9.7 | 1.4 | 6.6 |
| 5/2- T=3/2 | 5 263 / 15 789 | 8.1 | 4.5 | 0.5 / 0.6 / 2.9 | 0.7 | 5.9 |

Against the earlier GH200 numbers for 5/2+ T=1/2 (basis 93, `cfp W cfp^T` 46, scatter 11, three embeddings 31, flow 28): the scatter is 10x faster (1.1 s), the three embeddings 10x (3.1 s), the flow and the
basis are as before (the earlier force numbers also predate the vectorised angular coefficients, so the force total, 119 s then and 29 s now, is not an effect of item 1 alone). `to_host` takes 1 ms and
`device_wait` nothing.

**Not as estimated:** `cfp W cfp^T` inside the force takes 20.9 s, not the 1 s that the 3e13 flops suggest, although the same kernel runs three times in `embed` for 3.1 s in all. The force is the first
call with a given shape and the cache was cold, and the time falls with the matrix size faster than the flops would (2.9 s at 15 789 states): the most likely cause is the compilation (and GEMM autotuning)
of one large kernel per channel shape, about 15-20 s each, but this is a guess until a warm-cache run (or `XLA_FLAGS=--xla_gpu_autotune_level=0`) is measured.

**Lab transform of the largest channel** (J=9/2+ T=1/2, 32 259 states), cold / warm: products and copies 45.9 s / 4.2 s (before 46 s / 4.2 s), 21 + 21 compiled kernels (about 90 before), T coefficients 44 s,
peak host 20.4 GB and device 14.6 GB (both as before). The packed matrix is 2.08 GB (a full square would be 4.16 GB). So item 3 halves the stored matrix as intended, and item 2 cut the number of kernels
without cutting the cold time: 42 kernels in about 42 s, 1 s each instead of 0.4-0.5 s. The coarser ladder made the kernels bigger and did not pay; what a compile costs here is an open question,
the same one as for `cfp W cfp^T`.

## Verification C: what the first call costs (DeltaAI job 3317317, 2026-10-05, node `gh048`, commit `fc44232`)

Records: `benchmarks/gh200_stages_verify_c1_warm.jsonl`, `gh200_stages_verify_c2_autotune0.jsonl`, `baseline_gh200_me3j_lab_j9_autotune0.json`. 2J = 5 at N = 40, four channels, seconds summed:

| run | force (of it antisymmetrize) | embed | flow |
|---|---|---|---|
| A: cold compile cache, default | 66.2 (50.3) | 7.9 | 75.5 |
| C1: warm compile cache, default | 63.5 (44.2) | 7.3 | 74.8 |
| C2: `XLA_FLAGS=--xla_gpu_autotune_level=0`, empty cache | 29.1 (12.5) | 17.2 | 137.5 |

The 20.9 s of `cfp W cfp^T` in the first channel is **not compilation**: a warm compile cache gives the same 21.1 s. It is XLA's GEMM autotuning, which runs a few dozen algorithms on each new product
shape (2.6e13 flops each here), in every process and per channel shape; the same kernel inside `embed` is quick because the shape has been tuned by then. With autotuning off the antisymmetrisation is 4.2 s, but
the flow's products get 2x slower (62 s against 32 s for the first channel) and the embeddings 2.5x, so the global flag loses overall (183.8 s against 145.6 s for the three stages). The lab benchmark of the
largest channel with autotuning off: products cold 14.4 s (45.9 s), warm 5.6 s (4.2 s), peak device memory 11.4 GB (14.6 GB). So the cold lab cost of job A was autotuning too, not compilation (21 + 21
kernels in 14 s), and the coarser ladder of item 2 was not the lever.

Action: `kernels.compile.NO_AUTOTUNE` is set per kernel (`jit(..., compiler_options=...)`) on the antisymmetrisation and on the lab contraction, so the flow keeps its tuned products. Expected for the
production run, from the numbers above and not yet measured: about 15 s less for each large channel's antisymmetrisation, about 30 s less cold lab stage, over all channels some 10-20 minutes.

## Verification D: the per-kernel no-autotune option on the GH200 (DeltaAI job 3317423, 2026-10-05, node `gh015`, commit `ca13a3b`, 5 min, max RSS 21.4 GB)

Records: `benchmarks/baseline_gh200_3n_antisym_j5_n40.json`, `baseline_gh200_me3j_lab_j9_v4.json`, `gh200_stages_verify_d_j5.jsonl`.

- **Antisymmetrisation, 2J = 5 +, T = 1/2, N = 40** (n_as 11 340, n_nas 34 020, synchronised stages): first call 3.9 s (`as_cfp` 0.59, `as_gemm` 3.30), repeats 2.65 s (0.45 / 2.20), the same for a sparse input (2.7 and 2.6 s;
  the sparse matrix costs 0.1 s to go up). Job A: 20.9 s. Peak device memory of this benchmark 26.7 GB (the dense W, C and the intermediate).
- **Lab transform, largest channel** (J=9/2+ T=1/2): products cold 14.5 s, warm 5.5 s (job A: 45.9 / 4.2), peak device 11.3 GB (14.6), peak host 19.8 GB, T coefficients 44 s as before.
- **2J = 5 at N = 40, four channels, summed seconds**, default flags, V3 removed, compile cache holding the new antisymmetrisation kernel from D1:

| | A (cold, autotuned) | D3 |
|---|---|---|
| force | 66.2 | 15.6 |
| of it antisymmetrize | 50.3 | 3.6 (`as_cfp` 1.3, `as_gemm` 2.3) |
| embed | 7.9 | 6.8 |
| flow | 75.5 | 60.9 |

For 5/2+ T=1/2 alone: force 6.6 s (A: 29.0), antisymmetrize 1.5 s (20.9), embed 2.9 s, flow 27.3 s (32.7). The flow did not get slower, which is the point of setting the option per kernel instead of globally.
The `as_gemm` of 0.9 s here against 2.2 s in the benchmark process for the same shape is not explained (the benchmark's first call also compiled, and its process held the dense W alongside); nothing relies on it.

Summary of the three verification jobs for the items of the production-run list: item 1 (device antisymmetrisation, scatter and embeddings) took the force of the largest channel from 119 s (before the vectorised coefficients, too) to
6.6 s and the embeddings from 31 s to 2.9 s; item 3 halves the stored lab matrices; item 2's ladder was not the lever, autotuning was, now off for the two GEMM-heavy kernels; item 4 not done.

## Verification B: the Jacobi side with the production ramp (DeltaAI job 3317876, 2026-10-05, node `gh149`, commit `6fec19b`, 32 CPUs, 9 min 43 s, max RSS 13.8 GB)

Records: `benchmarks/gh200_stages_verify_b_ramp32.jsonl` (70 channels), `gh200_stages_verify_b_highj.jsonl` (20 channels). Both runs exited 0 with no error; no Jacobi cache existed in the new format, so the bases were rebuilt.

| ramp32, Jacobi part, all 70 channels | basis | force | embed | flow |
|---|---|---|---|---|
| before (`gh200_stages_jacobi_ramp32.jsonl`) | 101.9 s | 757.2 s | 34.1 s | 118.2 s |
| now | 99.3 s | 112.8 s | 24.3 s | 49.9 s |

The force of 2J = 5 at N = 32 is 4.1 s (49.9 s before), of it 1.7 s in the antisymmetrisation. The old force numbers predate both the vectorised angular coefficients and the device chain, so the factor 6.7 is
not the effect of one change. The flow got faster too (118 to 50 s); I did not look into why (the old run was another node and another commit). The basis is unchanged and is now the largest stage (99 of 286 s).

High-J channels with the production ramp, `--js 11,13,19,27,35` (20 channels, N = 28 for 2J = 11, N = 24 above): basis 93.5 s, force 77.9 s, embed 10.2 s, flow 52.0 s, 233 s in all. The largest, 11/2+ at N = 28
(5794 AS states), takes 25.5 s for the basis, 7.2 s for the force, 1.4 s for the embeddings and 5.1 s for the flow. These channels had not been measured at the production ramp before; nothing went wrong in them.

**What the profiles show** (main process of the flat16 run, 849 s): 162 s in 382 JAX compilations, 118 s in eager gathers of the force assembly (`_index_to_gather`, fixed afterwards by
`project_pairs_indexed`), waiting for workers 190 s, forking worker pools 15 s. In the workers (profiled on one CPU, J=11/2, N=20, `cProfile`) 95% of the force time is the **pure-Python angular coefficient
loops** (`_coefficients_tpe` and `_tpe_spin_sum`, 18 million scalar calls for one channel): the force is not limited by the projection or the device.

**Force time by channel** (T=1/2, parity +, 30 workers, seconds): the cost grows with `J` at fixed `N`; at N = 16 it is 2.3 (J=1/2), 5.7 (5/2), 11.3 (9/2), 16.8 (17/2), then falls again as the channels run out
of states (0.7 s at 35/2). A fit `ln t = -5.56 + 0.78 ln n_NAS + 0.85 ln(2J+1) - 0.72 [T=3/2]` reproduces the 140 channels to 16% (rms of the logarithm) but extrapolates the single measured large point
(J=5/2, N=40, 119 s) from below (64 s). Applied to the production ramp (J <= 5/2 at N=40, 7/2 at 36, 9/2 at 32, 11/2 at 28, J >= 13/2 at 24) it gives about 2400 s for the force, of which 1640 s are the 48 channels with
J >= 13/2 at N=24, which no run has measured yet (the smaller ramps put those at N=16). With the underestimate at the large end I take 2400-4800 s for the force. The basis (about 300-600 s the first time,
cached afterwards), embeddings (about 100-200 s) and flows (about 200-300 s) add up to 600-1100 s.

**Estimate of the production run** on 32 CPUs: Jacobi side 50-100 minutes, lab 5 minutes (286 s measured), write 1.5 minutes (77 s measured), NN projection and compilation 5 minutes: **about 1-2 hours**.
Dominated by the angular coefficients, which parallelise over the CPUs of the node (a GH200 node has 72 cores per GPU).

**The genuine force at high `J`** (`LocalForce.jmax3`, `--j3max-3nf`; NuHamil's `J3max_initial_3nf`, default all `J`): dropping it above J = 9/2 changes the me3j values at 8/16/8 by up to 0.036 MeV
(rms 4.4e-4 MeV against an rms value of 0.18 MeV; 2.3 of 5.2 million values differ by more than 1e-6), above J = 5/2 by up to 0.066 MeV. That is not negligible at e3max 8, so the default stays "all J";
it is the user's physics choice if production should cut it (and it would remove most of the force time).

**Guards added after this job:** worker count defaults to every CPU and is printed with the rest of the configuration on the first line; one worker on a machine with more than four CPUs warns; per-channel
stage log and cProfile options; finished `V3` of each Jacobi channel are cached (`cache_dir/v3`, keyed by interaction, hw, lambda, force, e3max, tolerance and ramp channel), so a run that hits its time limit or
crashes resumes with the channels it finished; `--js` times a few channels of a large run.
