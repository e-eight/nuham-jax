# Benchmark harness (Phase 4b)

`generate_me2j` takes an optional `StageTimer` (`nuham/timing.py`) and records `project`, `srg`, `lab_coeffs`, `lab_contract`,
`assemble` and `write` seconds. `path=None` runs compute-only (lab matrices built, nothing assembled or written). Timers and
compute-only do not change the output (`tests/test_timing.py`: byte-identical files).

`benchmarks/run_nn.py` (outside the package) runs repeats in one process and prints JSON with environment (host, backend, device,
versions, git commit), stage times, `unaccounted_s`, peak host RSS (whole process, only grows) and peak device memory (`null` on CPU).
Repeat 0 includes bracket-table build and kernel compilation; later repeats are warm.

```bash
uv run python benchmarks/run_nn.py ../NuHamil-public/input_nn_files/N3LO_EM500_kmax8_N100_Jmax8.bin \
    --emax 8 --e2max 16 --hw 16 --lam 2.0 --repeat 2 --json out.json [--compute-only]
```

## Local CPU baseline (2026-10-03, `darjeeling`, 16 cores, jax 0.11.2 CPU, commit 54e6719 + harness)

N3LO EM500, hw 16, lambda 2.0, emax 8, e2max 16 (2.50e6 values, 13.9 MB). Seconds. Raw record: `benchmarks/baseline_cpu_emax8.json`.

| run | total | project | srg | lab_coeffs | lab_contract | assemble | write | peak RSS |
|---|---|---|---|---|---|---|---|---|
| cold | 36.6 | 9.45 | 6.75 | 2.49 | 16.49 | 0.41 | 1.01 | 1.0 GB |
| warm | 4.7 | 0.04 | 1.83 | 0.50 | 1.02 | 0.44 | 0.84 | 1.1 GB |
| warm, compute-only | 3.7 | 0.04 | 2.16 | 0.50 | 1.02 | - | - | - |

Cold-run excess is compilation (project, lab_contract) and table building; the compile cache was empty for these shapes' first
use here. Writing (assemble + write) is ~27% of a warm run at emax 8, below the ~45% seen at emax 14 on GPU nodes, because the
GPU runs shrink the compute share. Compute-only run-to-run noise is about 0.5 s (srg 1.8-2.2 s), so compare by alternating runs.

## NuHamil reference on this machine (podman image `nuhamil`, same case)

21 s wall, cgroup peak memory 1.39 GB (includes page cache), output 11.8 MB (NuHamil gzip level differs). NuHamil's own timers:
SRG flow 8.5 s, TMTransChannel 8.9 s, file write 2.7 s. Thread count of the container build was not checked, so treat it as
one data point. Compare against the warm 4.7 s (cold 36.6 s) of this package; the cold figure is the fair one for a single run
on an empty cache.

## DeltaAI GH200 (job 3300774, 2026-10-03, one GPU, 16 cores allocated, ~3 min, jax 0.11.2 GPU)

Same case and settings as the CPU baseline; e2max = 2 emax. Raw records: `benchmarks/baseline_gh200_*.json` (`cpu_count` there is the
node's 288, not the allocation; `git_commit` is null because the code was uploaded with `git archive`). Seconds.

| case | run | total | project | srg | lab_coeffs | lab_contract | assemble | write | peak GPU |
|---|---|---|---|---|---|---|---|---|---|
| emax 8 full | cold | 80.6 | 38.79 | 8.47 | 2.07 | 30.39 | 0.31 | 0.46 | 0.19 GB |
| emax 8 full | warm | 1.8 | 0.18 | 0.06 | 0.27 | 0.55 | 0.30 | 0.44 | 0.19 GB |
| emax 8 compute-only | warm | 1.1 | 0.18 | 0.06 | 0.27 | 0.54 | - | - | 0.09 GB |
| emax 10 full | cold | 44.8 | 3.20 | 2.19 | 3.28 | 33.07 | 0.97 | 2.03 | 0.27 GB |
| emax 10 full | warm | 4.8 | 0.18 | 0.06 | 0.58 | 0.78 | 0.96 | 2.16 | 0.27 GB |
| emax 10 compute-only | warm | 1.6 | 0.18 | 0.06 | 0.59 | 0.77 | - | - | 0.22 GB |

- Warm emax 8: 1.8 s on the GH200 against 4.7 s on this CPU (compute-only 1.1 s against 3.7 s). The SRG flow is 0.06 s against ~2 s,
  and projection 0.18 s against 0.04 s on CPU (CPU wins there; it is a small stage).
- The first emax 8 job step started with an empty compile cache (project 38.8 s, lab_contract 30.4 s are compilation); later steps
  reused it. Cold times therefore mostly measure the compile cache, not the algorithm. Compare warm times.
- With the compute stages this fast, writing is 40% (emax 8) and 65% (emax 10) of a warm full run, which is the case for 4c.
- Only one job per case, so no run-to-run spread yet. A shared node can vary by tens of percent.

## Phase 4c: writer (CPU, 2026-10-03)

**Vectorized formatter** (`nuham.me2j._format_lines`). Python's `%16.8f` ran at ~5 M values/s, 71 MB/s, in the calling thread, and
threads cannot help it (the GIL). The values are now scaled by 1e8, rounded as integers and turned into text through lookup tables
(`_tables`: four-digit groups for the fraction, a signed right-aligned integer part plus the point for the head). It runs at
~27 M values/s, 5.6x faster. The output is byte-identical to Python's: scaling is only reliable away from half-way cases, so values whose scaled
fraction is within 4 ulp of 0.5 are formatted by Python, and chunks with non-finite values or |x| >= 1e5 take the Python path whole.
Checks: `tests/test_me2j.py` (adversarial ties, near-ties, signs, -0.0, powers of ten, fall-back) and 40 random sweeps of 500 k values with
no mismatch; the emax 8 file has the same size (13 908 826 bytes) as before.

**What is left in `write`** (emax 10, 1.27e7 values, 205 MB of text):

| output | write stage | file |
|---|---|---|
| plain text | 0.42 s | 205 MB |
| gzip level 1 | 3.7 s | 70.5 MB |
| gzip level 6 | 11.1 s | 59.8 MB |

Formatting is now cheap; gzip is the whole write cost. It runs in one thread, and only formatting overlaps with it, because the
lab matrices of all channels are finished before assembly starts. A faster compressor would be a separate decision (see below).

**NumPy contraction path for CPU: tried and dropped.** A plain GEMM microbenchmark favours OpenBLAS over XLA-CPU by 1.1-2.3x, but in
the pipeline the NumPy path was slower (emax 8 warm 4.65-4.87 s against 4.09-4.45 s; emax 10 11.7-12.1 s against 11.05-11.4 s,
alternating runs) and changed the output bytes (summation order flips a last-digit rounding). The XLA path stays on every backend.

**Open options for the gzip cost** (all produce valid single-stream gzip, none is byte-identical to the standard `gzip` module):
`python-isal` (igzip) as a drop-in; or pigz-style parallel compression of linked blocks. Not started.

## Phase 4d: parameter scans (CPU part, 2026-10-03)

`nuham.scan.generate_scan` writes one `me2j` per `(hw, lambda)` point (CLI `python -m nuham.scan`). Per batch of `batch_size` points:
the SRG flows of `flow_sets` points share a batch (`evolve_relative_nn_many`, per-entry `hw` and flow parameter in
`srg_evolve_batch`); the lab coefficients of each `(J, parity)` are built and uploaded once and contracted with every point's relative
interaction in one kernel (`accumulate_blocks_multi`, a single GEMM `P x (blocks*r) x (K*P)` instead of K). Brackets were already shared.
All lab matrices of a batch stay in memory until its files are written, so `batch_size` trades speed for memory (measured host RSS grows by about 6 GB per extra point at
emax 14, see the DeltaAI emax 12-14 section). Gate: every scan file equals the separate single run, `tests/test_scan.py` (several batch sizes and flow groupings, bare and
evolved points, atol 1e-8 on file values; relative potentials 1e-9). The single-run output is bit-identical to before the change.

CPU measurements (`benchmarks/run_scan.py`, emax 8, e2max 16, grid hw 12/16/20 x lambda 1.8/2.0 = 6 points, second pass of two;
raw record `benchmarks/baseline_cpu_scan_e8.json`). Seconds per point, `flow_sets` 1:

| batch size | compute-only | with writing | lab_coeffs (6 points) | lab_contract (6 points) | peak RSS |
|---|---|---|---|---|---|
| 0 (separate single runs) | 3.50 | 4.60 | 2.4 | 6.1 | 1.2 GB |
| 2 | 3.17 | 4.23 | 1.3 | 5.0 | 1.5 GB |
| 3 | 3.44 | 4.18 | 0.9 | 4.6 | 1.6 GB |
| 6 | 2.89 | 3.97 | 0.4 | 4.7 | 1.8 GB |

- Sharing the coefficients works as intended (`lab_coeffs` 2.4 -> 0.4 s) and the multi-set contraction gains ~20%. Both are small next to
  the SRG flow, which is ~70% of a CPU scan point (12.3 s of 17.4 s) and does not get cheaper per point.
- **Batching the flows across sets is slower on CPU.** With all six points in one flow batch the SRG stage took 34.9 s instead of 12.3 s.
  Flowing K identical sets together cost 1.96 s per set alone and 2.83/3.04/3.37 s per set for K = 2/3/6, so it is mostly cache and
  memory pressure on the dense dopri8 stages; different sets add imbalance, since a vmapped adaptive solver runs as long as its slowest
  entry (flow cost per set at hw 16: 1.18 s for lambda 2.4, 3.08 s for lambda 1.6). So on the CPU the default `flow_sets` is 1 and the speed-up above comes
  from coefficients and contraction only. On accelerators the default is all points of a batch together
  (`nuham.runtime.default_flow_sets`, chosen from the DeltaAI result below).
- Per-point cost is nearly flat in batch size on CPU, as expected: there is little spare parallelism to fill. The open question for DeltaAI
  is whether a GPU gains from `flow_sets` > 1 and from larger `batch_size`; the benchmark has both knobs (`--flow-sets`, `--batch-sizes`).

## DeltaAI GH200: 4c and 4d (job 3300961, 2026-10-03, commit b74e544, one GPU, ~4 min)

Code was pushed from the local repo and pulled on DeltaAI (`git_commit` in the records is the pulled commit). Raw records:
`benchmarks/baseline_gh200_4cd_*.json`. Seconds; N3LO EM500, hw 16, lambda 2.0 unless noted.

**4c, single runs, warm (before = job 3300774 at commit 91099bd).**

| case | write before | write after | assemble | total before | total after |
|---|---|---|---|---|---|
| emax 8 | 0.44 | 0.11 | 0.32 | 1.8 | 1.58 |
| emax 10 | 2.16 | 0.70 | 0.98 | 4.8 | 3.4 |

The vectorized formatter cuts `write` by 3-4x on the GH200 host as well (formatting was most of it; gzip level 1 is ~0.1 s per
emax-8 file and ~0.7 s at emax 10). Writing is no longer the largest stage: at emax 10 `assemble` (host Python, 0.98 s) is now bigger
than `write`, and `lab_coeffs` + `lab_contract` is 1.4 s.

**4d, scan of 6 points (hw 12/16/20 x lambda 1.8/2.0), emax 8, e2max 16, second pass of two, seconds per point.**

| batch size | flow_sets | mode | per point | project | srg | lab_coeffs | lab_contract | assemble | write |
|---|---|---|---|---|---|---|---|---|---|
| 0 (separate runs) | - | full | 1.58 | 1.07 | 0.42 | 1.67 | 3.55 | 1.89 | 0.68 |
| 2 | 1 | full | 1.22 | 1.07 | 0.42 | 0.83 | 2.34 | 1.90 | 0.67 |
| 6 | 1 | full | 0.98 | 1.06 | 0.42 | 0.28 | 1.53 | 1.87 | 0.68 |
| 6 | 1 | compute-only | 0.56 | 1.08 | 0.42 | 0.28 | 1.54 | - | - |
| 6 | 6 | compute-only | 0.53 | 1.09 | 0.20 | 0.28 | 1.55 | - | - |

(Stage columns are totals over the 6 points; the per-point column is total / 6.)

- A scan of 6 points costs 0.98 s per point against 1.58 s for separate runs (-38%); compute-only it is 0.56 s against ~1.1 s (the sum of the
  separate-run compute stages), about 2x. The gain comes from building the coefficients once (`lab_coeffs` 1.67 -> 0.28 s) and
  the batched contraction (3.55 -> 1.53 s).
- **The GPU prefers larger flow batches, the opposite of the CPU:** `flow_sets` 6 halves the SRG stage (0.42 -> 0.20 s for 6 points; a single
  set takes 0.06 s) while on the CPU it was 3x slower. At this size the stage is small either way (6% of a compute-only point).
- What is left per point: `assemble` (0.31 s, host Python) is the largest stage of a full scan point, then `project` (0.18 s) and
  `lab_contract` (0.26 s). `write` is 0.11 s. Peak GPU memory stayed below 0.33 GB and host RSS below 1.9 GB.
- Only one job, one pass per case: no spread. Larger emax (where contraction and writing grow) was not scanned.

## DeltaAI GH200: scans at emax 12 and 14 (job 3301003, 2026-10-03, commit a19fabd, ~15 min)

6 points (hw 12/16/20 x lambda 1.8/2.0), e2max = 2 emax, one GH200, `flow_sets` default (all points of a batch together on a GPU), second pass of
two. Raw records: `benchmarks/baseline_gh200_scan_*.json`. Stage columns are totals over the 6 points; seconds.

| emax | mode | batch | per point | srg | lab_coeffs | lab_contract | assemble | write | peak host RSS | peak GPU |
|---|---|---|---|---|---|---|---|---|---|---|
| 12 | compute-only | 0 | 2.93 | 0.42 | 8.77 | 6.86 | - | - | 4.0 GB | 0.7 GB |
| 12 | compute-only | 2 | 1.86 | 0.27 | 4.46 | 5.18 | - | - | 6.3 GB | 0.9 GB |
| 12 | compute-only | 6 | 1.13 | 0.20 | 1.48 | 3.96 | - | - | 10.7 GB | 1.9 GB |
| 14 | compute-only | 0 | 5.50 | 0.42 | 20.61 | 10.25 | - | - | 6.5 GB | 1.5 GB |
| 14 | compute-only | 2 | 3.36 | 0.27 | 10.35 | 8.14 | - | - | 19.4 GB | 2.1 GB |
| 14 | compute-only | 6 | 1.97 | 0.21 | 3.51 | 6.87 | - | - | 36.3 GB | 5.1 GB |
| 14 | full | 6 | 21.49 | 0.21 | 3.48 | 6.84 | 45.25 | 71.96 | 35.7 GB | 3.4 GB |

(`project` was 1.05 s for the 6 points in every row.)

- **Compute-only, a scan of 6 points is 2.6x (emax 12) and 2.8x (emax 14) faster per point than separate runs.** `lab_coeffs` is the big
  saving (20.6 -> 3.5 s at emax 14, built once instead of six times); `lab_contract` improves by a third (10.3 -> 6.9 s).
- **A full emax 14 scan point is dominated by the writer:** 21.5 s per point, of which assemble 7.5 s and write 12.0 s (2.7 GB of text per
  file, 225 MB/s through one gzip level-1 thread plus formatting). The same point costs about 25 s as a separate run (5.5 s of compute plus
  ~19.5 s of writing), so a full scan saves only ~14% at this size. This is the point at which the deferred gzip options become relevant
  (`python-isal`, or parallel blocks), and the host-side `assemble` is the second candidate.
- **Host memory grew by about 6 GB per extra point at emax 14** (6.5 GB for one point, 19.4 GB for two, 36.3 GB for six), three times the
  2 GB I estimated. With 64 GB requested the largest safe batch at emax 14 is about 8 points. The follow-up section below explains it: the lab
  matrices themselves are 3.9 GB per point, not 2 GB.
- The job took ~15 min against an estimate of ~20, about 0.26 GPU-hours.

## Writer and memory follow-up (DeltaAI jobs 3301083 and 3301101, CPU profiling, 2026-10-03)

**Compression: `isal` (igzip) against `gzip`.** `isal` writes standard gzip streams (`gzip -t` and `zcat` accept them, decompressed text identical).
Microbenchmark on 40 MB of emax-8 me2j text, MB/s and output size:

| compressor | local x86 CPU | GH200 (Grace) | size |
|---|---|---|---|
| gzip level 1 | 50 | 137 | 13.9 MB |
| isal level 1 | 484 | 267 | 14.2 MB (14.2 vs 13.9: +2%) |
| isal level 2 | 476 | 278 | 14.1 MB |
| isal level 0 | 493 | 374 | 16.9 MB |

ISA-L is much better optimised for x86 (10x) than for Grace (2x), but the whole `write` stage still improves a lot because formatting is cheap now.
End to end on the GH200 (warm second pass, two runs of each, `run_nn.py --isal`, `benchmarks/baseline_gh200_gz_*`):

| emax | compressor | total | assemble | write | file |
|---|---|---|---|---|---|
| 10 | gzip | 3.29 | 0.96 | 0.66 | 70.5 MB |
| 10 | isal | 2.87 | 0.96 | 0.26 | 71.5 MB |
| 12 | gzip | 8.95 | 2.78 | 3.27 | 278.4 MB |
| 12 | isal | 6.58 | 2.77 | 0.93 | 281.5 MB |

At emax 12 the write stage is 3.5x shorter and the run 26% shorter (local x86 emax 10: write 3.5 -> 0.4 s, run 12.4 -> 8.8 s). Extrapolating
to emax 14 (write 12 s per point now) gives about 3.5 s per point. An aarch64 wheel exists (`isal` 1.8.0, installed for the test into
`/projects/bhhk/soham/nuham/pylibs` with `PYTHONPATH`, the venv was not changed). Not yet in the package: it would be an optional dependency used
when importable, with `gzip` as the fallback; output bytes would differ from the `gzip` module's.

**`assemble`, batched** (`nuham.me2j.assemble`). Profile at emax 10 (local CPU): 1.37 s for 12.7 M values, 45% of it in 75 000 tiny `element()` calls, i.e.
numpy call overhead, not memory. It now batches consecutive `(a, b)` pairs into chunks of ~260 k tuples and handles each `(J, parity)` with one set of array
operations per chunk. Result: 0.69 s (2.0x), bit-identical to the old per-pair version (`tests/test_synthetic_pipeline.py` against
`tests/assemble_reference.py`, including tiny batches; the NuHamil reference comparisons pass unchanged). I expected the same on the GH200; it did not happen, see the next section.

**Memory.** A fresh process per point count at emax 14 on the GH200 (`benchmarks/bench_memory.py`; an earlier run of this tool assumed 4 KB pages and
was discarded, the kernel here has 64 KB pages), GB:

| pinned copy | points | RSS after | growth per point | matrices per point |
|---|---|---|---|---|
| on | 1 | 6.50 | 5.78 | 3.94 |
| on | 2 | 11.03 | 5.15 | 3.94 |
| off | 1 | 5.97 | 5.26 | 3.94 |
| off | 2 | 10.02 | 4.64 | 3.94 |

- **The lab matrices are 3.94 GB per point at emax 14**, twice the "about 2 GB" in `CLAUDE.md` and my earlier notes (that figure was too low).
  The marginal cost of an extra point is 4.0 GB (4.5 GB with the pinned copy), so almost all of it is the matrices; a fixed ~1.8 GB comes on top (coefficients, caches).
  The pinned copy costs about 0.5 GB per point (+12%).
- The 36 GB seen with 6 points in a scan (job 3301003) is above the 26-29 GB this predicts; the rest is probably allocator retention from the earlier
  passes of the same process and transient block stacks. For planning use ~6 GB per point at emax 14.
- The matrices are symmetric but stored as full square arrays, so packing the lower triangle would halve them (about 2 GB per point). That changes the values
  at the 1e-15 level (the two triangles of a GEMM result are not bit-equal), so output bytes would differ in rare last digits. Not done.

## DeltaAI GH200: batched `assemble` and the isal extra (job 3301348, 2026-10-03, commit 2cb9a9e, ~11 min)

Raw records `benchmarks/baseline_gh200_asm_*.json`. Single runs: second (warm) pass of two. Scans: 6 points (hw 12/16/20 x lambda 1.8/2.0), batch 6,
full mode, the pass reported is the second of two, seconds. Both compressors ran with the batched `assemble`.

| case | compressor | total | assemble | write | file |
|---|---|---|---|---|---|
| emax 12 single | gzip | 8.9 | 2.59 | 3.48 | 278 MB |
| emax 12 single | isal | 6.7 | 2.57 | 1.12 | 281 MB |
| emax 14 single | gzip | 25.4 | 8.51 | 11.29 | 914 MB |
| emax 14 single | isal | 17.7 | 8.44 | 3.61 | 922 MB |
| emax 14 scan, per point | gzip | 21.8 | 8.5 | 11.3 | |
| emax 14 scan, per point | isal | 13.95 | 8.4 | 3.5 | |

- **isal confirms its gain at emax 14:** write stage 11.3 -> 3.6 s (3.1x), a scan point 21.8 -> 13.95 s (-36%), a single run 25.4 -> 17.7 s (-30%); files +1%.
- **The batched `assemble` did not speed up on the GH200.** It took 8.4-8.5 s per point at emax 14 (7.5 s per point before in job 3301003) and 2.6 s at emax 12 (2.78 s before).
  My local measurements gave 2.0x at emax 10 but only 1.5x at emax 12 (4.74 -> 3.13 s) on this x86 CPU, so the gain shrinks as the matrices grow. Profile
  of the batched version at emax 12: `element()` 1.08 s and the batch function's own array operations 0.9 s, i.e. about 14 ns per looked-up element over
  ~76 M gathers, which is numpy's per-element cost (index arithmetic, masks, temporaries), not call overhead any more. On Grace this is no faster than before.
- `assemble` is now 60% of a full emax 14 scan point (8.5 of 13.95 s), then write (3.5 s); the compute stages total about 2 s.

Candidates, not started: (1) run assemble and formatting in worker threads over batches (numpy releases the GIL in these loops; 16 cores were allocated), keeping the
batches in order for the writer; (2) isal's multi-threaded compressor (`isal.igzip_threaded`), which writes one valid gzip stream from parallel blocks and would remove the
single compression thread (about 9 s of compression work per emax 14 file at 300 MB/s) as the floor once assembly is parallel; (3) assembling on the device, where the
matrices already are, and copying only the values back.

## Device-side assembly: CPU prototype (2026-10-03)

`nuham.me2j.assemble_device` (kernel `nuham.kernels.assemble.assemble_values`) builds the four values per tuple with JAX instead of numpy, selected with
`assembler="device"` (`--assembler device` in the CLIs and benchmark runners; `"numpy"` is still the default). All matrices of a run go into three flat arrays
(one per `Tz`) with per-group offsets and sizes, so the kernel has a static shape and compiles once per run; the host cuts the tuple stream into fixed windows of
`batch_tuples` (default 2^20) and prepares the next window while the device works on the current one.

Checks: bit-identical to the numpy `assemble` on the test problem and on real data (emax 10 and 12 on XLA-CPU: 0 of 12.7 M and 50.7 M values differ), windows of any size
including padded ones (`tests/test_assemble_device.py`), and whole files and scans written with it equal those written with numpy.

Local CPU timing, warm, seconds for the whole assembly (XLA-CPU, host index preparation included):

| emax | values | per-pair numpy (old) | batched numpy | device (XLA-CPU) |
|---|---|---|---|---|
| 10 | 12.7 M | 1.75 | 0.73 | 0.49 |
| 12 | 50.7 M | 5.00 | 3.10 | 1.86 |

At emax 12 the device version's 1.8 s splits into: flattening the matrices 0.7 s (one-off per call, a host concatenation of 1.2 GB; it goes away once matrices are
built into the flat layout on the device), tuple generation on the host 0.5 s, building the index columns 0.4 s, kernel and uploads about 0.5 s (about 25 M tuples/s on
the CPU). Host preparation does not depend on the matrices, so a scan can build it once and reuse it for every point. Not yet measured on a GPU.

**Device-resident matrices (added after the first prototype).** With `assembler="device"` the lab contraction keeps its results as device arrays (`on_device=True`, no
`to_host`; the stage timers then wait for the device so `lab_contract` stays honest) and `_flatten_channels` concatenates them on the device, so on an accelerator the
matrices never visit the host: no host-device round trip, and about 4 GB less host memory per point at emax 14. Equality with the numpy assembly is exact
(`tests/test_assemble_device.py`). On the CPU this path is not useful: emax 12 end to end, `assemble` 3.0 -> 1.2 s but `write` 1.6 -> 2.8 s (the XLA-CPU kernel and the
formatting share the same cores) and host RSS 2.7 -> 4.5 GB (matrices exist as device arrays and again as the flat copy), total unchanged. So `numpy` stays the default and
the device assembly is meant for accelerators.

## DeltaAI GH200: device assembly (job 3301528, 2026-10-03, commit 7007019, ~6 min, ~0.1 GPU-hours)

All runs with `--compressor isal`, second (warm) pass of two for single runs, seconds. Raw records `benchmarks/baseline_gh200_dev_*.json`. The numpy rows repeat the
isal rows of job 3301348 (same code, same numbers within 1%).

| case | assembler | total | assemble | write | host RSS | GPU peak |
|---|---|---|---|---|---|---|
| emax 12 single | numpy | 6.7 | 2.59 | 1.14 | 3.7 GB | 0.6 GB |
| emax 12 single | device | 6.2 | 0.66 | 2.66 | 2.7 GB | 3.1 GB |
| emax 14 single | numpy | 17.7 | 8.45 | 3.59 | 6.2 GB | 1.3 GB |
| emax 14 single | device | 15.9 | 1.98 | 8.35 | 3.7 GB | 10.3 GB |
| emax 14 scan, per point | numpy (job 3301348) | 13.95 | 8.4 | 3.5 | 35.5 GB | 3.4 GB |
| emax 14 scan, per point | device | 12.07 | 1.9 | 8.3 | 4.3 GB | 32.3 GB |

- **The device assembly is 4x faster than the numpy one** (2.59 -> 0.66 s at emax 12, 8.45 -> 1.98 s at emax 14), and the files are equivalent (sizes differ by a few bytes
  from the gzip streams; values equal in the CPU tests).
- **The end-to-end gain is only 10-13%** (emax 14 single 17.7 -> 15.9 s, scan point 13.95 -> 12.07 s), because `write` grew by the time `assemble` saved: assemble and write
  together went 12.0 -> 10.3 s per point. With assembly this fast, the writer is the floor. The main thread formats (about 6 s at emax 14, 27 M values/s per thread), and the
  compression thread needs about 10 s of igzip at ~270 MB/s for 2.7 GB of text; the pipeline now waits for it. (Before, the compression thread was hidden behind the 8.4 s of assembly.)
- **Host memory fell 8x in a scan** (35.5 -> 4.3 GB for 6 points at emax 14): the matrices stay on the GPU, which now holds 32 GB for 6 points (3.9 GB of matrices per point, plus a
  transient flat copy while a point is assembled). The batch size at emax 14 is limited by GPU memory (~120 GB) instead of 64 GB of host memory, so about 15 points fit.
- **Next floor: the writer.** Options, none started: isal's threaded compressor (`igzip_threaded`, 1.7x on this x86 machine at 8 threads with identical output size and valid gzip; not yet
  measured on Grace); formatting on the device (the table-driven formatter is integer arithmetic) or in worker threads; releasing the per-channel device arrays once the flat copy exists
  (halves the transient GPU memory); building the host index arrays once per scan instead of once per point.

## Device formatter and threaded compression: CPU prototype (2026-10-03)

`nuham.kernels.format.format_fields` formats the `(N, 4)` values of an assembled window as 16-byte `f16.8` fields on the device with the host formatter's integer
arithmetic and lookup tables (`formatter="device"`, `--formatter device`; needs `assembler="device"`). The values never leave the device: only the fields (16 bytes
per value) come back, and `write_me2j_fields` does the line layout (ten fields and a newline). Values whose scaled fraction is within 8 ulp of 0.5 are returned as a list
(capacity 4096 per window, redone with room for all if exceeded) and formatted on the host with Python's `%`, so the text is identical; a value that is not finite or has
magnitude >= 1e5 raises, because it would not fit a field. `compress_threads` (`--compress-threads`) selects `isal.igzip_threaded` for blocks compressed in parallel
(one valid stream, same size).

Checks (`tests/test_format_device.py`): fields equal Python's formatting on the adversarial set (ties, near-ties, signs, -0.0, powers of ten) and on random values at five
scales; the fields writer writes the same file as the host writer across windows that split lines; the patch and overflow paths; unformattable values; every
assembler/formatter combination writes identical files; threaded streams decompress to the same text.

Local CPU, emax 10, warm, isal, seconds (the CPU run is dominated by the contraction, 8-9 s in total, so only the writer stages are informative):

| assembler | formatter | compress threads | assemble | write | total |
|---|---|---|---|---|---|
| numpy | host | 1 | 0.78 | 0.46 | 8.85 |
| numpy | host | 8 | 0.89 | 0.49 | 8.97 |
| device | host | 1 | 0.38 | 0.83 | 8.38 |
| device | device | 1 | 0.52 | 0.44 | 8.12 |
| device | device | 8 | 0.57 | 0.22 | 8.32 |

On a CPU the XLA kernels share cores with the rest of the process, so there is nothing to gain there; the measurement that matters is on the GH200.

## DeltaAI GH200: device formatter and threaded compression (jobs 3301636 and 3301673, 2026-10-03)

All with `--compressor isal`, `--assembler device` except where noted. Single runs: warm second pass, seconds (`benchmarks/baseline_gh200_fmt_e*_*.json`; the `case` field of those
records does not list the options, the file name does, because the runner only recorded them from job 3301673 on). The job 3301528 row is the previous best.

| emax | formatter | compress threads | assemble | write | total |
|---|---|---|---|---|---|
| 12 | host | 1 (job 3301528) | 0.66 | 2.66 | 6.2 |
| 12 | host | 8 | 0.73 | 1.35 | 5.0 |
| 12 | device | 1 | 0.71 | 2.44 | 6.0 |
| 12 | device | 4 | 0.76 | 0.30 | 3.9 |
| 12 | device | 8 | 0.78 | 0.23 | 3.9 |
| 14 | host | 1 (job 3301528) | 1.98 | 8.35 | 15.9 |
| 14 | host | 8 | 2.16 | 4.35 | 11.9 |
| 14 | device | 1 | 2.06 | 7.91 | 15.3 |
| 14 | device | 4 | 2.27 | 0.65 | 8.29 |
| 14 | device | 8 | 2.32 | 0.56 | 8.09 |

- **The two changes need each other.** Threaded compression alone leaves the host formatter as the floor (write 8.35 -> 4.35 s); the device formatter alone leaves the single compression thread
  as the floor (7.91 s). Together `write` falls to 0.56 s and assemble plus write from 10.3 to 2.9 s. 4 threads are as good as 8 (16 cores were allocated).
- An emax 14 single run takes 8.1 s now: 25.4 s with numpy, host formatting and gzip, 17.7 s with isal. GPU peak 10.3 GB, host RSS 2.9 GB, file 922 MB.
- The largest stage of a single run is now `lab_coeffs` (about 3.5 s, host NumPy), then `assemble` (2.3 s, mostly host-side index preparation) and `lab_contract` (1.7 s).

Scans, 6 points (hw 12/16/20 x lambda 1.8/2.0) at batch 6 and 12 points (hw 12..22 step 2 x lambda 1.8/2.0) at batch 12, emax 14, `--formatter device --compress-threads 8`, seconds
(`baseline_gh200_fmt_scan*.json`). A first scan in job 3301636 ran with the wrong options because the runner did not pass them to `generate_scan`; its record was discarded and the runner fixed.

| points | batch | per point | compute (project+srg+lab) | assemble | write | host RSS | GPU peak |
|---|---|---|---|---|---|---|---|
| 6 | 6 | 4.81 | 1.87 | 2.34 | 0.58 | 3.9 GB | 32 GB |
| 12 | 12 | 4.58 | 1.65 | 2.34 | 0.58 | 5.2 GB | 57 GB |

- A full emax 14 scan point costs 4.6-4.8 s, against 25 s for a separate gzip/numpy run at the start of this work (5x) and 8.1 s for the best single run.
- Batch 12 fits easily (57 GB of the GPU, 5 GB of host memory) and is only 5% cheaper per point than batch 6: the per-point work is now dominated by `assemble` (host index preparation, not shared) and
  `lab_contract`; only `lab_coeffs` is shared.
