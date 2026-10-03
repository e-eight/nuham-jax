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
All lab matrices of a batch stay in memory until its files are written, so `batch_size` trades speed for memory (about 2 GB per point at
emax 14). Gate: every scan file equals the separate single run, `tests/test_scan.py` (several batch sizes and flow groupings, bare and
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
  entry (flow cost per set at hw 16: 1.18 s for lambda 2.4, 3.08 s for lambda 1.6). So `flow_sets` defaults to 1 and the speed-up above comes
  from coefficients and contraction only.
- Per-point cost is nearly flat in batch size on CPU, as expected: there is little spare parallelism to fill. The open question for DeltaAI
  is whether a GPU gains from `flow_sets` > 1 and from larger `batch_size`; the benchmark has both knobs (`--flow-sets`, `--batch-sizes`).
