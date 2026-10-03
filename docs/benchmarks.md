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
