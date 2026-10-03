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
