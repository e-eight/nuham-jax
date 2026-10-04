# Running on DeltaAI (NCSA, NVIDIA GH200)

First runs, 2026-10-03, on one GH200 of a shared node (`gh039`, `gh013`), 16 cores and 64 GB requested, dedicated GPU.
Total use: three jobs, about 15 GPU-minutes (~0.3 GPU-hours at the partition charge factors; the account balance had not updated
when checked).

## Access and layout

- Login `dtai-login.delta.ncsa.illinois.edu` needs NCSA password + Duo and SSH keys are disabled, so a person has to authenticate.
  Commands were run over an already-authenticated connection (the user's Emacs TRAMP control socket,
  `ssh -o ControlMaster=no -o ControlPath=~/.cache/emacs/tramp.%C -l soham dtai-login...`).
- Only the account `bhhk-dtai-gh` was used. Slurm: `--partition=ghx4` (charge 1.0) or `ghx4-interactive` (2 h, charge 2.0),
  `--gpus-per-node=1 --cpus-per-task=16 --mem=64g --constraint=projects` (jobs must declare a `/projects` dependency).
- `/work/nvme/bhhk` was almost full (988 of 1000 GB), so everything lives in `/projects/bhhk/soham/nuham`
  (venv 5.5 GB, repo, data file, caches 1.5 GB, job scripts and logs). Login node has internet; the environment was built there with
  `UV_PROJECT_ENVIRONMENT=... uv sync` then `uv pip install "jax[cuda12]==0.11.2"` (aarch64 wheels exist).
- Node: GH200 120GB (97.9 GB usable), driver 595.71, Neoverse-V2 host, 4 sockets x 72 cores, 478 GB RAM.

## Results

- Tests: 570 passed in 131 s on the GPU backend.
- Contraction kernel, float64: 39-49 TFLOP/s (ICRN H200: 23-52).
- Copy paths for a 152 MB float64 result:

| path | time | rate |
|---|---|---|
| host fill of a fresh `np.empty` buffer (page faults) | 8.9 ms | 17 GB/s |
| host fill of an already-touched buffer | 3.5 ms | 43.5 GB/s |
| device -> host, `np.asarray` into a new buffer (pageable) | 16.8 ms | 9.0 GB/s |
| device -> host, `device_put` to the `pinned_host` memory kind | 1.0 ms | 145.7 GB/s |
| host -> device, `jnp.asarray` of a touched array | 8.6 ms | 17.6 GB/s |

  The pageable device-to-host copy runs at about the speed of a fresh-buffer host fill, so it is limited by first-touch page faults,
  not by the link. Pinned memory is ~16x faster and the pinned array is then readable with no further copy.
- Largest emax-14 channel (J=6, parity +, 4234 pairs, 7133 rows), warm: coefficient build 0.16 s, upload 0.01 s,
  contraction for all three isospin cases 0.07 s (ICRN H200: 0.26 / 0.02 / 0.24 s).
- Full NN generation, SRG lambda 2.0, hw 16, e2max = 2 emax:

| emax | first run | warm run | values | file | peak host RSS | peak GPU |
|---|---|---|---|---|---|---|
| 10 | 62.5 s | 14.5 s | 1.27e7 | 71 MB | 1.6-1.8 GB | 0.2-0.3 GB |
| 12 | 65.0 s | 25.3 s | 5.07e7 | 278 MB | 3.2-3.5 GB | 0.6-0.7 GB |
| 14 | 114.5 s | 53.7 s | 1.68e8 | 914 MB | 7.2 GB | 1.3-1.4 GB |

  Warm emax 14 on ICRN's shared H200 was 75-80 s. "First run" includes building the bracket tables and compiling kernels
  (compile cache starts empty on a new machine).

## Findings that change the plan

1. **The bracket disk cache is too big and slow.** It stores full zero-padded matrices: 1.52 GB for N <= 28 (421 files). Loading it
   from `/projects` took 38 s in a fresh process, longer than regenerating the tables (about 15 s). That inflated an earlier profiled run
   (`t_matrix` 81 s instead of 3 s). Store only the valid block (~0.15 GB) or skip the cache on slow file systems.
2. **Use pinned host memory for the result copy** on GH200 (16x faster than the pageable copy), and avoid allocating a fresh host
   buffer for each result.
3. **File writing is now the main cost.** With tables loaded, warm emax 14 is 43-57 s; in the profile the text formatting expression is
   24 s, gzip 19.6 s (overlapped in a thread), tuple assembly 8 s. Everything GPU-friendly (contraction 0.07 s per large channel,
   the SRG flow) is a small share.
4. Run-to-run variation on a shared node is real (profiled 43.5 s vs unprofiled 56.9 s in one job); compare by alternating runs.


## Phase 4a results (2026-10-03, job on `gh136`, same node type)

Changes: `MoshinskyTable` keeps and stores only the compact valid block of each bracket matrix (`N<total>_lam<lam>_c.npy`), and
`lab_matrix` copies its result to the host through `kernels.transfer.to_host` (pinned host memory on accelerator backends,
`NUHAM_PINNED_COPY=0/1` to override). Output files are unchanged (max difference 0.0 at emax 10 and 12).

| gate | before | after |
|---|---|---|
| bracket cache on disk, N <= 28 | 1.55 GB, 421 files | 0.147 GB, 435 files |
| load the whole cache in a fresh process, from `/projects` | 38.3 s | 0.5 s |
| peak host memory while loading | ~1.5 GB resident tables | 0.19 GB |
| cold build of the compact cache (includes compiling the matrix exponential for ~20 padded shapes; 8.3 s locally on the CPU) | n/a | 38.2 s |
| largest emax-14 channel, `lab_matrix` for tz = -1, 0, +1 | 67.4 ms (pageable copy) | 48.4 ms (pinned copy) |
| full emax-14 run, warm, alternating | 54.2, 53.8 s (pageable), host RSS 5.9 GB | 55.5, 53.3, 53.2 s (pinned), host RSS 6.4 GB |

Pageable and pinned results are bitwise equal, and the pinned views stay valid after the device arrays are freed. All 581 tests
pass on the GH200.

Reading: the cache fix is a clear win. The pinned copy is correct and 16x faster per copy, but the copies are a small part of a
run, so the end-to-end difference is below noise and it costs ~0.5 GB of pinned host memory. It stays on by default for accelerator
backends (it will matter more when the matrices get larger, as in 3N); `NUHAM_PINNED_COPY=0` turns it off. The same A/B on ICRN's
PCIe H200 was not run because that server had been shut down (the token cannot start it).

## Phase 5.2.5 (2026-10-04, job 3308734 on `gh136`, 8 min wall, 32 CPUs, 96 GB)

Local N2LO 3N force on one GH200: 48 tests pass on the GPU backend (the worker processes of the coefficient stage are forked after CUDA is initialised and run fine), and the
force at N3max 16-40 for J=1/2+ and J=5/2+ (T=1/2) takes 2-13 s and 5-119 s of assembly respectively; full table, stage times and memory in `docs/study-local-3nf.md`.
Job script `job18.sbatch` (same pattern as before; `REPO` is the checkout `/work/hdd/bhhk/soham/nuham-repo`, pulled to `3946614` first), records in `benchmarks/baseline_gh200_3n_force_*.json`.
