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
