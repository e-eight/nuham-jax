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

## Jobs and budget

- Ask the user before every job.
- Account `bhhk-dtai-gh` only. On 2026-10-06, 113 of 618 hours were left; the fair-share factor was 0.14 and jobs waited 2-3 h. Phase 4 used about 2 GPU-hours, the 3N verification and production jobs of 5-6 October about 1.2 (billing 1.08 per hour).
- 3N jobs (`scripts/deltaai/`: `common.sh`, `verify_a..d.sbatch`, `production.sbatch`, `lnl.sbatch`, `imsrg_*.sbatch`): 32 CPUs, `--mem=120g` (billing counts requested memory: 180g cost 1623 against 1082 per hour, and Slurm capped the allocation at 113 GB anyway), `OMP_NUM_THREADS=1`, 3 h limit. The script aborts if the worker count differs from the allocated CPUs; the V3 cache resumes a run that stops. 72 CPUs and `OMP_NUM_THREADS=8` were not tried.
- imsrg++ runs here in Apptainer (aarch64 build, `scripts/deltaai/imsrg.def`; see `docs/imsrg-check.md`).

Measurements of the runs here: `docs/benchmarks.md` (NN), `docs/study-local-3nf.md`, `docs/study-3n-flow.md`, `docs/study-me3j.md` (3N), `docs/imsrg-check.md` (imsrg++).
