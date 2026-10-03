# Running on ICRN (NCSA JupyterHub, shared H200)

First GPU run of the package, 2026-10-03. ICRN is a JupyterHub with a per-user container; this note records what was
needed and what was measured.

## The machine (as seen from the container)

- 1 shared NVIDIA H200 NVL (143 GB; ~12 GB used by others during the runs), driver 570 / CUDA 12.8.
- cgroup limits: **10 CPUs** (the container shows 60 cores) and **32 GiB RAM**. `/tmp` is local disk (173 GB free).
- Image: `hub.ncsa.illinois.edu/icrn/pytorch:lab-4.6.1`, conda base Python 3.13, no `uv` (installed with `pip install --user uv`).
  GitHub and PyPI are reachable. `gh` is at `~/.pixi/bin/gh` and is logged in, but configured for SSH: clone over HTTPS with
  `git -c credential.helper= -c credential.helper='!gh auth git-credential' clone https://github.com/e-eight/nuham-jax.git`.

## Setup used

```bash
cd /tmp && git ... clone https://github.com/e-eight/nuham-jax.git      # see above
cd nuham-jax
export UV_PROJECT_ENVIRONMENT=/tmp/nuham-venv UV_LINK_MODE=copy
uv sync
uv pip install --python /tmp/nuham-venv/bin/python "jax[cuda12]==0.11.2"   # same version as uv.lock
export XDG_CACHE_HOME=/tmp/nuham-cache
/tmp/nuham-venv/bin/python -m nuham.fetch                                # EM500 input, checksum verified
/tmp/nuham-venv/bin/python -m nuham.generate $XDG_CACHE_HOME/nuham/data/N3LO_EM500_kmax8_N100_Jmax8.bin out.me2j.gz \
    --hw 16 --lam 2.0 --emax 14 --e2max 28 --bracket-cache $XDG_CACHE_HOME/brackets
```

The environment is 5.6 GB, so it goes in `/tmp`, not the home directory. Run long jobs detached (`setsid nohup ... &`) and poll a log.

## Results

- Tests (`NUHAM_FETCH=1`, no NuHamil checkout): **566 passed in 121 s** on the GPU backend (float64).
- Contraction kernel (`congruence_sum`, float64): 23-52 TFLOP/s on the H200 versus 0.084-0.095 TFLOP/s through XLA on the local CPU.
- Full generation, SRG lambda 2.0, hw 16, e2max = 2 emax:

| emax | first run (builds brackets, compiles) | warm caches, profiled | values | file | peak host RSS | peak GPU |
|---|---|---|---|---|---|---|
| 10 | 71.6 s | 14.8 s (unprofiled) | 1.27e7 | 71 MB | 5.1-6.3 GB | 0.1-0.4 GB |
| 12 | 57.9 s | 33.9 s | 5.07e7 | 278 MB | 6.5-7.5 GB | 0.4 GB |
| 14 | 120.8 s | 83.7 s | 1.68e8 | 914 MB | 10.7-11.7 GB | 0.8 GB |

  For comparison, the local CPU (16 cores, warm caches, unprofiled) took 18.5 s at emax 10 and 48 s at emax 12.
  The "first run" numbers include building the Moshinsky tables for the new N range and compiling kernels for new shapes.
  The profiled numbers carry profiler overhead; an unprofiled warm emax-14 run was not timed.
- Correctness: the emax-10 file generated on the GPU and the one generated on the local CPU have the same 12,741,400 values;
  the maximum difference is 0.0 at the file's 8-decimal precision.

## Where the time goes at emax 14 (warm, profiled, 83 s)

File writing and formatting ~41 s (49%, of which text formatting 14 s, tuple assembly 7.5 s, the rest waiting on the gzip thread);
lab matrices ~33 s (39%: contraction function 27.6 s, mostly host-side padding/gathers in NumPy (`_select` 4.8 s) and waiting on
the device 8 s; coefficient build 4.8 s); SRG flow and projection ~10 s. The GPU is no longer the bottleneck: the run is bound by
host-side Python/NumPy work and the 10-CPU quota.

## Access method (for repeating this from a terminal session)

The hub token in `~/.config/icrn/token` only has `access:servers` for the user's own *running* server (it cannot start one and
cannot list servers: the hub API shows none, while `/user/<name>/api/status` answers). A shell is reachable through the Jupyter
terminal API (`POST /user/<name>/api/terminals`, then the websocket `/user/<name>/terminals/websocket/<id>` with the token in the
`Authorization: token ...` header). Files go through the contents API, which refuses hidden directories and only serves paths under
the home directory. Do not print the token.
