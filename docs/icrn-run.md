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
/tmp/nuham-venv/bin/python -m nuham.nn.generate $XDG_CACHE_HOME/nuham/data/N3LO_EM500_kmax8_N100_Jmax8.bin out.me2j.gz \
    --hw 16 --lam 2.0 --emax 14 --e2max 28 --bracket-cache $XDG_CACHE_HOME/brackets
```

The environment is 5.6 GB, so it goes in `/tmp`, not the home directory. Run long jobs detached (`setsid nohup ... &`) and poll a log.

## Access method (for repeating this from a terminal session)

The hub token in `~/.config/icrn/token` only has `access:servers` for the user's own *running* server (it cannot start one and
cannot list servers: the hub API shows none, while `/user/<name>/api/status` answers). A shell is reachable through the Jupyter
terminal API (`POST /user/<name>/api/terminals`, then the websocket `/user/<name>/terminals/websocket/<id>` with the token in the
`Authorization: token ...` header). Files go through the contents API, which refuses hidden directories and only serves paths under
the home directory. Do not print the token.


## Shared-GPU memory (important)

By default JAX reserves ~75% of the GPU's memory when it starts. On the shared H200 a process that did a 1000x1000 matrix
product raised the used memory from 12 GB to 138 GB of 143 GB; with `XLA_PYTHON_CLIENT_PREALLOCATE=false` it stayed at the
baseline. The first runs on 2026-10-03 (tests, benchmarks, the emax 10-14 series) used the default, so they probably held most of
the card's free memory for those minutes, which can make other users' jobs fail with out-of-memory errors. `nuham` now sets
`XLA_PYTHON_CLIENT_PREALLOCATE=false` on import unless the user has set it (`tests/test_package_defaults.py`). With it, the highest
total GPU memory seen during the emax 12 and 14 runs was 62.3 GB while others used 59.5 GB, i.e. under 3 GB of ours.

Measurements of the runs here: `docs/benchmarks.md`, section "Before the harness".
