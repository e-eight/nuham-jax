# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch JAX package (`nuham`) that produces the same physics output as
[NuHamil](https://github.com/Takayuki-Miyagi/NuHamil-public) (NN and later 3N matrix elements for imsrg++).
It is not a translation of the Fortran code; NuHamil is a numerical oracle only.
Scope now: EM500 NN, Hamiltonian only, `me2j` output readable by imsrg++ (done), and the local N2LO 3N force with `me3j` output (done, production file made; validation with imsrg++ is next).
Develop and test on this CPU-only machine (float64, small cases); large runs go to a GPU cluster (Delta),
so nothing may assume a device.

**Status:** Phases 0-3 are done: spec and oracle; HO functions, projection, Racah/Moshinsky; Coulomb and SRG
flow (`nuham.nn.relative.evolve_relative_nn`); lab transform and `me2j` writer (`nuham.nn.generate.generate_me2j`,
CLI `uv run python -m nuham.nn.generate --help`; it enables JAX's persistent compile cache by default). Phase 4 (speed and GPU) is in progress: fast Moshinsky tables, JAX contraction, threaded writer, compile cache and vectorized 9j/6j are done, and runs on ICRN (H200) and DeltaAI (GH200) are recorded in `docs/`. 4a (compact bracket cache, pinned-host result copy), 4b (stage timers and `benchmarks/` runner; baselines in `docs/benchmarks.md`) and 4c (vectorized byte-identical text formatter; gzip kept as is, `python-isal` or parallel blocks only if it stays the bottleneck at scale) are done. 4c also has the device assembly/formatter and threaded compression for accelerators. These are chosen by one `nuham.runtime.WriterOptions` object (CLI flags `--assembler/--formatter/--compress-threads/--compressor/--gzip-level`); unset fields default by backend (CPU: numpy, host, 1 thread; accelerator: device, device, 4 threads), and an emax 14 run takes 8.1 s on the GH200. 4d (parameter scans, `nuham.nn.scan`) is done and measured on DeltaAI (emax 14: 4.6-4.8 s per point); the ICRN baseline is still to do, imsrg++ reads our files and reproduces the NuHamil-based 4He energies to ~1e-6 MeV (`docs/imsrg-check.md`, tools in `refs/imsrg/`); then 3N (Phase 5: 5.1-5.4 are done for the local N2LO force: reproduced against NuHamil and imsrg++ at 4/8/4, and the production file (14/28/16, 3NF400, hw 16, lambda 2.0, all relative J; 6.37e9 values, 31.5 GB) was made on one GH200 on 2026-10-06 in 30 minutes (DeltaAI job 3324394; it lives on DeltaAI under `/projects/bhhk/soham/nuham/out`, not in the repo). CLI `uv run python -m nuham.nnn.generate --help`, job scripts in `scripts/deltaai/`, details in `docs/study-me3j.md` and `PLAN.md`. Next is 5.5, the imsrg++ validation of that file; nothing has compared its values yet.) Code reaches the clusters by push from here and pull there, never by copying. Do not start a
phase unless the user asks.
`PLAN.md` holds the phases and decisions made so far.

## Conventions (set by the user)

- Dependencies: `uv` (`uv add`, `uv sync`, `uv run ...`). Do not use bare `pip`. Optional extra `fast` (`isal`, ISA-L's igzip): `uv sync --extra fast`. The `me2j` writer uses it when installed (`WriterOptions.compressor="auto"`): standard gzip, but not byte-identical to the `gzip` module's output, and 3.5x shorter write stage on the GH200. On the clusters install it with `uv pip install isal` into the venv (a plain `uv sync` there would strip the CUDA packages).
- Format and lint everything with `ruff` (`uv run ruff format`, `uv run ruff check`). Type checking and LSP: `ty` (`uv run ty check`).
- Testing: `pytest` (run through `uv run pytest`).
- Docstrings on every module, class and function. Type annotations everywhere.
- Comment only to explain a design choice, not what the code does.
- Before implementing code that serves a purpose the package may already serve (locating or loading data, option handling, timing, formatting, batching, test fixtures), look up the existing ability first: grep `src/nuham`, check `tests/conftest.py`, `docs/` and `PLAN.md`. Reuse or generalise it; do not add a parallel path next to it, and if the existing code is too narrow, widen it in one place. Say in the change what you reused.
- Commands: `uv sync`; `uv run pytest` (all), `uv run pytest tests/test_ho.py::test_name` (one); `uv run ruff format .`; `uv run ruff check .`; `uv run ty check`. `refs/` (throwaway oracle scripts) is excluded from ruff and ty.
- Test data: the only external inputs are NuHamil's momentum-space interaction files (about 8 MB each, data not code; 12 are listed with SHA-256 in `nuham.fetch`, EM500 is the default and most tests use it). `nuham.fetch` owns where they are and how to get them: `find_nn_file(name)` looks in `$NUHAMIL_PUBLIC/input_nn_files`, a sibling `../NuHamil-public`, then the cache `~/.cache/nuham/data`; `fetch_nn_file(name)` (also `uv run python -m nuham.fetch [name]`) downloads a verified copy from the pinned NuHamil commit. The `interaction_bin` fixture in `tests/conftest.py` only turns "not found" into a skip, and downloads when `NUHAM_FETCH=1`; `nn_bin` is `interaction_bin("N3LO_EM500")`. Tests needing a file skip if it is absent. Reference outputs are committed in `refs/work`.
- `tests/synthetic.py` builds a synthetic charge-independent interaction in NuHamil's file format; `tests/test_synthetic_pipeline.py` runs the whole pipeline on it with no NuHamil data, using exact identities (V_rel = 1 transforms to the identity; charge-independent V gives equal nn, pn T=1 and pp me2j columns; projecting the kinetic energy gives the HO T_rel). Deliberately breaking the pair normalisation, the exchange sign or the kinetic phase makes these fail. `tests/data/moshinsky_golden.json` holds brackets generated from the oracle. The package and its tests must pass without `refs/` (tests that need it skip).

## Design rules

- Layout (details in `PLAN.md`): shared infrastructure at the top of `src/nuham` (`constants`, `runtime`, `timing`, `fetch`) and in `angmom/`, `basis/` and `kernels/` (general kernels: `ho`, `rotation`, `srg`, `transfer`, `compile`: per-kernel compiler options, the large 3N GEMMs are compiled without XLA autotuning because it costs seconds per new shape); sector code in `nn/` (NN pipeline, CLIs `python -m nuham.nn.generate` and `nuham.nn.scan`) and `nnn/` (3N pipeline, `nnn/local_force/`), with their pure kernels in `kernels/nn/` and `kernels/nnn/`. Put new code in the sector it serves; move a module to the shared level only when both sectors use it.
- Kernels (`nuham/kernels/`) are pure JAX functions on explicit arrays: no classes, no I/O, no global state. This keeps a later Kokkos port easy. Channel loops, chunking and I/O live in drivers.
- Static shapes: bucket and pad channel dimensions so `jit` does not recompile per channel.
- Angular-momentum coefficients (6j/9j, Moshinsky brackets) are built on the host in NumPy with caching. Move them to JAX only if profiling shows a bottleneck.
- The interaction is a plugin ("give partial-wave momentum-space matrix elements for channel (J, parity, S, Tz)"); EM500-from-file is the first one.
- HO radial functions: use the normalized three-term recurrence in n (as in `../pre-quantum-projects/csm/csm/basis/harmonic_oscillator.py`) with closed-form coefficients as a `lax.scan`, not scipy/sympy.
- Output is streamed channel by channel, never held whole in RAM.

## Oracle workflow (works today)

NuHamil is not built on the host (no GSL/HDF5 dev headers, no sudo). Use the existing podman image `nuhamil`:

```bash
# in refs/work, with an Input_*.dat namelist
podman run --rm -v "$PWD:/work:Z" -w /work nuhamil:latest NuHamil.exe Input_x.dat
```

Input files must set `file_name_nn` explicitly (see `refs/work/Input_a.dat`); the NN input is at
`/root/NuHamil-public/input_nn_files/` inside the image. An emax=4 run takes about 11 s.

`refs/oracle_numpy.py` is a slow throwaway numpy chain (not the package) that reproduces the reference `me2j` files to ~1e-8:
`python refs/oracle_numpy.py --cases srg20_hw16 --n 300` (env `ORACLE_SOLVER=vode` emulates NuHamil's integrator).
The local NuHamil checkout is `../NuHamil-public` (read only).

## Physics facts that are easy to get wrong

Full detail: `docs/physics-spec.md` (summary), `docs/spec-1..4` (sources and checks). Key points:

- Pipeline: momentum-space `.bin` -> HO relative basis (quadrature on the file's 100-point mesh) -> add Coulomb to pp -> SRG in the **full Nmax=100 relative space** -> `V = H - T_rel` -> cut to `2n+L <= e2max` -> Talmi-Moshinsky to the lab frame -> `me2j`.
- The SRG kernel (`kernels/srg.py`) is an adaptive Dopri8 (Diffrax) flow on dense H with T kept as its diagonals; it takes ~2 s for all 54 channels at N_flow=100 on this CPU. Channels are zero-padded to a multiple of 16 and batched with `vmap`; padding does not change results.
- The lab transform (`nn/lab_transform.py`) builds T-coefficients per lab channel (J, parity) with host NumPy (array Racah symbols in `angmom/racah_array.py`, summed over lam = J-1..J+1 only) and caches Moshinsky bracket matrices per (N, lam) (`MoshinskyTable`; the cache keeps only the compact valid block of each matrix, 0.15 GB for N <= 28 instead of 1.5 GB, and `gather` fills in the zeros; hw-independent, reuse it). The tables come from a matrix exponential of a rotation generator (~15 s for all N <= 28). The explicit Kamuntavicius sum (a transcription of NuHamil's `gmosh`) is deliberately NOT in `src/`: it lives in `refs/moshinsky_reference.py` as a slow check, used only by `tests/test_reference_moshinsky.py`, which skips if `refs/` is absent. Keep NuHamil-derived transcriptions in `refs/`, not in the package. Do not use `jax.scipy.linalg.expm` for these matrices (3e-10 error); use `kernels/rotation.py`. The contraction `sum_blocks T^T V T` is a JAX kernel (`kernels/lab.py`): coefficients are uploaded once per channel, stacks are gathered on the device and accumulated there, so the result is copied back once (through `kernels/transfer.to_host`: pinned host memory on accelerator backends, `NUHAM_PINNED_COPY=0` to disable). `import nuham` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` (JAX otherwise grabs ~75% of a shared GPU; user setting wins). Remaining cost at emax 14 is mostly host side: file formatting/gzip and NumPy assembly (see `docs/icrn-run.md`).
- `me2j` tuples are enumerated in the file's order (`tuple_chunks`), identical to imsrg++'s reader loop; lab matrices of all (J, parity) stay in memory (about 4 GB per point at emax=14, measured).
- The SRG flow is independent of emax and always uses N_flow=100 by default. A smaller `n_flow` is a deviation from NuHamil (`docs/study-srg-truncation.md`); do not make it the default.
- Coulomb in pp is on by default and is evolved with the flow. Relative J > 8 is zero.
- Use NuHamil's constants (hc=197.32705, m_p=938.27231, m_n=939.56563), not SciPy's. One pn reduced mass is used for pp, nn and pn.
- The `me2j` file holds V only (no kinetic or Hcm). Four values per tuple in order: T=0, nn, pn T=1, pp; Darmstadt un-normalised convention; forbidden entries are 0.0. imsrg++ builds its own Trel.
- Test tolerances (absolute, MeV): full chain vs `me2j` 1e-6; projection vs relative cache 1e-5; lab transform 1e-8; SRG eigenvalue invariance 1e-8. NuHamil's own VODE error (rtol 1e-8) limits agreement for the 3S1-3D1 channel to about 3e-7 even with a converged integrator.
- The 7-digit relative cache file and the `.bin` give results that differ by ~2.5e-6; check which source a reference run used (`log_*.txt`).
