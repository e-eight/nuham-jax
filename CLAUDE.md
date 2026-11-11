# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch JAX package (`nuham`) that produces the same physics output as
[NuHamil](https://github.com/Takayuki-Miyagi/NuHamil-public): NN (`me2j`) and 3N (`me3j`) matrix elements for imsrg++.
It is not a translation of the Fortran code; NuHamil is a numerical oracle only.
Develop and test on this CPU-only machine (float64, small cases); large runs go to a GPU cluster (DeltaAI),
so nothing may assume a device. Usage and install: `README.md`.

## Status and PLAN.md

`PLAN.md` is the record of what has been done, what was decided and what is open (phases, gap tables). Read it before
starting work. Do not start a phase or a gap step unless the user asks.

After completing an implementation step, update `PLAN.md` in the same change:
- Mark the row or item `DONE <date>` and give the outcome in one line (the key number against its gate, e.g.
  "max diff 4.1e-6 vs NuHamil, 456 320 values"), plus a link to where the details are.
- Put the details (tables, job numbers, timings, how a problem was found) in the matching doc under `docs/`
  (`study-*`, `spec-*`, `benchmarks.md`, `imsrg-check.md`, cluster notes), not in `PLAN.md` and not here.
- Record new decisions made with the user, with the date, and new open questions or gaps.
- Update this file only if a rule or a pitfall changed.

Large output files live on DeltaAI under `/projects/bhhk/soham/nuham/out`, not in the repo. Code reaches the clusters by
push from here and pull there, never by copying. Ask before every cluster job (budget and rules: `docs/deltaai-run.md`).

## Conventions (set by the user)

- Dependencies: `uv` (`uv add`, `uv sync`, `uv run ...`). Do not use bare `pip`. Optional extra `fast` (`isal`, ISA-L's igzip): `uv sync --extra fast`; the writers use it when installed (`WriterOptions.compressor="auto"`; standard gzip, but not byte-identical to the `gzip` module's output). On the clusters install it with `uv pip install isal` into the venv (a plain `uv sync` there would strip the CUDA packages).
- Format and lint everything with `ruff` (`uv run ruff format`, `uv run ruff check`). Type checking and LSP: `ty` (`uv run ty check`).
- Testing: `pytest` (run through `uv run pytest`).
- Docstrings on every module, class and function. Type annotations everywhere. Implementation facts (cache sizes, padding, transfer paths) belong in the docstrings of the code they describe, not in Markdown.
- Comment only to explain a design choice, not what the code does.
- Before implementing code that serves a purpose the package may already serve (locating or loading data, option handling, timing, formatting, batching, test fixtures), look up the existing ability first: grep `src/nuham`, check `tests/conftest.py`, `docs/` and `PLAN.md`. Reuse or generalise it; do not add a parallel path next to it, and if the existing code is too narrow, widen it in one place. Say in the change what you reused.
- Commands: `uv sync`; `uv run pytest` (all), `uv run pytest tests/test_ho.py::test_name` (one); `uv run ruff format .`; `uv run ruff check .`; `uv run ty check`. `refs/` (throwaway oracle scripts) is excluded from ruff and ty.
- Test data: the only external inputs are NuHamil's momentum-space interaction files; `nuham.fetch` owns where they are and how to get them (see its docstrings and `README.md`). The `interaction_bin` fixture in `tests/conftest.py` turns "not found" into a skip and downloads when `NUHAM_FETCH=1`; `nn_bin` is `interaction_bin("N3LO_EM500")`. Reference outputs are committed in `refs/work`.
- `tests/synthetic.py` builds a synthetic charge-independent interaction in NuHamil's file format; `tests/test_synthetic_pipeline.py` runs the whole pipeline on it with no NuHamil data, using exact identities (V_rel = 1 transforms to the identity; charge-independent V gives equal nn, pn T=1 and pp me2j columns; projecting the kinetic energy gives the HO T_rel). Deliberately breaking the pair normalisation, the exchange sign or the kinetic phase makes these fail. `tests/data/moshinsky_golden.json` holds brackets generated from the oracle. The package and its tests must pass without `refs/` (tests that need it skip).
- Provenance: code transcribed from NuHamil's Fortran (e.g. the explicit Moshinsky sum, `refs/moshinsky_reference.py`) lives in `refs/`, never in `src/`.

## Design rules

- Layout: see `PLAN.md`. Shared code at the top of `src/nuham` and in `angmom/`, `basis/`, `kernels/`; sector code in `nn/` and `nnn/` with their pure kernels in `kernels/nn/` and `kernels/nnn/`. Put new code in the sector it serves; move a module to the shared level only when both sectors use it.
- Kernels (`nuham/kernels/`) are pure JAX functions on explicit arrays: no classes, no I/O, no global state. This keeps a later Kokkos port easy. Channel loops, chunking and I/O live in drivers.
- Static shapes: bucket and pad channel dimensions so `jit` does not recompile per channel.
- Angular-momentum coefficients (6j/9j, Moshinsky brackets) are built on the host in NumPy with caching. Move them to JAX only if profiling shows a bottleneck.
- The interaction is a plugin ("give partial-wave momentum-space matrix elements for channel (J, parity, S, Tz)"); EM500-from-file is the first one.
- HO radial functions: use the normalized three-term recurrence in n (as in `../pre-quantum-projects/csm/csm/basis/harmonic_oscillator.py`) with closed-form coefficients as a `lax.scan`, not scipy/sympy.
- Output files are written as a stream in file order. The lab matrices of all channels may be held in memory (about 4 GB at NN emax 14, 23 GiB packed for the 3N production file); nothing larger.

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

## Pitfalls

Physics conventions and tolerances in full: `docs/physics-spec.md` (NN) and `docs/spec-*.md`.

NN:
- Use NuHamil's constants (`nuham.constants`), not SciPy's. One pn reduced mass is used for pp, nn and pn.
- Coulomb is added to pp before the SRG flow and evolved with it (on by default). Relative J > 8 is zero.
- The SRG flow runs in the full N_flow=100 relative space regardless of emax. A smaller `n_flow` is a deviation from NuHamil (`docs/study-srg-truncation.md`); do not make it the default.
- Do not use `jax.scipy.linalg.expm` for Moshinsky matrices (3e-10 error); use `kernels/rotation.py`.
- Agreement with NuHamil `me2j` is 1e-6 MeV, not tighter: NuHamil's VODE error limits 3S1-3D1 to about 3e-7, and reference runs made from the 7-digit relative cache differ from the `.bin` by ~2.5e-6 (check `log_*.txt`).

3N:
- imsrg++ must be given `file3e1max`, `file3e2max` and `file3e3max` matching the `me3j` file; wrong values misalign the stream silently.
- The antisymmetric Jacobi basis is a gauge choice that depends on the platform: build the Jacobi and V3 caches on one machine, and compare gauge-invariant quantities or final `me3j` values with NuHamil.
- When `tests/test_v3_reference.py` fails because a small V3 moved, bump `CACHE_VERSION` and regenerate its reference.
- For comparison with NuHamil-based results, make the Jacobi operator at hw 30 and convert to hw 16 (`--hw 30 --hw-target 16`), as NuHamil's example script does (`docs/spec-5e-hw-conversion.md`).
