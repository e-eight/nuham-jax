# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch JAX package (`nuham`) that produces the same physics output as
[NuHamil](https://github.com/Takayuki-Miyagi/NuHamil-public) (NN and later 3N matrix elements for imsrg++).
It is not a translation of the Fortran code; NuHamil is a numerical oracle only.
Scope now: EM500 NN, Hamiltonian only, `me2j` output readable by imsrg++. 3N (local N2LO) comes later.
Develop and test on this CPU-only machine (float64, small cases); large runs go to a GPU cluster (Delta),
so nothing may assume a device.

**Status:** Phases 0-3 are done: spec and oracle; HO functions, projection, Racah/Moshinsky; Coulomb and SRG
flow (`nuham.relative_nn.evolve_relative_nn`); lab transform and `me2j` writer (`nuham.generate.generate_me2j`,
CLI `uv run python -m nuham.generate --help`; it enables JAX's persistent compile cache by default). Phase 4 (scale and GPU) is started (fast Moshinsky tables, JAX contraction, threaded writer and compile cache done; vectorized 9j/6j done; a GPU run and an emax 14 timing not yet); do not start a
phase unless the user asks.
`PLAN.md` holds the phases and decisions made so far.

## Conventions (set by the user)

- Dependencies: `uv` (`uv add`, `uv sync`, `uv run ...`). Do not use bare `pip`.
- Format and lint everything with `ruff` (`uv run ruff format`, `uv run ruff check`). Type checking and LSP: `ty` (`uv run ty check`).
- Testing: `pytest` (run through `uv run pytest`).
- Docstrings on every module, class and function. Type annotations everywhere.
- Comment only to explain a design choice, not what the code does.
- Commands: `uv sync`; `uv run pytest` (all), `uv run pytest tests/test_ho.py::test_name` (one); `uv run ruff format .`; `uv run ruff check .`; `uv run ty check`. `refs/` (throwaway oracle scripts) is excluded from ruff and ty.
- Tests that need NuHamil input files look in `../NuHamil-public` (override with `NUHAMIL_PUBLIC`) and in `refs/work`, and skip if absent. `tests/data/moshinsky_golden.json` holds brackets generated from the oracle. The package and its tests must pass without `refs/` (tests that need it skip).

## Design rules

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
- The lab transform (`lab_transform.py`) builds T-coefficients per lab channel (J, parity) with host NumPy (array Racah symbols in `angmom/racah_array.py`, summed over lam = J-1..J+1 only) and caches Moshinsky bracket matrices per (N, lam) (`MoshinskyTable`, optional disk cache; hw-independent, reuse it). The tables come from a matrix exponential of a rotation generator (~15 s for all N <= 28). The explicit Kamuntavicius sum (a transcription of NuHamil's `gmosh`) is deliberately NOT in `src/`: it lives in `refs/moshinsky_reference.py` as a slow check, used only by `tests/test_reference_moshinsky.py`, which skips if `refs/` is absent. Keep NuHamil-derived transcriptions in `refs/`, not in the package. Do not use `jax.scipy.linalg.expm` for these matrices (3e-10 error); use `kernels/rotation.py`. The contraction `sum_blocks T^T V T` is NumPy for now and is the largest remaining cost (see `docs/study-moshinsky-rotation.md`).
- `me2j` tuples are enumerated in the file's order (`tuple_chunks`), identical to imsrg++'s reader loop; lab matrices of all (J, parity) stay in memory (about 2 GB at emax=14).
- The SRG flow is independent of emax and always uses N_flow=100 by default. A smaller `n_flow` is a deviation from NuHamil (`docs/study-srg-truncation.md`); do not make it the default.
- Coulomb in pp is on by default and is evolved with the flow. Relative J > 8 is zero.
- Use NuHamil's constants (hc=197.32705, m_p=938.27231, m_n=939.56563), not SciPy's. One pn reduced mass is used for pp, nn and pn.
- The `me2j` file holds V only (no kinetic or Hcm). Four values per tuple in order: T=0, nn, pn T=1, pp; Darmstadt un-normalised convention; forbidden entries are 0.0. imsrg++ builds its own Trel.
- Test tolerances (absolute, MeV): full chain vs `me2j` 1e-6; projection vs relative cache 1e-5; lab transform 1e-8; SRG eigenvalue invariance 1e-8. NuHamil's own VODE error (rtol 1e-8) limits agreement for the 3S1-3D1 channel to about 3e-7 even with a converged integrator.
- The 7-digit relative cache file and the `.bin` give results that differ by ~2.5e-6; check which source a reference run used (`log_*.txt`).
