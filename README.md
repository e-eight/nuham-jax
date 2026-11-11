# nuham

A from-scratch [JAX](https://github.com/jax-ml/jax) package that produces nuclear
Hamiltonian matrix elements for in-medium and no-core many-body codes. It gives the
same physics output as [NuHamil](https://github.com/Takayuki-Miyagi/NuHamil-public),
but it is not a translation of the Fortran code. NuHamil is used only as a numerical
oracle to check results.

## What it computes

- **NN:** the chiral EM500 interaction, from NuHamil's momentum-space files, to
  oscillator-basis lab-frame matrix elements. Steps: HO projection, Coulomb for pp,
  SRG evolution in the full relative space, Talmi-Moshinsky transform, `me2j` output
  that [imsrg++](https://github.com/ragnarstroberg/imsrg) reads.
- **3N:** the local N2LO three-nucleon force, SRG-evolved in the Jacobi basis,
  transformed to the lab frame and written as `me3j`. Optional conversion of the
  oscillator frequency (`--hw` to `--hw-target`), as NuHamil's example script does.
- **Parameter scans** over NN settings (`nuham.nn.scan`).

The code is float64, runs on CPU for development and small cases, and runs on GPUs
(tested on H200 and GH200) for production sizes. Nothing assumes a device. For
reference, an emax 14 NN file takes about 8 s on one GH200, and the production 3N
file (emax 14, e2max 28, e3max 16, 6.4e9 values) took 30 minutes on one GH200.

Validation: NN output matches NuHamil `me2j` files to about 1e-6 MeV, 3N output
matches NuHamil to 6e-6 on small files, and imsrg++ reproduces the NuHamil-based
results. Details are in `docs/` and `PLAN.md`.

## Install

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12 or newer.

```bash
uv sync                  # core dependencies and the dev group
uv sync --extra fast     # also install isal (faster gzip for the me2j writer)
```

On a GPU cluster, install the CUDA build of JAX in the environment first, then add
`isal` with `uv pip install isal`. A plain `uv sync` there would remove the CUDA
packages.

## Input data

The only external input is NuHamil's momentum-space interaction files (about 8 MB
each). `nuham.fetch` finds them in `$NUHAMIL_PUBLIC/input_nn_files`, in a sibling
`../NuHamil-public`, or in the cache `~/.cache/nuham/data`. To download a verified
copy from the pinned NuHamil commit:

```bash
uv run python -m nuham.fetch              # EM500, the default
uv run python -m nuham.fetch --help
```

## Usage

```bash
# NN matrix elements (me2j), hw 16 MeV, SRG lambda 2.0 fm^-1
uv run python -m nuham.nn.generate N3LO_EM500.bin out.me2j.gz \
    --hw 16 --lam 2.0 --emax 8 --e2max 16

# 3N matrix elements (me3j) with the local N2LO force
uv run python -m nuham.nnn.generate N3LO_EM500.bin out.me3j.gz \
    --hw 16 --lam 2.0 --emax 4 --e2max 8 --e3max 4 --ramp 4

# parameter scan
uv run python -m nuham.nn.scan --help
```

Pass `--help` to any of these for all options, including output writer settings
(`--assembler`, `--formatter`, `--compress-threads`, `--compressor`) that default by
backend. Job scripts for the DeltaAI cluster are in `scripts/deltaai/`.

## Development

```bash
uv run pytest            # tests; those needing NuHamil data skip if it is absent
uv run ruff format .
uv run ruff check .
uv run ty check
```

`tests/synthetic.py` builds a synthetic interaction, so the whole NN pipeline is
tested without any NuHamil data. `NUHAM_FETCH=1` lets the tests download the real
input files.

## Layout

| Path | Content |
| --- | --- |
| `src/nuham/` | shared infrastructure (`constants`, `runtime`, `timing`, `fetch`) |
| `src/nuham/angmom/`, `basis/` | angular-momentum coefficients and HO basis |
| `src/nuham/kernels/` | pure JAX kernels, with no I/O or global state |
| `src/nuham/nn/`, `nnn/` | NN and 3N pipelines and their command-line tools |
| `docs/` | physics spec, studies, benchmark baselines, cluster run notes |
| `benchmarks/` | benchmark runner and recorded baselines |
| `refs/` | throwaway NuHamil oracle scripts and reference outputs (not part of the package) |
| `PLAN.md` | phases, decisions and open gaps |
| `CLAUDE.md` | conventions and physics pitfalls for contributors |

## Documentation

- `docs/physics-spec.md` is the summary of the physics; `docs/spec-*.md` give sources and checks.
- `docs/benchmarks.md` has timing baselines; `docs/deltaai-run.md` and `docs/icrn-run.md` cover access, setup and job rules for the clusters.
- `docs/imsrg-check.md` describes the check against imsrg++.

## Licence

GPL-3.0-only. See `LICENSE`.
