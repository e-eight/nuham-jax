# Study: Moshinsky brackets as a matrix exponential

Problem: the explicit Kamuntavicius sum costs 50-500 us per bracket and ~1.8e7 brackets are needed
at e2max=28 (a few CPU-hours, and the sum is not vectorizable).

## Result

For equal masses (any mass ratio d: theta = atan(sqrt(d))) the transformation between
(cm, rel) and (1, 2) oscillator states is a rotation of the two-oscillator Fock space. At fixed total
quanta N and coupled orbital momentum lam the bracket matrix is

    M[(ncm,lcm,nrel,lrel), (n1,l1,n2,l2)] = (-1)^(N+lam) * [exp(theta G) P]

- `G = -2 (S - S^T)`, `S = a1^dagger . a2` in the coupled basis: elements are
  `(-1)^(l1 + l2' + lam) {l1' l2' lam; l2 l1 1} <l1'||a^dagger||l1> <l2'||a||l2>` with the reduced elements
  of the 3D oscillator ladder operators (`l1 -> l1+1` with `sqrt((l+1)(n+l+3/2))`, `(n,l) -> (n+1,l-1)` with
  `sqrt(l(n+1))`, and the two lowering ones). `G` is real, antisymmetric, ~8 nonzeros per row, spectrum i*(integers, |k| <= N).
- `P` exchanges the cm and rel labels: row (ncm,lcm,nrel,lrel) <-> column (nrel,lrel,ncm,lcm). The sign `(-1)^(N+lam)`
  was fitted (all N <= 8) and equals the d -> 0 limit of the bracket.

The factor 2 in `G` and the sign were found by extracting the generator numerically from brackets at tiny d,
not derived; the match is the evidence.

## Evidence

- exp(theta G) P reproduces the Kamuntavicius brackets to 2e-14 for every N <= 8 and all lam, to 5e-14 for N <= 12
  (d = 1), and to 5e-15 for d = 2 (N = 6, lam = 2).
- Orthogonality of each matrix: 4e-14 up to N = 28 (the explicit sum: 5e-14 already at N = 12).
- All 421 matrices for N <= 28 (e2max = 28) take ~15 s in total on this CPU, versus an estimated few CPU-hours.
- The stretched state (0,6,0,6) -> (0,7,0,5) at N = 12, lam = 12 is exactly zero by symmetry
  (`(x^2 - y^2)^6` has no odd powers); the table reproduces it to <1e-13.

## Pitfall: `jax.scipy.linalg.expm`

For these matrices (norm ~ 9) `jax.scipy.linalg.expm` in float64 was off by 3e-10 (a Schur-based exponential and scipy agree to
5e-15), although its result is orthogonal to 1e-15, so an orthogonality check does not catch it. The package uses
`kernels/rotation.py`: scaling, a degree-20 Taylor series and squaring (matrix products only).

## Where it lives

`angmom/moshinsky_table.py`: `MoshinskyTable(cache_dir, d, method="rotation" | "bracket")`. The bracket method is kept as the
independent check (`tests/test_moshinsky_table.py`). The rotation form also covers the unequal-mass brackets needed for 3N
Jacobi transformations.

## Effect on the full pipeline (this machine, 16 cores, unprofiled, cold bracket cache)

| emax / e2max | before | after | note |
|---|---|---|---|
| 6 / 12 | ~8 s | 3.2 s | |
| 8 / 16 | ~60-90 s | 8.2 s | |
| 10 / 20 | not run | 41 s | |
| 12 / 24 | not run | 302 s, 2.4 GB peak | 5.07e7 values |

Remaining cost at emax=10 (profiled): block contraction `sum_b T_b^T V_b T_b` ~50%, T-coefficient build ~15%,
file writing (formatting + gzip) ~25%, brackets ~7%. The emax=14 time is not measured; extrapolating the
factor of ~7 per two units of emax suggests tens of minutes.

## Next steps (not done)

1. Contraction: the per-block rank updates touch the full `P x P` matrix (memory-bound). Regroup by (N, N')
   column blocks, which are dense GEMMs, and run them in JAX (also the GPU path).
2. T-coefficient 9j/6j factors: python loops; the 9j/6j depend only on a few quantum numbers and can be tabulated and
   applied with array operations.
3. Writer: gzip level and text formatting (~25% of the run).


## Update: contraction (done)

The per-block rank update of the full `P x P` matrix was replaced by `kernels/lab.congruence_sum`: blocks are
zero-padded into stacks of equal shape (sizes 4, 8, 16, ...; stacks capped at ~0.5 GB) and
`sum_b T_b^T V_b T_b` is two einsums, i.e. one dense GEMM of size `P x (n_blocks r) x P` plus small batched products.
The old routine is kept as `lab_matrix_reference` and tested against the new one in every channel at emax 4 (1e-11).

| Case (this CPU) | before | after |
|---|---|---|
| contraction, emax 10, J=4 P=- pn (P=1302) | 1.35 s | 0.23 s |
| contraction, emax 10, J=8 P=+ pn (P=1034) | 1.06 s | 0.10 s |
| full run emax 10 / e2max 20, cold | 41 s | 25 s (21 s with warm JAX cache) |
| full run emax 12 / e2max 24, cold | 302 s | 76 s, 2.8 GB peak |

Per-channel results agree to 2e-15. At emax=10 (profiled, 29 s) the remaining shares are roughly: file writing 11 s (gzip 7 s,
formatting), channel matrices 16 s (T-coefficient build 7 s, contraction 8 s including JAX compiles of new shapes),
brackets ~3 s. The emax=14 time is still not measured.


## Update: writer and compile cache (done)

- **Compile cache** (`nuham/runtime.py`, `--compile-cache`, on by default in the CLI, default dir `~/.cache/nuham/jax`):
  the SRG flow and the lab contraction compile ~380 kernels at emax 10 (2.7 MB on disk). Fresh-process CLI runs at
  emax 10: 35.7 s with no cache, 36.6 s with a cold cache (about 1 s overhead), 24.2 s with a warm cache.
- **Writer**: formatting stays in the calling thread; gzip and disk writes run in a background thread behind a bounded
  queue. Isolated test, 2.4e7 values: 20.1 s serial vs 16.9 s threaded (decompressed output byte-identical). Compression
  dominates, so the default gzip level is now 1 (`--gzip-level`): on real me2j text level 1 = 150 MB/s, ratio 2.89;
  level 4 = 69 MB/s, ratio 3.20; level 6 = 36 MB/s, ratio 3.39.
- Fresh-process CLI timings with warm compile cache: emax 10 / e2max 20: 21.3 s (68 MB file; 25.0 s and 61 MB at gzip level 4);
  emax 12 / e2max 24: ~56-60 s (266 MB file, 5.07e7 values). Both include reading the NN file and the SRG flow.
- Still not done: tabulated 9j/6j (T-coefficient build), an emax 14 run, GPU timing.


## Update: vectorized 9j/6j (done)

- `angmom/racah_array.py`: array versions of the 6j and 9j (doubled-integer arguments, broadcasting, zeros for violated
  triangles), tested against the scalar versions on 40000 random tuples each (1e-13; 1062 nonzero 6j and 192 nonzero 9j in the samples).
- `t_matrix` now evaluates the recoupling factors A (9j) and B (6j) for all pairs and rows at once, and sums only
  `lam = J-1, J, J+1`: both factors vanish unless `|J-S| <= lam <= J+S` with `S` = 0 or 1. The old scalar routine is kept as
  `t_matrix_reference` and compared to the new one in every emax-4 channel (1e-12).
- Profile, emax 12 / e2max 24: `t_matrix` 10.2 s -> 2.3 s; total (profiled) 51 s -> 41 s. Fresh-process CLI with warm caches:
  emax 10: 21.3 -> 18.5 s; emax 12: ~56-60 -> 48 s.
- Remaining at emax 12 (profiled, 41 s): contraction ~18 s (10.6 s of it waiting on the JAX products), file formatting ~10 s
  (overlapped with the gzip thread), SRG flow ~2 s, T-coefficients 2.3 s, assembly ~3 s.
- Observation: for the contraction shapes (n_blocks 512-1024, r 8-16, P 1280-2432) XLA's CPU product reaches 84-95 GFLOP/s here,
  NumPy BLAS 128-203 GFLOP/s. A NumPy path on CPU (JAX on GPU) would save perhaps 5 s at emax 12; not done.
