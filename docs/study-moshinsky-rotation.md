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
