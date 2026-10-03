# Study: is the full N=100 SRG flow necessary?

Script: `refs/srg_truncation_study.py` (throwaway, uses `refs/oracle_numpy.py`).
Raw output: `refs/srg_truncation_study_output.txt`.

Method: for each (hw, lambda), flow every channel (J <= 8, all Tz, Coulomb in pp) in a
truncated space 2n+L <= N_flow, with the full-precision .bin as bare input, DOP853 rtol=atol=1e-10.
Compare the evolved V, cut to 2n+L <= e2max, with the N_flow=100 result. Max abs error over all
channels and elements, in MeV. Times are scipy CPU-seconds summed over 54 channels (single thread per
channel, so only a relative scale).

## Max abs error (MeV) in V sub-block

hw=16, lambda=2.0 (the others are in the output file):

| N_flow | e2max=8 | 12 | 16 | 20 | 28 | CPU s |
|---|---|---|---|---|---|---|
| 30 | 2.6e-2 | 4.1e-2 | 4.1e-2 | 4.8e-2 | 8.2e-1 | 0.6 |
| 40 | 1.0e-4 | 1.6e-4 | 1.8e-4 | 4.4e-4 | 3.2e-3 | 0.9 |
| 50 | 9.3e-6 | 1.4e-5 | 3.9e-5 | 1.0e-4 | 2.4e-4 | 1.2 |
| 60 | 5.7e-6 | 8.8e-6 | 2.3e-5 | 5.8e-5 | 1.2e-4 | 1.5 |
| 80 | 2.0e-6 | 3.0e-6 | 7.6e-6 | 1.8e-5 | 3.7e-5 | 3.3 |
| 100 | reference | | | | | 42.5 |

Same pattern for lambda=1.8 and 2.8, and hw=24, 40. At hw=10 the truncation needs more:
N_flow=60 gives 5e-4..1e-3 and N_flow=80 gives 3e-6..2e-5. At hw=40, N_flow=60 gives 6e-4 at e2max=28.

## Findings

1. The truncation error is a UV-cutoff effect: it falls steeply with N_flow, and it needs
   more N_flow at small hw (cutoff ~ sqrt(N hw)) and at large e2max (the margin N_flow - e2max matters).
2. No N_flow below 100 reaches the 1e-6 full-chain test tolerance for e2max >= 12. N_flow=80 is
   within 4e-5 everywhere tested except hw=10 corners. N_flow ~ 50-60 gives 1e-4..1e-5 for hw >= 16.
3. The cost does not scale like emax: this is the fixed cost of the flow. It is ~40 CPU-s total
   for all 54 channels in scipy (a minute on one core), against ~10 s wall on 16 threads in NuHamil.
   N_flow=60 is ~25x cheaper than 100 (cost ~ N_flow^4 in this implementation).

## Implications

- The full flow is cheap in absolute terms. In JAX the flow is a batch of dense N x N products
  (and the commutator with T is banded, so it can be done in O(N^2)); it should take seconds on a CPU
  core-set and less on a GPU. The Fortran code's real cost at emax=14 is the lab transform, not this.
- Therefore: keep N_flow=100 as the default so results match NuHamil to ~1e-7, and expose `n_flow`
  as a documented speed/accuracy knob. Do not make a smaller value the default.
- A smaller N_flow is a deviation from NuHamil, not an equivalent algorithm. Use it only with an
  error check (flow at N_flow and N_flow+20 and compare), and note the error grows at small hw.
- The 3N case is different: Jacobi SRG matrices are much larger, so a truncation study will matter
  there. Not covered here.

## Not tested

hw < 10; lambda < 1.8; momentum-space SRG; adaptive/banded implementations (timings are scipy dense);
effect on nuclear observables (only matrix elements are compared).
