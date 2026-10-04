# Study: cost and accuracy of the 3-body SRG flow (Phase 5.3.5)

Measured on this CPU (16 cores, float64) with `benchmarks/bench_3n_dense.py flow`, hw 16, lambda 2.0, EM500 NN plus
the bare local 3NF400, channel at `Nmax = n` (the flow space). Records: `benchmarks/baseline_cpu_3n_flow_j1.json`,
`..._j1_n40.json`, `..._j5.json`. The flow is `kernels/srg.srg_flow_dense`: Dopri8 (13 right-hand-side evaluations
per step, 2 dense GEMMs each), max-norm control, tolerances rtol = atol in units of hw.

| channel | Nmax | n_AS | steps at rtol 1e-8 / 1e-10 / 1e-12 | time at 1e-8 | V3 error at 1e-8 | rho(H0) alpha hw |
|---|---|---|---|---|---|---|
| J=1/2+ T=1/2 | 16 | 351 | 6 / 10 / 15 | 0.13 s | 1.7e-9 | 0.18 |
| | 24 | 1033 | 11 / 17 / 28 | 4.4 s | 3.3e-10 | 0.26 |
| | 32 | 2280 | 14 / 22 / 37 | 48 s | 1.1e-10 | 0.33 |
| | 40 | 4263 | 16 / 26 / - | 323 s | 1.5e-10 | 0.40 |
| J=5/2+ T=1/2 | 16 | 792 | 4 / 6 / - | 0.7 s | 6.6e-10 | 0.16 |
| | 24 | 2548 | 9 / 14 / - | 41 s | 4.3e-10 | 0.24 |

"V3 error" is the largest deviation of V3 from the run with the tightest tolerance of that row (1e-12, or 1e-10 where
no 1e-12 run was made). The spectrum of H(s) equals that of H0 to 3e-12 at rtol 1e-12 (checked up to Nmax 32).

Findings:

- **The flow is not stiff.** The estimate rho(H0) * alpha * hw stays below 0.4, and the step count grows only roughly
  linearly with Nmax (6 -> 16 steps from Nmax 16 to 40 at rtol 1e-8). Higher J needs fewer steps. The NN flow of
  spec 2 took 350-520 right-hand-side calls; here 16 steps are 208 calls.
- **Tolerance.** rtol = atol = 1e-8 already gives V3 to 2e-9 or better, far below the NN data accuracy (about 1e-6,
  spec 5c) and NuHamil's own VODE error. It saves 40% of the time against 1e-10. Suggested default for production:
  1e-8; keep 1e-10 for validation against NuHamil.
- **Cost per right-hand-side call**: 1.55 s at n = 4263 on this CPU (about 0.2 TFLOP/s, `4 n^3` flop); 0.26 s at
  n = 2280. Setting up J=1/2 at Nmax 40 on the CPU took 102 s for the 3NF and 8.5 s for the basis.
- Peak host memory at n = 4263 was 4.8 GB, including the dense 3NF assembly.

Estimate for the production ramp (72 channels, sum of n^3 = 8.2e12 from spec 5c, `4 n^3` flop per call, 208-340 calls
per channel at rtol 1e-8 to 1e-10): 7e15 to 1.1e16 flop in all, i.e. 10-15 hours on this CPU and, at the 40 TFLOP/s
measured for FP64 products on the GH200 (`baseline_gh200_3n_gemm.json`), 3-5 minutes. The largest channel (J=5/2+,
n = 11 340): 5.8e12 flop per call, about 0.15 s on the GH200 and 30-50 s for the flow; Dopri8 holds about 15 matrices of
1.03 GB, so about 16 GB of device memory. The J=5/2 step count at Nmax 40 was not measured (Nmax 24 needs fewer steps
than J=1/2). The assembly of the 3NF (119 s for J=5/2) and its Jacobi basis (90 s) then dominate the channel, not the
flow. The GH200 time of the two largest channels (n = 8080 and 11 340) is the one number still to measure; it needs
about 16 GB of device memory and a few minutes.
