# Spec 5b2: the non-local regulator of the `LNL` 3N force (gap D, step D0)

Sources: `NuHamil-public/src/ThreeBody/NNNForceHOIsospin.F90` ("NFH") lines 238-268 (`set_nnn_interaction_chEFT_n2lo`),
296-413 (`multiply_non_local_regulator_hospace`), 425-455 and 681 (`Regulator` dispatch, local part); `MyLibrary.F90:750`
(`ho_radial_wf_norm`); `exe/NuHamil_3BME.py` (preset `N3LOEMlnl`); P. Navratil, Few-Body Syst. 41, 117 (2007) for the
force. Section 6 of `spec-5b-local-n2lo-3nf.md` has a first summary; this file settles its open points. Read from the source, nothing run yet
(that is D3).

## 1. What `LNL` is

`Regulator = "LNL"` computes the **same local force** as `Local` (the `Local` and `LNL` cases share one branch, NFH:429-451, and the file name
uses "Local" for both, NFH:431), with the preset `lambda_3nf_local = 650`, then multiplies a non-local regulator on both sides. Preset
`N3LOEMlnl` (`NuHamil_3BME.py`, `helium4.py`): NN `N3LO_EM500`, `RegulatorPower = 2`, c1, c3, c4, cD, cE = -0.81, -3.2, 5.4, **0.7, -0.06**,
`lambda_3nf_local = 650`, `lambda_3nf_nonlocal = 500` (MeV). One `RegulatorPower` n serves both regulators (comment NFH:681).
The example script's defaults are `TNF = "N3LOEMlnl"`, the ramp `ramp40-5-36-7-32-9-28-11-24` (ours for the production file), `lambda = 2.0`,
`hw = 20` and e3max 16; NuHamil's README table is for hw 16, so it did not use the script's default hw (its other settings are unknown).

## 2. The operation, in order, per Jacobi channel (J, P, T)

1. `work` = NAS matrix of the local force (all five LEC terms added, as for `Local`).
2. `DMat = CFPᵀ work CFP` (antisymmetric basis, size `north`): our `local_3nf_matrix` returns exactly this (`antisymmetrize(w, channel)`).
3. Regulator (only for LNL): `R` = matrix of `F` in the NAS basis (section 3), `Fa = CFPᵀ R CFP` (the same antisymmetrizer call),
   `DMat <- Faᵀ DMat Fa`. The regulator multiplies once on each side and is applied **before the SRG flow** (the flow sees the regulated force).

## 3. The matrix R

`F(p, k) = exp(-( (p² + k²)/2 · hc² / Λ_nl² )^n)`, p and k the momenta (fm⁻¹) conjugate to the two Jacobi vectors xi1, xi2
(`(p²+k²)/2 = p_phys² + 3 q_phys²/4`, so `F` depends only on the hyperradius in momentum space and is invariant under permutations of the nucleons).
Diagonal in `l12, s12, j12, t12, l3, j3` (the other quantum numbers drop out); in the radial quantum numbers:

`R[(n12, n3), (n12', n3')] = Σ_i Σ_k w_i w_k  u_{n12,L}(p_i) u_{n3,l}(k_k) F(p_i, k_k) u_{n12',L}(p_i) u_{n3',l}(k_k)`

with `L = l12`, `l = l3`. NuHamil's numbers:

| item | value | source |
|---|---|---|
| p mesh | Gauss-Legendre, 500 points on `[0, pmax]`, the **same mesh for p and k** | NFH:308, 318, 322 |
| `pmax` | `25·6/√hw` fm⁻¹ (hw in MeV): 37.5 at hw 16, 33.5 at hw 20 | NFH:318 |
| radial function | `u(p) = ho_radial_wf_norm(n, l, a, p) · (-1)ⁿ`, normalised so that `∫ u² dp = 1` (the factor `p` is inside, the weights have no `p²`) | NFH:376-393; `MyLibrary.F90:750` |
| scale | `a = (m_p + m_n) hc² / (2 m_p m_n hw)` = the squared oscillator length b² of the Jacobi basis (fm²); `x = √a · p` | NFH:388 |
| n range | `n12, n3 <= Nmax/2` and `l <= Nmax` of the channel (the ramp's Nmax) | NFH:386-387 |
| constants | `m_p`, `m_n`, `hc` of NuHamil (`MyLibrary`), same as ours | |

The `(-1)ⁿ` is the Fourier-transform phase of the HO function (the `i^l` factors cancel between bra and ket as `l` is equal); it does not cancel in
`R` between `n` and `n'` of different parity, so it matters.

**Reuse.** `nuham.kernels.ho.momentum_radial(k, l, n_max, b)` is exactly this function: `(-1)ⁿ √b g_nl(b k)`, normalised on `∫ R² dk = 1`, used by
`kernels/nn/projection.py` with the same NuHamil convention. Use it with `b = √a` (equals `hc/√(m hw)`; check against our `nu` in `local_force/radial.py`).
`R` is then two batched products: `M12[a, a', i] = u_a(p_i) u_a'(p_i) w_i`, `M3` likewise, `R = einsum(M12, F, M3)`. Cost per `(L, l)` pair:
`n_max² · 500² + n_max⁴ · 500` flops, small beside the flow.

## 4. Points that the first summary left open, now settled

- **Phase and normalisation of u:** as in the table; `momentum_radial` already has both.
- **Role of truncation:** `R` is built in the NAS space of the channel (ramp Nmax, e.g. 40), then projected with `CFPᵀ R CFP`. `Fa Fa` is then the product of two
  truncated operators, intermediate sums stay in this space. NuHamil does the same, and the space is the same ramp, so no deviation is expected. R is not diagonal in N
  (it couples different `n12, n3`): the truncation error of `F` at high N (where `F` is small for Λ_nl = 500 MeV and hw 16: `F` has fallen to e⁻¹ at `(p²+k²)/2 hc² = Λ²`)
  is NuHamil's too.
- **Mesh error is a real part of the reference:** 500 points to `pmax = 150/√hw` with HO functions up to `n = 20`, `l = 40`: the highest functions
  oscillate fast (about `√(4n + 2l + 3) · b` fm⁻¹ turning point, `b ≈ 2.2 fm` at hw 16, so up to `p ≈ 10 fm⁻¹`), so 500 points over 37.5 fm⁻¹ may be thin at the top of the ramp.
  D1 includes a convergence study of `R` against the mesh; D3 tells whether NuHamil's fixed mesh matters at N ≤ 12.
- **Local part of LNL:** unchanged code path with `cutoff = 650`, `power = n`, cD = 0.7, cE = -0.06. Nothing new except the LECs and cutoff.
- **Symmetry check available without NuHamil:** `F` commutes with the antisymmetrizer (it depends on the hyperradius only), so `R` restricted to the antisymmetric
  subspace must satisfy `(1 - A) R A = 0` on the states away from the truncation edge, and `Fa` must be symmetric with eigenvalues in `(0, 1]`.

## 5. Where it goes in the package (D1, D2)

- Kernel `kernels/nnn/regulator.py` (new, pure JAX): `nonlocal_regulator_blocks(...)` returning `R` on the NAS states of a channel.
- Driver: in `nnn/generate.py` after `local_3nf_matrix`, `fa = antisymmetrize(R, channel)`, `v3n = fa.T @ v3n @ fa`. `antisymmetrize` is reused for `CFPᵀ R CFP`.
- `LocalForce` gets `nonlocal_cutoff: float | None = None`. It is part of `repr(force)`, so the cache key of the evolved operators changes automatically for LNL
  (and `None` leaves existing keys, output and `CACHE_VERSION` alone). CLI `--cutoff-nl`, preset `--preset N3LOEMlnl` (sets LECs, both cutoffs, power).
- `jmax3` (`J3max_initial_3nf`) for the force still applies; for channels above it there is no genuine force, so no regulator.

## 6. Checks (D1-D3)

1. `R -> 1` as `Λ_nl -> ∞`; `n = 1`: `F` factorises, so `R = R12 ⊗ R3` with 1D integrals (independent code path in the test).
2. `R` symmetric, positive; `(1 - A) R A` small; `Fa` eigenvalues in `(0, 1]`.
3. Mesh convergence of `R` (points 250/500/1000, `pmax` factor 1/1.5/2) at N 12, 24, 40.
4. Against NuHamil: 4/8/4 and 6/12/6 `me3j` files with the preset, tolerance 1e-5 (as F, G, H); 6/12/12 if needed.
5. Regression: `nonlocal_cutoff=None` gives the 3NF400 files unchanged (existing `tests/test_v3_reference.py`).
