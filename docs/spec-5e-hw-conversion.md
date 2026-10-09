# Spec 5e: frequency conversion (`hw_target`) of the 3N Jacobi operator

Gap K of `PLAN.md`. Sources are NuHamil files read on 2026-10-09 (read only, `../NuHamil-public`); `[V]` = read in the
source, `[U]` = not verified. Nothing from NuHamil is copied into `src/`: the overlap is a textbook quantity and is
implemented from its definition.

## 1. Why this matters

`exe/NuHamil_3BME.py`, the script the README names for its 3N interaction and its imsrg++ benchmark table, sets in
`main()` `hw = 30` and `hw_target = 16` (`emax 14, e2max 28, e3max 16`), with `ramp40-5-36-7-32-9-28-11-24` and the
`N3LOEMlnl` preset. [V] Our production files use the same ramp and the same preset but make the Jacobi operator at
`hw = 16` directly. `exe/NuHamil_2BME.py` makes the NN file at `hw = 16` directly (our NN files agree with it). [V] The git
history has one import commit (2023-01-12) for the script and for the README table, so that the table came from this
`main()` is an inference. [U]

## 2. What NuHamil does

- Input: `hw` (Jacobi frequency, the one of the flow) and `hw_target` (default `hw`; if different, `hw_conversion = .true.`,
  `NuHamilInput.F90:470-476`). [V] The me3j file does not store `hw`. [V]
- Per Jacobi channel `(J, P, T)` the evolved operator is built in the space `N <= Nmax` of the ramp and converted there
  (`ThreeBodyJacOpsChanIso.F90`, `set_three_body_evolved_scalar_isospin`, lines 852-856): `op3 = UT^T op UT - op3sub`
  (the SRG transformation applied to the bare operator, minus the embedded evolved NN force), then
  `op3 = FreqConv(op3, Nmax, hw, hw_target)`, then the leading `n_as(e3max)` block is kept. [V] So the conversion acts on
  `V3 = H(s) - T - V_NN(s)`, before the cut to `e3max`, in the ramp's `Nmax` space. This is the same object and the same
  order as `nuham.nnn.srg3.three_body_operator` (`v3 = evolved_h - t - nn_jacobi(...)`, then the cut).
- `FrequencyConversion` (lines 242-273, 887-935): `M' = U^T M U` with `U = C^T W C`, the antisymmetrisation of
  `W[bra, ket] = delta(l12, s12, j12, t12, l3, j3) O_{l12}(n12, n12') O_{l3}(n3, n3')` on the NAS states of the channel (`C`
  is the CFP matrix). `bra` runs over the original frequency, `ket` over the target. [V]
- `O_l(n, n') = sum_i w_i R_{nl}(r_i; nu) R_{n'l}(r_i; nu_target)` on 500 Gauss points on `[0, 25]` fm, with the
  coordinate-space radial functions without the `(-1)^n` phase, `nu = 2 m_red hw / (hbar c)^2` (`m_red` the pn reduced mass,
  the oscillator parameter of the Jacobi coordinates, `b = 1 / sqrt(nu)`), `n <= Nmax/2`, `l <= Nmax`, `2n + l <= Nmax`
  (`init_overlaps`, lines 1128-1168). [V]
- Only the operators named `hamil` and `NNNint` are converted on the read path (line 295-301); in the evolved path the
  operator is converted whatever its name. [V] Our V3 is `NNNint`.

## 3. Consequences for nuham

- `U` is not square-orthogonal in the truncated space: `O_l` between two frequencies couples every `n` to every `n'` of the
  same `l`, and the rows are cut at `n <= (Nmax - l)/2`. The converted operator therefore differs from the one made
  directly at `hw_target` by the tail of the overlap beyond `Nmax`. This is the effect to be measured (gap K5/K6).
- The scale transformation acts on both Jacobi coordinates alike, so it commutes with permutations and `U` can be
  antisymmetrised like the regulator `R` (`nnn/regulator.py`, `antisymmetrize`): `U_A = C W C^T` in our orientation
  (`antisymmetrize` returns `cfp W cfp^T`).
- The flow, the NN potentials, `T` and `V3n` all stay at `hw`; only the final `V3` is converted. The lab transform
  (`T` coefficients, brackets) does not depend on the frequency.
- Cache: the cached `V3` of a channel is the one after the cut, so the conversion must be done before the cache write and
  `hw_target` must enter the cache key when it differs from `hw` (the key stays unchanged otherwise).
- Quadrature: NuHamil's 500 points on `[0, 25]` fm give the overlap to near machine precision for the N that occur at
  `b` of 1.3 to 1.8 fm (functions reach about 12 fm at `N = 40`). Tests check `O` against closed forms and orthonormality
  rather than copying the mesh. [U for `hw = 30` at `N = 40`: the tests cover it.]

## 4. Checks

- `hw_target = hw`: `O = identity` to 1e-12.
- `n = 0` closed form: `<0 l(b1)|0 l(b2)> = (2 b1 b2 / (b1^2 + b2^2))^(l + 3/2)`.
- Large basis: the rows of the lowest states of `O` are orthonormal (`sum_n' O_{n n'}^2 -> 1`) with `n'` up to 40.
- Round trip `hw -> hw' -> hw` is the identity on the low block.
- Against NuHamil (gap K4): 4/8/4 and 6/12/6 at `hw 30 -> 16`, and `hw 30 -> 20`, element by element, tolerance 1e-5.
