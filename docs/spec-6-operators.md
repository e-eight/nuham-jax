# Spec 6: consistently evolved two-body operators (gap L, step L0)

Sources are NuHamil and imsrg++ files read on 2026-10-11 (read only, `../NuHamil-public`, `../imsrg` at `2acba947`) and two
small NuHamil runs (`refs/work/ops/r2_e4`). `[V]` = read in the source or seen in a run, `[U]` = not verified. Nothing from
NuHamil is copied into `src/`; the operators and the transformation are implemented from their definitions.

## 1. What is needed and why

The Hamiltonian file gives ground-state energies. Other observables need the operator carried through the same SRG
transformation `U` as `H`, `O' = U^T O U`, in the two-body space; otherwise `O` and `H` are not in the same basis.

The first target is the point-nucleon mean-square radius. With `r_i - r_j` the pair separation,
`sum_i (r_i - R)^2 = (1/A) sum_{i<j} (r_i - r_j)^2`, so `<r^2> = (1/A^2) sum_{i<j} (r_i - r_j)^2`: a pure two-body operator of
the relative coordinate, with no one-body part. NuHamil's `R2` is exactly the pair operator `(r_i - r_j)^2`; its one-body
block in the file is all zero. [V] (The factor `1/A^2` is applied by the user of the file; how imsrg++ combines it with its
own radius operators was not checked. [U])

## 2. What NuHamil does for a relative scalar operator

Pipeline `manage_trans_to_lab` -> `trans_to_lab_op` (`TwoBody/TwoBodyManager.F90`). [V]

1. The NN interaction `V` is projected to the HO relative basis at the file's `hw` and the SRG flow is run in the **full**
   `N2max = 100` space of each channel `(J, parity, S, Tz)` with the kinetic generator, Coulomb included in pp and evolved
   (`evolve_coulomb = .true.`); this is the flow we already have. `Nmax_srg_edge` defaults to -1, which means a plain flow
   (no Neumann edge).
2. The same flow returns `U` per channel (`renorm_ho_space` -> `HOSRGChannel`, `u = sol%U`).
3. The operator is built in the relative basis `|n l S J, Tz>` of the same space: for `R2`
   `<n l|r^2|n' l> = b_rel^2 (2n + l + 3/2)` on the diagonal and `-b_rel^2 sqrt(n'(n' + l + 1/2))`-type off-diagonals
   (`two_body_hopot` times `2 b2`, `b_rel^2 = hc^2 / (m_red hw)`), diagonal in `l`, `S`, `J`, `Tz` (`CalcMERel`,
   `OperatorDefinitions.F90`, `two_body_r2`). `jmax2` defaults to 12, so relative `J` runs to 12 for operators, although
   the NN potential is zero above 8 (then `U` is the identity).
4. `oprel%UT(U, U)` forms `U^T O U` (`TwoBodyRelOp::UnitaryTransformationTensor`), channel by channel.
5. The result is cut to `e2max` (`truncate`), then `TMtrans` takes it to the lab frame with the same
   Talmi-Moshinsky coefficients as `H` (centre-of-mass part is the identity for these operators).
6. The writer is `write_general_operator_gzip` for `*.op.me2j.gz`. Operators other than `hamil` also reach it when the file
   name ends in `.me2j.gz` (`write_scalar_operator`). With `Operators = "R2"` and `file_name_nn = "x.op.me2j.gz"` the
   file is named `R2_x.op.me2j.gz` (a prefix with the operator name). [V]

### What `U` is

`HOSRGChannel` uses the flow mode `Hflow` (`sol%init(h, 'Hflow')`; the integrated-`U` mode `HUflow` is commented out).
`SRGHflow` integrates only `H` (VODE, relative tolerance `rtol`, default `1e-8`) and then builds
`U = V_bare V_eff^T`, where `V_bare` and `V_eff` are the eigenvector matrices of `H(0)` and `H(s)` in ascending eigenvalue
order, and the sign of every column of `V_eff` is flipped when its overlap with the bare column is negative (a comment
says this assumes that the transformation does not change the eigenvectors much). [V] Then `U^T H(0) U = H(s)` holds
exactly, and `U^T O U` is the evolved operator provided that every eigenvector pair is matched and signed as the flow
would match it.

This is a definition, not the flow's own `U`. The two coincide when the eigenvalues are non-degenerate and every overlap
keeps its sign along the flow; they can differ in channels with near-degenerate eigenvalues or strongly rotated states. Which
one to implement is the decision of L1: the flow-integrated `U` (`dW/ds = eta W` with `H(s) = W H(0) W^T`, `U = W^T`) is the
physically defined transformation; NuHamil's pairing is what the reference files contain. L1 measures the difference
(a numpy replica of the pairing in `refs/`, same channels) before committing to either.

## 3. File format read by imsrg++ (`.op.me2j.gz`) [V]

Writer `write_general_operator_gzip` (NuHamil) and reader `ReadWrite::ReadOperator2b_Miyagi` (imsrg++) agree:

- line 1: free text header; line 2: `J P Z emax e2max lmax` (operator rank, parity, charge change); the reader takes the first
  five and builds the operator with `rank_J = J`, parity `P`, charge `Z`; line 3: zero-body value;
- one-body block: for `a` in orbits, `b` in orbits (all `b`, not `b <= a`), skipping pairs that violate parity or the
  triangle `(j_a, j_b, 2J)`: four values `pp nn np pn` (`OneBody(ap,bp)`, `(an,bn)`, `(an,bp)`, `(ap,bn)`); all zero for `R2`;
- two-body block: loops `a` over orbits, `b <= a`, `c` over orbits (all), `d <= c`, with `e_a + e_b <= e2max` and
  `e_c + e_d <= e2max`, parity `(-1)^(l_a+l_b+l_c+l_d) = P`, for every `J_ab` and `J_cd` satisfying the triangle
  `(J_ab, J_cd, J_op)`; ten values per line in the order `pppp pppn ppnp ppnn pnpn pnnp pnnn npnp npnn nnnn`
  (`%16.6e`, one line per `(J_ab, J_cd)`), orbit order by `e = 2n + l`, then `l`, then `j`;
- the values are in the normalised convention of imsrg++'s `SetTBME_J`: the `me2j` writer for `H` multiplies by `sqrt 2`
  for `a = b` and for `c = d` (the Darmstadt un-normalised convention), this one does not. In the run below the bare
  `<(0s1/2)^2 J=0| (r_1 - r_2)^2 |same>` is 7.775827 = `3 b^2` (`b^2 = hc^2/(m hw)`, 2.5919 fm^2 at hw 16 for the mean nucleon mass 938.919 MeV; the relative-coordinate length of step 3 is `b_rel^2 = 2 b^2`, so `<r^2>` of the relative `0s` state is `3 b^2`), not twice that. [V]

Our `me2j` writer for `H` (`nn/lab_transform.py`, `nn/me2j.py`) enumerates a different loop (`a <= b`, `c <= d`, `J` shared),
so the operator writer is a new enumeration, not a flag on the existing one; the formatting, gzip and chunking of
`WriterOptions` can be reused.

## 4. Oracle runs (done)

`refs/work/ops/r2_e4/{bare,srg}` (`Input.dat`, `log.txt`; the `.op.me2j.gz` outputs are not tracked): NuHamil, `R2`, EM500,
`hw 16`, `emax 4`, `e2max 8`, `lambda 2.0` (`srg`) or bare, relative cache reused from `refs/work`. Each file has 13 294 lines
(header, 2 lines of preamble, the one-body block and the two-body block). First two-body line of the bare file:
`7.775827 0 0 0 7.775827 7.775827 0 7.775827 0 7.775827` (`pppp` = `pnpn` = `pnnp` = `npnp` = `nnnn` for the `(0s1/2)^2`
`J = 0` element; `pppn`, `ppnp`, `ppnn`, `pnnn`, `npnn` vanish by charge conservation); the srg file's first line is
`7.710609 ... 7.702570 ... 7.705587` for `pppp`, `pnpn`, `nnnn`: the flow lowers the element by about 0.9 % and the
Coulomb in pp separates the three charge columns (7.7106, 7.7026, 7.7056). The largest elements are 36.287 (bare) and
36.286 (srg). The one-body block is zero in both.

These files are the references of L3 (lab-level, 1e-6 target as for `H`). For L1 and L2 no relative-space file exists:
NuHamil does not write `U` or the evolved relative operator for this path, so those two steps are checked through the
lab file and through identities.

## 5. Plan consequences (updates to the L-table in `PLAN.md`)

- **First operator set:** `R2` (scalar, relative only, no one-body block). A second scalar check needs no new operator:
  `U^T (T + V_0) U` must equal our flowed `H` to the flow tolerance (a self-consistency gate for L1 and L2 that does not
  depend on NuHamil). `Sigma`, `E1`, `E2`, `M1`, `Gamow-Teller` have a nonzero rank or charge change and wait for L5.
- **L1** carries the decision flow-`U` against pairing-`U` (section 2) and the tolerance: NuHamil's VODE `rtol` of `1e-8`
  limited the `H` agreement in 3S1-3D1 to about 3e-7.
- **L2** needs relative channels with `J` up to 12 (the potential is zero above 8, so `U = I` there and no flow is needed),
  `R2` over the full `N2max = 100` space, `U^T O U`, then the cut to `e2max`.
- **L3** needs the new writer of section 3, the transform for relative operators `O(n l S J Tz)` diagonal in all labels
  (the existing lab transform handles scalar positive-parity operators; the check is that it applies without change to an
  operator that is not `H`), and the `jmax2` rule: lab pairs with relative `J > jmax2` get no contribution, as in NuHamil.
- **Risks found here:** the pairing-`U` heuristic; the `jmax2 = 12` truncation changes `R2` at large `e2max` (the operator
  does not vanish for `J > 8` the way `V` does); the numerical `rtol`; the normalisation convention differs from the `H`
  file (section 3).
