# imsrg++ reads our me2j files (2026-10-03)

The first end-to-end check of the deliverable: imsrg++ (sibling checkout `../imsrg`, commit 2acba947, built in podman from a copy with `refs/imsrg/build.sh`) reads files written by
`nuham` and gives the same 4He energies as it does for NuHamil's file. imsrg++ reads `.gz` through Boost's `gzip_decompressor` (`ReadWrite.cc`), which is why the compressors
needed testing: this package now writes igzip streams by default when `isal` is installed. Commands: `refs/imsrg/run_he4.sh`.

Setup: N3LO EM500, hw 16, SRG lambda 2.0, Coulomb on, NN only (`3bme=none`), 4He reference, HF basis, IMSRG(2) with the Magnus method and the imaginary-time generator.

**emax 4, e2max 8, full flow (smax 500, 281 steps).** Energies in MeV:

| file | compressor | EHF | E(IMSRG(2)) |
|---|---|---|---|
| NuHamil | (its own gzip) | -21.9125640 | -26.695921921 |
| nuham | gzip | -21.9125652 | -26.695922758 |
| nuham | isal, 1 thread | -21.9125652 | -26.695922758 |
| nuham | isal, 4 threads | -21.9125652 | -26.695922758 |
| nuham | none (plain text) | -21.9125652 | -26.695922758 |

All four of our files give identical numbers to every digit imsrg++ prints; the difference to NuHamil's file is 1.2e-6 (HF) and 8.4e-7 (IMSRG(2)), which is the level of
the element-wise agreement (1e-6 test tolerance; NuHamil's own VODE error ~3e-7).

**emax 8, e2max 16, short flow (smax 0.05, 10 steps; only the read, HF and MBPT energies are meaningful).** The files are 11.8 MB (NuHamil), 13.9 MB (gzip) and 13.9 MB
(igzip with 4 threads, one gzip member with many deflate blocks, checked with `gzip -t`):

| file | EHF | EMP2 | E after 10 steps |
|---|---|---|---|
| NuHamil | -21.9371121 | -6.63539219 | -28.116952931 |
| nuham, gzip | -21.9371134 | -6.63539203 | -28.116954071 |
| nuham, isal 4 threads | -21.9371134 | -6.63539203 | -28.116954071 |

Differences to NuHamil: 1.3e-6 (HF), 1.6e-7 (MP2). Each run took 45 s and 1.6 GB.

Element-wise at emax 8: all 2 496 152 values of our file agree with NuHamil's with a maximum difference of 2.4e-7 MeV (none above 5e-7); NuHamil took 21 s on this machine for the same file.

**What this does and does not show.** It shows that imsrg++ accepts our file format and every compressor variant, including threaded igzip streams with many blocks, and that the physics agrees
with the NuHamil-based run to about 1e-6 MeV in 4He. Not covered: other nuclei, emax above 8, 3N, a converged flow at emax 8, and files above a few GB (the largest file read was 13.9 MB; a
threaded stream above ~1 GB would be worth one more test). Build notes: gcc 13 needs `-include cstdint`, and `libIMSRG.so` needs `-fPIC`, which the project only sets with Python bindings on.

## Other interactions at emax 4 (2026-10-03)

The interaction is a plug-in (momentum-space partial-wave file in, nothing else assumed), and every other file in NuHamil's `input_nn_files` works without a code change. Each was run through
NuHamil (podman image, about 9 s each) and through `nuham` with hw 16, lambda 2.0, emax 4, e2max 8, Coulomb on; element-wise over all 27 336 values:

| interaction | max diff (MeV) | | interaction | max diff (MeV) |
|---|---|---|---|---|
| LO_EMN500 | 1.1e-7 | | N2LO_sat | 4.7e-7 |
| NLO_EMN500 | 1.1e-7 | | N3LO_texas | 6.5e-7 |
| N2LO_EMN500 | 9.0e-8 | | DNLOGO450 | 7.0e-8 |
| N3LO_EMN500 | 1.2e-7 | | DN2LOGO394 | 1.5e-7 |
| N4LO_EMN500 | 2.2e-7 | | DN2LOGO450 | 4.8e-7 |
| N2LO_opt | 3.5e-7 | | (N3LO_EM500, earlier) | 2.4e-7 at emax 8 |

No value exceeds 1e-6 and the counts and order agree. The largest differences (N3LO_texas, N2LO_sat, DN2LOGO450) are the ones with the stiffest interactions; they are at the level of NuHamil's own integrator error.
Five of the references (LO_EMN500, N4LO_EMN500, N2LO_opt, N2LO_sat, N3LO_texas) are committed in `refs/work` and tested in `tests/test_generate.py` (they skip when the momentum-space files are not available).

4He with imsrg++ for N2LO_opt (full flow, smax 500, emax 4), MeV:

| file | e2hf | EHF | EMP2 | E(IMSRG(2)) |
|---|---|---|---|---|
| NuHamil | -79.6567062 | -23.8799955 | -4.09134418 | -28.221148 |
| nuham | -79.6567067 | -23.8799959 | -4.09134416 | -28.221148 |

Differences: 4e-7 (HF), 2e-8 (MP2), none visible in the IMSRG(2) energy at the printed precision.

## Large files: emax 12 and 14 (2026-10-03)

**imsrg++ stops reading early.** Given a file with `file2e1max=14, file2e2max=28` but a model space of `emax=4`, imsrg++ finished in 0.4 s: the tuples are in file order, so it has everything it needs after
the first part and never reads or decompresses the rest (so the gzip trailer is never checked). Such a run says nothing about the big stream. Two tests replace it.

**1. Boost decoding of whole emax 14 streams** (`refs/imsrg/boost_cat.cc`, built exactly like imsrg++'s reader: `ifstream` -> `gzip_decompressor` -> `filtering_istream`). One computation on this CPU (matrices 98 s),
written three ways (emax 14, e2max 28, hw 16, lambda 2.0, 168 066 656 values, 2 705 873 196 bytes of text, which is above 2^31):

| writer | write time | file | Boost decode | decoded SHA-256 |
|---|---|---|---|---|
| `gzip` level 1 | 54 s | 914 MB | 2 705 873 196 bytes, eof, no error | 4409e7d4... |
| isal level 1, 4 threads | 13 s | 904 MB | same | same |
| isal level 0, 4 threads | 13 s | 1.08 GB | same | same |

All three are single gzip members (`gzip -t` passes, the size field is right) and decode to byte-identical text, each in 15 s. This covers the streams above 1 GB compressed and above 2^31 bytes decompressed.

**2. imsrg++ consuming a whole file: emax 12, e2max 24, full model space** (4He, hw 16, lambda 2.0, `smax=0`: read, HF and MBPT only; 50 715 864 values). NuHamil needed 555 s and 5.9 GB for this file on this machine, `nuham` 36 s.
Energies in MeV; each imsrg++ run took about 200 s and 8.9 GB:

| file | e2hf | EHF | EMP2 |
|---|---|---|---|
| NuHamil (235 MB) | -73.8069259 | -21.9435394 | -6.77482643 |
| nuham, gzip (278 MB) | -73.8069298 | -21.9435407 | -6.77482628 |
| nuham, isal level 1, 4 threads (276 MB) | -73.8069298 | -21.9435407 | -6.77482628 |

The two of ours are identical; the difference to NuHamil is 1.3e-6 (HF) and 1.5e-7 (MP2). Element-wise, all 50 715 864 values agree with NuHamil's to 2.4e-7 MeV (none above 5e-7).

**Remaining gap:** imsrg++ itself consuming an emax 14 file. It used 8.9 GB at emax 12 and 2.9 GB at emax 10, so emax 14 would need roughly 25-30 GB, more than this machine's 19 GB. It would need a larger machine
(imsrg++ built for aarch64 on DeltaAI or run elsewhere). The decoder test above covers the one risk that differs between emax 12 and 14 (stream size); the parsing is the same code that read emax 12.

## 3N: imsrg++ reads our me3j file (2026-10-04, Phase 5.4)

4He, N3LO EM500 with the local 3NF400 (c1 -0.81, c3 -3.2, c4 5.4, cD -0.2, cE 0.098), hw 16, lambda 2.0, `emax=4, e3max=4`, ramp flat8, full flow (smax 500), `3bme_type=full`,
`file3e1max=4`, `file3e3max=4`, **`file3e2max=8`**. Both runs read an me2j and an me3j file, NuHamil's or ours (`nuham.nn.generate`, `nuham.nnn.generate.generate_me3j`):

| files | e2hf | e3hf | EHF | EMP2 | E(IMSRG(2)) |
|---|---|---|---|---|---|
| NuHamil | -29.2196072 | 14.5720766 | 5.8490810 | -3.44611700 | 1.761305 |
| nuham | -29.2196093 | 14.5720818 | 5.8490859 | -3.44611837 | 1.761308 |

The differences are 5e-6 (EHF) and 3e-6 (IMSRG(2)), at the level of the element-wise agreement of the two files (2e-6). The absolute energies are not physical: a flat8 ramp with e3max 4
keeps the induced 3N force large (spec 5c, section 3.4); the run only checks that imsrg++ reads our file and gets the same numbers.

**`file3e2max` must be given.** With the default (-1) this imsrg++ build stores zero 3N values ("Stored 0 floating point numbers", no error, `e3hf = 0`), for NuHamil's file as well as ours;
the run then gives the NN-only energies. Spec 5d section 7.3 said the default is fine: it is not. Run with `BME3=FILE E3MAX=4 EXTRA="file3e2max=8" refs/imsrg/run_he4.sh FILE2 4 8 500`
(`GREP=` overrides the output filter; check that "Stored N floating point numbers" equals the file's value count).


## 16O with 3N at small size (2026-10-06): not a usable comparison, and why

Step 1 of 5.5 was imsrg++ on 16O (emax 4, hw 16, EM500 SRG 2.0, 3NF400) with NuHamil's 4/8/4 files against ours. It cannot be done at this size. Run with the helper `refs/imsrg/run_he4.sh` (now `NUC=O16 A=16`; the
full imsrg++ output goes to `$WORK/imsrg/run/last_run.log`, one flow-table line per step), model space emax 4, e2max 8, short flow (smax 3):

| 3N file (model e3max) | NN file | HF | e1hf | e2hf | e3hf | EHF (MeV) |
|---|---|---|---|---|---|---|
| none (NN only, NuHamil) | NuHamil emax 4 | converged, 31 iterations | 350.0 | -480.6 | - | -130.6 |
| NuHamil 4/8/4, N3max 8, **induced only** (4) | NuHamil emax 4 | not converged after 800 iterations | 271.5 | -109.3 | +3.4 | +165.6 |
| NuHamil 4/8/4, N3max 8 (4) | NuHamil emax 4 | not converged after 800 iterations | 263.8 | -118.8 | +10.0 | +155.0 |
| NuHamil 6/12/6, N3max 6 (6, read as a prefix of emax 6) | NuHamil emax 6 | not converged after 800 iterations | 336.9 | -97.9 | +53.6 | +292.5 |
| ours 6/12/12, N3max 12 (12, prefix of emax 6) | ours emax 6 | converged, 45 iterations | 200.0 | -149.3 | +493.6 | +544.3 |

With NN only 16O behaves (the flow settles at E = -150.7559425 MeV by s = 2 and only drifts at 1e-9 to 1e-6 after that; with smax 500 it would run for hours). With any of these 3N files HF fails or gives a huge
e3hf that grows with e3max, in NuHamil's own files too, so it is not a property of ours. The induced-only file (no genuine local force) fails the same way and the genuine force adds only +6.6 MeV to e3hf at 4/8/4, so the locality of the 3NF400 force is not what breaks HF. Likely cause (not confirmed): a model space of emax 4 needs the 3N force up to e3max = 12, and a Jacobi flow space
N3max as small as e3max (4 to 12 here; production uses 24-40) leaves the induced 3N force unconverged and large. What the runs do show: (i) our 6/12/6 file equals NuHamil's over all 456 320 values, maximum
difference 4.1e-6, nothing above 1e-5, largest value 24.6 (this is the N > 4 check that 5.1b had asked for, up to N = 6); (ii) the values of our 6/12/12 file (144 345 500, 719 MB) are all below 14 in magnitude
with no NaN; (iii) NuHamil at 6/12/12 (N3max 12) took 2 h 08 min for its 3N step on this machine (about 30 min of Jacobi part and 1 h 38 min of lab transform and write, with the machine shared with other runs
and the process paused for a few minutes), so the element-wise comparison of that file is now done (below). The README table of NuHamil is for another 3N force (`N3LOEMlnl`) and cannot be matched.

What can be done instead: imsrg++ on a prefix of the production file (its N3max is 24-40). A model space of emax 4 needs the first 13.1 million values (about 70 MB of gzip), emax 6 the first 693 million
(about 3.4 GB), emax 8 the first 3.4e9 (about 17 GB).


## 16O on the production file (2026-10-06)

The production file (14/28/16, 3NF400, N3max 24-40, `out/N3LO_3NF400_hw16_e14_e2_28_e3_16.me3j.gz`, 31 532 859 061 bytes, fetched from DeltaAI with the same size) read by imsrg++ in place (`BME3_INPLACE=1`; the
container mounts its directory read-only) with a small model space. NN: our own me2j (hw 16, SRG 2.0, Coulomb on) at the model's emax. The run script takes the file's limits (`F3EMAX=14`, `F3E3MAX=16`,
`EXTRA="file3e2max=28"`) apart from the model's (`E3MAX`).

**emax 4, e2max 8, model e3max 12 (the full 3N of the space), smax 3:** imsrg++ read **13 131 655** values and stopped, exactly the number of values that precede the first bra with a leading orbit above e = 4 in
the file order (counted from `nuham.nnn.me3j`), so the prefix property and the enumeration agree with the reader. HF converged in 29 iterations: e1hf 295.472, e2hf -419.004, e3hf +29.439, EHF -94.093 MeV,
EMP2 -22.878 MeV. IMSRG(2) (Magnus, imaginary-time generator, short flow): **E = -119.824798 MeV** in 119 s and 138 MB; it was -119.8247965 at s = 0.7, so the short flow is converged to 1e-6.
For scale: the README of NuHamil gives -115.540 MeV for 16O at emax 4, hw 16, e3max 16, lambda 2.0 with the `N3LOEMlnl` 3N force, a different force (local plus non-local regulator), so this is context and not a reference.
This is the first 16O result with the 3N force of the production file; nothing independent confirms it yet (see the 6/12/12 comparison in PLAN.md 5.5).


**Our 6/12/12 file against NuHamil's, element by element (2026-10-06).** NuHamil (3NF400, hw 16, lambda 2.0, emax 6, e2max 12, e3max 12, N3max 12, jmax3 27, flat ramp; the 3N step took 7677 s = 2 h 08 min, 17:59 to 20:15 for NN plus 3N, with 16 threads and up to 7 GB, on a shared machine and with a pause, so not a clean timing; its NN step took 477 s) wrote 144 345 500 values, as does `nuham` (8 min with 2 workers). All values agree: **maximum difference 5.65e-6, no value off by more than 1e-5**
(largest value 13.8). This covers N up to 12, all 56 channels (2J up to 27, both T and parities) and the writer's value order. **Which file of ours:** the one generated locally on this CPU (`nuham.nnn.generate --emax 6 --e2max 12 --e3max 12 --ramp flat12`, 2 workers, 8 min), not the production file from DeltaAI, which has N3max 24-40 and e3max 16 and
was made on the GH200; the production run's backend is therefore not covered by this comparison. Together with the 6/12/6 file (456 320 values, 4.1e-6) and the 4/8/4 file
(24 890 values, 1.2e-6) it is the independent correctness check of the code against NuHamil; the production file's own settings (Jacobi flow space N3max 24-40 by J, N from 13 to 16, e3max 16) have no NuHamil
reference at that size. The imsrg++ numbers in the table above for "ours 6/12/12" (e3hf +494) therefore are what NuHamil's own file gives too: a small N3max, not a bug.

**16O at emax 6 on the production file (smax 2, flow not settled).** The reader took 693 209 575 values (2.58 GB), again exactly the number that precedes the first bra with a leading orbit above e = 6.
HF converged in 30 iterations: e1hf 294.702, e2hf -419.590, e3hf +30.087, EHF -94.802 MeV (emax 4: -94.093). The IMSRG(2) flow does not settle at emax 6 with imsrg++'s default step control: E alternates between two
values on consecutive steps (-126.34 and -126.24 at s = 2 for model e3max 16; -126.35 and -126.32 for e3max 12), ||Eta_2|| stays at about 11-13, and the last step, which lands on s = smax, gives another value
(-126.480 and -126.569), so no energy is quoted. The same happens with NN only (-160.05 and -159.88, ||Eta_2|| about 11-15, final -159.996), so it is a property of the flow at this size and not of the 3N file.
Model e3max 12 against 16 differs by 0.06-0.08 MeV along the flow and by 1.7e-4 MeV in EHF. The 3N shift at fixed s is similar at both emax: +30.9 MeV at emax 4 (-150.756 to -119.825) and about +33.7 MeV at emax 6.
Memory: imsrg++ 5.6 GB for e3max 16 and 3.4 GB for e3max 12 at emax 6; 811 s and 545 s on this CPU for 240 steps.


## Why the 16O flow stalled at emax 6, and the settings that work (2026-10-07)

**Cause.** imsrg++'s imaginary-time generator is `eta = H_od * sign(denominator)` (`Generator.cc`), a discontinuous function: a matrix element whose denominator is near zero flips sign from step to step. The step size is
`ds = domega / ||eta||` (`domega` 0.1 by default), so a generator that does not decay (||Eta_2|| of 11-35 here) gives tiny steps (0.0075), the flow alternates between two energies on consecutive steps, and the energy
depends on the step size. It is independent of our file: NN only behaves the same. The HF reference is fine (holes at -48.3, -22.7, -12.2 MeV, the lowest particle orbit 0d5/2 well above them).

16O, emax 6, e2max 12, NN only (our me2j), smax 1 unless noted; E near the end of the flow in MeV, ||Eta_2|| at the end, and whether the flow settled:

| setting | E | ||Eta_2|| | settled |
|---|---|---|---|
| imaginary-time, default (`domega=0.1`, smax 2) | about -160.05 and -159.88 alternating, last step -159.996 | 11 | no |
| `domega=0.5` | -158.24 and -158.66 alternating | 34 | no |
| `domega=0.02` | -160.26 (chatter 6 keV, drift 11 keV over s) | 2-3 | yes |
| `omega_norm_max=1.0` | -160.13 and -159.92 alternating | 10-15 | no |
| `core_generator=white`, smax 1 | -156.69, still falling | 0.5 | smooth, too short |
| `core_generator=white`, smax 30 | **-160.352995** | 8e-4 | yes (70 steps, 86 s) |

**Recommendation:** use the `white` generator with `smax` 30 (`GEN=white`, smax 30 in `refs/imsrg/run_he4.sh`), or the imaginary-time generator with `domega` 0.02 or smaller; never imaginary-time with the default `domega` at emax 6 and above.
With `white` the flow is smooth, takes 70 steps (the `white` flow can take steps up to `dsmax` 0.5) and is converged to 1e-5 MeV.

**16O, hw 16, SRG lambda 2.0, 3NF400 (production file read as a prefix), E(IMSRG(2)) in MeV:**

| | emax 4 (model e3max 12) | emax 6 (model e3max 16) |
|---|---|---|
| `white`, smax 30, NN only | -150.796767 | -160.352995 |
| `white`, smax 30, NN + 3N | **-119.946195** | **-126.741247** |
| 3N shift | +30.85 | +33.61 |
| imaginary-time (emax 4: default, converged; emax 6: `domega=0.02`, smax 2), NN + 3N | -119.824798 | -126.533 (chatter 3 keV, ||Eta_2|| 2) |
| HF energy (EHF) | -94.093 | -94.802 |

The generators differ by 0.04 (NN only) and 0.12 MeV (3N) at emax 4, and by 0.09 and 0.21 MeV at emax 6, where the imaginary-time flow is the less converged one; the generator dependence of IMSRG(2) is normally of this size, so quote
the generator with the number. Going from emax 4 to 6 lowers the energy by 6.80 MeV (`white`) or 6.71 MeV (imaginary-time). NuHamil's README (another 3N force, `N3LOEMlnl`, e3max 16) has -115.540 and -123.445 MeV
at emax 4 and 6, a step of 7.9 MeV: the same order, no more can be said. imsrg++ used 5.6 GB at emax 6 (e3max 16); 940 s for the imaginary-time flow of 314 steps and 565 s for `white` (including about 4 minutes of reading the 693 million values).
Logs of these runs are in the session scratch only; the commands are `refs/imsrg/run_he4.sh` with `NUC=O16 A=16`, `BME3`, `BME3_INPLACE=1`, `F3EMAX=14 F3E3MAX=16 E3MAX=<model>`, `EXTRA="file3e2max=28"`.
