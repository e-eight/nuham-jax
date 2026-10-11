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


## Gap A: the GPU path against the CPU path (DeltaAI job 3327601, 2026-10-07)

The 6/12/12 case (3NF400, hw 16, lambda 2.0, emax 6, e2max 12, e3max 12, `flat12`) was made twice on one node (`gh058`, commit `0ccc1a7`, 32 workers, 7 min 10 s in all, max RSS 12 GB): once on the GPU backend and once with
`JAX_PLATFORMS=cpu` (the aarch64 host). Both exited 0, both wrote 144 345 500 values (738 MB each, `gzip -t` ok). Element by element (`scripts/compare_me3j.py`): **maximum difference 1.0e-8** (one unit of the last printed decimal,
at value 139 691 025), no value above 1e-5 (and none above 1e-3); the decompressed texts differ (different md5), through the last printed digit of some values, which comes from rounding differences between the
backends; the number of differing values was not counted. Stage times, GPU / CPU backend: force 49.5 / 64.4 s, embed 10.5 / 7.9 s, flow 23.6 / 29.1 s, lab 24.2 / 27.3 s (the local x86 CPU run with 2 workers took 245 s for the lab stage),
write 3.3 / 4.0 s. The files are on DeltaAI in `/projects/bhhk/soham/nuham/out/gapA_{gpu,cpu}_e6_e3_12.me3j.gz`.

**The GPU file against NuHamil's, fetched and compared here (2026-10-07):** 144 345 500 values, **maximum difference 5.650e-6** (at value 3: 13.82467937 against NuHamil's 13.82468502, the same maximum and the same value
as for the local x86 CPU file), none above 1e-5. The GPU file against the local x86 CPU file: maximum difference 1.2e-7 (at value 2 356 618), none above 1e-5. So the GPU path agrees with NuHamil as well as the CPU path does, for this case (N3max 12). The case has N3max 12; the production ramp (N3max 24-40) is covered only by gap B.

## Gaps F, G and H: further comparisons with NuHamil (2026-10-07)

All are 4/8/4 files (3NF400 unless noted, N3LO_EM500, `--jmax3 11`), ours made on the local CPU, NuHamil's in the podman image, compared with `scripts/compare_me3j.py`. Each has 24 890 values and none differs by more than 1e-5.

| Gap | Case | Max difference |
|---|---|---|
| F | `ramp8-5-6-9-4`, hw 16, lambda 2.0 | 1.23e-6 |
| G | `flat8`, hw 20, lambda 2.0 | 3.34e-6 |
| G | `flat8`, hw 16, lambda 1.8 | 2.03e-6 |
| H | `flat6`, hw 16, lambda 2.0, 3NF500 (cD 0.83, cE -0.052), the `n500_N6` reference | 1.55e-6 |

F: NuHamil's `GetRampNmax` (`ThreeBodyJacobiSpaceIso.F90`) switches to the next N when doubled J is greater than the threshold, as `ramp_nmax` does. The ramp changes the result: NuHamil's `flat8` file and its ramp file differ by up to 0.60 (8148 values above 1e-5), so the agreement tests the boundaries. The NuHamil inputs and `run_fg.sh` are in `refs/work/nnn3f/{f_ramp,g_hw20,g_lam18}`; the me3j files are not tracked.

## Gap D3: the LNL force against NuHamil (2026-10-07)

`N3LOEMlnl` of `NuHamil_3BME.py`: N3LO_EM500, `Regulator="LNL"`, power 2, c1, c3, c4, cD, cE = -0.81, -3.2, 5.4, 0.7, -0.06, `lambda_3nf_local` 650, `lambda_3nf_nonlocal` 500, hw 16, lambda 2.0. Ours with `--preset N3LOEMlnl`, local CPU; compared with `scripts/compare_me3j.py`. Inputs, logs and `run_lnl.sh` are in `refs/work/nnn3f/lnl_448` and `lnl_6126`.

| Case | Values | Max difference | Above 1e-5 |
|---|---|---|---|
| 4/8/4, `flat8`, jmax3 11 | 24 890 | 1.98e-6 | 0 |
| 6/12/6, `flat6`, jmax3 15 | 456 320 | 4.13e-6 | 0 |

The regulator changes the force (unit test: the Jacobi operators differ by more than 1e-2 from the local force), so the agreement tests it. The mesh study (`refs/regulator_mesh_study.py`, spec 5b2) shows NuHamil's 500-point mesh is converged to 5e-13 up to N = 40, so the agreement at N <= 8 carries over to the mesh at the production N. What is not tested against NuHamil is N3max above 8 (as for gap B).

## Gap D4: imsrg++ with the LNL force against NuHamil's README table (2026-10-07)

Files: `N3LO_N3LOEMlnl_hw16_e6_e2_16_e3_16.me3j.gz` (DeltaAI job 3330416, 11 min 19 s on the GH200, 693 209 575 values, 3 336 434 168 bytes, `gzip -t` ok; the same count as the production file's prefix for emax 6, as it must be) and the emax 8 file (job 3330462, 8 min 38 s with the Jacobi cache of the first job, 3 424 228 845 values, 16.9 GB). Both: `N3LOEMlnl` (EM500, c1, c3, c4, cD, cE = -0.81, -3.2, 5.4, 0.7, -0.06, Lambda_local 650, Lambda_nonlocal 500, n = 2), hw 16, lambda 2.0, ramp 40-5-36-7-32-9-28-11-24, e3max 16; `--e2max 16` because the generator requires e3max <= e2max (the file is the same as with e2max 12: emax 2 test, byte-identical). The emax 4 and 6 models read prefixes of these files. NN: our me2j at the model's emax (hw 16, lambda 2.0, Coulomb on). imsrg++ at `2acba947`: x86 in podman (local) and aarch64 in Apptainer on DeltaAI (`scripts/deltaai/imsrg.def`, `imsrg_build.sbatch`, `imsrg_run.sbatch`; the build takes 81 s; it needs `--cleanenv` because the host sets CXX=CC, and the build directory must be named `build`).

**4He** (smax 500, imaginary-time; `file3e2max=16`, model e3max 12 at emax 4 and 16 at emax 6), core energy in MeV:

| emax | ours, local x86 | ours, DeltaAI aarch64 | NuHamil README | ours minus README |
|---|---|---|---|---|
| 4 | -27.224725 (95 s) | -27.224725 (11 s) | -27.221 | -3.7 keV |
| 6 | -28.218471 (821 s, 5.6 GB) | -28.218471 (771 s) | -28.221 | +2.5 keV |

e3hf is -0.0202 MeV (emax 4) and -0.0088 MeV (emax 6): the 3N file is read and enters. The x86 and aarch64 builds agree in all printed digits (e1hf, e2hf, EHF, core energy).

**16O** (model e3max 12 at emax 4, 16 at emax 6):

| emax | generator, smax | core energy (MeV) | README | difference |
|---|---|---|---|---|
| 4 | white, 30 | -115.282590 (50 s; DeltaAI aarch64: identical, 8 s) | -115.540 | +0.257 |
| 4 | white, 100 | -115.282592 | | +0.257 |
| 4 | imaginary-time, 100 and 500 | -115.101066 | | +0.439 |
| 6 | white, 30 | -123.089793 (548 s, 5.6 GB) | -123.445 | +0.355 |

The emax 4 to 6 step is 7.807 MeV (white; the README 7.905). White is converged in smax (30 against 100: 2e-6); the generator changes 16O at emax 4 by 0.18 MeV, neither reaches the README's value, which has no recorded imsrg++ settings (generator, step control, normal-ordering options; nothing in NuHamil's repository states them). 4He, which is much more sensitive to the 3N force, agrees to 4 keV, so the difference in 16O is attributed to the IMSRG(2) settings and not to the file. This is plausibility agreement, not a bit-level reference.

**emax 8 (DeltaAI jobs 3335164 and 3335165, 2026-10-07/08, aarch64).** Model emax 8, e2max 16, e3max 16, reading the whole emax 8 file (3 424 228 845 values, 12.76 GB stored). Both runs used `white`, smax 30 (4 He earlier used imaginary-time, smax 500). HF converged (24 iterations for 16O, 28 for 4He). The 16O run took 18 min 33 s, the 4He run 17 min 14 s; both used 27.5 GB of imsrg++ memory.

| nucleus | EHF | EMP2 | core energy (MeV) | README | difference |
|---|---|---|---|---|---|
| 4He | -21.897947 | -6.293158 | -28.576389 | -28.569 | -7 keV |
| 16O | -85.320499 | -36.454081 | -125.851198 | -126.270 | +0.419 |

Convergence of the flow (from `imsrg_run/<jobid>/imsrg.log`): 16O took 73 steps, the energy changed by 6e-8 MeV on the last step and the MBPT correction is -1.4e-7 MeV, so it is converged. 4He took 63 steps; the energy was still falling by 8e-6 MeV per step and the MBPT correction is -1.1e-5 MeV, so it is converged to about 1e-5 MeV, which does not matter here.

Steps from emax 6 to 8: 16O 2.761 MeV (README 2.825), 4He 0.358 MeV (README 0.348; this uses the imaginary-time emax 6 value, the generators differ). The 16O offset to the README is +0.257, +0.355 and +0.419 MeV at emax 4, 6 and 8: it is smooth and slowly growing, always on the same side. 4He agrees to under 10 keV at all three emax. Reading: the 3N file is right (4He is the sensitive test) and the 16O offset comes from IMSRG(2) settings the README does not record; the generator alone moves 16O by 0.18 MeV at emax 4. This was not tested further. Not done: other generators at emax 6 or 8, and the 4He emax 8 run with imaginary-time.

## Gap D4 extension: 16O at emax 10 and where the imsrg++ time goes (2026-10-09)

Approved by the user for this one run. LNL file for emax 10 (DeltaAI job 3345248, 12 min 21 s, 5 486 676 675 values, 27.2 GB, `gzip -t` ok), then imsrg++ 16O on it (job 3345249, 33 min 37 s of a 1.5 h limit, same settings as emax 8: `white`, smax 30, e3max 16, 32 CPUs, aarch64 container).

16O emax 10: EHF -85.3657815, EMP2 -36.97602077, core energy **-126.569920 MeV**. HF converged in 24 iterations (e3hf 30.22 MeV). The flow took 75 steps to s = 30 and the energy changed by 4e-8 MeV on the last step. Step from emax 8 (-125.8512): -0.719 MeV; the README's step is about 0.75, so the size is right. The README value at emax 10 was not looked up, only its step; with the README's emax 8 value minus 0.75 the offset would be about +0.45 MeV, continuing the drift (+0.257, +0.355, +0.419). 4He was not run at emax 10.

**Where the imsrg++ time goes** (the `TIMES (s)` block of `imsrg.log`, profiler of imsrg++; `Read_3body_file` = `Read_3BME` (serial text parse) + `Store_3BME` (OpenMP) + a few seconds):

| job | values read | Read_3BME | Store_3BME | Read_3body_file | whole run (`real`) | memory |
|---|---|---|---|---|---|---|
| emax 6 (3330502, DeltaAI) | 6.93e8 | 154 s | 98 s | 253 s | 305 s | 5.6 GB |
| emax 8 (3335164) | 3.42e9 | 770 s | 142 s | 917 s | 1105 s | 27.5 GB |
| emax 10 (3345249) | 5.49e9 | 1262 s | 141 s | 1414 s | 1943 s | 45.0 GB |

- The parse runs at 4.4 M values/s at every size, on one thread (`for (i<nread) infile >> ThreeBME[i]` in `ReadWrite.cc`, no OpenMP). Reading is 73% of the emax 10 run and 83% of the emax 8 run. Store scales with the threads and does not matter.
- Test of the gunzip share (local, emax 6 file, 693 M values, 16 cores): `Read_3BME` 151 s from the `.gz` and 125 s from the same file as plain text (11.2 GB); `gzip -dc` alone 47 s, `igzip` 11 s. So gunzip is about 17% of the read and the text parse about 83%; decompressing outside imsrg++ does not pay. Only a parallel parse or a binary format would shorten the read. The same read took 154 s on the GH200 and 151 s here: a faster core does not help.
- The rest (NN read, HF, normal ordering, flow) took 188 s at emax 8 and 530 s at emax 10 (2.8x); at emax 6 it was about 52 s. Largest items at emax 10: `Standard_BCH_Transform` 307 s and `CommutatorScalarScalar` 332 s (overlapping), `HF_GetNormalOrderedH` 69 s, `HF_Solve` 30 s.
- The same file read took 154 s and 287 s in two emax 6 jobs (3330502, 3330503): allow a factor 2 in the read time.
- Forecast, not measured: emax 12 about 50 min (read about 24 min, the file's 6.37e9 values are the limit; rest about 25 min), memory about 55 GB; emax 14 about 1.7 h. Time limits so far were 1.5 to 100 times the run time; the only timeout was 3330668 (4He emax 8 with imaginary-time and smax 500, 1 h), a flow that did not settle, not the read.

## Gap D4: settings scan for the 16O offset to the README, emax 4 (2026-10-09)

Question: can imsrg++ settings explain the 16O offset (+0.26 MeV at emax 4, growing to about +0.45 at emax 10) while 4He agrees to a few keV? Local x86 podman build, 16O, emax 4, e2max 8, e3max 12, the 13.1 M-value prefix of the emax 6 LNL file, hw 16, `method=magnus`, one setting changed per run from the baseline `white`, smax 30 (about 1 to 2.5 min each). README value: -115.540 MeV. EHF = -84.6017095 and EMP2 = -27.03088631 in every run.

| run | core energy (MeV) | change from baseline |
|---|---|---|
| baseline: `white`, smax 30 | -115.282590 | 0 |
| `white`, smax 100 | -115.282592 | -2e-6 |
| `white`, `ode_tolerance=1e-8` | -115.282590 | 0 |
| `white`, `dsmax=0.1` | -115.283059 | -0.5 keV |
| `atan`, smax 30 | -115.282025 | +0.6 keV |
| `white`, `denominator_partitioning=Moller_Plesset` | -115.293041 | -10 keV |
| imaginary-time, smax 500 | -115.101066 | +0.18 MeV |
| imaginary-time, `Moller_Plesset` | -115.101066 | same as imaginary-time |
| `white`, `basis=oscillator` | -115.328182 | -45 keV (first attempt gave the baseline: `run_he4.sh` appended `basis=HF` after `EXTRA`, so the option was overridden; the script now takes `BASIS`; HF is skipped, EMP2 = -29.964 instead of -27.031) |

Result: no setting lowers the energy by more than 45 keV (the oscillator basis); the most bound value (-115.328) is still 0.212 MeV above the README. The flow is converged (smax, tolerance and step size move it by at most 0.5 keV). The generator choice moves it by 0.18 MeV in the wrong direction. imsrg++'s default `hw` is 20, but every run here passes hw 16 explicitly, so that is not the cause. Remaining candidates: a perturbative triples correction or another IMSRG truncation in the README's runs, and the frequency conversion: NuHamil's example script `NuHamil_3BME.py` uses the same ramp as ours but flows at hw 30 and converts to hw 16 (`hw_target`), which we did not do (gap K in `PLAN.md`, `docs/spec-5e-hw-conversion.md`). The README's emax 10 value is -127.016, so the offset at emax 10 is +0.446 MeV.

Perturbative triples (`perturbative_triples=true`, all 58 246 three-body states kept, dE3max 99, 16O emax 4, same setup; the flow energy is unchanged): `white`, smax 30: -2.66347 MeV, giving -117.94605 MeV in total; imaginary-time, smax 500: -2.60099 MeV, giving -117.70205 MeV. The shift is about ten times the 0.26 MeV offset and has the same sign, so a triples correction does not explain the README value (-115.540): that number is an IMSRG(2) energy. Settings of imsrg++ are therefore not the cause of the offset at emax 4 (every variation above, and this one, misses the README by 0.25 MeV or more in one direction or the other); what remains is the Jacobi truncation of the force and the README's own provenance. 4He, the sensitive test of the force, agrees to a few keV.

Oscillator basis with the other settings (`BASIS=oscillator` in `refs/imsrg/run_he4.sh`, 16O, emax 4, e3max 12; HF skipped, EMP2 = -29.964284 in all runs):

| generator, smax | HF basis (MeV) | oscillator basis (MeV) | change |
|---|---|---|---|
| `white`, 30 | -115.282590 | -115.328182 | -45.6 keV |
| `atan`, 30 | -115.282025 | -115.327261 | -45.2 keV |
| imaginary-time, 500 | -115.101066 | -115.154671 | -53.6 keV |
| `white`, 30, `Moller_Plesset` | -115.293041 | -115.341686 | -48.6 keV |

The basis effect is 45 to 54 keV for every generator and denominator choice, so it adds to the others independently. The most bound result of the whole scan, -115.341686 (`white`, Moller_Plesset, oscillator), is still 0.198 MeV above the README (-115.540). The emax 6 run with `white`, smax 30 and the oscillator basis was killed with exit status 137 (SIGKILL, most likely the out-of-memory killer; not confirmed, `dmesg` is not readable here) right after `Solving` on the 19 GB local machine; the HF basis needs 5.6 GB at this size, the HO-basis normal ordering evidently more. DeltaAI job 3346009 (the same run with `basis=oscillator`, 120 GB, 1 h limit) was OOM-killed after 6 min 29 s (Slurm `oom_kill event`, state OUT_OF_MEMORY) right after printing EMP2 = -40.617391, so the HO-basis normal ordering of the 3N force at emax 6, e3max 16 needs more than 120 GB (the HF basis needs 5.6 GB); the emax dependence of the basis effect is not measured, and a larger-memory node would be needed to measure it.

## Gap K4: the frequency conversion against NuHamil (2026-10-09)

Our `hw 30 -> hw_target` file equals NuHamil's (`NuHamil_3BME`-style input with `hw_target`) element by element: 4/8/4 full 3NF400 at 16 MeV 7.4e-6, induced only at 16, 20 and 12 MeV 3.8e-6, 4.5e-6, 2.7e-6, and 6/12/6 (456 320 values) 3.1e-6; none above 1e-5 (details in `docs/spec-5e-hw-conversion.md`, runs in `refs/work/nnn3f/k4_*`). The first attempt disagreed by 10 MeV because `antisymmetrize` symmetrised `C W C^T` and the frequency overlap `W` is not symmetric; the kernel now takes `symmetric=False` for it. K5 and K6 (imsrg++ on a converted file, DeltaAI) are not run yet.

### Oscillator-basis memory (cause of job 3346009, analysis only)

The OOM above is not intrinsic to the 3N size. In `../imsrg/src/imsrg++.cc` the oscillator branch calls `Hbare.DoNormalOrdering()`; `Operator::DoNormalOrdering3` ends with `ThreeBody.SetMode("pn")`, which allocates a zero-filled pn 3-body block (17.07 GB at emax 6, e3max 16) that stays attached to the operator (particle rank 3, IMSRG3 off). Every flow operator then carries it (17.5 GB each; 85.8 GB at the first Magnus step), so 120 GB, sized from HF runs, was too little. The HF branch uses `GetNormalOrderedH(2)`, which returns a rank-2 operator. A small imsrg++ patch (return rank 2 unless IMSRG3) should bring the run to about 10 GB; it is not applied (on hold by the user).

## Gap K5/K6: imsrg++ on the file converted from hw 30 to hw 16 (2026-10-09/10)

Files made with `lnl.sbatch` `HW_SRC=30` (`--hw 30 --hw-target 16`, N3LOEMlnl, ramp40-5-36-7-32-9-28-11-24, e3max 16, lambda 2.0): emax 6 (DeltaAI job 3348873, 13 min 44 s, 693 209 575 values) and emax 10 (job 3349285, 12 min 11 s, 5 486 676 675 values, `gzip -t` ok; the emax 8 model reads its prefix). imsrg++ as before (`imsrg_run.sbatch` with `TAG=hw30to16`, HF basis, hw 16, 16O `white` smax 30, 4He imaginary-time smax 500; the NN me2j is the unchanged hw 16 file). Core energies in MeV:

| nucleus, emax | README | hw 16 direct | hw 30 -> 16 (job) | converted minus README | direct minus README |
|---|---|---|---|---|---|
| 16O, 4 | -115.540 | -115.2826 | -115.544602 (3348874) | -4.6 keV | +257 keV |
| 16O, 6 | -123.445 | -123.0898 | -123.449394 (3348875) | -4.4 keV | +355 keV |
| 16O, 8 | -126.270 | -125.8512 | -126.273402 (3349286) | -3.4 keV | +419 keV |
| 16O, 10 | -127.016 | -126.5699 | -127.019467 (3349287) | -3.5 keV | +446 keV |
| 4He, 4 | -27.221 | -27.2247 | -27.217134 (3348876) | +3.9 keV | -3.7 keV |
| 4He, 6 | -28.221 | -28.2185 | -28.211122 (3348877) | +9.9 keV | +2.5 keV |

Result: the frequency conversion removes the 16O offset to the README at every emax (a few keV, flat in emax, where it grew from +0.26 to +0.45 MeV without conversion). 16O step from emax 8 to 10: -0.746 MeV, README -0.746. 4He moves by 7 to 8 keV in the other direction and stays within 10 keV of the README (the README's 4He generator and step settings are not recorded). Gap K is closed: the direct-hw-16 production file differs from NuHamil's recipe by the conversion, so a production file for comparison with NuHamil-based results should be made with `hw 30 -> 16`. Whether to regenerate it is the user's decision (about 30 min of GH200 plus a cold V3 cache at hw 30).

## The production file at hw 30 -> 16 (2026-10-10)

Regenerated with the frequency conversion: DeltaAI job 3350092 (`production.sbatch`, `HW_SRC=30`, 14/28/16, 3NF400 at all relative J, lambda 2.0, same ramp), 22 min 42 s, 6 367 911 165 values (as the hw 16 file), 31.3 GB, `gzip -t` ok, 75.6 GB peak memory: `out/N3LO_3NF400_hw30to16_e14_e2_28_e3_16.me3j.gz`. imsrg++ reads prefixes of both production files (jobs 3356201-3356204, 16O, `white`, smax 30, HF basis, hw 16, `file3e2max=28`; the hw 16 numbers reproduce the earlier runs):

| emax (model e3max) | file | e3hf | EHF | EMP2 | core energy (MeV) |
|---|---|---|---|---|---|
| 4 (12) | hw 16 direct | 29.4393 | -94.0927 | -22.8781 | -119.946195 |
| 4 (12) | hw 30 -> 16 | 29.1476 | -94.5752 | -23.0389 | -120.574087 |
| 6 (16) | hw 16 direct | 30.0869 | -94.8019 | -28.9547 | -126.741247 |
| 6 (16) | hw 30 -> 16 | 29.8131 | -95.3025 | -29.2113 | -127.451782 |

The conversion lowers 16O by 0.63 MeV at emax 4 and 0.71 MeV at emax 6 for the 3NF400 force. The file reads correctly and the values are plausible, but there is no NuHamil reference for these production settings (N3max 24-40); the conversion itself is verified against NuHamil (gap K4) and against the README on the LNL force (gap K5/K6). The hw 30 -> 16 file is the one to use when comparing with NuHamil-based results.

