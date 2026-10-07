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
with no NaN; (iii) NuHamil at 6/12/12 (N3max 12) takes 30 min for the Jacobi part and over 2 hours for its lab transform on this machine (started 18:37; it took 14 min per 2J step from 2J = 7 to 9), so the
element-wise comparison of that file is still running. The README table of NuHamil is for another 3N force (`N3LOEMlnl`) and cannot be matched.

What can be done instead: imsrg++ on a prefix of the production file (its N3max is 24-40). A model space of emax 4 needs the first 13.1 million values (about 70 MB of gzip), emax 6 the first 693 million
(about 3.4 GB), emax 8 the first 3.4e9 (about 17 GB).
