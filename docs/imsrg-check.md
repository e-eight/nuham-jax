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
