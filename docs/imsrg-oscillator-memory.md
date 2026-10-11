# imsrg++: oscillator basis keeps an empty 3N block (handoff note, 2026-10-11)

Not a task of this repository: the fix belongs to the imsrg++ project (`../imsrg`, commit `2acba947` of ragnarstroberg/imsrg).
Nothing here will patch it; this note is what to hand over.

## Symptom

DeltaAI job 3346009: 16O, emax 6, e3max 16, `basis=oscillator`, 120 GB, `white`, smax 30. Killed by the out-of-memory killer
after 6 min 29 s (MaxRSS 105 GB), right after printing EMP2 = -40.617391. The same run with `basis=HF` needs 5.6 GB.
The log shows "DONE ALLOCATING PN 3-body, size of MatEl is 2290497028 -> 17.0655 GB" and "particle rank is 3 IMSRG3 is 0";
each Magnus operator then costs 17.5 GB (85.8 GB in use at the first omega push).
A local x86 run of the same model was also killed on a 19 GB machine.

## Cause

`src/imsrg++.cc` line 692 (oscillator branch): `HNO = Hbare.DoNormalOrdering();`. `Operator::DoNormalOrdering3`
(`src/Operator.cc`, from line 582) ends with `opNO2.ThreeBody.SetMode("pn")` (line 652), which allocates a zero-filled pn 3-body
block. The returned operator keeps it (particle rank 3) although IMSRG(3) is off, so every operator copied from it during the
flow carries the 17 GB block. The HF branch calls `hf.GetNormalOrderedH(2)`, which returns a rank-2 operator with no block.

## Suggested fix (untested)

In `DoNormalOrdering3`, return the rank-2 operator (no `SetMode("pn")`, free the 3-body storage) unless IMSRG(3) is requested,
or do that in `imsrg++.cc` after the call. Expected memory for the failed job: about 10 GB.

## What would show it is fixed

Rerun job 3346009's setup (`scripts/deltaai/imsrg_run.sbatch` with `EXTRA="basis=oscillator"`, 16O, `EMAX=6`, `E3MAX=16`,
`FILE_EMAX=6`, `GEN=white`, `SMAX=30`; the file `N3LO_N3LOEMlnl_hw16_e6_e2_16_e3_16.me3j.gz` is on DeltaAI) and compare with the
HF run (-123.0898 MeV). At emax 4 the oscillator basis lowers 16O by 45 keV (`docs/imsrg-check.md`, settings scan); the emax
dependence of that shift is the open measurement.
