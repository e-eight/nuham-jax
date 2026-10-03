#!/bin/bash
# Run imsrg++ for 4He on one me2j file and print the energies.
# usage: run_he4.sh FILE.me2j[.gz] EMAX E2MAX [SMAX]     (FILE is copied into $WORK/imsrg/data)
# SMAX 500 converges the IMSRG(2) flow at emax 4 (about 90 s on 16 cores); use a small SMAX at larger
# emax to check only the read and the HF/MBPT energies. hw is fixed to 16 (the hw of the files used).
set -euo pipefail
WORK=${WORK:-/tmp/nuham-imsrg}
F=$(basename "$1"); EMAX=$2; E2MAX=$3; SMAX=${4:-500}
cp "$1" "$WORK/imsrg/data/$F"
podman run --rm -e LD_LIBRARY_PATH=/work/build -v "$WORK/imsrg:/work:Z" -w /work/run imsrg-dev \
  /work/build/imsrg++ 2bme="/work/data/$F" 3bme=none emax="$EMAX" e2max="$E2MAX" \
  file2e1max="$EMAX" file2e2max="$E2MAX" file2lmax="$EMAX" hw=16 A=4 reference=He4 \
  valence_space=He4 method=magnus smax="$SMAX" core_generator=imaginary-time basis=HF \
  | grep -E "^(e1hf|e2hf|EHF|EMP2|Core Energy)|^real:"
