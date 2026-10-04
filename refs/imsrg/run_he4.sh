#!/bin/bash
# Run imsrg++ for 4He on one me2j file and print the energies.
# usage: run_he4.sh FILE.me2j[.gz] EMAX E2MAX [SMAX [FILE_EMAX FILE_E2MAX]]   (FILE is copied into $WORK/imsrg/data)
# EMAX/E2MAX is the imsrg++ model space; FILE_EMAX/FILE_E2MAX (default the same) is what the file contains. With a
# larger file than model space imsrg++ reads and skips the rest: a full read of a big file in a small run.
# With 3N: set BME3=FILE.me3j.gz (copied like FILE) and E3MAX (file e3max = model e3max, default 4); the file
# is read with file3e1max=FILE_EMAX, file3e3max=E3MAX and 3bme_type=full (a wrong value misaligns the stream silently).
# SMAX 500 converges the IMSRG(2) flow at emax 4 (about 90 s on 16 cores); use a small SMAX at larger
# emax to check only the read and the HF/MBPT energies. hw is fixed to 16 (the hw of the files used).
set -euo pipefail
WORK=${WORK:-/tmp/nuham-imsrg}
F=$(basename "$1"); EMAX=$2; E2MAX=$3; SMAX=${4:-500}; FEMAX=${5:-$EMAX}; FE2MAX=${6:-$E2MAX}
[ "$(realpath "$1")" = "$(realpath -m "$WORK/imsrg/data/$F")" ] || cp "$1" "$WORK/imsrg/data/$F"
BME3=${BME3:-}; E3MAX=${E3MAX:-4}
ARGS3="3bme=none"
if [ -n "$BME3" ]; then
  F3=$(basename "$BME3")
  [ "$(realpath "$BME3")" = "$(realpath -m "$WORK/imsrg/data/$F3")" ] || cp "$BME3" "$WORK/imsrg/data/$F3"
  ARGS3="3bme=/work/data/$F3 3bme_type=full file3e1max=$FEMAX file3e3max=$E3MAX e3max=$E3MAX"
fi
podman run --rm -e LD_LIBRARY_PATH=/work/build -v "$WORK/imsrg:/work:Z" -w /work/run imsrg-dev \
  /work/build/imsrg++ 2bme="/work/data/$F" $ARGS3 ${EXTRA:-} emax="$EMAX" e2max="$E2MAX" \
  file2e1max="$FEMAX" file2e2max="$FE2MAX" file2lmax="$FEMAX" hw=16 A=4 reference=He4 \
  valence_space=He4 method=magnus smax="$SMAX" core_generator=imaginary-time basis=HF \
  | grep -E "${GREP:-^(e1hf|e2hf|EHF|EMP2|Core Energy)|^real:|^Max Used:}"
