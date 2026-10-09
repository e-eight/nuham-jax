#!/bin/bash
# Run imsrg++ for 4He (or another closed-shell nucleus: NUC=O16 A=16) on one me2j file and print the energies.
# usage: run_he4.sh FILE.me2j[.gz] EMAX E2MAX [SMAX [FILE_EMAX FILE_E2MAX]]   (FILE is copied into $WORK/imsrg/data)
# EMAX/E2MAX is the imsrg++ model space; FILE_EMAX/FILE_E2MAX (default the same) is what the file contains. With a
# larger file than model space imsrg++ reads and skips the rest: a full read of a big file in a small run.
# With 3N: set BME3=FILE.me3j.gz (copied like FILE) and E3MAX (file e3max = model e3max, default 4); the file
# is read with file3e1max=FILE_EMAX, file3e3max=E3MAX and 3bme_type=full (a wrong value misaligns the stream silently).
# NUC and A (default He4 and 4) are the reference and mass number: for 16O set NUC=O16 A=16; file3e2max (needed for
# 3N) goes in EXTRA, e.g. EXTRA="file3e2max=8".
# The full imsrg++ output goes to $WORK/imsrg/run/last_run.log as it runs: `tail -f` it for progress (the flow table has
# one line per step: step, s, E, ...; HF convergence lines are there too; a flow that does not settle is a sign of a bad reference).
# For a 3N file bigger than the model space (a prefix is read): F3EMAX, F3E3MAX, EXTRA="file3e2max=...", BME3_INPLACE=1.
# GEN sets the core generator (default imaginary-time); step control goes in EXTRA (domega, dsmax, omega_norm_max, ode_tolerance).
# BASIS sets the single-particle basis (HF default, oscillator, NAT).
# SMAX 500 converges the IMSRG(2) flow at emax 4 (about 90 s on 16 cores); use a small SMAX at larger
# emax to check only the read and the HF/MBPT energies. hw is fixed to 16 (the hw of the files used).
set -euo pipefail
WORK=${WORK:-/tmp/nuham-imsrg}
F=$(basename "$1"); EMAX=$2; E2MAX=$3; SMAX=${4:-500}; FEMAX=${5:-$EMAX}; FE2MAX=${6:-$E2MAX}
[ "$(realpath "$1")" = "$(realpath -m "$WORK/imsrg/data/$F")" ] || cp "$1" "$WORK/imsrg/data/$F"
BME3=${BME3:-}; E3MAX=${E3MAX:-4}; NUC=${NUC:-He4}; A=${A:-4}
# F3EMAX/F3E3MAX are the emax and e3max of the 3N FILE (default FILE_EMAX and E3MAX); E3MAX is the model space's.
# BME3_INPLACE=1 mounts the directory of BME3 read-only instead of copying the file (for files too big for $WORK).
F3EMAX=${F3EMAX:-$FEMAX}; F3E3MAX=${F3E3MAX:-$E3MAX}; MOUNT3=()
ARGS3="3bme=none"
if [ -n "$BME3" ]; then
  F3=$(basename "$BME3")
  if [ "${BME3_INPLACE:-0}" = 1 ]; then
    MOUNT3=(-v "$(dirname "$(realpath "$BME3")"):/work/bme3:ro,Z"); P3=/work/bme3/$F3
  else
    [ "$(realpath "$BME3")" = "$(realpath -m "$WORK/imsrg/data/$F3")" ] || cp "$BME3" "$WORK/imsrg/data/$F3"
    P3=/work/data/$F3
  fi
  ARGS3="3bme=$P3 3bme_type=full file3e1max=$F3EMAX file3e3max=$F3E3MAX e3max=$E3MAX"
fi
podman run --rm -e LD_LIBRARY_PATH=/work/build -v "$WORK/imsrg:/work:Z" "${MOUNT3[@]}" -w /work/run imsrg-dev \
  /work/build/imsrg++ 2bme="/work/data/$F" $ARGS3 ${EXTRA:-} emax="$EMAX" e2max="$E2MAX" \
  file2e1max="$FEMAX" file2e2max="$FE2MAX" file2lmax="$FEMAX" hw=16 A="$A" reference="$NUC" \
  valence_space="$NUC" method=magnus smax="$SMAX" core_generator="${GEN:-imaginary-time}" basis="${BASIS:-HF}" \
  | tee "$WORK/imsrg/run/last_run.log" | grep -E "${GREP:-^(e1hf|e2hf|EHF|EMP2|Core Energy)|^real:|^Max Used:|not converged|Not converged}"
