#!/bin/bash
cd "$(dirname "$0")"
for d in n400_N16_bare n400_N16_bare_ind; do
  (cd $d && podman run --rm -v "$PWD:/work:Z" -w /work nuhamil:latest NuHamil.exe Input.dat > log.txt 2>&1)
done
