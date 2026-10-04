#!/bin/bash
# Run the three oracle cases one after another: concurrent OpenMP runs oversubscribe the cores.
cd "$(dirname "$0")"
for d in n400_N6 n500_N6 n400_N8; do
  (cd $d && podman run --rm -v "$PWD:/work:Z" -w /work nuhamil:latest NuHamil.exe Input.dat > log.txt 2>&1)
done
