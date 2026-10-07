#!/bin/bash
# Gaps F and G: NuHamil 4/8/4 runs one after another (concurrent OpenMP runs oversubscribe the cores).
cd "$(dirname "$0")"
for d in f_ramp g_hw20 g_lam18; do
  (cd $d && podman run --rm -v "$PWD:/work:Z" -w /work nuhamil:latest NuHamil.exe Input.dat > log.txt 2>&1)
done
