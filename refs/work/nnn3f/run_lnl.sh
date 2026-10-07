#!/bin/bash
# Gap D3: NuHamil LNL runs one after another (concurrent OpenMP runs oversubscribe the cores).
cd "$(dirname "$0")"
for d in lnl_448 lnl_6126; do
  (cd $d && podman run --rm -v "$PWD:/work:Z" -w /work nuhamil:latest NuHamil.exe Input.dat > log.txt 2>&1)
done
