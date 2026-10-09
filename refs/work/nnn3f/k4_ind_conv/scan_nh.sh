#!/bin/bash
# NuHamil induced-only conversions hw 30 -> t, for comparison with ours scan_t.
cd "$(dirname "$0")/.."
for t in 20 12; do
  d=k4_ind_$t; mkdir -p $d; sed "s/hw_target=16/hw_target=$t/" k4_ind_conv/Input.dat > $d/Input.dat; cp k4_hw30_16/NNint_A2*hw30.gz $d/
  (cd $d && podman run --rm -v "$PWD:/work:Z" -w /work nuhamil:latest NuHamil.exe Input.dat > log.txt 2>&1)
  (cd ../../.. && uv run python - $t <<'P'
import sys, numpy as np
sys.path.insert(0,"tests")
from ref3n import read_me3j
t=sys.argv[1]; D="refs/work/nnn3f/"
a=read_me3j(D+f"k4_ind_{t}/nnn.me3j.gz"); b=read_me3j(D+f"k4_ind_conv/scan_{t}.me3j.gz")
print("hw_target", t, "max", abs(a-b).max(), "norms", np.linalg.norm(a), np.linalg.norm(b), "first", a[3], b[3], flush=True)
P
  )
done
