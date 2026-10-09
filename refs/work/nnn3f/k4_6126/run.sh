#!/bin/bash
# Gap K4: NuHamil 6/12/6 (flat8, hw 30 -> 16) against nuham.
cd "$(dirname "$0")"
podman run --rm -v "$PWD:/work:Z" -w /work nuhamil:latest NuHamil.exe Input.dat > log.txt 2>&1
(cd ../../../.. && uv run python -m nuham.nnn.generate --hw 30 --hw-target 16 --lam 2.0 --emax 6 --e2max 12 --e3max 6 \
  --ramp flat8 --jmax3 15 --preset 3NF400 --cE 0.098 --cD -0.2 \
  ../NuHamil-public/input_nn_files/N3LO_EM500_kmax8_N100_Jmax8.bin refs/work/nnn3f/k4_6126/ours.me3j.gz) > ours.log 2>&1
(cd ../../../.. && uv run python - <<'P'
import sys, numpy as np
sys.path.insert(0, "tests")
from ref3n import read_me3j
d = "refs/work/nnn3f/k4_6126/"
a, b = read_me3j(d + "nnn.me3j.gz"), read_me3j(d + "ours.me3j.gz")
e = np.abs(a - b)
print("count", a.size, b.size, "max diff", e.max(), "n>1e-5", int((e > 1e-5).sum()), "max |v|", np.abs(a).max())
P
) > compare.log 2>&1
