#!/bin/bash
# Gap K4: NuHamil (hw 30 -> hw_target 16) against nuham --hw 30 --hw-target 16, 4/8/4, flat8.
cd "$(dirname "$0")"
REPO=../../../..
podman run --rm -v "$PWD:/work:Z" -w /work nuhamil:latest NuHamil.exe Input.dat > log.txt 2>&1
(cd $REPO && uv run python -m nuham.nnn.generate --hw 30 --hw-target 16 --lam 2.0 --emax 4 --e2max 8 --e3max 4 \
  --ramp flat8 --jmax3 11 --preset 3NF400 --cE 0.098 --cD -0.2 \
  ../NuHamil-public/input_nn_files/N3LO_EM500_kmax8_N100_Jmax8.bin refs/work/nnn3f/k4_hw30_16/ours.me3j.gz) > ours.log 2>&1
(cd $REPO && uv run python - <<'P'
import numpy as np, sys
sys.path.insert(0, "tests")
from ref3n import read_me3j
d = "refs/work/nnn3f/k4_hw30_16/"
a, b = read_me3j(d + "nnn.me3j.gz"), read_me3j(d + "ours.me3j.gz")
print("count", a.size, b.size)
e = np.abs(a - b)
print("max diff", e.max(), "at", e.argmax(), "n>1e-5", int((e > 1e-5).sum()), "max |v|", np.abs(a).max())
P
) > compare.log 2>&1
