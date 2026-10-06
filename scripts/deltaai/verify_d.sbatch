#!/bin/bash
#SBATCH --account=bhhk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120g
#SBATCH --gpus-per-node=1
#SBATCH --constraint=projects
#SBATCH --time=00:40:00
#SBATCH --job-name=nuham-verify-d
#SBATCH --output=/projects/bhhk/soham/nuham/verify_d_%j.log
# Verification D (about 12 min): the per-kernel no-autotune option (`kernels/compile.py`) on the GH200.
#  D1  antisymmetrisation of 2J = 5, N = 40 with synchronised stages: first call against repeats, dense against sparse.
#      Job A: 20.9 s inside the force. Expected now: a few seconds at the first call.
#  D2  lab benchmark of the largest channel with default XLA flags. Job A: 45.9 s cold / 4.2 s warm; with the global
#      flag (C3) 14.4 / 5.6 s. Expected now: about the C3 numbers.
#  D3  2J = 5 at N = 40 again (V3 removed, default flags): the flow must stay at about 32 s for the first channel
#      (job A) and the force and embed fall as in C2.
source /projects/bhhk/soham/nuham/repo/scripts/deltaai/common.sh
echo "=== D1: antisymmetrisation stages, 2J = 5 +, T = 1/2, N = 40"
$PY benchmarks/bench_3n_dense.py antisym --n 40 --j2 5 --parity 1 --t2 1 --workers $SLURM_CPUS_PER_TASK --jacobi-cache $ROOT/cache/jacobi \
  --json $ROOT/bench3n/verify_d_antisym.json 2>&1 | grep -v Warn | cut -c1-1800
echo "=== D2: lab benchmark, largest channel, default flags"
$PY benchmarks/bench_me3j.py lab --emax 14 --e2max 28 --e3max 16 --j2 9 --parity 1 --t2 1 --repeat 2 --json $ROOT/bench3n/verify_d_lab_j9.json 2>&1 | grep -v Warn | cut -c1-1200
echo "=== D3: 2J = 5 at N = 40, V3 removed"
find $ROOT/cache/jacobi/v3 -name 'v3_*_j5_p*' -delete 2>/dev/null
rm -f $ROOT/bench3n/verify_d_stages.jsonl
run_checked $ROOT/out/verify_d_jac.log $PY -m nuham.nnn.generate $NN $ROOT/out/unused.me3j.gz --hw 16 --lam 2.0 --emax 14 --e2max 28 --e3max 16 \
  --ramp ramp40-5-36-7-32-9-28-11-24 --js 5 --stop-after jacobi --jacobi-cache $ROOT/cache/jacobi \
  --stage-log $ROOT/bench3n/verify_d_stages.jsonl
$PY - <<PYEOF
import json
tot = {}
for line in open("$ROOT/bench3n/verify_d_stages.jsonl"):
    r = json.loads(line)
    print({k: r[k] for k in ("j2", "parity", "t2", "n_as") if k in r}, {k: round(v, 1) for k, v in r["seconds"].items()})
    for k, v in r["seconds"].items():
        tot[k] = tot.get(k, 0.0) + v
print("TOTAL", {k: round(v, 1) for k, v in tot.items()})
PYEOF
rm -rf $TMPDIR; echo "JOB_DONE $(date -Is)"
