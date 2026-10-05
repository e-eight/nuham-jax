#!/bin/bash
#SBATCH --account=bhhk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120g
#SBATCH --gpus-per-node=1
#SBATCH --constraint=projects
#SBATCH --time=01:30:00
#SBATCH --job-name=nuham-verify-b
#SBATCH --output=/projects/bhhk/soham/nuham/verify_b_%j.log
# Verification B (run only after A passed; about 1 h, 1 GPU-h): the Jacobi side with the production ramp.
#  1. ramp32 Jacobi part again (before: basis 102 s, force 757 s, embed 34 s, flow 118 s, about 17 min);
#  2. the high-J channels at N = 24 and 28 that no run has measured: --js 11,13,19,27,35 with the production ramp.
# 32 CPUs and workers as in the earlier measured runs (CPUs are not billed; 72 is a later test: host memory with that many workers is unmeasured). Set OMP_NUM_THREADS=8 in the submitting environment to test the BLAS threads
# of the main process (sbatch --export=ALL,OMP_NUM_THREADS=8 ...); the default 1 is what earlier runs used.
source /projects/bhhk/soham/nuham/repo/scripts/deltaai/common.sh

echo "=== Jacobi part, ramp32, e3max 16"
rm -f $ROOT/bench3n/verify_b_stages_ramp32.jsonl
run_checked $ROOT/out/verify_b_ramp32.log $PY -m nuham.nnn.generate $NN $ROOT/out/unused.me3j.gz --hw 16 --lam 2.0 --emax 14 --e2max 28 --e3max 16 \
  --ramp ramp32-5-28-7-24-9-20-11-16 --stop-after jacobi --jacobi-cache $ROOT/cache/jacobi \
  --stage-log $ROOT/bench3n/verify_b_stages_ramp32.jsonl
echo "=== production ramp, high-J channels 2J = 11,13,19,27,35"
rm -f $ROOT/bench3n/verify_b_stages_highj.jsonl
run_checked $ROOT/out/verify_b_highj.log $PY -m nuham.nnn.generate $NN $ROOT/out/unused.me3j.gz --hw 16 --lam 2.0 --emax 14 --e2max 28 --e3max 16 \
  --ramp ramp40-5-36-7-32-9-28-11-24 --js 11,13,19,27,35 --stop-after jacobi --jacobi-cache $ROOT/cache/jacobi \
  --stage-log $ROOT/bench3n/verify_b_stages_highj.jsonl
for f in ramp32 highj; do
  echo "--- totals $f"
  $PY - <<PYEOF
import json
tot = {}
for line in open("$ROOT/bench3n/verify_b_stages_$f.jsonl"):
    for k, v in json.loads(line)["seconds"].items():
        tot[k] = tot.get(k, 0.0) + v
print({k: round(v, 1) for k, v in tot.items()})
PYEOF
done
rm -rf $TMPDIR; echo "JOB_DONE $(date -Is)"
