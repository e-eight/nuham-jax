#!/bin/bash
#SBATCH --account=bhhk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120g
#SBATCH --gpus-per-node=1
#SBATCH --constraint=projects
#SBATCH --time=03:00:00
#SBATCH --job-name=nuham-me3j-prod
#SBATCH --output=/projects/bhhk/soham/nuham/production_%j.log
# Production run: the 3N me3j file for imsrg++ (Phase 5.4). 14/28/16, hw 16, lambda 2.0, local N2LO 3NF400 at all
# relative J (no --j3max-3nf), EM500 NN, ramp 40-5-36-7-32-9-28-11-24, float32 lab matrices, 32 workers.
# Estimate about 26 min (20-40; docs/study-me3j.md), the 3 h limit is far above it. The V3 cache resumes a run that
# stops: submit the same script again and finished channels are read from $ROOT/cache/jacobi/v3 (the key holds
# the physics settings and the exact basis). The file stays in $ROOT/out; nothing is deleted at the end.
source /projects/bhhk/soham/nuham/repo/scripts/deltaai/common.sh
# HW_SRC (default: none) flows at that frequency and converts to hw 16 afterwards (gap K, as in lnl.sbatch); the file is then
# named ..._hw${HW_SRC}to16_... The default run is unchanged.
HW_SRC=${HW_SRC:-}
if [ -n "$HW_SRC" ]; then HWARGS="--hw $HW_SRC --hw-target 16"; TAG=hw${HW_SRC}to16; else HWARGS="--hw 16"; TAG=hw16; fi
OUT=$ROOT/out/N3LO_3NF400_${TAG}_e14_e2_28_e3_16.me3j.gz
STAGES=$ROOT/bench3n/production_stages_$SLURM_JOB_ID.jsonl
echo "=== production run, output $OUT"
run_checked $ROOT/out/production_$SLURM_JOB_ID.log $PY -m nuham.nnn.generate $NN $OUT $HWARGS --lam 2.0 --emax 14 --e2max 28 --e3max 16 \
  --ramp ramp40-5-36-7-32-9-28-11-24 --jacobi-cache $ROOT/cache/jacobi --stage-log $STAGES
echo "=== checks"
ls -la $OUT
gzip -t $OUT && echo "gzip -t ok"
zcat $OUT | head -c 400 | head -3 | cut -c1-200
echo "=== stage totals"
$PY - <<PYEOF
import json
tot = {}
for line in open("$STAGES"):
    r = json.loads(line)
    for k, v in r["seconds"].items():
        tot[(r["kind"], k)] = tot.get((r["kind"], k), 0.0) + v
for (kind, k), v in sorted(tot.items()):
    print(kind, k, round(v, 1))
PYEOF
sacct -j $SLURM_JOB_ID -n -o JobID,Elapsed,MaxRSS | head -4
rm -rf $TMPDIR; echo "JOB_DONE $(date -Is)"
