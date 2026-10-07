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
#SBATCH --job-name=nuham-me3j-lnl
#SBATCH --output=/projects/bhhk/soham/nuham/lnl_%j.log
# Gap D4: the LNL 3N file (preset N3LOEMlnl) for 4He and 16O at emax 4 and 6 in imsrg++, to compare with NuHamil's README table
# (docs/spec-5b2-lnl-regulator.md). One file serves both model spaces: emax 6, e2max 12, e3max 16, hw 16, lambda 2.0, EM500, the
# production ramp. imsrg++ reads the emax 4 model as a prefix of this file (same enumeration order, docs/imsrg-check.md).
# Estimate about 25 min: the Jacobi side (basis, force, regulator, flow of 72 channels) costs what it cost in the production run
# (about 22 min) and the lab side is far smaller at emax 6; the Jacobi bases come from the cache, the V3 cache starts empty for LNL.
# The V3 cache resumes a run that stops: submit the same script again.
source /projects/bhhk/soham/nuham/repo/scripts/deltaai/common.sh
OUT=$ROOT/out/N3LO_N3LOEMlnl_hw16_e6_e2_12_e3_16.me3j.gz
STAGES=$ROOT/bench3n/lnl_stages_$SLURM_JOB_ID.jsonl
echo "=== LNL run, output $OUT"
run_checked $ROOT/out/lnl_$SLURM_JOB_ID.log $PY -m nuham.nnn.generate $NN $OUT --preset N3LOEMlnl --hw 16 --lam 2.0 --emax 6 --e2max 12 \
  --e3max 16 --ramp ramp40-5-36-7-32-9-28-11-24 --jacobi-cache $ROOT/cache/jacobi --stage-log $STAGES
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
