#!/bin/bash
#SBATCH --account=bhhk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --constraint=projects
#SBATCH --time=00:45:00
#SBATCH --job-name=nuham-gap-a
#SBATCH --output=/projects/bhhk/soham/nuham/gap_a_%j.log
# Gap A of PLAN.md (5.5): is the GPU path as right as the CPU path? The 6/12/12 case (3NF400, hw 16, lambda 2.0, emax 6,
# e2max 12, e3max 12, flat12 ramp, 144 345 500 values) equals NuHamil's file to 5.65e-6 when made on the local x86 CPU.
# Here the same command runs twice on one node, once on the GPU backend and once with JAX_PLATFORMS=cpu (aarch64 host),
# and the two files are compared value by value. The GPU file stays in $ROOT/out so that it can be fetched and compared
# with NuHamil's file and with the local CPU file (scripts/compare_me3j.py). Estimate about 15-20 min, under 0.4 GPU-hours.
source /projects/bhhk/soham/nuham/repo/scripts/deltaai/common.sh
GEN="-m nuham.nnn.generate --hw 16 --lam 2.0 --emax 6 --e2max 12 --e3max 12 --ramp flat12"
G=$ROOT/out/gapA_gpu_e6_e3_12.me3j.gz
C=$ROOT/out/gapA_cpu_e6_e3_12.me3j.gz
rm -f $G $C

echo "=== run 1: GPU backend"
run_checked $ROOT/out/gapA_gpu.log $PY $GEN $NN $G --stage-log $ROOT/bench3n/gapA_gpu_stages.jsonl
echo "=== run 2: CPU backend (JAX_PLATFORMS=cpu), same node"
JAX_PLATFORMS=cpu run_checked $ROOT/out/gapA_cpu.log $PY $GEN $NN $C --stage-log $ROOT/bench3n/gapA_cpu_stages.jsonl

echo "=== files"
ls -la $G $C
gzip -t $G && echo "gzip -t GPU file ok"; gzip -t $C && echo "gzip -t CPU file ok"
echo "=== element-wise comparison GPU against CPU (tolerance 1e-5)"
$PY $REPO/scripts/compare_me3j.py $G $C; echo "compare exit status $?"
echo "=== decompressed text identical?"
a=$(zcat $G | md5sum | cut -c1-32); b=$(zcat $C | md5sum | cut -c1-32); echo "GPU $a"; echo "CPU $b"; [ "$a" = "$b" ] && echo "IDENTICAL TEXT" || echo "texts differ"
sacct -j $SLURM_JOB_ID -n -o JobID,Elapsed,MaxRSS | head -3
rm -rf $TMPDIR; echo "JOB_DONE $(date -Is)"
