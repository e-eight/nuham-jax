#!/bin/bash
#SBATCH --account=bhhk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120g
#SBATCH --gpus-per-node=1
#SBATCH --constraint=projects
#SBATCH --time=00:45:00
#SBATCH --job-name=nuham-verify-c
#SBATCH --output=/projects/bhhk/soham/nuham/verify_c_%j.log
# Verification C (about 12-15 min): what a compile costs. Job A showed `cfp W cfp^T` at 20.9 s inside the force
# (the same kernel is 1 s per call in `embed`) and a cold lab stage of 45.9 s for 42 kernels (4.2 s warm).
#  C1  2J = 5 at N = 40 again with the compile cache that job A filled (V3 removed, Jacobi bases kept):
#      if `antisymmetrize` falls to a few seconds, the 20.9 s was compilation.
#  C2  the same with XLA_FLAGS=--xla_gpu_autotune_level=0 and an empty compile cache: does the cold cost go
#      away, and does the flow (GEMM speed) stay at about 30 s?
#  C3  the lab benchmark of the largest channel with autotune level 0, cold then warm (A: 45.9 s / 4.2 s).
source /projects/bhhk/soham/nuham/repo/scripts/deltaai/common.sh
AT0="--xla_gpu_autotune_level=0"
summary() {  # per-channel stage times and their totals of a stage log
  $PY - <<PYEOF
import json
tot = {}
for line in open("$1"):
    r = json.loads(line)
    print({k: r[k] for k in ("j2", "parity", "t2", "n_as") if k in r}, {k: round(v, 1) for k, v in r["seconds"].items()})
    for k, v in r["seconds"].items():
        tot[k] = tot.get(k, 0.0) + v
print("TOTAL", {k: round(v, 1) for k, v in tot.items()})
PYEOF
}
jac_run() {  # LABEL COMPILE_CACHE_DIR: 2J = 5 at N = 40 without the cached V3 (the Jacobi bases stay)
  local label=$1 cc=$2
  find $ROOT/cache/jacobi/v3 -name 'v3_*_j5_p*' -delete 2>/dev/null
  rm -f $ROOT/bench3n/verify_c_$label.jsonl
  run_checked $ROOT/out/verify_c_$label.log $PY -m nuham.nnn.generate $NN $ROOT/out/unused.me3j.gz --hw 16 --lam 2.0 --emax 14 --e2max 28 --e3max 16 \
    --ramp ramp40-5-36-7-32-9-28-11-24 --js 5 --stop-after jacobi --jacobi-cache $ROOT/cache/jacobi \
    --stage-log $ROOT/bench3n/verify_c_$label.jsonl --compile-cache $cc
  summary $ROOT/bench3n/verify_c_$label.jsonl
}

echo "=== check that this XLA accepts the autotune flag"
XLA_FLAGS="$AT0" $PY -c "import jax, jax.numpy as jnp; print(jax.devices(), float((jnp.ones((64,64))@jnp.ones((64,64))).sum()))" 2>&1 | tail -2

echo "=== C1: warm compile cache, default autotuning"
jac_run c1_warm $ROOT/cache/jax
echo "=== C2: empty compile cache, autotune level 0"
rm -rf $ROOT/cache/jax_at0
XLA_FLAGS="$AT0" jac_run c2_at0 $ROOT/cache/jax_at0
echo "=== C3: lab benchmark, largest channel, autotune level 0 (no persistent cache in the benchmark)"
XLA_FLAGS="$AT0" $PY benchmarks/bench_me3j.py lab --emax 14 --e2max 28 --e3max 16 --j2 9 --parity 1 --t2 1 --repeat 2 --json $ROOT/bench3n/verify_c_lab_j9_at0.json 2>&1 | grep -v Warn | cut -c1-1200
rm -rf $ROOT/cache/jax_at0 $TMPDIR; echo "JOB_DONE $(date -Is)"
