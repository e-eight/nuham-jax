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
#SBATCH --job-name=nuham-verify-a
#SBATCH --output=/projects/bhhk/soham/nuham/verify_a_%j.log
# Verification A (estimated 35-50 min, under 1 GPU-h): the changes to the production chain at real size, one job.
#  1. tests on the GPU backend (device antisymmetrisation and scatter, packed lab matrices, cache keys);
#  2. lab transform of the largest channel, cold and warm: compiled kernels (item 2) and matrix size (item 3);
#  3. Jacobi part of 2J = 5 at N = 40, both parities, T = 1/2 and 3/2 (item 1): force stages (projection,
#     scatter, antisymmetrize), embed, flow, to_host. Before: cfp W cfp^T 46 s, scatter 11 s, embeddings 31 s.
# Pass/fail rule written before the run: tests pass (the V3 reference test may fail in its sampled elements
# only, because the AS gauge depends on the platform; its trace and norm must agree), and the stage log shows
# no stage slower than before.
source /projects/bhhk/soham/nuham/repo/scripts/deltaai/common.sh

echo "=== tests on the GPU backend (the workers test forks after CUDA is initialised)"
timeout 1500 $PY -m pytest tests/test_antisym_device.py tests/test_local3n_assembly.py tests/test_nnn_hamiltonian.py \
  tests/test_lab_transform3.py tests/test_me3j.py tests/test_generate3.py tests/test_jacobi3.py -q 2>&1 | tail -4
echo "=== V3 reference (gauge-dependent elements may differ across platforms: informative)"
timeout 600 $PY -m pytest tests/test_v3_reference.py -q 2>&1 | grep -E "passed|failed|assert|changed" | cut -c1-300

echo "=== lab transform, largest channel (J=9/2+ T=1/2, 14/28/16): cold then warm; compiled kernels, matrix size"
$PY benchmarks/bench_me3j.py lab --emax 14 --e2max 28 --e3max 16 --j2 9 --parity 1 --t2 1 --repeat 2 --json $ROOT/bench3n/verify_a_lab_j9.json 2>&1 | grep -v Warn | cut -c1-1200

echo "=== Jacobi part, 2J = 5, N = 40 (production ramp), device chain"
rm -f $ROOT/bench3n/verify_a_stages.jsonl
run_checked $ROOT/out/verify_a_jac.log $PY -m nuham.nnn.generate $NN $ROOT/out/unused.me3j.gz --hw 16 --lam 2.0 --emax 14 --e2max 28 --e3max 16 \
  --ramp ramp40-5-36-7-32-9-28-11-24 --js 5 --stop-after jacobi --jacobi-cache $ROOT/cache/jacobi \
  --stage-log $ROOT/bench3n/verify_a_stages.jsonl
$PY - <<PYEOF
import json
for line in open("$ROOT/bench3n/verify_a_stages.jsonl"):
    r = json.loads(line)
    print({k: r[k] for k in ("j2", "parity", "t2", "nmax", "n_as", "n_nas") if k in r}, {k: round(v, 1) for k, v in r["seconds"].items()})
PYEOF
rm -rf $TMPDIR; echo "JOB_DONE $(date -Is)"
