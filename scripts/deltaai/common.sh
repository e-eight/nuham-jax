# Shared setup of the DeltaAI job scripts (sourced, not run). Layout: see docs/deltaai-run.md.
ROOT=/projects/bhhk/soham/nuham
REPO=$ROOT/repo
export XDG_CACHE_HOME=$ROOT/cache TMPDIR=$ROOT/tmp PYTHONPATH=$REPO/src
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
mkdir -p $TMPDIR $ROOT/bench3n $ROOT/out $ROOT/cache/jacobi
PY=$ROOT/venv/bin/python
NN=$XDG_CACHE_HOME/nuham/data/N3LO_EM500_kmax8_N100_Jmax8.bin
cd $REPO
echo "host: $(hostname) $(date -Is) commit: $(git rev-parse --short HEAD) cpus: $SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$OMP_NUM_THREADS"
git status --short | grep -v '^??' && { echo "ABORT: the checkout has local changes (pull a pushed commit, do not edit it)"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# run_checked LOG CMD...: start the command, and abort the whole job if its first config line does not show
# one worker per allocated CPU. A wrong worker count once cost a GPU-hour of slow runs.
run_checked() {
  local log=$1; shift
  "$@" > "$log" 2>&1 &
  local pid=$!
  for i in $(seq 1 180); do grep -q '^config:' "$log" && break; sleep 1; done
  local w; w=$(grep -m1 '^config:' "$log" | sed 's/.*"workers": \([0-9]*\).*/\1/')
  if [ "$w" != "$SLURM_CPUS_PER_TASK" ]; then echo "ABORT: workers=$w, allocated $SLURM_CPUS_PER_TASK CPUs"; kill $pid; exit 1; fi
  grep -m1 '^config:' "$log" | cut -c1-600
  wait $pid; echo "exit status $?"
  grep -E "wrote|stages:|Error|Traceback" "$log" | cut -c1-900
}
