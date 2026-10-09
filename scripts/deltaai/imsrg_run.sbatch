#!/bin/bash
#SBATCH --account=bhhk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120g
#SBATCH --gpus-per-node=1
#SBATCH --constraint=projects
#SBATCH --time=02:00:00
#SBATCH --job-name=imsrg-run
#SBATCH --output=/projects/bhhk/soham/nuham/imsrg_run_%j.log
# imsrg++ (aarch64, built by imsrg_build.sbatch) on a nuham file: NN me2j made here (nuham.nn.generate, hw 16, lambda 2.0, Coulomb on)
# and the 3N me3j made by lnl.sbatch. The same settings as refs/imsrg/run_he4.sh, which runs the x86 build locally.
# Settings through the environment (sbatch --export=ALL,NUC=O16,...):
#   NUC (He4 or O16) and A, EMAX (model emax, e2max = 2 EMAX), E3MAX (model, at most 3 EMAX and the file's),
#   FILE_EMAX (emax of the me3j file, at least EMAX; the model reads a prefix; default 6), SMAX, GEN (core generator).
# The me3j file is N3LO_N3LOEMlnl_hw16_e${FILE_EMAX}_e2_16_e3_16.me3j.gz (file3e2max 16 and file3e3max 16: lnl.sbatch).
# For 4He use the defaults (smax 500, imaginary-time); for 16O use GEN=white SMAX=30 (docs/imsrg-check.md: the default generator chatters).
source /projects/bhhk/soham/nuham/repo/scripts/deltaai/common.sh
NUC=${NUC:-He4}; A=${A:-4}; EMAX=${EMAX:-4}; E3MAX=${E3MAX:-12}; FILE_EMAX=${FILE_EMAX:-6}
SMAX=${SMAX:-500}; GEN=${GEN:-imaginary-time}
E2MAX=$((2 * EMAX))
SRC=/projects/bhhk/soham/imsrg
SIF=/projects/bhhk/soham/apptainer/imsrg-dev.sif
# TAG names the me3j file's frequency setting (hw16, or hw30to16 for a converted file, lnl.sbatch HW_SRC); the model hw is 16 either way.
TAG=${TAG:-hw16}
F3=$ROOT/out/N3LO_N3LOEMlnl_${TAG}_e${FILE_EMAX}_e2_16_e3_16.me3j.gz
F2=$ROOT/out/NN_N3LO_EM500_srg2.0_hw16_emax${EMAX}_e2max${E2MAX}.me2j.gz
RUN=$ROOT/imsrg_run/$SLURM_JOB_ID
mkdir -p $RUN/output
[ -s "$F3" ] || { echo "ABORT: $F3 is missing"; exit 1; }
[ -s "$F2" ] || $PY -m nuham.nn.generate $NN $F2 --hw 16 --lam 2.0 --emax $EMAX --e2max $E2MAX 2>&1 | grep -E "wrote|Error|Traceback"
echo "=== imsrg++ $NUC emax $EMAX e3max $E3MAX file emax $FILE_EMAX smax $SMAX generator $GEN, commit of imsrg: $(git -C $SRC rev-parse --short HEAD)"
cd $RUN
time apptainer exec --cleanenv --bind /projects:/projects --env LD_LIBRARY_PATH=$SRC/build --env OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK \
  $SIF $SRC/build/imsrg++ 2bme=$F2 3bme=$F3 3bme_type=full file3e1max=$FILE_EMAX file3e3max=16 file3e2max=16 e3max=$E3MAX \
  emax=$EMAX e2max=$E2MAX file2e1max=$EMAX file2e2max=$E2MAX file2lmax=$EMAX hw=16 A=$A reference=$NUC valence_space=$NUC \
  method=magnus smax=$SMAX core_generator=$GEN basis=HF $EXTRA > $RUN/imsrg.log 2>&1
echo "exit status $?"
grep -E "^(e1hf|e2hf|e3hf|EHF|EMP2|Core Energy)|not converged|Not converged|^real:|^Max Used:" $RUN/imsrg.log
echo "JOB_DONE $(date -Is)"
