#!/bin/bash
#SBATCH --account=bhhk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60g
#SBATCH --gpus-per-node=1
#SBATCH --constraint=projects
#SBATCH --time=00:40:00
#SBATCH --job-name=imsrg-build
#SBATCH --output=/projects/bhhk/soham/nuham/imsrg_build_%j.log
# Compile imsrg++ for aarch64 inside the Apptainer image built from scripts/deltaai/imsrg.def (the compute nodes have no
# internet: the image and the source checkout are made on the login node first, see docs/deltaai-run.md). The same recipe
# as refs/imsrg/build.sh: gcc 13 needs -include cstdint, and libIMSRG.so needs -fPIC. About 10 minutes with 32 cores.
set -euo pipefail
AP=/projects/bhhk/soham/apptainer
SRC=/projects/bhhk/soham/imsrg
echo "host: $(hostname) $(date -Is) imsrg commit: $(git -C $SRC rev-parse --short HEAD)"
mkdir -p $SRC/build-aarch64
apptainer exec --bind $SRC:/work $AP/imsrg-dev.sif bash -c \
  "cd /work/build-aarch64 && cmake -DIMSRG_BUILD_PYTHON_BINDINGS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
   -DCMAKE_CXX_FLAGS='-include cstdint' ../src > cmake.log 2>&1 && make -j$SLURM_CPUS_PER_TASK imsrg++ > make.log 2>&1"
ls -la $SRC/build-aarch64/imsrg++
file $SRC/build-aarch64/imsrg++ 2>/dev/null || true
echo "JOB_DONE $(date -Is)"
