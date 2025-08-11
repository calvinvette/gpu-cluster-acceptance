#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
#SBATCH -J nccl-mn
#SBATCH -t 00:20:00
#SBATCH -N ${NODES}
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --ntasks-per-node=${GPUS_PER_NODE}
module load mpi || true
srun bash -lc '/opt/nccl-tests/build/all_reduce_perf -b 8M -e 512M -f 2'
