#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
#SBATCH -N ${NODES}
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --ntasks-per-node=${GPUS_PER_NODE}
#SBATCH -t 00:30:00
#SBATCH -J fsdp-lora
#SBATCH -o %x.%j.out

module load cuda || true
export NCCL_DEBUG=INFO
srun bash -lc 'torchrun --nproc_per_node=${GPUS_PER_NODE} --nnodes=${NODES} \
  --rdzv_backend=c10d --rdzv_endpoint=$HOSTNAME:29500 \
  scripts/08_train_fsdp_lora.py --model ${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}'
