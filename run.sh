docker run \
        --rm \
        --gpus all \
        --ipc=host \
        --network host \
        ghcr.io/calvinvette/gpu-cluster-acceptance \
        bash -lc 'set -e; \
                ./scripts/00_env_probe.sh && \
                python3 scripts/01_cuda_probe.py && \
                ./scripts/02_nccl_probe.sh && \
                ./scripts/03_dcgm_diag.sh && \
                ./scripts/04_nvlink_matrix.sh && \
                ./scripts/05.1_gpu_burn.sh && \
                ./scripts/07_fio.sh'
