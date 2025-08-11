<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# GPU Cluster Acceptance & Training Smoke

Portable, progressive acceptance tests for NVIDIA GPU nodes (Orin/Thor, H100/H200), plus an optional FSDP+LoRA training smoke on open models/datasets. Builds a container, runs checks, and (optionally) logs metrics to MLflow.

## Quick start
```bash
# build (amd64)
docker build -f Dockerfile.amd64 -t ghcr.io/OWNER/REPO:amd64-local .
# run progressive checks
docker run --rm --gpus all --ipc=host --network host ghcr.io/OWNER/REPO:amd64-local   bash -lc 'set -e; ./scripts/00_env_probe.sh && python3 scripts/01_cuda_probe.py &&             ./scripts/02_nccl_probe.sh && ./scripts/03_dcgm_diag.sh &&             ./scripts/04_nvlink_matrix.sh && ./scripts/07_fio.sh'
```

## Acknowledgments
Test progression and methodology inspired by
[A Practitioner’s Guide to Testing and Running Large GPU Clusters for Training Generative AI Models](https://www.together.ai/blog/a-practitioners-guide-to-testing-and-running-large-gpu-clusters-for-training-generative-ai-models) by Together AI.

## License
AGPL‑3.0. See [LICENSE](./LICENSE).

### SPDX
All source files include `SPDX-License-Identifier: AGPL-3.0-only`.
