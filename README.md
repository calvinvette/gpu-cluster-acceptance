<!-- SPDX-License-Identifier: AGPL-3.0-only -->
[![REUSE status](../../actions/workflows/reuse.yml/badge.svg)](../../actions/workflows/reuse.yml)
[![Container build](../../actions/workflows/build-push.yml/badge.svg)](../../actions/workflows/build-push.yml)

# GPU Cluster Acceptance & Distributed Training Test Harness (Smoke Testing)

This repository provides a portable, open-source containerized test harness for **GPU cluster acceptance testing** and **distributed model training**.  

It is designed for **Cloud/Infrastructure engineers** who need to validate NVIDIA GPU systems ranging from embedded developer kits to large-scale NVLink/NVSwitch clusters.

This can be run as a container to verify the installation of key software, key performance metrics
such as Model-Flops Utilization, memory bandwidth, FS I/O, NVLink and Infiniband networking throughput and latencies.

At the moment it focuses on NVIDIA H100/H200 and Orin AGX/Thor AGX Developer kits. Future versions
will support other environments including RKNN/RKNPU, ROCm, and Apple MLX.

Methodology: Fail-Fast Progressive Testing

The test harness follows a fail-fast progression inspired by Together AI’s GPU cluster testing methodology:

	1. Quick sanity checks first — cheap and fast to catch obvious misconfigurations.
	2. Hardware/driver validation — fail early if the GPUs, drivers, or libraries are wrong/missing.
	3. Progressively heavier tests — storage, networking, NCCL, NVLink, then multi-node scale-out.
	4. Full distributed model training — only run after all infrastructure checks pass.

This ensures we don’t waste expensive engineer time or cluster time on nodes that will fail under load.

Please note that this repo is an initial launch and will be migrated to the NextGen HomeLabML project soon.

## Quick start
```bash
# build (amd64)
docker build -f Dockerfile.amd64 -t ghcr.io/calvinvette/gpu-acceptance-testing:amd64-local .
# run progressive checks
docker run --rm --gpus all --ipc=host --network host ghcr.io/calvinvette/gpu-acceptance-testing:amd64-local   bash -lc 'set -e; ./scripts/00_env_probe.sh && python3 scripts/01_cuda_probe.py &&             ./scripts/02_nccl_probe.sh && ./scripts/03_dcgm_diag.sh &&             ./scripts/04_nvlink_matrix.sh && ./scripts/07_fio.sh'
```


---

## Project Layout

```text
.
├── .github/workflows/           # CI workflows (REUSE compliance, container build/push)
├── configs/
│   ├── expected.yaml             # Expected driver, CUDA, NCCL, etc. versions
│   └── fio/                      # Storage performance job files
├── docker/
│   ├── Dockerfile.amd64          # H100/H200 + x86_64 base
│   └── Dockerfile.arm64          # Orin/Thor + Jetson ARM base
├── jobs/                         # SLURM job files for NCCL & multi-node tests
├── scripts/                      # Test scripts (env probe, CUDA probe, NCCL tests, DCGM, etc.)
├── training/
│   └── fsdp_train.py             # PyTorch FSDP LoRA training script
├── LICENSE                       # Root AGPL-3.0 license
├── LICENSES/                     # SPDX-compliant license texts (symlink to LICENSE)
├── pyproject.toml                 # Python project metadata
├── setup.cfg                      # Packaging config
├── fetch_license.sh               # Script to fetch AGPL license text
└── README.md                      # This file
```


Test Stages

1. Environment & Driver Checks
	•	Verify GPU driver presence and version via nvidia-smi.
	•	Compare against expected driver versions (configs/expected.yaml).
	•	Capture GPU model, vRAM, and topology (nvidia-smi topo -m).
	•	For Jetson/embedded devices, optionally use jetson_stats.

2. CUDA & Library Verification
	•	Confirm CUDA toolkit and runtime versions.
	•	Check availability and version of key acceleration libs:
	•	TensorRT
	•	FlashAttention
	•	PyTorch
	•	vLLM

3. NCCL & DCGM
	•	Confirm NCCL is installed and matches expected version.
	•	Run DCGM diagnostics with configurable iteration count (default 5) in fail-early mode.
	•	Log GPU health metrics.

4. Optional Stress Test
	•	Run gpu-burn for a configurable duration.
	•	Useful for catching marginal cooling/power issues.

5. NVLink / NVSwitch Validation
	•	Use NCCL tests (all_reduce_perf, all_gather_perf, etc.) and nvbandwidth to produce GPU-to-GPU bandwidth matrix.
	•	Flag anomalies vs expected NVLink/NVSwitch performance.

6. Multi-Node & Infiniband Tests (8-GPU boxes only)
	•	Check Infiniband connectivity/config (ibstat, ibping).
	•	Measure latency and throughput with ib_read_bw / ib_write_bw.
	•	Run NCCL tests with SLURM across node pairs → scale up to full cluster.

7. Storage Performance
	•	Run fio jobs:
	•	Sequential read
	•	Sequential write
	•	Random read
	•	Random write
	•	Configurable via configs/fio/*.fio.

8. Distributed Model Training (FSDP LoRA)
	•	Train an open-source model (e.g., meta-llama/Llama-3.2-3B-Instruct) with PyTorch FSDP + LoRA.
	•	Monitor:
	•	Training throughput (tokens/sec)
	•	MFU (Model FLOPs Utilization)
	•	GPU utilization
	•	Network latency (all-reduce timings)
	•	Metrics sent to MLFlow (default) or Weights & Biases.

⸻

Base Images
	•	x86_64 / H100 / H200: nvcr.io/nvidia/pytorch:24.07-py3
	•	ARM64 / Orin / Thor: nvcr.io/nvidia/l4t-ml:r36.3.0-py3

## Future Work
The framework is flexible enough that additional scripts can be added for additional tests.

While each of these scripts works independently, it would be nice to be able to extract their running values and submit them to Prometheus/OpenTelemetry to be graphed with Grafana.


## Acknowledgments
Test progression and methodology inspired by
[A Practitioner’s Guide to Testing and Running Large GPU Clusters for Training Generative AI Models](https://www.together.ai/blog/a-practitioners-guide-to-testing-and-running-large-gpu-clusters-for-training-generative-ai-models) by Together AI.

[GPU Burn from Ville Timonen](http://wili.cc/blog/gpu-burn.html)
[GPU Burn GitHiub](https://github.com/wilicc/gpu-burn?tab=readme-ov-file)
```
GPU Burn
Usage: gpu_burn [OPTIONS] [TIME]

-m X   Use X MB of memory
-m N%  Use N% of the available GPU memory
-d     Use doubles
-tc    Try to use Tensor cores (if available)
-l     List all GPUs in the system
-i N   Execute only on GPU N
-h     Show this help message

Example:
gpu_burn -tc -d 3600
```

[NVIDIA's "Important Packages and their installation"](https://docs.nvidia.com/networking/display/ubuntu2204/important+packages+and+their+installation)

## License
AGPL‑3.0. See [LICENSE](./LICENSE).

### SPDX
All source files include `SPDX-License-Identifier: AGPL-3.0-only`.
