#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2025 Calvin Vette
set -euo pipefail

echo "=== GPU & Driver Probe ==="
nvidia-smi -L || { echo "nvidia-smi not found"; exit 1; }
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 || true)
echo "Driver: $DRV"
echo "=== GPU Topology (if available) ==="
nvidia-smi topo -m || true

python3 - <<'PY'
import subprocess, yaml
from pathlib import Path
cfg = yaml.safe_load(open("configs/expected.yaml"))
q = ["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"]
out = subprocess.check_output(q).decode().strip().splitlines()
print("GPUs:", len(out))
for i,l in enumerate(out):
    parts=[x.strip() for x in l.split(",")]
    if len(parts)>=2:
        name,mem = parts[0], parts[1]
    else:
        name,mem = parts[0], "unknown"
    print(f"  [{i}] {name} | vRAM {mem}")
PY
