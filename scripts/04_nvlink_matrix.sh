#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
if [ -x /opt/nvbandwidth/nvbandwidth ]; then
  /opt/nvbandwidth/nvbandwidth -l || true
  /opt/nvbandwidth/nvbandwidth -m g2g || true
fi
/opt/nccl-tests/build/all_reduce_perf -b 8M -e 512M -f 2 -g 1 || true
