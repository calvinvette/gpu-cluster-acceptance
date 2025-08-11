#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
if [ -x /opt/nccl-tests/build/all_reduce_perf ]; then
  /opt/nccl-tests/build/all_reduce_perf -b 8M -e 256M -f 2 -g 1
else
  echo "nccl-tests not found"
fi
