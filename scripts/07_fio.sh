#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
for job in configs/fio/*.fio; do
  echo "=== fio: $job ==="
  fio "$job" || exit 1
done
