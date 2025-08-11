#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
ibv_devices || true
which ib_read_bw >/dev/null 2>&1 && ib_read_bw -F -q || echo "ib_read_bw not found"
which ib_write_bw >/dev/null 2>&1 && ib_write_bw -F -q || echo "ib_write_bw not found"
