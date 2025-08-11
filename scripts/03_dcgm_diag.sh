#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
REPEATS=${REPEATS:-5}
LEVEL=${LEVEL:-1}
for i in $(seq 1 $REPEATS); do
  echo "DCGM diag run $i/$REPEATS"
  dcgmi diag -r $LEVEL --fail-early || exit 1
done
