#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
curl -fsSL https://www.gnu.org/licenses/agpl-3.0.txt -o LICENSE
echo "Fetched AGPL-3.0 license to LICENSE"
