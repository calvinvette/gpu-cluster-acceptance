#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
if [ ! -f LICENSE ]; then
	curl -fsSL https://www.gnu.org/licenses/agpl-3.0.txt -o LICENSE
	echo "Fetched AGPL-3.0 license to LICENSE"
fi

if [ ! -f LICENSES/AGPL-3.0-only.txt ]; then
	mkdir -p LICENSES && cd LICENSES && ln -s ../LICENSE AGPL-3.0-only.txt
	echo "Created link to LICENSES/AGPL-3.0-only.txt"
fi
