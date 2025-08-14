#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2025 Calvin Vette

mkdir -p out

docker build . \
    -f Dockerfile.flash-attention.amd64 \
    -v ./out:/out \
    -t nge/builder.flash-attention