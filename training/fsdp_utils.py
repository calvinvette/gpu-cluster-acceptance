# SPDX-License-Identifier: AGPL-3.0-only
import os, torch, torch.distributed as dist

def init_dist():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

def model_flos_per_token(model, seq_len=512):
    # rough estimator; good enough for MFU trend
    # assumes decoder-only transformer
    n_params = sum(p.numel() for p in model.parameters())
    return 6 * n_params  # ballpark per token
