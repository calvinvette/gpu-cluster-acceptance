# SPDX-License-Identifier: AGPL-3.0-only
import torch

def make_synthetic_loader(tokenizer, batch_size=1, seq_len=512):
    pad = tokenizer.pad_token_id or 0
    while True:
        ids = torch.randint(low=100, high=tokenizer.vocab_size-1, size=(batch_size, seq_len))
        attn = torch.ones_like(ids)
        yield {"input_ids": ids, "attention_mask": attn, "labels": ids.clone()}
