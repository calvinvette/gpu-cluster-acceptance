# SPDX-License-Identifier: AGPL-3.0-only
import importlib, sys
import torch
print("Torch:", torch.__version__, "CUDA runtime:", torch.version.cuda, "bf16:", torch.cuda.is_bf16_supported())
mods = ["xformers","flash_attn","tensorrt","vllm"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "present"))
    except Exception as e:
        print(m, "missing:", e)
