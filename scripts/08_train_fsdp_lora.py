# SPDX-License-Identifier: AGPL-3.0-only
import os, time, math
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import mlflow
from training.dataset_stub import make_synthetic_loader
from training.fsdp_utils import init_dist, model_flos_per_token

MODEL = os.getenv("MODEL_NAME","meta-llama/Llama-3.2-3B-Instruct")
BF16 = torch.cuda.is_bf16_supported()
STEPS = int(os.getenv("STEPS","200"))
BATCH = int(os.getenv("BATCH","1"))

def main():
  init_dist()
  ml_uri = os.getenv("MLFLOW_TRACKING_URI")
  if ml_uri: mlflow.set_tracking_uri(ml_uri)
  if ml_uri: mlflow.start_run(run_name="fsdp-lora-smoke")

  tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
  base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16 if BF16 else torch.float16, device_map=None)
  lora = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","v_proj","k_proj","o_proj"])
  model = get_peft_model(base, lora)

  policy = transformer_auto_wrap_policy
  model = FSDP(model, auto_wrap_policy=policy, device_id=torch.cuda.current_device())
  optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

  loader = make_synthetic_loader(tok, batch_size=BATCH, seq_len=512)
  tokens = 0
  t0 = time.time()
  flops_per_tok = model_flos_per_token(model, seq_len=512)

  for step, batch in zip(range(STEPS), loader):
    with torch.cuda.amp.autocast(dtype=torch.bfloat16 if BF16 else torch.float16):
      out = model(**{k:v.cuda() for k,v in batch.items()})
      loss = out.loss
    optim.zero_grad()
    loss.backward()
    optim.step()
    tokens += batch["input_ids"].numel()
    if step % 10 == 0 and (ml_uri):
      elapsed = time.time()-t0
      tps = tokens/elapsed
      mfu = (tps * flops_per_tok) / (torch.cuda.get_device_properties(0).flops * 1e12) if hasattr(torch.cuda.get_device_properties(0), "flops") else None
      mlflow.log_metric("loss", float(loss.item()), step=step)
      mlflow.log_metric("tokens_per_sec", float(tps), step=step)
      if mfu is not None: mlflow.log_metric("mfu", float(mfu), step=step)

  if ml_uri: mlflow.end_run()

if __name__ == "__main__":
  main()
