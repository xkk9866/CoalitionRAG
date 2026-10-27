"""Probe a GPU-direct 7B load. Does not build the Wikipedia index."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM

print("streaming", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    dtype=torch.bfloat16,
    local_files_only=True,
    low_cpu_mem_usage=True,
    device_map={"": "cuda:0"},
)
print("loaded", model.device, flush=True)
del model
torch.cuda.empty_cache()
