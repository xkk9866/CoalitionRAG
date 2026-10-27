"""Load Qwen2.5-14B the same way the reader does, then exit."""

from __future__ import annotations

import os

os.environ.setdefault("OPEN_CUDA", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen2.5-14B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    name,
    dtype=torch.bfloat16,
    local_files_only=True,
    low_cpu_mem_usage=True,
)
print("loaded_cpu", flush=True)
model.to("cuda:0")
model.eval()
messages = [{"role": "user", "content": "Reply with only the short answer.\nQuestion: What is 2+2?\nAnswer:"}]
chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
enc = tokenizer(chat, return_tensors="pt")
enc = {k: v.to(model.device) for k, v in enc.items()}
with torch.inference_mode():
    out = model.generate(**enc, max_new_tokens=8, do_sample=False)
text = tokenizer.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)
print("probe_ok", text.strip(), flush=True)
