"""Where does the RAM go? Loads the v1 setup and prints a memory breakdown.

Usage:
    .venv/bin/python -m profiling.mem_report
"""

import re
import subprocess

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from benchmarks.bench_spec import PROMPTS
from spec_decode import TARGET, prefill, speculative_generate
from stream_model import process_gb

DRAFT = "models/llama3.2-1b-mlx-4bit"
GB = 1e9


def weights_gb(model):
    return sum(v.nbytes for _, v in tree_flatten(model.parameters())) / GB


def kv_gb(model, tokens):
    # Fill a fresh KV cache with these tokens and measure its size.
    cache = make_prompt_cache(model)
    prefill(model, tokens, cache)
    size = sum(c.nbytes for c in cache) / GB
    del cache
    return size


def swap_gb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(re.search(r"used = ([\d.]+)M", out).group(1)) / 1000


def mlx_line(label):
    print(f"{label:28} active {mx.get_active_memory() / GB:5.2f}  buffer cache {mx.get_cache_memory() / GB:5.2f}  "
          f"peak {mx.get_peak_memory() / GB:5.2f}  process {process_gb() or 0:5.2f}  swap {swap_gb():5.2f}  (GB)")


def main():
    print(f"swap before loading: {swap_gb():.2f} GB\n")
    target, tokenizer = load(TARGET)
    draft, _ = load(DRAFT)
    print(f"3B weights:    {weights_gb(target):.2f} GB")
    print(f"1B weights:    {weights_gb(draft):.2f} GB\n")
    mlx_line("after loading")

    chat = lambda text: tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True)
    short, long = chat(PROMPTS["explain"]), chat(PROMPTS["long_input"])
    long2 = chat(PROMPTS["long_input"] * 3)  # ~2,000 tokens, like a long job page
    print()
    for name, toks in [("short", short), ("long", long), ("long x3", long2)]:
        print(f"KV cache, {name:8} ({len(toks):5} tokens): 3B {kv_gb(target, toks):.3f} GB   1B {kv_gb(draft, toks):.3f} GB")
    print()

    for name, toks in [("short", short), ("long x3", long2)]:
        mx.reset_peak_memory()
        speculative_generate(target, draft, toks, tokenizer.eos_token_ids, k=2, max_tokens=200)
        mlx_line(f"after generate ({name})")


if __name__ == "__main__":
    main()
