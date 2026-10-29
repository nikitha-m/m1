"""Measure how much GPU memory each part of speculative decoding uses.

Usage:
    .venv/bin/python measure_memory.py --draft models/llama3.2-1b-mlx-4bit
"""

import argparse

import mlx.core as mx
from mlx_lm import load

from spec_decode import DRAFT, TARGET, speculative_generate

GB = 1e9


def show(label):
    # active = memory MLX uses now
    # cache  = memory MLX freed but keeps for later (other apps can't use it)
    # peak   = highest active memory so far
    print(f"{label:<16} active {mx.get_active_memory() / GB:5.2f} GB   "
          f"cache {mx.get_cache_memory() / GB:5.2f} GB   "
          f"peak {mx.get_peak_memory() / GB:5.2f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", default=DRAFT, help="path to the draft model")
    parser.add_argument("--draft-tokens", type=int, default=2)
    parser.add_argument("--prompt", default="Explain how a CPU works in a short paragraph.")
    args = parser.parse_args()

    show("start")
    target, tokenizer = load(TARGET)
    show("+ 3B loaded")
    draft, _ = load(args.draft)
    show("+ draft loaded")

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], add_generation_prompt=True
    )
    out, _ = speculative_generate(
        target, draft, prompt, tokenizer.eos_token_ids, k=args.draft_tokens
    )
    show(f"+ {len(out)} tokens")

    limit = mx.device_info()["max_recommended_working_set_size"]
    print(f"\nGPU limit set by macOS: {limit / GB:.2f} GB")


if __name__ == "__main__":
    main()
