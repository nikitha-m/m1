"""Speculative decoding with a streamed 3B: one SSD read of the 3B checks k guesses at once.

The 1B draft stays fully in RAM. The 3B keeps `--in-ram` layers in RAM; the rest come from the SSD.
Usage:
    .venv/bin/python spec_stream.py              # 6 layers in RAM, k=4, 4-bit SSD layers
"""

import argparse
import time

import mlx.core as mx
from mlx_lm import load

from spec_decode import speculative_generate
from stream_model import STATS, load_streamed, process_gb

DRAFT = "models/llama3.2-1b-mlx-4bit"
PROMPT = "Explain how a CPU works in a short paragraph."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-ram", type=int, default=6)
    parser.add_argument("--draft-tokens", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=40)
    parser.add_argument("--ssd-bits", type=int, default=4, help="bit width of the layers read from the SSD")
    args = parser.parse_args()

    target, tok = load_streamed(args.in_ram, args.ssd_bits)
    draft, _ = load(DRAFT)
    print(f"3B layers in RAM: {args.in_ram} (8-bit), from SSD: {28 - args.in_ram} ({args.ssd_bits}-bit), k = {args.draft_tokens}")
    print(f"after loading: process {process_gb():.2f} GB")

    prompt = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True)
    eos = tok.eos_token_ids
    speculative_generate(target, draft, prompt, eos, k=args.draft_tokens, max_tokens=4)  # warm up
    mx.reset_peak_memory()
    STATS["ssd_s"] = STATS["ssd_gb"] = 0.0
    t = time.perf_counter()
    out, rate = speculative_generate(target, draft, prompt, eos, k=args.draft_tokens, max_tokens=args.tokens)
    secs = time.perf_counter() - t

    print(tok.decode(out))
    print(f"\n{len(out)} tokens in {secs:.1f} s = {len(out) / secs:.2f} tok/s (includes reading the prompt)")
    print(f"draft guesses accepted: {rate:.0%}")
    print(f"SSD: {STATS['ssd_gb']:.1f} GB read, {STATS['ssd_s'] / secs:.0%} of the time")
    print(f"peak MLX memory {mx.get_peak_memory() / 1e9:.2f} GB, process {process_gb():.2f} GB")


if __name__ == "__main__":
    main()
