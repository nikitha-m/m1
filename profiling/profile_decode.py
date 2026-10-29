"""Measure decode speed and the real memory speed (GB/s) of an MLX model.

Usage:
    .venv/bin/python profile_decode.py --model models/llama3.2-3b-mlx-8bit
"""

import argparse
import statistics
import time

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

M1_GBS = 68  # Apple's maximum memory speed for the M1


def weight_bytes(model):
    # Add up the size of every weight array.
    return sum(p.nbytes for _, p in tree_flatten(model.parameters()))


def time_tokens(model, prompt, n):
    # Make n tokens one at a time; return the time for each token.
    cache = make_prompt_cache(model)
    logits = model(mx.array(prompt)[None], cache=cache)  # read the prompt
    y = mx.argmax(logits[0, -1])
    mx.eval(y)  # MLX is lazy: make the GPU finish now
    times = []
    for _ in range(n):
        start = time.perf_counter()
        logits = model(y.reshape(1, 1), cache=cache)
        y = mx.argmax(logits[0, -1])
        mx.eval(y)
        times.append(time.perf_counter() - start)
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/llama3.2-3b-mlx-8bit")
    parser.add_argument("--tokens", type=int, default=100)
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Explain how a CPU works."}], add_generation_prompt=True
    )
    gb = weight_bytes(model) / 1e9
    time_tokens(model, prompt, 5)  # warm-up: first use loads weights from disk
    secs = statistics.median(time_tokens(model, prompt, args.tokens))
    gbs = gb / secs
    print(f"weights:    {gb:.2f} GB")
    print(f"per token:  {secs * 1000:.1f} ms = {1 / secs:.1f} tok/s")
    print(f"real speed: {gbs:.1f} GB/s = {gbs / M1_GBS:.0%} of {M1_GBS} GB/s")


if __name__ == "__main__":
    main()
