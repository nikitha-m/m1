"""Time 1 pass of the model with n new tokens, for several n.

If a pass with many tokens costs about the same as a pass with 1 token,
there is "free math" that methods like lookahead decoding can use.

Usage:
    .venv/bin/python profile_batch.py
"""

import statistics
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from spec_decode import TARGET

SIZES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
REPEATS = 20


def time_pass(model, cache, n):
    # Run n tokens after the prompt, then remove them from the cache again.
    tokens = mx.zeros((1, n), dtype=mx.int32)
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        mx.eval(mx.argmax(model(tokens, cache=cache), axis=-1))
        times.append(time.perf_counter() - start)
        trim_prompt_cache(cache, n)
    return statistics.median(times)


def main():
    model, tokenizer = load(TARGET)
    cache = make_prompt_cache(model)
    prompt = tokenizer.encode("Explain how a CPU works in a short paragraph.")
    mx.eval(model(mx.array(prompt)[None], cache=cache))
    time_pass(model, cache, 1)  # warm-up: load weights from disk

    one = time_pass(model, cache, 1)
    print(f"{'tokens':>6} {'ms/pass':>8} {'vs 1 token':>11} {'ms/token':>9}")
    for n in SIZES:
        t = time_pass(model, cache, n)
        print(f"{n:>6} {t * 1000:>8.1f} {t / one:>10.2f}x {t * 1000 / n:>9.1f}")


if __name__ == "__main__":
    main()
