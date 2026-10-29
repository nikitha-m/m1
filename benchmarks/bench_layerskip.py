"""Self-speculative decoding: the 3B drafts with some of its own layers, then checks with all.

Usage:
    .venv/bin/python -m benchmarks.bench_layerskip
"""

import json
import time
from pathlib import Path

from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from layer_skip import LayerSubset
from spec_decode import TARGET, greedy, speculative_generate

PROMPT = "Explain how a CPU works in a short paragraph."
N = 28  # layers in Llama 3.2 3B

# name -> layers the draft runs
DRAFTS = {
    "early exit 14": list(range(14)),
    "early exit 21": list(range(21)),
    "every 2nd (14)": list(range(0, N, 2)),
    "drop every 3rd (19)": [i for i in range(N) if i % 3 != 2],
    "drop middle 12-19 (20)": [i for i in range(N) if not 12 <= i <= 19],
    "drop middle 14-17 (24)": [i for i in range(N) if not 14 <= i <= 17],
}


def plain_generate(target, prompt, eos, max_tokens=256):
    # Normal decoding: 1 full pass of the 3B for each token.
    cache = make_prompt_cache(target)
    y = greedy(target, prompt, cache)[-1]
    out = [y]
    while len(out) < max_tokens and y not in eos:
        y = greedy(target, [y], cache)[-1]
        out.append(y)
    return out


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def main():
    target, tokenizer = load(TARGET)
    eos = tokenizer.eos_token_ids
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True
    )
    plain_generate(target, prompt, eos, max_tokens=8)  # warm-up: load weights from disk
    ref, secs = timed(lambda: plain_generate(target, prompt, eos))
    base = len(ref) / secs
    print(f"{'3B alone':<24} {base:5.1f} tok/s")

    rows = []
    for name, layers in DRAFTS.items():
        draft = LayerSubset(target, layers)
        for k in (1, 2, 3):
            (out, rate), secs = timed(
                lambda: speculative_generate(target, draft, prompt, eos, k=k)
            )
            row = {"draft": name, "layers": len(layers), "k": k, "tok_s": round(len(out) / secs, 1),
                   "accepted": round(rate, 2), "same_text": out == ref}
            rows.append(row)
            print(f"{name:<24} k={k}  {row['tok_s']:5.1f} tok/s  accepted {rate:4.0%}  same text: {row['same_text']}")

    out = Path("results") / "layerskip.json"
    out.write_text(json.dumps({"prompt": PROMPT, "baseline_tok_s": round(base, 1),
                               "date": time.strftime("%Y-%m-%d %H:%M"), "runs": rows}, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
