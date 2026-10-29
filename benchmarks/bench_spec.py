"""Compare the 3B alone vs speculative decoding, on 10 prompts.

Usage (close Cursor and Ollama first, the 8 GB M1 needs the memory):
    .venv/bin/python -m benchmarks.bench_spec --draft models/llama3.2-1b-mlx-4bit --draft-tokens 2
"""

import argparse
import json
import time
from pathlib import Path

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from benchmarks.bench import OPTIONS, PROMPTS
from spec_decode import TARGET, speculative_generate

# The 4 prompts from bench.py, plus 6 more kinds of task.
PROMPTS = dict(PROMPTS) | {
    "translate": "Translate to French: 'The train leaves at eight, so we should eat breakfast early.'",
    "list": "List five tips for staying focused while studying.",
    "math": "A shop sells pens at 3 dollars each. Sam buys 7 pens and pays with a 50 dollar note. How much change does Sam get? Show the steps.",
    "email": "Write a short email to a teacher asking to move a meeting from Monday to Wednesday.",
    "story": "Write a four-sentence story about a robot that learns to paint.",
    "review": "Find the bug in this Python code and fix it:\n\ndef average(nums):\n    return sum(nums) / len(nums) + 1",
}
MAX_TOKENS = OPTIONS["num_predict"]


def baseline(model, tokenizer, prompt):
    # The 3B alone, with mlx_lm's own generate loop. Returns token ids.
    sampler = make_sampler(temp=0.0)
    return [r.token for r in stream_generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS, sampler=sampler)]


def mlx_lm_spec(target, draft, prompt, eos, k, max_tokens, tokenizer):
    # mlx_lm's built-in speculative decoding, to compare with ours.
    # "rate" here = share of output tokens that came from the draft (mlx_lm does not report drafted count).
    sampler = make_sampler(temp=0.0)
    rs = list(stream_generate(target, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler,
                              draft_model=draft, num_draft_tokens=k))
    return [r.token for r in rs], sum(r.from_draft for r in rs) / max(len(rs), 1)


def timed(fn):
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--draft", default="models/llama3.2-1b-mlx-4bit")
    parser.add_argument("--draft-tokens", type=int, default=2)
    parser.add_argument("--label", default="spec_4bit_k2")
    parser.add_argument("--mlx-lm", action="store_true", help="use mlx_lm's speculative decoding instead of ours")
    args = parser.parse_args()

    target, tokenizer = load(args.target)
    draft, _ = load(args.draft)
    eos = tokenizer.eos_token_ids
    k = args.draft_tokens
    spec_fn = (lambda *a, **kw: mlx_lm_spec(*a, **kw, tokenizer=tokenizer)) if args.mlx_lm else speculative_generate

    results = {}
    for i, (name, text) in enumerate(PROMPTS.items()):
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True)
        if i == 0:  # warm up both models once: MLX loads weights from disk on first use
            baseline(target, tokenizer, prompt)
            spec_fn(target, draft, prompt, eos, k=k, max_tokens=8)
        base, base_s = timed(lambda: baseline(target, tokenizer, prompt))
        (spec, rate), spec_s = timed(lambda: spec_fn(target, draft, prompt, eos, k=k, max_tokens=MAX_TOKENS))
        # mlx_lm leaves the end token out; ours keeps it. Compare without it.
        base_ids = [t for t in base if t not in eos]
        spec_ids = [t for t in spec if t not in eos]
        results[name] = {
            "tokens": len(spec_ids),
            "base_tok_s": len(base_ids) / base_s,
            "spec_tok_s": len(spec_ids) / spec_s,
            "accepted": rate,
            "same_text": base_ids == spec_ids,
            "answer": tokenizer.decode(spec_ids),
        }
        r = results[name]
        print(f"{name:11} base {r['base_tok_s']:5.1f}  spec {r['spec_tok_s']:5.1f} tok/s  "
              f"accepted {rate:4.0%}  same text: {r['same_text']}", flush=True)

    base_avg = sum(r["base_tok_s"] for r in results.values()) / len(results)
    spec_avg = sum(r["spec_tok_s"] for r in results.values()) / len(results)
    print(f"\naverage     base {base_avg:5.1f}  spec {spec_avg:5.1f} tok/s  ({spec_avg / base_avg - 1:+.0%})")

    out = Path("results") / f"{args.label}.json"
    out.write_text(json.dumps({"target": args.target, "draft": args.draft, "draft_tokens": k,
                               "date": time.strftime("%Y-%m-%d %H:%M"), "prompts": results}, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
