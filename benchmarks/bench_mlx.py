"""Measure speed and memory of an MLX model, same prompts as bench.py.

Usage:
    .venv/bin/python -m benchmarks.bench_mlx --model models/llama3.2-3b-mlx-8bit --label mlx_8bit
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from benchmarks.bench import OPTIONS, PROMPTS


def generate(model, tokenizer, prompt):
    # Same chat format Ollama uses, so the model sees a normal conversation.
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
    )
    sampler = make_sampler(temp=OPTIONS["temperature"])  # temp 0 = always pick the top word
    start = time.perf_counter()
    first = None
    answer = ""
    for r in stream_generate(model, tokenizer, text, max_tokens=OPTIONS["num_predict"], sampler=sampler):
        if first is None:
            first = time.perf_counter() - start
        answer += r.text
    return {
        "output_tokens_per_s": r.generation_tps,
        "input_tokens_per_s": r.prompt_tps,
        "time_to_first_token_s": first,
        "total_s": time.perf_counter() - start,
        "input_tokens": r.prompt_tokens,
        "output_tokens": r.generation_tokens,
        "answer": answer,
        "peak_memory_gb": r.peak_memory,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--runs", type=int, default=5)
    args = p.parse_args()

    t = time.perf_counter()
    model, tokenizer = load(args.model)
    cold_load_s = time.perf_counter() - t
    print(f"Cold load time: {cold_load_s:.2f} s")
    generate(model, tokenizer, PROMPTS["short"])  # warm-up, not counted

    results = {}
    peak = 0.0
    for name, prompt in PROMPTS.items():
        runs = [generate(model, tokenizer, f"(Request {i})\n{prompt}") for i in range(args.runs)]
        keys = ["output_tokens_per_s", "input_tokens_per_s", "time_to_first_token_s", "total_s"]
        results[name] = {k: statistics.median(r[k] for r in runs) for k in keys}
        for k in ["input_tokens", "output_tokens", "answer"]:
            results[name][k] = runs[0][k]
        peak = max(peak, max(r["peak_memory_gb"] for r in runs))

    summary = {
        "model": args.model,
        "label": args.label,
        "engine": "mlx",
        "runs_per_prompt": args.runs,
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "cold_load_s": cold_load_s,
        "memory_gb": peak,
        "prompts": results,
    }
    out = Path("results") / f"{args.label}.json"
    out.write_text(json.dumps(summary, indent=2))

    print(f"Peak memory:    {peak:.2f} GB\n")
    print(f"{'prompt':<12} {'out tok/s':>10} {'in tok/s':>10} {'1st token s':>12} {'total s':>8}")
    for name, r in results.items():
        print(
            f"{name:<12} {r['output_tokens_per_s']:>10.1f} {r['input_tokens_per_s']:>10.1f}"
            f" {r['time_to_first_token_s']:>12.2f} {r['total_s']:>8.2f}"
        )
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
