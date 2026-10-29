"""Run vLLM Metal's speculative decoding on the same 10 prompts as bench_spec.py.

Same models, same k, same 200-token limit, greedy. Time = full request (prompt + answer),
the same way bench_spec.py measures. Runs in its own venv:
    .venv-vllm-metal/bin/python -m benchmarks.bench_vllm_spec               # their speculative decoding
    .venv-vllm-metal/bin/python -m benchmarks.bench_vllm_spec --no-spec     # their 3B alone
"""

import argparse
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams

from benchmarks.bench_spec import MAX_TOKENS, PROMPTS

TARGET = "models/llama3.2-3b-mlx-8bit"
DRAFT = "models/llama3.2-1b-mlx-4bit"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--draft-tokens", type=int, default=2)
    parser.add_argument("--no-spec", action="store_true", help="run the 3B alone")
    parser.add_argument("--gpu-mem", type=float, default=1.0, help="fraction of the GPU memory limit vLLM may use")
    parser.add_argument("--mine", default="results/spec_4bit_k2.json", help="my results, to compare text")
    args = parser.parse_args()

    spec = None if args.no_spec else {"method": "draft_model", "model": DRAFT, "num_speculative_tokens": args.draft_tokens}
    llm = LLM(
        model=args.target,
        speculative_config=spec,
        max_model_len=2048,
        gpu_memory_utilization=args.gpu_mem,  # 8 GB M1: needs 1.0 for both models + KV cache
        async_scheduling=False,  # required for speculative decoding on Metal
    )
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    mine = json.loads(Path(args.mine).read_text())["prompts"]

    # Warm up once: the first request loads weights and compiles kernels.
    llm.chat([{"role": "user", "content": "Hi"}], params, use_tqdm=False)

    results = {}
    for name, text in PROMPTS.items():
        start = time.perf_counter()
        out = llm.chat([{"role": "user", "content": text}], params, use_tqdm=False)[0].outputs[0]
        secs = time.perf_counter() - start
        answer = out.text
        n = len(out.token_ids)
        results[name] = {
            "tokens": n,
            "tok_s": n / secs,
            "same_text_as_mine": answer.strip() == mine[name]["answer"].strip(),
            "answer": answer,
        }
        r = results[name]
        print(f"{name:11} vllm-metal {r['tok_s']:5.1f} tok/s  mine {mine[name]['spec_tok_s']:5.1f}  "
              f"tokens {n:3}  same text as mine: {r['same_text_as_mine']}", flush=True)

    theirs = sum(r["tok_s"] for r in results.values()) / len(results)
    ours = sum(mine[k]["spec_tok_s"] for k in results) / len(results)
    print(f"\naverage     vllm-metal {theirs:5.1f}  mine {ours:5.1f} tok/s  (mine is {ours / theirs - 1:+.0%})")

    label = "vllm_metal_base" if args.no_spec else f"vllm_metal_spec_k{args.draft_tokens}"
    label += "_" + Path(args.target).name
    out_path = Path("results") / f"{label}.json"
    out_path.write_text(json.dumps({"target": args.target, "draft": None if args.no_spec else DRAFT,
                                    "draft_tokens": None if args.no_spec else args.draft_tokens,
                                    "date": time.strftime("%Y-%m-%d %H:%M"), "prompts": results}, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
