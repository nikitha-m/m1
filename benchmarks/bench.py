"""Measure speed and memory of an Ollama model.

Usage:
    python3 bench.py                          # baseline for llama3.2
    python3 bench.py --model llama3.2 --label my-change --runs 5
"""

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

OLLAMA = "http://localhost:11434"

# Fixed prompts so every test is comparable. Mix of short and long inputs.
PROMPTS = {
    "short": "What is the capital of France? Answer in one sentence.",
    "explain": "Explain how a CPU works in a short paragraph.",
    "code": "Write a Python function that checks if a number is prime.",
    # Original, non-repeating text (~800 tokens). Repeated text made models refuse.
    "long_input": (
        "Summarize this text in two sentences:\n\n"
        "In November 2020, Apple released the M1, its first chip designed for Mac computers. "
        "Before that, Macs had used Intel processors for about fifteen years. The switch was a "
        "large risk, because every Mac application had been written for Intel's x86 instruction "
        "set, while the M1 uses the ARM instruction set found in iPhones and iPads.\n\n"
        "The M1 is a system on a chip. This means that the CPU, the GPU, the Neural Engine, the "
        "memory controller, and several other parts sit on one piece of silicon. In a traditional "
        "laptop, these parts are often separate chips on a circuit board, connected by wires that "
        "add delay and use power. Putting them together lets data move faster and more efficiently.\n\n"
        "The CPU has eight cores, split into two groups. Four performance cores handle demanding "
        "work such as compiling code or exporting video. Four efficiency cores handle light work "
        "such as checking email or playing music, and they use only a fraction of the power. The "
        "operating system decides which group gets each task, so the laptop stays fast when needed "
        "and saves battery the rest of the time.\n\n"
        "One of the most important design choices is unified memory. On most computers, the CPU "
        "and the GPU each have their own memory. When the GPU needs data that the CPU prepared, "
        "the data must be copied from one memory to the other, which takes time. On the M1, both "
        "processors read from the same pool of memory, so no copy is needed. This is especially "
        "useful for machine learning, where large arrays of numbers move between processors many "
        "times per second. The trade-off is that memory is fixed when the computer is built; a user "
        "cannot add more later.\n\n"
        "The Neural Engine is a separate unit with sixteen cores built only for the math used in "
        "neural networks. Apple says it can perform eleven trillion operations per second. Features "
        "such as face recognition in photos, live text in images, and voice dictation run on it. "
        "However, many open-source machine learning tools do not use the Neural Engine directly; "
        "they run on the GPU through Apple's Metal framework instead.\n\n"
        "To keep old applications working, Apple built a translation layer called Rosetta 2. When a "
        "user opens an Intel application for the first time, Rosetta 2 converts its instructions to "
        "ARM instructions and saves the result. Many reviewers were surprised to find that some "
        "translated applications ran faster on the M1 than they did on recent Intel Macs.\n\n"
        "Reviews of the first M1 laptops focused on battery life and heat. The MacBook Air with the "
        "M1 has no fan at all, yet it stays quiet and cool during most tasks. Apple claimed up to "
        "eighteen hours of video playback on a single charge, and independent tests found results "
        "close to that number.\n\n"
        "The M1 also had limits. It supported only one external display on most models, and the "
        "maximum memory was sixteen gigabytes. Later chips, such as the M1 Pro, M1 Max, and the M2 "
        "family, added more cores, more memory, and support for more displays. Still, the original "
        "M1 changed how people thought about laptop chips, and it pushed other companies to invest "
        "more in ARM designs for personal computers."
    ),
}

# Same settings every run: no randomness, same max answer length.
OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 200}


def post(path, body):
    req = urllib.request.Request(
        OLLAMA + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def get(path):
    with urllib.request.urlopen(OLLAMA + path) as resp:
        return json.load(resp)


def generate(model, prompt):
    r = post("/api/generate", {"model": model, "prompt": prompt, "stream": False, "options": OPTIONS})
    ns = 1e9  # Ollama reports times in nanoseconds
    return {
        "output_tokens_per_s": r["eval_count"] / (r["eval_duration"] / ns),
        "input_tokens_per_s": r["prompt_eval_count"] / (r["prompt_eval_duration"] / ns),
        "time_to_first_token_s": (r["load_duration"] + r["prompt_eval_duration"]) / ns,
        "total_s": r["total_duration"] / ns,
        "load_s": r["load_duration"] / ns,
        "input_tokens": r["prompt_eval_count"],
        "output_tokens": r["eval_count"],
        "answer": r["response"],
    }


def memory_gb(model):
    for m in get("/api/ps")["models"]:
        if m["name"].startswith(model):
            return m["size"] / 1e9
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llama3.2")
    p.add_argument("--label", default="baseline")
    p.add_argument("--runs", type=int, default=5)
    args = p.parse_args()

    # Cold start: unload the model, then time how long loading takes.
    post("/api/generate", {"model": args.model, "keep_alive": 0})
    time.sleep(1)
    cold = generate(args.model, PROMPTS["short"])
    print(f"Cold load time: {cold['load_s']:.2f} s")

    results = {}
    for name, prompt in PROMPTS.items():
        # Different first line each run, so Ollama can't reuse a cached prompt.
        runs = [generate(args.model, f"(Request {i})\n{prompt}") for i in range(args.runs)]
        keys = ["output_tokens_per_s", "input_tokens_per_s", "time_to_first_token_s", "total_s"]
        results[name] = {k: statistics.median(r[k] for r in runs) for k in keys}
        results[name]["input_tokens"] = runs[0]["input_tokens"]
        results[name]["output_tokens"] = runs[0]["output_tokens"]
        results[name]["answer"] = runs[0]["answer"]  # for checking quality later

    summary = {
        "model": args.model,
        "label": args.label,
        "runs_per_prompt": args.runs,
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "cold_load_s": cold["load_s"],
        "memory_gb": memory_gb(args.model),
        "prompts": results,
    }

    out = Path("results") / f"{args.label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print(f"Memory used:    {summary['memory_gb']:.2f} GB\n")
    print(f"{'prompt':<12} {'out tok/s':>10} {'in tok/s':>10} {'1st token s':>12} {'total s':>8}")
    for name, r in results.items():
        print(
            f"{name:<12} {r['output_tokens_per_s']:>10.1f} {r['input_tokens_per_s']:>10.1f}"
            f" {r['time_to_first_token_s']:>12.2f} {r['total_s']:>8.2f}"
        )
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
