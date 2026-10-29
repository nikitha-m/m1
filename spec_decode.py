"""

Usage:
    .venv/bin/python spec_decode.py --draft-tokens 3
"""

import argparse
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

TARGET = "models/llama3.2-3b-mlx-8bit"
DRAFT = "models/llama3.2-1b-mlx-8bit"

# Seconds spent in each part. greedy() ends in .tolist(), so the GPU work is done when it returns.
TIMES = {"prompt": 0.0, "draft": 0.0, "check": 0.0, "rounds": 0}

def greedy(model, tokens, cache):
    # returns score for each word
    logits = model(mx.array(tokens)[None], cache=cache)
    # picks the one with the highest score  
    return mx.argmax(logits[0], axis=-1).tolist()


def prefill(model, tokens, cache):
    # Fill the KV cache only. We never ask for word scores, so MLX (lazy) skips the
    # final layer that turns each position into 128k word scores.
    model(mx.array(tokens)[None], cache=cache)
    mx.eval([c.state for c in cache])  # waits until done, so the timer stays correct


def next_token(model, tokens, cache):
    # Top next token after the last position, kept on the GPU (no .tolist(), no wait).
    logits = model(tokens[None], cache=cache)
    return mx.argmax(logits[0, -1:], axis=-1)


def speculative_generate(target, draft, prompt, eos, k=3, max_tokens=256):
    # Initializing cache
    t_cache, d_cache = make_prompt_cache(target), make_prompt_cache(draft)

    # Read the prompt (all but the last token) into both caches as the last token gives first new prediction
    t0 = time.perf_counter()
    prefill(target, prompt[:-1], t_cache)
    prefill(draft, prompt[:-1], d_cache)
    TIMES["prompt"] += time.perf_counter() - t0
    y = prompt[-1]  
    pending = [y]  
    out, drafted, accepted = [], 0, 0
    while len(out) < max_tokens:
        # Fix 2: guesses stay on the GPU. async_eval starts each one without waiting,
        # so Python prepares the next step while the GPU works. One wait per round (.tolist below).
        # Timers: "draft" = time to SEND the draft work; "check" now also waits for it to finish.
        t0 = time.perf_counter()
        tok, drafts = mx.array(pending), []
        for _ in range(k):
            tok = next_token(draft, tok, d_cache)
            mx.async_eval(tok)
            drafts.append(tok)
        guesses_mx = mx.concatenate(drafts)
        t1 = time.perf_counter()
        logits = target(mx.concatenate([mx.array([y]), guesses_mx])[None], cache=t_cache)
        checks_mx = mx.argmax(logits[0], axis=-1)
        mx.eval(guesses_mx, checks_mx)  # the one wait per round
        guesses, checks = guesses_mx.tolist(), checks_mx.tolist()
        t2 = time.perf_counter()
        TIMES["draft"] += t1 - t0
        TIMES["check"] += t2 - t1
        TIMES["rounds"] += 1
        n = 0
        while n < k and guesses[n] == checks[n]:
            n += 1
        drafted, accepted = drafted + k, accepted + n
        # Remove the rejected guesses from both caches.
        trim_prompt_cache(t_cache, k - n)
        if n < k:
            trim_prompt_cache(d_cache, k - 1 - n)
            pending = [checks[n]]
        else:
            pending = [guesses[-1], checks[n]]  # draft never saw its last guess
        y = checks[n]
        for tok in guesses[:n] + [y]:
            out.append(tok)
            if tok in eos:
                return out, accepted / drafted
    return out, accepted / drafted


def main():
    # Step 1: Read stuff from cmd line
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-tokens", type=int, default=3)
    parser.add_argument("--draft", default=DRAFT, help="path to the draft model")
    parser.add_argument("--prompt", default="Explain how a CPU works in a short paragraph.")
    args = parser.parse_args()

    # Step 2: Model only accepts numbers, so you convert your input prompt into numbers
    # load(TARGET) gives model and its tokenizer
    target, tokenizer = load(TARGET)
    draft, _ = load(args.draft)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], add_generation_prompt=True
    )

    # This is for warmup, output thrown away
    speculative_generate(target, draft, prompt, tokenizer.eos_token_ids, k=args.draft_tokens, max_tokens=8)
    
    for key in TIMES:  # forget the warm-up times
        TIMES[key] = 0
    start = time.perf_counter()
    # This is the actual call
    out, rate = speculative_generate(
        target, draft, prompt, tokenizer.eos_token_ids, k=args.draft_tokens
    )

    secs = time.perf_counter() - start
    print(tokenizer.decode(out))
    print(f"\n{len(out)} tokens in {secs:.2f} s = {len(out) / secs:.1f} tok/s")
    print(f"draft guesses accepted: {rate:.0%}")
    r = TIMES["rounds"]
    other = secs - TIMES["prompt"] - TIMES["draft"] - TIMES["check"]
    print(f"\n{r} rounds, {len(out) / r:.2f} tokens per round")
    for name in ("prompt", "draft", "check"):
        print(f"{name:>6}: {TIMES[name]:6.2f} s ({TIMES[name] / secs:4.0%})  {TIMES[name] / r * 1000:6.1f} ms/round")
    print(f" other: {other:6.2f} s ({other / secs:4.0%})  {other / r * 1000:6.1f} ms/round  (trim + Python)")


if __name__ == "__main__":
    main()
