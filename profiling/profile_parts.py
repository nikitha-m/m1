"""Time each part of 1 layer of the 3B for 1 token, and its memory speed (GB/s).

Usage:
    .venv/bin/python profile_parts.py
"""

import statistics
import time

import mlx.core as mx
from mlx_lm import load

from spec_decode import TARGET

M1_GBS = 68
REPEATS = 50
CALLS = 20  # calls per timing, so the fixed GPU start/stop wait is shared


def time_op(fn, dim):
    # Median time of fn on 1 token, in seconds.
    xs = [mx.random.normal((1, 1, dim)).astype(mx.float16) for _ in range(CALLS)]
    mx.eval(xs)
    fn(xs[0])  # warm-up
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        mx.eval([fn(x) for x in xs])
        times.append((time.perf_counter() - start) / CALLS)
    return statistics.median(times)


def nbytes(module):
    return sum(v.nbytes for v in (module.weight, module.scales, module.biases))


def main():
    model, _ = load(TARGET)
    mx.eval(model.parameters())
    layer = model.model.layers[0]
    attn, mlp = layer.self_attn, layer.mlp
    dim, ffn = model.args.hidden_size, model.args.intermediate_size
    parts = {
        "q_proj": (attn.q_proj, dim), "k_proj": (attn.k_proj, dim),
        "v_proj": (attn.v_proj, dim), "o_proj": (attn.o_proj, dim),
        "gate_proj": (mlp.gate_proj, dim), "up_proj": (mlp.up_proj, dim),
        "down_proj": (mlp.down_proj, ffn),
    }
    print(f"{'part':<12} {'MB':>6} {'us':>6} {'GB/s':>6} {'% of 68':>8}")
    layer_us = 0
    for name, (mod, d) in parts.items():
        t = time_op(mod, d)
        layer_us += t * 1e6
        mb = nbytes(mod) / 1e6
        print(f"{name:<12} {mb:>6.1f} {t * 1e6:>6.0f} {mb / 1e3 / t:>6.1f} {mb / 1e3 / t / M1_GBS:>8.0%}")

    whole = time_op(lambda x: layer(x, None, None), dim) * 1e6
    print(f"\nsum of 7 multiplies: {layer_us:6.0f} us")
    print(f"whole layer:         {whole:6.0f} us  (rest = norms, rope, attention: {whole - layer_us:.0f} us)")

    emb = model.model.embed_tokens
    head = time_op(emb.as_linear, dim)
    mb = nbytes(emb) / 1e6
    print(f"output head:         {head * 1e6:6.0f} us  ({mb:.0f} MB, {mb / 1e3 / head:.1f} GB/s)")
    est = (28 * whole + head * 1e6) / 1e3
    print(f"\nestimate for 1 token: 28 x layer + head = {est:.1f} ms")


if __name__ == "__main__":
    main()
