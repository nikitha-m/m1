"""Time 1 quantized matrix multiply (1 real weight of the 3B) for n tokens.

If the time jumps between 4 and 6 tokens here too, the jump in
profile_batch.py comes from this kernel.

Usage:
    .venv/bin/python -m profiling.profile_kernel
"""

import statistics
import time

import mlx.core as mx
from mlx_lm import load

from spec_decode import TARGET

SIZES = [1, 2, 3, 4, 5, 6, 8, 16, 32]
REPEATS = 50
CALLS = 20  # multiplies per timing, so the fixed GPU start/stop wait is shared


def time_matmul(layer, n):
    dim = layer.weight.shape[1] * 32 // layer.bits
    xs = [mx.random.normal((1, n, dim)).astype(mx.float16) for _ in range(CALLS)]
    mx.eval(xs)
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        mx.eval([layer(x) for x in xs])
        times.append((time.perf_counter() - start) / CALLS)
    return statistics.median(times)


def main():
    model, _ = load(TARGET)
    layer = model.model.layers[0].mlp.gate_proj  # 3072 -> 8192, 8-bit
    mx.eval(layer.parameters())
    mb = sum(v.nbytes for v in (layer.weight, layer.scales, layer.biases)) / 1e6
    time_matmul(layer, 1)  # warm-up
    print(f"weight: {mb:.1f} MB\n{'tokens':>6} {'us':>7} {'vs 1':>6} {'GB/s':>6}")
    one = time_matmul(layer, 1)
    for n in SIZES:
        t = time_matmul(layer, n)
        print(f"{n:>6} {t * 1e6:>7.0f} {t / one:>5.2f}x {mb / 1e3 / t:>6.1f}")


if __name__ == "__main__":
    main()
