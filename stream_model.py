"""Run the 3B with only the first N layers in RAM. The other layers are read from the SSD for every token.

Needs the per-layer files from split_layers.py.
Usage:
    .venv/bin/python stream_model.py --in-ram 12      # 12 layers in RAM, 16 from the SSD
    .venv/bin/python stream_model.py --in-ram 28      # all in RAM (for comparison)
"""

import argparse
import fcntl
import json
import struct
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_map
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from mem_report import process_gb

MODEL = "models/llama3.2-3b-mlx-8bit"
DTYPES = {"U32": (np.uint32, None), "BF16": (np.uint16, mx.bfloat16), "F16": (np.float16, None), "F32": (np.float32, None)}
STATS = {"ssd_s": 0.0, "ssd_gb": 0.0}


def read_from_ssd(path):
    # Read a safetensors file. F_NOCACHE: macOS must not keep it in its file cache,
    # or the next "SSD read" would really come from RAM and the speed would be fake.
    t = time.perf_counter()
    with open(path, "rb", buffering=0) as f:
        fcntl.fcntl(f.fileno(), fcntl.F_NOCACHE, 1)
        data = f.read()
    n = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8 + n])
    header.pop("__metadata__", None)
    out = {}
    for name, info in header.items():
        a, b = info["data_offsets"]
        np_type, view = DTYPES[info["dtype"]]
        arr = np.frombuffer(data, np_type, (b - a) // np.dtype(np_type).itemsize, 8 + n + a)
        arr = mx.array(arr.reshape(info["shape"]))
        out[name] = arr.view(view) if view else arr
    STATS["ssd_s"] += time.perf_counter() - t
    STATS["ssd_gb"] += len(data) / 1e9
    return out


def empty(module):
    # Replace every weight with a tiny placeholder, so the real weights leave RAM.
    module.update(tree_map(lambda _: mx.array(0), module.parameters()))


class StreamedLayer(nn.Module):
    # Wraps one layer: read its weights from the SSD, run it, then free the weights again.
    def __init__(self, layer, path, prefix):
        super().__init__()
        self.layer, self.path, self.prefix = layer, path, prefix
        self.use_sliding = getattr(layer, "use_sliding", False)  # mlx_lm reads this to build the KV cache

    def __call__(self, x, mask=None, cache=None):
        ws = read_from_ssd(self.path)
        self.layer.load_weights([(k[len(self.prefix):], v) for k, v in ws.items()], strict=False)
        out = self.layer(x, mask, cache=cache)
        mx.eval(out, cache.state)  # compute now, so the weights are not needed later
        empty(self.layer)
        return out


def load_streamed(in_ram, ssd_bits=8):
    # The 3B with only the first `in_ram` layers in RAM (8-bit); the others become StreamedLayers.
    # ssd_bits=4: the SSD layers are read from the 4-bit model instead, so half the bytes.
    mx.set_cache_limit(256 * 2**20)  # MLX must not keep freed layers in its buffer cache
    model, tok = load(MODEL, lazy=True)  # lazy: nothing is read yet
    ssd_dir = MODEL if ssd_bits == 8 else MODEL.replace("8bit", f"{ssd_bits}bit")
    layers = model.model.layers
    for i in range(in_ram, len(layers)):
        empty(layers[i])
        for _, m in layers[i].named_modules():
            if isinstance(m, nn.QuantizedLinear):
                m.bits = ssd_bits  # each matrix must unpack its numbers with the right bit width
        layers[i] = StreamedLayer(layers[i], f"{ssd_dir}/layers/layer_{i:02}.safetensors", f"model.layers.{i}.")
    mx.eval(model.parameters())  # read only the RAM part now
    return model, tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-ram", type=int, default=12, help="how many layers stay in RAM")
    parser.add_argument("--tokens", type=int, default=20)
    args = parser.parse_args()

    model, tok = load_streamed(args.in_ram)
    layers = model.model.layers
    print(f"layers in RAM: {args.in_ram}, from SSD: {len(layers) - args.in_ram}")
    print(f"after loading: active {mx.get_active_memory() / 1e9:.2f} GB, process {process_gb():.2f} GB")

    prompt = tok.apply_chat_template([{"role": "user", "content": "Explain how a CPU works in a short paragraph."}],
                                     add_generation_prompt=True)
    cache = make_prompt_cache(model)
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    y = mx.argmax(model(mx.array(prompt)[None], cache=cache)[0, -1])
    out = [y.item()]
    t1 = time.perf_counter()
    STATS["ssd_s"] = STATS["ssd_gb"] = 0.0  # count the SSD only while writing tokens
    for _ in range(args.tokens - 1):
        y = mx.argmax(model(y.reshape(1, 1), cache=cache)[0, -1])
        out.append(y.item())
    t2 = time.perf_counter()

    n = len(out) - 1
    print(tok.decode(out))
    print(f"\nfirst token: {t1 - t0:.2f} s   then {n / (t2 - t1):.2f} tok/s")
    print(f"SSD: {STATS['ssd_gb'] / n:.2f} GB per token, {STATS['ssd_s'] / (t2 - t1):.0%} of the time, "
          f"{STATS['ssd_gb'] / max(STATS['ssd_s'], 1e-9):.2f} GB/s")
    print(f"peak MLX memory {mx.get_peak_memory() / 1e9:.2f} GB, process {process_gb():.2f} GB")


if __name__ == "__main__":
    main()
