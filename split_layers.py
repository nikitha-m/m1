"""Split an MLX model into one weights file per layer, so each layer can be read from the SSD alone.

Output (inside <model>/layers/):
    base.safetensors        embedding table (also the lm_head) + final norm, always kept in RAM
    layer_00.safetensors    ... layer_27.safetensors

Usage:
    .venv/bin/python split_layers.py models/llama3.2-3b-mlx-8bit
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import mlx.core as mx


def main():
    model_dir = Path(sys.argv[1])
    out_dir = model_dir / "layers"
    out_dir.mkdir(exist_ok=True)

    weights = mx.load(str(model_dir / "model.safetensors"))
    groups = defaultdict(dict)
    for name, w in weights.items():
        m = re.match(r"model\.layers\.(\d+)\.", name)
        groups[f"layer_{int(m.group(1)):02}" if m else "base"][name] = w

    for group, ws in sorted(groups.items()):
        mx.save_safetensors(str(out_dir / f"{group}.safetensors"), ws)
        size = sum(w.nbytes for w in ws.values()) / 1e9
        print(f"{group:10} {len(ws):3} arrays  {size:.3f} GB")


if __name__ == "__main__":
    main()
