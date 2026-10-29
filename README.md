| Setup | Engine / code | RAM (GB) | Output tok/s |
|---|---|---|---|
| 3B Q4_K_M | Ollama | 2.34 | 25.6 |
| 3B q3_K_M | Ollama | 2.01 | 25.4 |
| 3B q4_0 | Ollama | 2.24 | 28.4 |
| 3B q8_0 | Ollama | 3.74 | 16.2 |
| 3B 8-bit | MLX | 3.93 | 17.3 |
| 3B 4-bit | MLX | – | 24.4 |
| 3B 8-bit + 1B 8-bit draft, k=1 | spec_decode.py | 4.88 | 16.1 |
| 3B 8-bit + 1B 4-bit draft, k=2 | mlx_lm speculative decoding | – | 21.0 |
| 3B 8-bit + 1B 4-bit draft, k=2 | spec_decode.py | – | 20.0 |
| 3B 8-bit + 1B 4-bit draft, k=2, prefill fix | spec_decode.py | – | 20.2 |
| 3B 8-bit + 1B 4-bit draft, k=2, one sync per round | spec_decode.py | – | 21.3 |
| 3B 4-bit + 1B 4-bit draft, k=2 | spec_decode.py | – | 20.6 |
| 3B 8-bit + 1B 4-bit draft, k=2 | vLLM Metal | does not fit in 8 GB | – |
| 3B 8-bit, self-speculative (19 of 28 layers), k=1 | layer_skip.py | – | 10.6 |
| 3B 8-bit, 12 layers in RAM, 16 from SSD | stream_model.py | 2.18 | 1.65 |
| 3B 8-bit, 6 layers in RAM, 22 from SSD (8-bit) + 1B 4-bit draft, k=2 | spec_stream.py | 2.26 | 1.68 |
| 3B 8-bit, 6 layers in RAM, 22 from SSD (4-bit) + 1B 4-bit draft, k=4 | spec_stream.py | 2.24 | 4.7–5.5 |
