"""Self-speculative draft: the target model, run with only some of its layers.

The draft shares the target's weights, so it uses no extra memory for weights.
It acts like a model, so spec_decode.py can use it in place of the 1B draft.
"""

from mlx_lm.models.base import create_attention_mask


class LayerSubset:
    def __init__(self, model, layer_ids):
        self.model = model
        # make_prompt_cache reads .layers: it makes 1 KV cache per chosen layer.
        self.layers = [model.model.layers[i] for i in layer_ids]

    def __call__(self, inputs, cache):
        inner = self.model.model
        h = inner.embed_tokens(inputs)
        mask = create_attention_mask(h, cache[0])  # Llama 3.2: all layers use full attention
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)
        h = inner.norm(h)
        # Llama 3.2 uses the same table for input words and output words ("tied").
        return inner.embed_tokens.as_linear(h)
