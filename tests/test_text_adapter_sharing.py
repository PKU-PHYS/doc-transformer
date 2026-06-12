import unittest

import torch

from config import ModelConfig
from model.json_parser import LeafNode
from model.token_embedding import TokenEmbedding


class _DummyFrozenLM:
    def __init__(self, dim=4):
        self._dim = dim

    def dim(self):
        return self._dim

    def encode(self, texts):
        values = torch.arange(len(texts) * self._dim, dtype=torch.float32)
        return values.reshape(len(texts), self._dim)


def _number_leaf():
    return LeafNode(
        value=1.25,
        value_type="number",
        path=["material", "band_gap"],
        path_types=[0, 0],
        path_ids=[1, 2],
        group_ids=[],
    )


class TextAdapterSharingTest(unittest.TestCase):
    def test_default_path_gru_consumes_frozen_lm_dim(self):
        config = ModelConfig(d_model=6, frozen_lm_dim=4, n_fourier_feats=2)
        embedder = TokenEmbedding(config)

        self.assertEqual(embedder.path_encoder.gru.input_size, 4)
        self.assertEqual(embedder.path_encoder.node_type_emb.embedding_dim, 4)

    def test_shared_text_projection_feeds_path_gru_in_model_dim(self):
        config = ModelConfig(
            d_model=6,
            frozen_lm_dim=4,
            n_fourier_feats=2,
            share_text_proj_to_path=True,
        )
        embedder = TokenEmbedding(config)

        calls = []

        def _hook(_module, inputs, output):
            calls.append((tuple(inputs[0].shape), tuple(output.shape)))

        handle = embedder.value_encoder.text_proj.register_forward_hook(_hook)
        try:
            out = embedder([_number_leaf()], _DummyFrozenLM(dim=4))
        finally:
            handle.remove()

        self.assertEqual(tuple(out.shape), (1, 6))
        self.assertEqual(embedder.path_encoder.gru.input_size, 6)
        self.assertEqual(embedder.path_encoder.node_type_emb.embedding_dim, 6)
        self.assertEqual(calls, [((1, 2, 4), (1, 2, 6))])


if __name__ == "__main__":
    unittest.main()
