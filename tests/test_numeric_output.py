import unittest

import torch

from config import ModelConfig
from model.document_transformer import DocumentTransformer


class _DummyFrozenLM:
    def encode(self, texts):
        return torch.zeros(len(texts), 384)


class NumericOutputTest(unittest.TestCase):
    def test_numeric_loss_l1_is_supported(self):
        config = ModelConfig(numeric_loss="l1")
        model = DocumentTransformer(config, _DummyFrozenLM())
        out = torch.zeros(1, 1, config.d_model)
        loss = model.compute_loss(out, [[object()]], [{0: (1.25, "number")}])

        self.assertTrue(torch.isfinite(loss))

    def test_numeric_huber_delta_changes_loss(self):
        out = torch.zeros(1, 1, ModelConfig().d_model)

        small_delta = ModelConfig(numeric_loss="huber", numeric_huber_delta=0.25)
        small_model = DocumentTransformer(small_delta, _DummyFrozenLM())
        small_loss = small_model.compute_loss(out, [[object()]], [{0: (3.0, "number")}])

        large_delta = ModelConfig(numeric_loss="huber", numeric_huber_delta=2.0)
        large_model = DocumentTransformer(large_delta, _DummyFrozenLM())
        large_loss = large_model.compute_loss(out, [[object()]], [{0: (3.0, "number")}])

        self.assertNotAlmostEqual(float(small_loss.detach()), float(large_loss.detach()))

    def test_unknown_numeric_loss_raises(self):
        config = ModelConfig(numeric_loss="bogus")
        model = DocumentTransformer(config, _DummyFrozenLM())
        out = torch.zeros(1, 1, config.d_model)

        with self.assertRaisesRegex(ValueError, "Unknown numeric_loss"):
            model.compute_loss(out, [[object()]], [{0: (1.25, "number")}])


if __name__ == "__main__":
    unittest.main()
