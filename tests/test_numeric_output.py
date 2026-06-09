import unittest

import torch

from config import ModelConfig
from model.document_transformer import DocumentTransformer
from model.decode_head import DecodeHead


class _DummyFrozenLM:
    def encode(self, texts):
        return torch.zeros(len(texts), 384)


class NumericOutputTest(unittest.TestCase):
    def test_linear_output_is_raw_head(self):
        config = ModelConfig(d_model=2, numeric_output="linear")
        head = DecodeHead(config)
        with torch.no_grad():
            head.num_head.weight.fill_(0.0)
            head.num_head.bias.fill_(-0.5)

        pred = head.predict_number(torch.zeros(3, 2))
        self.assertTrue(torch.allclose(pred, torch.full((3,), -0.5)))

    def test_softplus_output_is_non_negative(self):
        config = ModelConfig(d_model=2, numeric_output="softplus", numeric_softplus_beta=1.0)
        head = DecodeHead(config)
        with torch.no_grad():
            head.num_head.weight.fill_(0.0)
            head.num_head.bias.fill_(-10.0)

        pred = head.predict_number(torch.zeros(3, 2))
        self.assertTrue(torch.all(pred >= 0))

    def test_numeric_loss_l1_is_supported(self):
        config = ModelConfig(numeric_loss="l1")
        model = DocumentTransformer(config, _DummyFrozenLM())
        out = torch.zeros(1, 1, config.d_model)
        loss = model.compute_loss(out, [[object()]], [{0: (1.25, "number")}])

        self.assertTrue(torch.isfinite(loss))

    def test_unknown_numeric_loss_raises(self):
        config = ModelConfig(numeric_loss="bogus")
        model = DocumentTransformer(config, _DummyFrozenLM())
        out = torch.zeros(1, 1, config.d_model)

        with self.assertRaisesRegex(ValueError, "Unknown numeric_loss"):
            model.compute_loss(out, [[object()]], [{0: (1.25, "number")}])


if __name__ == "__main__":
    unittest.main()
