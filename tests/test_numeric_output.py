import unittest

import torch

from config import ModelConfig
from model.decode_head import DecodeHead


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


if __name__ == "__main__":
    unittest.main()
