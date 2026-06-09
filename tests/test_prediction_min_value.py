import unittest
from types import SimpleNamespace

import torch

from eval import apply_numeric_postprocessing, evaluate


class _ConstantNumberHead:
    def __init__(self, value):
        self.value = value

    def predict_number(self, mask_repr):
        return torch.full((mask_repr.shape[0],), self.value, device=mask_repr.device)

    def predict_is_zero(self, mask_repr):
        return torch.full((mask_repr.shape[0],), 0.5, device=mask_repr.device)


class _ConstantModel(torch.nn.Module):
    def __init__(
        self,
        value,
        prediction_min_value,
        use_zero_head=False,
        zero_logit_threshold=0.0,
    ):
        super().__init__()
        self.config = SimpleNamespace(
            prediction_scale=1.0,
            prediction_bias=0.0,
            prediction_min_value=prediction_min_value,
            prediction_zero_threshold=None,
            use_zero_head=use_zero_head,
            zero_logit_threshold=zero_logit_threshold,
        )
        self.decode_head = _ConstantNumberHead(value)

    def forward(self, batched_leaves, padding_mask, bias_indices=None):
        batch = len(batched_leaves)
        return torch.zeros(batch, 1, 2, device=padding_mask.device)


def _single_number_batch(target):
    return [(
        [[object()]],
        [{0: (target, "number")}],
        torch.zeros(1, 1, dtype=torch.bool),
        {},
    )]


class PredictionMinValueTest(unittest.TestCase):
    def test_default_keeps_negative_prediction(self):
        model = _ConstantModel(value=-0.25, prediction_min_value=None)
        score = evaluate(model, _single_number_batch(0.0), "cpu", metric="mae")
        self.assertAlmostEqual(score, 0.25)

    def test_min_value_clamps_numeric_prediction(self):
        model = _ConstantModel(value=-0.25, prediction_min_value=0.0)
        score = evaluate(model, _single_number_batch(0.0), "cpu", metric="mae")
        self.assertAlmostEqual(score, 0.0)

    def test_postprocessing_order_is_bias_min_then_zero_threshold(self):
        config = SimpleNamespace(
            prediction_scale=2.0,
            prediction_bias=-0.2,
            prediction_min_value=0.0,
            prediction_zero_threshold=0.15,
        )
        self.assertAlmostEqual(apply_numeric_postprocessing(0.4, config), 0.6)
        self.assertAlmostEqual(apply_numeric_postprocessing(0.18, config), 0.16)
        self.assertAlmostEqual(apply_numeric_postprocessing(0.05, config), 0.0)

    def test_zero_head_threshold_can_gate_zero_prediction(self):
        model = _ConstantModel(
            value=1.0,
            prediction_min_value=None,
            use_zero_head=True,
            zero_logit_threshold=0.0,
        )
        score = evaluate(model, _single_number_batch(0.0), "cpu", metric="mae")
        self.assertAlmostEqual(score, 0.0)

        model.config.zero_logit_threshold = 1.0
        score = evaluate(model, _single_number_batch(0.0), "cpu", metric="mae")
        self.assertAlmostEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
