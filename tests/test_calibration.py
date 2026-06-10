import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from calibration import apply_calibration_report
from eval import apply_numeric_postprocessing
from scripts.calibrate_matbench_gap import (
    _apply_binned_residual,
    _collect_checkpoint_predictions,
    _fit_cv_binned_residual,
    _parse_float_list,
    _parse_int_list,
)


class CalibrationReportTests(unittest.TestCase):
    def test_applies_scalar_and_binned_residual_report(self):
        report = {
            "calibration": {
                "prediction_scale": 0.9,
                "prediction_bias": 0.1,
                "prediction_min_value": 0.0,
                "prediction_zero_threshold": 0.2,
            },
            "alternative_calibrations": {
                "binned_residual": {
                    "mapping": {
                        "centers": [0.0, 1.0],
                        "corrections": [0.0, 0.1],
                    },
                    "calibration": {
                        "prediction_zero_threshold": 0.3,
                    },
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            path.write_text(json.dumps(report))
            config = SimpleNamespace()

            apply_calibration_report(config, path)

        self.assertEqual(config.prediction_scale, 0.9)
        self.assertEqual(config.prediction_bias, 0.1)
        self.assertEqual(config.prediction_min_value, 0.0)
        self.assertEqual(config.prediction_zero_threshold, 0.3)
        self.assertEqual(config.prediction_residual_centers, [0.0, 1.0])
        self.assertEqual(config.prediction_residual_corrections, [0.0, 0.1])

    def test_runtime_postprocessing_matches_calibration_residual_mapping(self):
        config = SimpleNamespace(
            prediction_scale=0.9,
            prediction_bias=0.1,
            prediction_min_value=0.0,
            prediction_residual_centers=[0.0, 1.0, 2.0],
            prediction_residual_corrections=[0.0, 0.2, -0.2],
            prediction_zero_threshold=None,
        )
        raw_preds = [-1.0, 0.5, 1.5, 3.0]
        scaled = [max(p * 0.9 + 0.1, 0.0) for p in raw_preds]
        mapping = {
            "centers": config.prediction_residual_centers,
            "corrections": config.prediction_residual_corrections,
        }
        expected = _apply_binned_residual(
            __import__("numpy").asarray(scaled),
            mapping,
            config.prediction_min_value,
        )

        actual = [apply_numeric_postprocessing(p, config) for p in raw_preds]

        for lhs, rhs in zip(actual, expected):
            self.assertAlmostEqual(lhs, rhs)

    def test_cv_binned_residual_selects_candidate_from_train_only_grid(self):
        np = __import__("numpy")
        preds = np.linspace(0.0, 3.0, 60)
        targets = preds + np.where(preds < 1.5, 0.1, -0.1)

        best, candidates, mapping = _fit_cv_binned_residual(
            preds,
            targets,
            bins_grid=[2, 3],
            shrinkage_grid=[0.0, 10.0],
            n_folds=3,
            min_bin_size=5,
            min_value=0.0,
        )

        self.assertIn(best["bins"], [2, 3])
        self.assertIn(best["shrinkage"], [0.0, 10.0])
        self.assertEqual(len(candidates), 4)
        self.assertTrue(mapping["centers"])

    def test_grid_parsers(self):
        self.assertEqual(_parse_int_list("4,6,8"), [4, 6, 8])
        self.assertEqual(_parse_float_list("1,2.5"), [1.0, 2.5])

    def test_collect_checkpoint_predictions_averages_ensemble(self):
        class Head:
            def __init__(self, model):
                self.model = model

            def predict_number(self, mask_repr):
                return torch.full((mask_repr.shape[0],), self.model.value)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = 0.0
                self.decode_head = Head(self)

            def load_state_dict(self, state):
                self.value = float(state["value"])

            def eval(self):
                return self

            def forward(self, batched_leaves, padding_mask, bias_indices=None):
                return torch.zeros(len(batched_leaves), 1, 2)

        loader = [
            (
                [[object()]],
                [{0: (2.0, "number")}],
                torch.zeros(1, 1, dtype=torch.bool),
                {},
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path1 = Path(tmp) / "a.pth"
            path2 = Path(tmp) / "b.pth"
            torch.save({"model": {"value": 1.0}}, path1)
            torch.save({"model": {"value": 3.0}}, path2)

            train_preds, train_targets, val_preds, val_targets = (
                _collect_checkpoint_predictions(
                    Model(),
                    [path1, path2],
                    loader,
                    loader,
                    "cpu",
                )
            )

        self.assertEqual(train_preds.tolist(), [2.0])
        self.assertEqual(val_preds.tolist(), [2.0])
        self.assertEqual(train_targets.tolist(), [2.0])
        self.assertEqual(val_targets.tolist(), [2.0])


if __name__ == "__main__":
    unittest.main()
