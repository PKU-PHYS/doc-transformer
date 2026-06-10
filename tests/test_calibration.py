import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from calibration import apply_calibration_report
from eval import apply_numeric_postprocessing
from scripts.calibrate_matbench_gap import _apply_binned_residual


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


if __name__ == "__main__":
    unittest.main()
