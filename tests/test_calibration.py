import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from calibration import apply_calibration_report


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


if __name__ == "__main__":
    unittest.main()
