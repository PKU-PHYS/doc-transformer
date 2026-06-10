"""Utilities for applying saved numeric calibration reports."""

import json
from pathlib import Path


def apply_calibration_report(config, report_path, *, residual_key="binned_residual"):
    """Mutate a model config from a calibration JSON report.

    The scalar calibration is always applied. If the requested residual mapping
    exists, its piecewise correction and residual-specific zero threshold are
    applied as well.
    """
    with Path(report_path).open() as f:
        report = json.load(f)

    scalar = report.get("calibration", {})
    config.prediction_scale = scalar.get("prediction_scale", 1.0)
    config.prediction_bias = scalar.get("prediction_bias", 0.0)
    config.prediction_min_value = scalar.get("prediction_min_value")
    config.prediction_zero_threshold = scalar.get("prediction_zero_threshold")
    config.prediction_residual_centers = None
    config.prediction_residual_corrections = None

    alternative = report.get("alternative_calibrations", {}).get(residual_key)
    if alternative is not None:
        mapping = alternative.get("mapping", {})
        config.prediction_residual_centers = mapping.get("centers")
        config.prediction_residual_corrections = mapping.get("corrections")
        residual_calibration = alternative.get("calibration", {})
        if "prediction_zero_threshold" in residual_calibration:
            config.prediction_zero_threshold = residual_calibration[
                "prediction_zero_threshold"
            ]

    return report
