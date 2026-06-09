"""Fit train-only scalar calibration for matbench_mp_gap checkpoints.

The script keeps the official Matbench test fold blind by default: it loads test
documents without targets and only reports train/internal-val metrics.
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import get_configs
from data.base import collate_fn
from data.matbench import MatbenchDataset, MatbenchLoader
from data.matbench.configs import (
    build_cache_tag,
    build_dataset_options,
    get_matbench_config,
    register_args as register_matbench_args,
    validate_args as validate_matbench_args,
)
from eval import apply_numeric_postprocessing
from model.document_transformer import DocumentTransformer
from model.frozen_lm import FrozenLM


def _mae(preds, targets):
    return float(np.mean(np.abs(preds - targets)))


def _collect_predictions(model, loader, device):
    preds = []
    targets = []
    model.eval()
    with torch.no_grad():
        for batched_leaves, batched_masks, padding_mask, bias_indices in loader:
            padding_mask = padding_mask.to(device)
            bias_indices = {k: v.to(device) for k, v in bias_indices.items()}
            with torch.amp.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=(device == "cuda" and torch.cuda.is_bf16_supported()),
            ):
                out = model(batched_leaves, padding_mask, bias_indices=bias_indices)

            for sample_idx, masks in enumerate(batched_masks):
                for mask_pos, (true_val, val_type) in masks.items():
                    if val_type != "number":
                        continue
                    mask_repr = out[sample_idx, mask_pos].unsqueeze(0)
                    pred = model.decode_head.predict_number(mask_repr).float().item()
                    preds.append(pred)
                    targets.append(float(true_val))
    return np.asarray(preds, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def _postprocess_array(preds, *, bias=0.0, min_value=None, zero_threshold=None):
    cfg = argparse.Namespace(
        prediction_bias=bias,
        prediction_min_value=min_value,
        prediction_zero_threshold=zero_threshold,
    )
    return np.asarray([apply_numeric_postprocessing(float(p), cfg) for p in preds])


def _fit_median_bias(preds, targets):
    return float(np.median(targets - preds))


def _fit_zero_threshold(preds, targets):
    quantile_grid = np.quantile(preds, np.linspace(0.0, 0.65, 326))
    fixed_grid = np.linspace(0.0, 1.0, 501)
    candidates = np.unique(np.concatenate([quantile_grid, fixed_grid]))
    best_mae = float("inf")
    best_threshold = None
    for threshold in candidates:
        thresholded = preds.copy()
        thresholded[preds <= threshold] = 0.0
        score = _mae(thresholded, targets)
        if score < best_mae:
            best_mae = score
            best_threshold = float(threshold)
    return best_threshold, best_mae


def _summarize_split(raw_preds, targets, *, bias, min_value, zero_threshold):
    clipped = _postprocess_array(raw_preds, min_value=min_value)
    biased_clipped = _postprocess_array(raw_preds, bias=bias, min_value=min_value)
    calibrated = _postprocess_array(
        raw_preds,
        bias=bias,
        min_value=min_value,
        zero_threshold=zero_threshold,
    )
    return {
        "n": int(len(targets)),
        "raw_mae": _mae(raw_preds, targets),
        "min_clamped_mae": _mae(clipped, targets),
        "bias_min_clamped_mae": _mae(biased_clipped, targets),
        "calibrated_mae": _mae(calibrated, targets),
        "raw_negative_fraction": float(np.mean(raw_preds < 0.0)),
        "target_zero_fraction": float(np.mean(np.abs(targets) < 0.01)),
        "calibrated_zero_fraction": float(np.mean(calibrated == 0.0)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fit train-only scalar calibration for matbench_mp_gap",
        allow_abbrev=False,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pth")
    parser.add_argument("--output", default=None, help="Calibration JSON output path")
    parser.add_argument("--dataset", default="matbench_mp_gap")
    parser.add_argument("--model-size", default="large")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-cpu-workers", type=int, default=4)
    parser.add_argument("--matbench-fold", type=int, default=0)
    parser.add_argument("--matbench-val-ratio", type=float, default=0.1)
    parser.add_argument("--prediction-min-value", type=float, default=0.0)
    parser.add_argument("--loss-compression-scale", type=float, default=5.0)
    register_matbench_args(parser)
    args = parser.parse_args()
    validate_matbench_args(args, parser)

    if args.dataset != "matbench_mp_gap":
        raise ValueError("This calibration script is currently specialized for matbench_mp_gap")

    model_config, _ = get_configs(args.model_size)
    model_config.loss_compression_scale = args.loss_compression_scale
    model_config.numeric_output = "linear"
    model_config.use_zero_head = False
    model_config.prediction_bias = 0.0
    model_config.prediction_min_value = None
    model_config.prediction_zero_threshold = None

    mb_config = get_matbench_config(args.dataset)
    dataset_options = build_dataset_options(args, mb_config)
    if not dataset_options:
        dataset_options = {
            "add_composition": True,
            "add_comp_ewald": True,
            "add_comp_nn": True,
            "drop_coords": True,
        }

    mb_loader = MatbenchLoader(
        args.dataset,
        max_tokens=model_config.max_tokens,
        test_ratio=0.2,
        seed=42,
        dataset_options=dataset_options,
        max_cpu_workers=args.max_cpu_workers,
        split_strategy="official",
        fold=args.matbench_fold,
        val_ratio=args.matbench_val_ratio,
        include_test_targets=False,
    )
    if any("target" in doc.get("crystal", {}) for doc in mb_loader.test_docs):
        raise RuntimeError("Refusing to calibrate: held-out test targets are present")

    opts_tag = build_cache_tag(dataset_options)
    split_tag = f"official_f{args.matbench_fold}_val{args.matbench_val_ratio:g}"
    train_ds = MatbenchDataset(
        docs=mb_loader.train_docs,
        max_tokens=model_config.max_tokens,
        cache_tag=f"{args.dataset}{opts_tag}_{split_tag}_train",
    )
    val_ds = MatbenchDataset(
        docs=mb_loader.val_docs,
        max_tokens=model_config.max_tokens,
        cache_tag=f"{args.dataset}{opts_tag}_{split_tag}_val",
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frozen_lm = FrozenLM(model_config.frozen_lm_name, device=device)
    model = DocumentTransformer(model_config, frozen_lm).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    train_preds, train_targets = _collect_predictions(model, train_loader, device)
    val_preds, val_targets = _collect_predictions(model, val_loader, device)

    bias = _fit_median_bias(train_preds, train_targets)
    train_biased_clipped = _postprocess_array(
        train_preds,
        bias=bias,
        min_value=args.prediction_min_value,
    )
    zero_threshold, train_threshold_mae = _fit_zero_threshold(
        train_biased_clipped,
        train_targets,
    )

    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "split": {
            "strategy": "official",
            "fold": args.matbench_fold,
            "val_ratio": args.matbench_val_ratio,
            "loader_metadata": mb_loader.split_metadata,
        },
        "dataset_options": dataset_options,
        "fit_policy": {
            "bias": "median(target - prediction) on official train-only subset",
            "zero_threshold": "MAE-minimizing threshold on train predictions after bias and min clamp",
            "test_targets": "omitted",
        },
        "calibration": {
            "prediction_bias": bias,
            "prediction_min_value": args.prediction_min_value,
            "prediction_zero_threshold": zero_threshold,
            "train_threshold_fit_mae": train_threshold_mae,
        },
        "metrics": {
            "train": _summarize_split(
                train_preds,
                train_targets,
                bias=bias,
                min_value=args.prediction_min_value,
                zero_threshold=zero_threshold,
            ),
            "val": _summarize_split(
                val_preds,
                val_targets,
                bias=bias,
                min_value=args.prediction_min_value,
                zero_threshold=zero_threshold,
            ),
        },
    }

    output = args.output
    if output is None:
        checkpoint_path = pathlib.Path(args.checkpoint)
        output = checkpoint_path.with_name(f"{checkpoint_path.stem}_calibration.json")
    output_path = pathlib.Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report["calibration"], indent=2))
    print(json.dumps(report["metrics"], indent=2))
    print(f"Saved calibration report: {output_path}")


if __name__ == "__main__":
    main()
