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


def _collect_checkpoint_predictions(
    model,
    checkpoint_paths,
    train_loader,
    val_loader,
    device,
):
    train_predictions = []
    val_predictions = []
    train_targets_ref = None
    val_targets_ref = None

    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        train_preds, train_targets = _collect_predictions(model, train_loader, device)
        val_preds, val_targets = _collect_predictions(model, val_loader, device)

        if train_targets_ref is None:
            train_targets_ref = train_targets
            val_targets_ref = val_targets
        elif not (
            np.array_equal(train_targets_ref, train_targets)
            and np.array_equal(val_targets_ref, val_targets)
        ):
            raise RuntimeError("Target order changed while collecting ensemble predictions")

        train_predictions.append(train_preds)
        val_predictions.append(val_preds)

    train_mean = np.mean(np.stack(train_predictions, axis=0), axis=0)
    val_mean = np.mean(np.stack(val_predictions, axis=0), axis=0)
    return train_mean, train_targets_ref, val_mean, val_targets_ref


def _postprocess_array(
    preds,
    *,
    scale=1.0,
    bias=0.0,
    min_value=None,
    zero_threshold=None,
):
    cfg = argparse.Namespace(
        prediction_scale=scale,
        prediction_bias=bias,
        prediction_min_value=min_value,
        prediction_zero_threshold=zero_threshold,
    )
    return np.asarray([apply_numeric_postprocessing(float(p), cfg) for p in preds])


def _fit_median_bias(preds, targets):
    return float(np.median(targets - preds))


def _fit_scale_and_bias(preds, targets, min_value, scale_min, scale_max, scale_steps):
    best_score = float("inf")
    best_scale = None
    best_bias = None
    for scale in np.linspace(scale_min, scale_max, scale_steps):
        scaled = preds * scale
        bias = _fit_median_bias(scaled, targets)
        calibrated = _postprocess_array(
            preds,
            scale=scale,
            bias=bias,
            min_value=min_value,
        )
        score = _mae(calibrated, targets)
        if score < best_score:
            best_score = score
            best_scale = float(scale)
            best_bias = bias
    return best_scale, best_bias, best_score


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


def _fit_isotonic(train_preds, train_targets, min_value):
    from sklearn.isotonic import IsotonicRegression

    model = IsotonicRegression(
        y_min=min_value,
        y_max=None,
        increasing=True,
        out_of_bounds="clip",
    )
    model.fit(train_preds, train_targets)
    return model


def _fit_binned_residual(
    preds,
    targets,
    *,
    n_bins,
    min_bin_size,
    shrinkage,
    edges=None,
):
    if edges is None:
        edges = np.quantile(preds, np.linspace(0.0, 1.0, n_bins + 1))
    else:
        edges = np.asarray(edges, dtype=np.float64)
        if edges[0] > np.min(preds):
            edges = np.concatenate([[np.min(preds)], edges])
        if edges[-1] < np.max(preds):
            edges = np.concatenate([edges, [np.max(preds)]])
    edges = np.unique(edges)
    if len(edges) < 3:
        return {
            "centers": [],
            "corrections": [],
            "counts": [],
            "edges": [float(x) for x in edges],
        }

    centers = []
    corrections = []
    counts = []
    residuals = targets - preds
    for bin_idx in range(len(edges) - 1):
        left = edges[bin_idx]
        right = edges[bin_idx + 1]
        if bin_idx == len(edges) - 2:
            mask = (preds >= left) & (preds <= right)
        else:
            mask = (preds >= left) & (preds < right)

        count = int(mask.sum())
        if count < min_bin_size:
            continue

        center = float(np.median(preds[mask]))
        residual = float(np.median(residuals[mask]))
        shrink = count / (count + shrinkage)
        centers.append(center)
        corrections.append(float(residual * shrink))
        counts.append(count)

    return {
        "centers": centers,
        "corrections": corrections,
        "counts": counts,
        "edges": [float(x) for x in edges],
    }


def _apply_binned_residual(preds, mapping, min_value):
    centers = np.asarray(mapping["centers"], dtype=np.float64)
    corrections = np.asarray(mapping["corrections"], dtype=np.float64)
    if len(centers) < 2:
        corrected = preds.copy()
    else:
        corrected = preds + np.interp(preds, centers, corrections)
    if min_value is not None:
        corrected = np.maximum(corrected, min_value)
    return corrected


def _parse_int_list(text):
    return [int(part) for part in text.split(",") if part]


def _parse_float_list(text):
    return [float(part) for part in text.split(",") if part]


def _parse_optional_float_list(text):
    if not text:
        return None
    return _parse_float_list(text)


def _fit_cv_binned_residual(
    preds,
    targets,
    *,
    bins_grid,
    shrinkage_grid,
    n_folds,
    min_bin_size,
    min_value,
):
    indices = np.arange(len(preds))
    folds = np.array_split(indices, n_folds)
    candidates = []
    best = None

    for n_bins in bins_grid:
        for shrinkage in shrinkage_grid:
            fold_maes = []
            for fold_idx, valid_idx in enumerate(folds):
                train_idx = np.concatenate(
                    [folds[i] for i in range(n_folds) if i != fold_idx]
                )
                mapping = _fit_binned_residual(
                    preds[train_idx],
                    targets[train_idx],
                    n_bins=n_bins,
                    min_bin_size=min_bin_size,
                    shrinkage=shrinkage,
                )
                valid_preds = _apply_binned_residual(
                    preds[valid_idx],
                    mapping,
                    min_value,
                )
                threshold, _ = _fit_zero_threshold(
                    _apply_binned_residual(
                        preds[train_idx],
                        mapping,
                        min_value,
                    ),
                    targets[train_idx],
                )
                valid_preds = _postprocess_array(
                    valid_preds,
                    zero_threshold=threshold,
                )
                fold_maes.append(_mae(valid_preds, targets[valid_idx]))

            candidate = {
                "bins": int(n_bins),
                "shrinkage": float(shrinkage),
                "fold_mae": [float(score) for score in fold_maes],
                "mean_mae": float(np.mean(fold_maes)),
            }
            candidates.append(candidate)
            if best is None or candidate["mean_mae"] < best["mean_mae"]:
                best = candidate

    final_mapping = _fit_binned_residual(
        preds,
        targets,
        n_bins=best["bins"],
        min_bin_size=min_bin_size,
        shrinkage=best["shrinkage"],
    )
    return best, candidates, final_mapping


def _load_run_metadata_for_checkpoint(checkpoint_path):
    metadata_path = pathlib.Path(checkpoint_path).with_name("run_metadata.json")
    if not metadata_path.exists():
        return {}
    with metadata_path.open() as f:
        return json.load(f)


def _apply_architecture_metadata(model_config, metadata):
    """Restore config fields that change checkpoint parameter shapes."""
    saved_model_config = metadata.get("model_config", {})
    for key in (
        "bias_is_group_fork",
        "bias_first_diff",
        "bias_tree_dist",
        "bias_same_parent",
        "bias_shared_group_depth",
        "bias_same_path_template",
        "bias_value_type_pair",
        "numeric_path_film",
    ):
        if key in saved_model_config:
            setattr(model_config, key, saved_model_config[key])


def _summarize_split_with_scale(raw_preds, targets, *, scale, bias, min_value, zero_threshold):
    clipped = _postprocess_array(raw_preds, min_value=min_value)
    scale_bias_clipped = _postprocess_array(
        raw_preds,
        scale=scale,
        bias=bias,
        min_value=min_value,
    )
    calibrated = _postprocess_array(
        raw_preds,
        scale=scale,
        bias=bias,
        min_value=min_value,
        zero_threshold=zero_threshold,
    )
    return {
        "n": int(len(targets)),
        "raw_mae": _mae(raw_preds, targets),
        "min_clamped_mae": _mae(clipped, targets),
        "scale_bias_min_clamped_mae": _mae(scale_bias_clipped, targets),
        "calibrated_mae": _mae(calibrated, targets),
        "raw_negative_fraction": float(np.mean(raw_preds < 0.0)),
        "target_zero_fraction": float(np.mean(np.abs(targets) < 0.01)),
        "calibrated_zero_fraction": float(np.mean(calibrated == 0.0)),
    }


def _summarize_precomputed(preds, targets):
    return {
        "n": int(len(targets)),
        "mae": _mae(preds, targets),
        "negative_fraction": float(np.mean(preds < 0.0)),
        "target_zero_fraction": float(np.mean(np.abs(targets) < 0.01)),
        "prediction_zero_fraction": float(np.mean(preds == 0.0)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fit train-only scalar calibration for matbench_mp_gap",
        allow_abbrev=False,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pth")
    parser.add_argument(
        "--ensemble-checkpoint",
        action="append",
        default=[],
        help="Additional checkpoint to average at prediction level before calibration",
    )
    parser.add_argument("--output", default=None, help="Calibration JSON output path")
    parser.add_argument("--dataset", default="matbench_mp_gap")
    parser.add_argument("--model-size", default="large")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-cpu-workers", type=int, default=4)
    parser.add_argument("--matbench-fold", type=int, default=0)
    parser.add_argument("--matbench-val-ratio", type=float, default=0.1)
    parser.add_argument("--prediction-min-value", type=float, default=0.0)
    parser.add_argument("--loss-compression-scale", type=float, default=5.0)
    parser.add_argument("--scale-min", type=float, default=0.85)
    parser.add_argument("--scale-max", type=float, default=1.08)
    parser.add_argument("--scale-steps", type=int, default=117)
    parser.add_argument(
        "--also-fit-isotonic",
        action="store_true",
        help="Also report train-only monotonic calibration without changing scalar output",
    )
    parser.add_argument(
        "--isotonic-input",
        choices=("raw", "scale_bias"),
        default="scale_bias",
        help="Prediction values used as the isotonic calibration input",
    )
    parser.add_argument(
        "--also-fit-binned-residual",
        action="store_true",
        help="Also report train-only binned residual calibration without changing scalar output",
    )
    parser.add_argument("--binned-residual-bins", type=int, default=8)
    parser.add_argument(
        "--binned-residual-edges",
        default=None,
        help="Optional comma-separated fixed prediction-space bin edges",
    )
    parser.add_argument("--binned-residual-min-bin-size", type=int, default=1000)
    parser.add_argument("--binned-residual-shrinkage", type=float, default=5000.0)
    parser.add_argument(
        "--also-fit-cv-binned-residual",
        action="store_true",
        help="Select binned residual hyperparameters by train-only K-fold CV",
    )
    parser.add_argument("--cv-binned-residual-bins", default="4,6,8")
    parser.add_argument("--cv-binned-residual-shrinkage", default="5000,10000")
    parser.add_argument("--cv-binned-residual-folds", type=int, default=5)
    register_matbench_args(parser)
    args = parser.parse_args()
    validate_matbench_args(args, parser)

    if args.dataset != "matbench_mp_gap":
        raise ValueError("This calibration script is currently specialized for matbench_mp_gap")

    metadata = _load_run_metadata_for_checkpoint(args.checkpoint)
    model_config, _ = get_configs(args.model_size)
    _apply_architecture_metadata(model_config, metadata)
    model_config.loss_compression_scale = args.loss_compression_scale
    model_config.numeric_output = "linear"
    model_config.use_zero_head = False
    model_config.prediction_scale = 1.0
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
    checkpoint_paths = [args.checkpoint, *args.ensemble_checkpoint]
    train_preds, train_targets, val_preds, val_targets = _collect_checkpoint_predictions(
        model,
        checkpoint_paths,
        train_loader,
        val_loader,
        device,
    )

    scale, bias, train_scale_bias_mae = _fit_scale_and_bias(
        train_preds,
        train_targets,
        args.prediction_min_value,
        args.scale_min,
        args.scale_max,
        args.scale_steps,
    )
    train_biased_clipped = _postprocess_array(
        train_preds,
        scale=scale,
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
        "ensemble_checkpoints": args.ensemble_checkpoint,
        "dataset": args.dataset,
        "split": {
            "strategy": "official",
            "fold": args.matbench_fold,
            "val_ratio": args.matbench_val_ratio,
            "loader_metadata": mb_loader.split_metadata,
        },
        "dataset_options": dataset_options,
        "run_metadata_path": str(pathlib.Path(args.checkpoint).with_name("run_metadata.json"))
        if metadata else None,
        "fit_policy": {
            "scale": "grid search on train MAE after median-bias and min clamp",
            "bias": "median(target - scale * prediction) on official train-only subset",
            "zero_threshold": "MAE-minimizing threshold on train predictions after scale, bias, and min clamp",
            "test_targets": "omitted",
        },
        "calibration": {
            "prediction_scale": scale,
            "prediction_bias": bias,
            "prediction_min_value": args.prediction_min_value,
            "prediction_zero_threshold": zero_threshold,
            "train_scale_bias_fit_mae": train_scale_bias_mae,
            "train_threshold_fit_mae": train_threshold_mae,
        },
        "metrics": {
            "train": _summarize_split_with_scale(
                train_preds,
                train_targets,
                scale=scale,
                bias=bias,
                min_value=args.prediction_min_value,
                zero_threshold=zero_threshold,
            ),
            "val": _summarize_split_with_scale(
                val_preds,
                val_targets,
                scale=scale,
                bias=bias,
                min_value=args.prediction_min_value,
                zero_threshold=zero_threshold,
            ),
        },
    }

    if args.also_fit_isotonic:
        train_scale_bias_clipped = _postprocess_array(
            train_preds,
            scale=scale,
            bias=bias,
            min_value=args.prediction_min_value,
        )
        val_scale_bias_clipped = _postprocess_array(
            val_preds,
            scale=scale,
            bias=bias,
            min_value=args.prediction_min_value,
        )
        if args.isotonic_input == "raw":
            train_iso_input = train_preds
            val_iso_input = val_preds
            input_policy = "raw checkpoint predictions"
        else:
            train_iso_input = train_scale_bias_clipped
            val_iso_input = val_scale_bias_clipped
            input_policy = "scale/bias/min-clamped predictions fitted on train only"

        iso_model = _fit_isotonic(
            train_iso_input,
            train_targets,
            args.prediction_min_value,
        )
        train_iso_preds = np.asarray(iso_model.predict(train_iso_input), dtype=np.float64)
        val_iso_preds = np.asarray(iso_model.predict(val_iso_input), dtype=np.float64)
        report["alternative_calibrations"] = {
            "isotonic": {
                "fit_policy": {
                    "method": "sklearn.isotonic.IsotonicRegression",
                    "input": input_policy,
                    "target": "official train-only targets",
                    "y_min": args.prediction_min_value,
                    "out_of_bounds": "clip",
                    "test_targets": "omitted",
                },
                "mapping": {
                    "x_thresholds": [float(x) for x in iso_model.X_thresholds_],
                    "y_thresholds": [float(y) for y in iso_model.y_thresholds_],
                },
                "metrics": {
                    "train": _summarize_precomputed(train_iso_preds, train_targets),
                    "val": _summarize_precomputed(val_iso_preds, val_targets),
                },
            }
        }

    if args.also_fit_binned_residual:
        train_scale_bias_clipped = _postprocess_array(
            train_preds,
            scale=scale,
            bias=bias,
            min_value=args.prediction_min_value,
        )
        val_scale_bias_clipped = _postprocess_array(
            val_preds,
            scale=scale,
            bias=bias,
            min_value=args.prediction_min_value,
        )
        residual_mapping = _fit_binned_residual(
            train_scale_bias_clipped,
            train_targets,
            n_bins=args.binned_residual_bins,
            min_bin_size=args.binned_residual_min_bin_size,
            shrinkage=args.binned_residual_shrinkage,
            edges=_parse_optional_float_list(args.binned_residual_edges),
        )
        train_residual_preds = _apply_binned_residual(
            train_scale_bias_clipped,
            residual_mapping,
            args.prediction_min_value,
        )
        val_residual_preds = _apply_binned_residual(
            val_scale_bias_clipped,
            residual_mapping,
            args.prediction_min_value,
        )
        residual_zero_threshold, residual_train_threshold_mae = _fit_zero_threshold(
            train_residual_preds,
            train_targets,
        )
        train_residual_calibrated = _postprocess_array(
            train_residual_preds,
            zero_threshold=residual_zero_threshold,
        )
        val_residual_calibrated = _postprocess_array(
            val_residual_preds,
            zero_threshold=residual_zero_threshold,
        )
        report.setdefault("alternative_calibrations", {})["binned_residual"] = {
            "fit_policy": {
                "method": "quantile-bin median residual correction with shrinkage",
                "input": "scale/bias/min-clamped predictions fitted on train only",
                "target": "official train-only targets",
                "bins": args.binned_residual_bins,
                "fixed_edges": _parse_optional_float_list(args.binned_residual_edges),
                "min_bin_size": args.binned_residual_min_bin_size,
                "shrinkage": args.binned_residual_shrinkage,
                "zero_threshold": "refit on residual-corrected train predictions",
                "test_targets": "omitted",
            },
            "mapping": residual_mapping,
            "calibration": {
                "prediction_zero_threshold": residual_zero_threshold,
                "train_threshold_fit_mae": residual_train_threshold_mae,
            },
            "metrics": {
                "train": _summarize_precomputed(train_residual_calibrated, train_targets),
                "val": _summarize_precomputed(val_residual_calibrated, val_targets),
            },
        }

    if args.also_fit_cv_binned_residual:
        train_scale_bias_clipped = _postprocess_array(
            train_preds,
            scale=scale,
            bias=bias,
            min_value=args.prediction_min_value,
        )
        val_scale_bias_clipped = _postprocess_array(
            val_preds,
            scale=scale,
            bias=bias,
            min_value=args.prediction_min_value,
        )
        bins_grid = _parse_int_list(args.cv_binned_residual_bins)
        shrinkage_grid = _parse_float_list(args.cv_binned_residual_shrinkage)
        cv_best, cv_candidates, cv_mapping = _fit_cv_binned_residual(
            train_scale_bias_clipped,
            train_targets,
            bins_grid=bins_grid,
            shrinkage_grid=shrinkage_grid,
            n_folds=args.cv_binned_residual_folds,
            min_bin_size=args.binned_residual_min_bin_size,
            min_value=args.prediction_min_value,
        )
        train_cv_preds = _apply_binned_residual(
            train_scale_bias_clipped,
            cv_mapping,
            args.prediction_min_value,
        )
        val_cv_preds = _apply_binned_residual(
            val_scale_bias_clipped,
            cv_mapping,
            args.prediction_min_value,
        )
        cv_zero_threshold, cv_train_threshold_mae = _fit_zero_threshold(
            train_cv_preds,
            train_targets,
        )
        train_cv_calibrated = _postprocess_array(
            train_cv_preds,
            zero_threshold=cv_zero_threshold,
        )
        val_cv_calibrated = _postprocess_array(
            val_cv_preds,
            zero_threshold=cv_zero_threshold,
        )
        report.setdefault("alternative_calibrations", {})["cv_binned_residual"] = {
            "fit_policy": {
                "method": "train-only K-fold selection of binned residual calibration",
                "input": "scale/bias/min-clamped predictions fitted on train only",
                "target": "official train-only targets",
                "bins_grid": bins_grid,
                "shrinkage_grid": shrinkage_grid,
                "folds": args.cv_binned_residual_folds,
                "min_bin_size": args.binned_residual_min_bin_size,
                "zero_threshold": "refit on full train after CV hyperparameter selection",
                "test_targets": "omitted",
            },
            "selection": {
                "best": cv_best,
                "candidates": cv_candidates,
            },
            "mapping": cv_mapping,
            "calibration": {
                "prediction_zero_threshold": cv_zero_threshold,
                "train_threshold_fit_mae": cv_train_threshold_mae,
            },
            "metrics": {
                "train": _summarize_precomputed(train_cv_calibrated, train_targets),
                "val": _summarize_precomputed(val_cv_calibrated, val_targets),
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
    if args.also_fit_isotonic:
        print(json.dumps(report["alternative_calibrations"]["isotonic"]["metrics"], indent=2))
    if args.also_fit_binned_residual:
        print(json.dumps(report["alternative_calibrations"]["binned_residual"], indent=2))
    if args.also_fit_cv_binned_residual:
        print(json.dumps(report["alternative_calibrations"]["cv_binned_residual"], indent=2))
    print(f"Saved calibration report: {output_path}")


if __name__ == "__main__":
    main()
