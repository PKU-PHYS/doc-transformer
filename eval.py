"""
评估模块 — 在 test set 上计算预测指标。

支持指标:
  - mae:  Mean Absolute Error (Matbench 标准)
  - rmse: Root Mean Squared Error (Tabular 回归)
"""

import math
import torch


def evaluate(model, test_loader, device, metric="mae"):
    """
    在 test set 上评估模型。

    Args:
        model:       DocumentTransformer 模型
        test_loader: 预创建的 DataLoader（外部管理生命周期）
        device:      设备
        metric:      "mae" 或 "rmse"

    Returns:
        float: 评估指标值
    """
    model.eval()

    errors = []
    with torch.no_grad():
        for batched_leaves, batched_masks, padding_mask, fork_bias_indices in test_loader:
            padding_mask = padding_mask.to(device)
            fork_bias_indices = fork_bias_indices.to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                    enabled=(device == "cuda" and torch.cuda.is_bf16_supported())):
                out = model(batched_leaves, padding_mask, fork_bias_indices=fork_bias_indices)

            for sample_idx in range(len(batched_leaves)):
                masks = batched_masks[sample_idx]
                for mask_pos, (true_val, val_type) in masks.items():
                    if val_type == "number":
                        mask_repr = out[sample_idx, mask_pos].unsqueeze(0)
                        pred_val = model.decode_head.predict_number(mask_repr).item()
                        true_float = float(true_val)

                        if metric == "mae":
                            errors.append(abs(pred_val - true_float))
                        elif metric == "rmse":
                            errors.append((pred_val - true_float) ** 2)

    model.train()

    if not errors:
        return float("inf")

    if metric == "mae":
        return sum(errors) / len(errors)
    elif metric == "rmse":
        return math.sqrt(sum(errors) / len(errors))
    else:
        raise ValueError(f"Unknown metric: {metric}")
