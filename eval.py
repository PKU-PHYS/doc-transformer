"""
评估模块 — 在 test set 上计算预测指标。

支持指标:
  - mae:  Mean Absolute Error (Matbench 标准)
  - rmse: Root Mean Squared Error (Tabular 回归)
"""

import math
import time
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
    was_training = model.training  # 记录入场模式，退出时还原
    model.eval()

    errors = []
    t_start = time.time()
    t_data = 0.0   # 等数据的时间
    t_fwd = 0.0    # GPU forward 的时间
    t_pred = 0.0   # 提取预测的时间
    n_batches = 0

    try:
        t0 = time.time()
        with torch.no_grad():
            for batched_leaves, batched_masks, padding_mask, bias_indices in test_loader:
                t_data += time.time() - t0

                t1 = time.time()
                padding_mask = padding_mask.to(device)
                bias_indices = {k: v.to(device) for k, v in bias_indices.items()}

                with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                        enabled=(device == "cuda" and torch.cuda.is_bf16_supported())):
                    out = model(batched_leaves, padding_mask, bias_indices=bias_indices)
                if device == "cuda":
                    torch.cuda.synchronize()
                t_fwd += time.time() - t1

                t2 = time.time()
                for sample_idx in range(len(batched_leaves)):
                    masks = batched_masks[sample_idx]
                    for mask_pos, (true_val, val_type) in masks.items():
                        if val_type == "number":
                            mask_repr = out[sample_idx, mask_pos].unsqueeze(0)
                            pred_val = model.decode_head.predict_number(mask_repr).item()
                            true_float = float(true_val)

                            # 零值分类头：logit > 0 (sigmoid > 0.5) → 直接输出 0
                            if model.config.use_zero_head:
                                zero_logit = model.decode_head.predict_is_zero(mask_repr).item()
                                if zero_logit > 0:
                                    pred_val = 0.0
                            prediction_min_value = getattr(
                                model.config, "prediction_min_value", None
                            )
                            if prediction_min_value is not None:
                                pred_val = max(pred_val, prediction_min_value)

                            if metric == "mae":
                                errors.append(abs(pred_val - true_float))
                            elif metric == "rmse":
                                errors.append((pred_val - true_float) ** 2)
                        else:
                            raise NotImplementedError(
                                f"evaluate() only supports number masks; got val_type={val_type!r}"
                            )
                t_pred += time.time() - t2

                n_batches += 1
                t0 = time.time()

        t_total = time.time() - t_start
        print(f"    ⏱ Eval breakdown: total={t_total:.1f}s, "
              f"data={t_data:.1f}s, fwd={t_fwd:.1f}s, pred={t_pred:.1f}s "
              f"({n_batches} batches)")
    finally:
        model.train(was_training)

    if not errors:
        return float("inf")

    if metric == "mae":
        return sum(errors) / len(errors)
    elif metric == "rmse":
        return math.sqrt(sum(errors) / len(errors))
    else:
        raise ValueError(f"Unknown metric: {metric}")
