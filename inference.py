"""
Document Transformer 推理入口。

支持：
  - --checkpoint <path> 指定权重路径
  - 若未指定，自动查找 checkpoints/ 下最新的 .pth
  - 构建完整的 structural bias 字典以确保推理结果与训练一致
"""

import os
import glob
import argparse
from typing import Optional, List
import torch

from calibration import apply_calibration_report
from config import ModelConfig, MODEL_PRESETS, get_configs
from eval import apply_numeric_postprocessing
from model.frozen_lm import FrozenLM
from model.document_transformer import DocumentTransformer
from model.json_parser import LeafNode, JSONParser
from data.base import compute_single_structural_bias


def _find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> Optional[str]:
    """自动查找 checkpoint 目录下最新的 .pth 文件。"""
    if not os.path.isdir(checkpoint_dir):
        return None
    pth_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not pth_files:
        return None
    # 按修改时间倒序，取最新的
    return max(pth_files, key=os.path.getmtime)



def predict(model, frozen_lm, doc, device, root_name: str = "doc"):
    """
    对一个 JSON 文档执行推理，返回所有 [MASK] 位置的预测结果。
    
    Args:
        model: DocumentTransformer 模型
        frozen_lm: FrozenLM 实例
        doc: 待推理的 JSON 文档（dict 或 list），其中需要预测的位置用 "[MASK]" 标记
        device: 推理设备
        root_name: 解析时的根节点名称
    
    Returns:
        predictions: List of (path, predicted_value) 
    """
    parser = JSONParser()
    leaves = parser.parse(
        doc, [root_name], [0], [JSONParser._key_hash(root_name)], []
    )

    if not leaves:
        print("  ⚠️  文档解析后无叶子节点")
        return []

    # 找到所有 mask 位置
    mask_indices = [i for i, l in enumerate(leaves) if l.value_type == "mask"]
    if not mask_indices:
        print("  ⚠️  文档中未找到 [MASK] 标记")
        return []

    # 构建 padding mask 和 structural bias
    seq_len = len(leaves)
    padding_mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
    bias_dict = compute_single_structural_bias(leaves)
    bias_indices = {k: v.long().unsqueeze(0).to(device) for k, v in bias_dict.items()}

    # 前向推理
    with torch.no_grad():
        out = model([leaves], padding_mask, bias_indices=bias_indices)

    # 提取预测
    predictions = []
    for idx in mask_indices:
        mask_repr = out[0, idx].unsqueeze(0)
        path_str = ".".join(leaves[idx].path)

        # 数值预测（默认）
        pred_val = model.decode_head.predict_number(mask_repr).item()
        pred_val = apply_numeric_postprocessing(pred_val, model.config)
        
        # 零值分类头：logit > 0 (sigmoid > 0.5) → 直接输出 0
        if model.config.use_zero_head:
            zero_logit = model.decode_head.predict_is_zero(mask_repr).item()
            zero_logit_threshold = getattr(model.config, "zero_logit_threshold", 0.0)
            if zero_logit > zero_logit_threshold:
                pred_val = 0.0
        
        predictions.append((path_str, pred_val))

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Document Transformer Inference")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint .pth file. "
                             "If not specified, auto-finds the latest in checkpoints/")
    parser.add_argument("--model-size", type=str, default="large",
                        choices=list(MODEL_PRESETS.keys()),
                        help=f"Model size preset ({', '.join(MODEL_PRESETS.keys())})")
    parser.add_argument("--calibration", type=str, default=None,
                        help="Optional calibration JSON from scripts/calibrate_matbench_gap.py")
    args = parser.parse_args()

    print("=== Document Transformer Inference ===\n")
    model_config, _ = get_configs(args.model_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frozen_lm = FrozenLM(model_config.frozen_lm_name, device=str(device))
    model = DocumentTransformer(model_config, frozen_lm).to(device)

    # 加载 checkpoint
    ckpt_path = args.checkpoint or _find_latest_checkpoint()
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
        stage = ckpt.get("stage", "unknown") if isinstance(ckpt, dict) else "unknown"
        print(f"  ✅ Loaded checkpoint: {ckpt_path} (stage: {stage})")
    else:
        print("  ⚠️  No checkpoint found. Running with random weights.\n")

    model.eval()
    if args.calibration:
        apply_calibration_report(model.config, args.calibration)
        print(f"  ✅ Loaded calibration: {args.calibration}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_params/1e6:.1f}M params, device={device}\n")

    # ── Demo 1: 单条材料预测 ──
    print("─" * 50)
    print("Demo 1: 单材料 band_gap 预测")
    print("─" * 50)
    doc1 = {
        "formula": "TiO2",
        "band_gap": "[MASK]",
        "lattice": {"a": 4.59, "c": 2.96}
    }
    print(f"  输入: {doc1}")
    preds = predict(model, frozen_lm, doc1, device, root_name="materials")
    for path, val in preds:
        print(f"  预测: {path} = {val:.6f}")

    # ── Demo 2: 数学关系预测 ──
    print(f"\n{'─' * 50}")
    print("Demo 2: 数学关系预测 (3 + 7 = ?)")
    print("─" * 50)
    doc2 = {
        "function": "add",
        "a": 3.0,
        "b": 7.0,
        "result": "[MASK]",
    }
    print(f"  输入: {doc2}")
    preds = predict(model, frozen_lm, doc2, device)
    for path, val in preds:
        print(f"  预测: {path} = {val:.6f} (期望: 10.0)")

    # ── Demo 3: In-Context 推理 ──
    print(f"\n{'─' * 50}")
    print("Demo 3: In-Context 推理 (推断隐含规则)")
    print("─" * 50)
    doc3 = [
        {"x": 1.0, "y": 2.0},
        {"x": 2.0, "y": 4.0},
        {"x": 3.0, "y": 6.0},
        {"x": 5.0, "y": "[MASK]"},
    ]
    print(f"  输入: {doc3}")
    preds = predict(model, frozen_lm, doc3, device, root_name="records")
    for path, val in preds:
        print(f"  预测: {path} = {val:.6f} (期望: 10.0, 规则: y=2x)")

    print(f"\n{'=' * 50}")
    print("Inference complete.")


if __name__ == "__main__":
    main()
