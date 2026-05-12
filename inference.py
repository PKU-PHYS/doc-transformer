import os
import glob
import argparse
from typing import Optional
import torch
from config import ModelConfig
from model.frozen_lm import FrozenLM
from model.document_transformer import DocumentTransformer
from model.json_parser import LeafNode, JSONParser


def _find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> Optional[str]:
    """自动查找 checkpoint 目录下最新的 .pth 文件。"""
    if not os.path.isdir(checkpoint_dir):
        return None
    pth_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not pth_files:
        return None
    # 按修改时间倒序，取最新的
    return max(pth_files, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description="Document Transformer Inference")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint .pth file. "
                             "If not specified, auto-finds the latest in checkpoints/")
    args = parser.parse_args()

    print("=== Inference Demo ===")
    model_config = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    frozen_lm = FrozenLM(model_config.frozen_lm_name, device=device)
    model = DocumentTransformer(model_config, frozen_lm).to(device)
    
    # 加载 checkpoint：优先使用 --checkpoint 参数，否则自动查找最新
    ckpt_path = args.checkpoint or _find_latest_checkpoint()
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # 支持两种格式：纯 state_dict 或包含 "model" key 的完整 checkpoint
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("No checkpoint found. Running with random weights.")
        
    model.eval()
    
    # 构建测试文档
    doc = {
        "formula": "TiO2",
        "band_gap": "[MASK]",
        "lattice": {"a": 4.59, "c": 2.96}
    }
    
    parser = JSONParser()
    leaves = parser.parse(doc, ["materials"], [])
    
    # 找到 mask 位置
    mask_idx = next(i for i, l in enumerate(leaves) if l.value_type == "mask")
    
    padding_mask = torch.zeros((1, len(leaves)), dtype=torch.bool, device=device)
    
    with torch.no_grad():
        out = model([leaves], padding_mask)
        # 预测 band_gap (数字)
        pred_val = model.decode_head.predict_number(out[0, mask_idx].unsqueeze(0)).item()
        
    print(f"Document:")
    print(doc)
    print(f"\nPredicted band_gap: {pred_val:.4f}")

if __name__ == "__main__":
    main()
