import sys
import os
import torch
from torch.optim import AdamW

# 确保能导入外层模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ModelConfig
from model.json_parser import LeafNode
from model.frozen_lm import FrozenLM
from model.document_transformer import DocumentTransformer

def test_stage1():
    print("=== Stage 1: Zero-Loss Overfit Test ===")
    
    config = ModelConfig(
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        max_tokens=64
    )
    frozen_lm = FrozenLM(config.frozen_lm_name, device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {frozen_lm.device}")
    model = DocumentTransformer(config, frozen_lm).to(frozen_lm.device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    
    # 固定的单样本
    # 测试能否在一个固定的输入上过拟合
    # 输入: {"val1": 1.0, "val2": 2.0, "pred": [MASK]}
    # 真实值应该是 3.0 (代表某些组合逻辑，只是让模型背下来)
    
    leaves = [
        LeafNode(value=1.0, value_type="number", path=["val1"], group_ids=[]),
        LeafNode(value=2.0, value_type="number", path=["val2"], group_ids=[]),
        LeafNode(value="[MASK]", value_type="mask", path=["pred"], group_ids=[])
    ]
    
    target_masks = [{2: (3.0, "number")}]
    
    # padding mask: 全是 False (没有 pad)
    padding_mask = torch.zeros((1, 3), dtype=torch.bool, device=frozen_lm.device)
    
    model.train()
    for step in range(100):
        optimizer.zero_grad()
        out = model([leaves], padding_mask)
        loss = model.compute_loss(out, [leaves], target_masks)
        
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            pred_val = model.decode_head.predict_number(out[0, 2].unsqueeze(0)).item()
            print(f"Step {step:03d}: Loss = {loss.item():.6f}, Pred = {pred_val:.6f}")
            
        if loss.item() < 1e-4:
            pred_val = model.decode_head.predict_number(out[0, 2].unsqueeze(0)).item()
            print(f"Converged at step {step}: Loss = {loss.item():.6f}, Pred = {pred_val:.6f}")
            print("=== Stage 1 PASSED ===")
            return
            
    pred_val = model.decode_head.predict_number(out[0, 2].unsqueeze(0)).item()
    print(f"Failed to converge. Final Loss = {loss.item():.6f}, Pred = {pred_val:.6f}")
    assert False, "Stage 1 Failed"

if __name__ == "__main__":
    test_stage1()
