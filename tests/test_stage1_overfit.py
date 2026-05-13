import sys
import os
import torch
from torch.optim import AdamW

# 确保能导入外层模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ModelConfig
from model.json_parser import LeafNode, JSONParser
from model.frozen_lm import FrozenLM
from model.document_transformer import DocumentTransformer
from data.synthetic import compute_fork_bias_indices

def _make_fork_bias(leaves_batch, device):
    """为测试构建 fork bias 矩阵"""
    B = len(leaves_batch)
    max_len = max(len(l) for l in leaves_batch)
    max_path_len = max(len(leaf.path_ids)
                       for leaves in leaves_batch for leaf in leaves) if max_len > 0 else 1
    
    path_ids_t = torch.zeros(B, max_len, max_path_len, dtype=torch.long)
    is_group_t = torch.zeros(B, max_len, max_path_len, dtype=torch.long)
    valid_path_lens_t = torch.zeros(B, max_len, dtype=torch.long)
    
    for b, leaves in enumerate(leaves_batch):
        for i, leaf in enumerate(leaves):
            L = len(leaf.path_ids)
            valid_path_lens_t[b, i] = L
            if L > 0:
                path_ids_t[b, i, :L] = torch.tensor(leaf.path_ids, dtype=torch.long)
                is_group_t[b, i, :L] = torch.tensor(leaf.path_types, dtype=torch.long)
    
    return compute_fork_bias_indices(path_ids_t, is_group_t, valid_path_lens_t).to(device)

def _leaf(value, value_type, path):
    """创建带完整新字段的 LeafNode（无数组嵌套的简单叶子）"""
    path_types = [0] * len(path)  # 全是 Dict Key
    path_ids = [JSONParser._key_hash(p) for p in path]
    return LeafNode(value=value, value_type=value_type, path=path,
                    path_types=path_types, path_ids=path_ids, group_ids=[])

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
    leaves = [
        _leaf(1.0, "number", ["val1"]),
        _leaf(2.0, "number", ["val2"]),
        _leaf("[MASK]", "mask", ["pred"]),
    ]
    
    target_masks = [{2: (3.0, "number")}]
    
    padding_mask = torch.zeros((1, 3), dtype=torch.bool, device=frozen_lm.device)
    fork_bias = _make_fork_bias([leaves], frozen_lm.device)
    
    model.train()
    for step in range(100):
        optimizer.zero_grad()
        out = model([leaves], padding_mask, fork_bias_indices=fork_bias)
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
