import sys
import os
import torch
import random
from torch.optim import AdamW

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ModelConfig
from model.json_parser import LeafNode, JSONParser
from model.frozen_lm import FrozenLM
from model.document_transformer import DocumentTransformer
from tests.test_stage1_overfit import _make_fork_bias, _leaf

def generate_copy_task():
    # 动态生成一条复制任务
    val = random.uniform(0.0, 10.0)
    target_id = random.choice(["Fe", "O", "Ti", "Si"])
    
    leaves = [
        _leaf(target_id, "string", ["source", "id"]),
        _leaf(val, "number", ["source", "val"]),
        _leaf(target_id, "string", ["target", "id"]),
        _leaf("[MASK]", "mask", ["target", "pred"]),
    ]
    
    target_masks = [{3: (val, "number")}]
    return leaves, target_masks

def test_stage2():
    print("=== Stage 2: Logic Copy Test ===")
    
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
    optimizer = AdamW(model.parameters(), lr=3e-4)
    
    model.train()
    
    batch_size = 16
    for step in range(2001):
        batched_leaves = []
        batched_masks = []
        for _ in range(batch_size):
            leaves, target_masks = generate_copy_task()
            batched_leaves.append(leaves)
            batched_masks.append(target_masks[0])
            
        padding_mask = torch.zeros((batch_size, 4), dtype=torch.bool, device=frozen_lm.device)
        fork_bias = _make_fork_bias(batched_leaves, frozen_lm.device)
        
        optimizer.zero_grad()
        out = model(batched_leaves, padding_mask, fork_bias_indices=fork_bias)
        loss = model.compute_loss(out, batched_leaves, batched_masks)
        
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            pred_val = model.decode_head.predict_number(out[0, 3].unsqueeze(0)).item()
            truth_val = batched_masks[0][3][0]
            print(f"Step {step:03d}: Loss = {loss.item():.6f}, Truth = {truth_val:.4f}, Pred = {pred_val:.4f}")
            
    # Evaluation over a batch to avoid single-sample noise
    model.eval()
    error_sum = 0.0
    eval_batch = 16
    with torch.no_grad():
        for _ in range(eval_batch):
            leaves, target_masks = generate_copy_task()
            padding_mask = torch.zeros((1, 4), dtype=torch.bool, device=frozen_lm.device)
            fork_bias = _make_fork_bias([leaves], frozen_lm.device)
            out = model([leaves], padding_mask, fork_bias_indices=fork_bias)
            pred_val = model.decode_head.predict_number(out[0, 3].unsqueeze(0)).item()
            truth_val = target_masks[0][3][0]
            error_sum += abs(pred_val - truth_val)
            
    error = error_sum / eval_batch
    print(f"Final Eval Average Error (Batch {eval_batch}): {error:.6f}")
    
    if error < 0.25:
        print("=== Stage 2 PASSED ===")
    else:
        assert False, f"Stage 2 Failed, Error {error:.6f} > 0.25"

if __name__ == "__main__":
    test_stage2()
