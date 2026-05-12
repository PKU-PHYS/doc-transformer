import sys
import os
import torch
import random
from torch.optim import AdamW

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ModelConfig
from model.json_parser import LeafNode
from model.frozen_lm import FrozenLM
from model.document_transformer import DocumentTransformer
from tests.test_stage2_copy import generate_copy_task

def test_stage3():
    print("=== Stage 3: Padding Block Test ===")
    
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
    
    model.train()
    
    batch_size = 4
    
    for step in range(3001):
        batched_leaves = []
        batched_masks = []
        
        # 为了测试 Padding，我们在批次内插入不同长度的噪声节点
        max_len = 0
        for b in range(batch_size):
            leaves, masks = generate_copy_task()
            num_noise = random.randint(0, 5)
            for _ in range(num_noise):
                leaves.append(LeafNode(value="noise", value_type="string", path=["noise"], group_ids=[]))
            batched_leaves.append(leaves)
            batched_masks.append(masks[0])
            if len(leaves) > max_len:
                max_len = len(leaves)
                
        padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool, device=frozen_lm.device)
        for b, leaves in enumerate(batched_leaves):
            padding_mask[b, :len(leaves)] = False
            
        optimizer.zero_grad()
        out = model(batched_leaves, padding_mask)
        loss = model.compute_loss(out, batched_leaves, batched_masks)
        
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Step {step:03d}: Loss = {loss.item():.6f}")
            
    # Evaluation
    model.eval()
    leaves, masks = generate_copy_task()
    batched_leaves = [leaves, leaves + [LeafNode("noise", "string", ["n"], [])]*10]
    padding_mask = torch.ones((2, 14), dtype=torch.bool, device=frozen_lm.device)
    padding_mask[0, :4] = False
    padding_mask[1, :14] = False
    
    with torch.no_grad():
        out = model(batched_leaves, padding_mask)
        pred_val1 = model.decode_head.predict_number(out[0, 3].unsqueeze(0)).item()
        pred_val2 = model.decode_head.predict_number(out[1, 3].unsqueeze(0)).item()
        truth_val = masks[0][3][0]
        
    error1 = abs(pred_val1 - truth_val)
    error2 = abs(pred_val2 - truth_val)
    
    print(f"Final Eval (No Pad) - Truth: {truth_val:.4f}, Pred: {pred_val1:.4f}, Error: {error1:.6f}")
    print(f"Final Eval (With Pad) - Truth: {truth_val:.4f}, Pred: {pred_val2:.4f}, Error: {error2:.6f}")
    
    if error1 < 0.4 and error2 < 0.4:
        print("=== Stage 3 PASSED ===")
    else:
        assert False, f"Stage 3 Failed, Errors: {error1:.6f}, {error2:.6f}"

if __name__ == "__main__":
    test_stage3()
