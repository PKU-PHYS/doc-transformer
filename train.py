import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from config import TrainConfig, ModelConfig
from model.frozen_lm import FrozenLM
from model.document_transformer import DocumentTransformer
from data.synthetic import SyntheticDataset, collate_fn

def main():
    print("=== Training Document Transformer ===")
    
    model_config = ModelConfig()
    train_config = TrainConfig()
    
    device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Initialize models
    frozen_lm = FrozenLM(model_config.frozen_lm_name, device=device)
    model = DocumentTransformer(model_config, frozen_lm).to(device)
    
    # 2. Dataset & DataLoader
    train_dataset = SyntheticDataset(size=2000, mask_ratio=train_config.mask_ratio)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    optimizer = AdamW(model.parameters(), lr=train_config.lr)
    
    # 3. Training Loop
    model.train()
    for epoch in range(1, train_config.epochs + 1):
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{train_config.epochs}")
        for batched_leaves, batched_masks, padding_mask in pbar:
            padding_mask = padding_mask.to(device)
            
            optimizer.zero_grad()
            out = model(batched_leaves, padding_mask)
            loss = model.compute_loss(out, batched_leaves, batched_masks)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        
    # 4. Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/document_transformer.pth")
    print("Model saved to checkpoints/document_transformer.pth")

if __name__ == "__main__":
    main()
