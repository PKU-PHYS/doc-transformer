import torch
from config import ModelConfig
from model.frozen_lm import FrozenLM
from model.document_transformer import DocumentTransformer
from model.json_parser import LeafNode, JSONParser

def main():
    print("=== Inference Demo ===")
    model_config = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    frozen_lm = FrozenLM(model_config.frozen_lm_name, device=device)
    model = DocumentTransformer(model_config, frozen_lm).to(device)
    
    try:
        model.load_state_dict(torch.load("checkpoints/document_transformer.pth", weights_only=True))
        print("Loaded pre-trained weights.")
    except FileNotFoundError:
        print("No checkpoint found. Running with random weights.")
        
    model.eval()
    
    # 构建测试文档
    doc = {
        "formula": "TiO2",
        "band_gap": "[MASK]",
        "lattice": {"a": 4.59, "c": 2.96}
    }
    
    parser = JSONParser()
    leaves = parser.parse(doc, ["materials"], [1])
    
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
