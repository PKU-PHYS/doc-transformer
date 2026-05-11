import torch
import random
from typing import List, Dict, Any, Tuple
from torch.utils.data import Dataset
from model.json_parser import LeafNode, JSONParser

def generate_synthetic_document() -> Dict[str, Any]:
    """
    生成一条测试用的随机嵌套文档。
    包含材料的 formula，数值属性（band_gap, lattice），以及嵌套数组（sites）。
    """
    elements = ["Fe", "O", "Ti", "Si", "C", "H", "N", "S", "Cl", "Na"]
    
    doc = {
        "formula": "".join(random.sample(elements, 2)) + str(random.randint(1, 4)),
        "is_metal": random.choice([True, False]),
        "band_gap": random.uniform(0.0, 5.0),
        "lattice": {
            "a": random.uniform(2.0, 10.0),
            "b": random.uniform(2.0, 10.0),
            "c": random.uniform(2.0, 20.0),
            "alpha": random.uniform(60, 120),
            "beta": random.uniform(60, 120),
            "gamma": random.uniform(60, 120)
        },
        "sites": []
    }
    
    num_sites = random.randint(1, 5)
    for _ in range(num_sites):
        site = {
            "element": random.choice(elements),
            "x": random.uniform(0.0, 1.0),
            "y": random.uniform(0.0, 1.0),
            "z": random.uniform(0.0, 1.0),
            "occupancy": 1.0
        }
        doc["sites"].append(site)
        
    return doc

class SyntheticDataset(Dataset):
    def __init__(self, size: int, mask_ratio: float = 0.15):
        self.size = size
        self.mask_ratio = mask_ratio
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx) -> Tuple[List[LeafNode], Dict[int, Any]]:
        # 生成两个文档作为一个样本（上下文）
        doc1 = generate_synthetic_document()
        doc2 = generate_synthetic_document()
        
        parser = JSONParser()
        # 将两个文档包装为数组，让 parser 统一管理 group_id，避免手工 ID 与 counter 冲突
        leaves = parser.parse([doc1, doc2], ["materials"], [])
        
        target_masks = {}
        # 随机 Mask 一部分叶子节点
        num_masks = max(1, int(len(leaves) * self.mask_ratio))
        mask_indices = random.sample(range(len(leaves)), num_masks)
        
        for i in mask_indices:
            orig_node = leaves[i]
            # 记录真实值和类型
            target_masks[i] = (orig_node.value, orig_node.value_type)
            # 替换为 [MASK]
            leaves[i] = LeafNode(
                value="[MASK]", 
                value_type="mask", 
                path=orig_node.path, 
                group_ids=orig_node.group_ids
            )
            
        return leaves, target_masks

def collate_fn(batch: List[Tuple[List[LeafNode], Dict[int, Any]]]):
    """
    处理变长 Batch，生成 padding mask。
    """
    batched_leaves = []
    batched_masks = []
    
    max_len = max(len(leaves) for leaves, _ in batch)
    
    # src_key_padding_mask: True 表示 <PAD> 需要被阻断
    padding_mask = torch.ones((len(batch), max_len), dtype=torch.bool)
    
    for b, (leaves, masks) in enumerate(batch):
        batched_leaves.append(leaves)
        batched_masks.append(masks)
        # 将有效部分置为 False
        padding_mask[b, :len(leaves)] = False
        
    return batched_leaves, batched_masks, padding_mask
