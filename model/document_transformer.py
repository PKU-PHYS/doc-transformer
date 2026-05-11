import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Dict, Any

from config import ModelConfig
from .json_parser import LeafNode
from .frozen_lm import FrozenLM
from .token_embedding import TokenEmbedding
from .transformer import GlobalTransformer
from .decode_head import DecodeHead

class DocumentTransformer(nn.Module):
    def __init__(self, config: ModelConfig, frozen_lm: FrozenLM):
        super().__init__()
        self.config = config
        self.frozen_lm = frozen_lm
        
        self.token_embedder = TokenEmbedding(config)
        self.transformer = GlobalTransformer(config)
        self.decode_head = DecodeHead(config)

    def forward(self, 
                batched_leaves: List[List[LeafNode]], 
                padding_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        batched_leaves: batch 中每个样本的叶子节点列表
        padding_mask: (B, max_len)
        """
        B = len(batched_leaves)
        device = padding_mask.device
        max_len = padding_mask.size(1)
        
        x_emb = torch.zeros((B, max_len, self.config.d_model), device=device)
        
        for b, leaves in enumerate(batched_leaves):
            if leaves:
                embs = self.token_embedder(leaves, self.frozen_lm)
                x_emb[b, :len(leaves), :] = embs
                
        # (B, max_len, d_model)
        out = self.transformer(x_emb, padding_mask=padding_mask)
        return out
        
    def compute_loss(self, 
                     out: Tensor, 
                     batched_leaves: List[List[LeafNode]], 
                     target_masks: List[Dict[int, Any]]) -> Tensor:
        """
        计算损失。
        target_masks: 长度为 B 的列表。
                      每个元素是一个 dict: {token_idx: (truth_val, truth_type)}
        """
        device = out.device
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        count = 0
        
        num_preds = []
        num_targets = []
        
        bool_preds = []
        bool_targets = []
        
        str_preds = []
        str_targets = []
        
        for b, masks in enumerate(target_masks):
            for idx, (truth_val, truth_type) in masks.items():
                mask_repr = out[b, idx] # (d_model,)
                
                if truth_type == "number":
                    pred = self.decode_head.predict_number(mask_repr.unsqueeze(0))
                    num_preds.append(pred)
                    num_targets.append(float(truth_val))
                    
                elif truth_type == "boolean":
                    pred = self.decode_head.predict_boolean(mask_repr.unsqueeze(0))
                    bool_preds.append(pred)
                    bool_targets.append(1.0 if truth_val else 0.0)
                    
                elif truth_type == "string":
                    pred = self.decode_head.predict_string(mask_repr.unsqueeze(0))
                    str_preds.append(pred)
                    str_targets.append(truth_val)

        if num_preds:
            preds_t = torch.cat(num_preds)
            targets_t = torch.tensor(num_targets, dtype=torch.float32, device=device)
            # Huber Loss for numbers
            loss = loss + F.huber_loss(preds_t, targets_t)
            count += len(num_preds)
            
        if bool_preds:
            preds_t = torch.cat(bool_preds)
            targets_t = torch.tensor(bool_targets, dtype=torch.float32, device=device)
            loss = loss + F.binary_cross_entropy_with_logits(preds_t, targets_t)
            count += len(bool_preds)
            
        if str_preds:
            preds_t = torch.cat(str_preds) # (N_str, lm_dim)
            with torch.no_grad():
                targets_t = self.frozen_lm.encode(str_targets) # (N_str, lm_dim)
            # Cosine Embedding Loss
            # targets for cosine_embedding_loss should be 1 or -1. 1 means similar.
            y = torch.ones(len(str_preds), device=device)
            loss = loss + F.cosine_embedding_loss(preds_t, targets_t, y)
            count += len(str_preds)
            
        if count > 0:
            return loss / count
        return loss
