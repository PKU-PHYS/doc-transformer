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
                padding_mask: Tensor,
                fork_bias_indices: Tensor = None) -> Tensor:
        """
        batched_leaves: batch 中每个样本的叶子节点列表
        padding_mask: (B, max_len)
        fork_bias_indices: (B, max_len, max_len) fork level 矩阵
        """
        B = len(batched_leaves)
        device = padding_mask.device
        max_len = padding_mask.size(1)
        
        x_emb = torch.zeros((B, max_len, self.config.d_model), device=device)
        
        # 嵌入阶段关闭 autocast：手动张量赋值与 BF16 不兼容
        with torch.amp.autocast('cuda', enabled=False):
            # 展平整个 batch 的 leaves，一次性通过 token_embedder
            all_leaves = []
            batch_offsets = []  # (batch_idx, start_in_flat, count)
            for b, leaves in enumerate(batched_leaves):
                if leaves:
                    start = len(all_leaves)
                    all_leaves.extend(leaves)
                    batch_offsets.append((b, start, len(leaves)))
            
            if all_leaves:
                # 单次调用处理全部 leaves（内部已按类型分组批量化）
                all_embs = self.token_embedder(all_leaves, self.frozen_lm)
                # scatter 回 (B, max_len, d_model)
                for b, start, count in batch_offsets:
                    x_emb[b, :count, :] = all_embs[start:start + count]
                
        # Transformer 主干在外层 autocast 下运行
        out = self.transformer(x_emb, padding_mask=padding_mask,
                               fork_bias_indices=fork_bias_indices)
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

        losses = []

        if num_preds:
            preds_t = torch.cat(num_preds)
            targets_t = torch.tensor(num_targets, dtype=torch.float32, device=device)
            # arcsinh 压缩空间中计算 Huber Loss，消除极端值导致的梯度方差
            losses.append(F.huber_loss(torch.arcsinh(preds_t), torch.arcsinh(targets_t), reduction='mean'))
            
        if bool_preds:
            preds_t = torch.cat(bool_preds)
            targets_t = torch.tensor(bool_targets, dtype=torch.float32, device=device)
            losses.append(F.binary_cross_entropy_with_logits(preds_t, targets_t, reduction='mean'))
            
        if str_preds:
            preds_t = torch.cat(str_preds) # (N_str, lm_dim)
            with torch.no_grad():
                targets_t = self.frozen_lm.encode(str_targets) # (N_str, lm_dim)
            y = torch.ones(len(str_preds), device=device)
            losses.append(F.cosine_embedding_loss(preds_t, targets_t, y, reduction='mean'))
            
        if losses:
            return sum(losses) / len(losses)
        return torch.tensor(0.0, device=device, requires_grad=True)
