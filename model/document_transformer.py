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
        
        zero_preds = []
        
        for b, masks in enumerate(target_masks):
            for idx, (truth_val, truth_type) in masks.items():
                mask_repr = out[b, idx] # (d_model,)
                
                if truth_type == "number":
                    pred = self.decode_head.predict_number(mask_repr.unsqueeze(0))
                    num_preds.append(pred)
                    num_targets.append(float(truth_val))
                    
                    # 零值分类预测（独立分支）
                    zero_logit = self.decode_head.predict_is_zero(mask_repr.unsqueeze(0))
                    zero_preds.append(zero_logit)
                    
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
            # 尾数空间 Loss：用真实值的量级 2^E 归一化 pred 和 target
            # clamp E 到 [E_min, E_max] 控制 scale 范围：
            #   E_min=0 → 小值/零值 scale=1，不放大梯度
            #   E_max=4 → 大值 scale≤16，适度归一化
            _, e_true = torch.frexp(targets_t)
            e_clamped = e_true.clamp(
                self.config.loss_exponent_min,
                self.config.loss_exponent_max,
            )
            scale = torch.pow(2.0, e_clamped.float())
            m_pred = preds_t / scale
            m_true = targets_t / scale
            s = self.config.loss_compression_scale
            k = self.config.loss_scale_power
            per_sample = F.huber_loss(
                torch.arcsinh(m_pred / s) * s,
                torch.arcsinh(m_true / s) * s,
                reduction='none')
            # scale^k 量级补偿：k=0 无补偿, k=1 均匀梯度, k>1 偏重大值
            if k != 0:
                per_sample = per_sample * torch.pow(scale, k)
            losses.append(per_sample.mean())
            
            # ── 零值分类 loss（独立，不影响回归 loss）──
            zero_logits = torch.cat(zero_preds)
            zero_labels = (targets_t.abs() < self.config.zero_threshold).float()
            zero_loss = F.binary_cross_entropy_with_logits(
                zero_logits, zero_labels, reduction='mean')
            losses.append(zero_loss * self.config.zero_loss_weight)
            
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
