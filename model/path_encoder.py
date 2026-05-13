import math
import torch
import torch.nn as nn
from torch import Tensor


class MLPPathEncoder(nn.Module):
    """
    MLP Fusion 路径编码器。
    
    将 FrozenLM 文本特征 + 节点类型嵌入 + 正弦深度编码 三路相加，
    经 LayerNorm + MLP 非线性投影后，Masked Sum Pooling 得到最终路径表示。
    
    核心优势：
    - 非线性 MLP 在 sum 前打破加法交换律 → 路径顺序敏感
    - 正弦 PE 是公式计算 → 无深度上限
    - node_type_emb 显式区分 Dict Key 与 Array Instance
    """
    def __init__(self, frozen_lm_dim: int, d_model: int):
        super().__init__()
        self.frozen_lm_dim = frozen_lm_dim
        
        # 节点类型嵌入 (0: Dict Key, 1: Array Instance)
        self.node_type_emb = nn.Embedding(2, frozen_lm_dim)
        
        # 特征融合前归一化（稳定 464M 大模型训练）
        self.pre_norm = nn.LayerNorm(frozen_lm_dim)
        
        # 非线性投影：打破加法交换律
        self.mlp = nn.Sequential(
            nn.Linear(frozen_lm_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def _sinusoidal_pe(self, depths: Tensor) -> Tensor:
        """
        On-the-fly 正弦位置编码，无深度上限。
        
        depths: 任意形状的整数张量
        Returns: (*depths.shape, frozen_lm_dim) 的浮点张量
        """
        dim = self.frozen_lm_dim
        half = dim // 2
        div_term = torch.exp(
            torch.arange(half, device=depths.device, dtype=torch.float32)
            * (-math.log(10000.0) / half)
        )
        pos = depths.unsqueeze(-1).float()  # (..., 1)
        pe = torch.zeros(*depths.shape, dim, device=depths.device)
        pe[..., 0::2] = torch.sin(pos * div_term)
        pe[..., 1::2] = torch.cos(pos * div_term)
        return pe

    def forward(self, path_text_embs: Tensor, path_depths: Tensor,
                path_types: Tensor, valid_lens: Tensor) -> Tensor:
        """
        Args:
            path_text_embs: (N, max_L, frozen_lm_dim) FrozenLM 编码的路径节点文本
            path_depths:    (N, max_L) 每个节点的深度索引 (0, 1, 2, ...)
            path_types:     (N, max_L) 节点类型 (0=Dict Key, 1=Array Instance)
            valid_lens:     (N,) 每条路径的有效长度

        Returns:
            (N, d_model) 路径编码
        """
        type_bias = self.node_type_emb(path_types)   # (N, max_L, frozen_lm_dim)
        depth_pe  = self._sinusoidal_pe(path_depths)  # (N, max_L, frozen_lm_dim)
        
        fused = self.pre_norm(path_text_embs + type_bias + depth_pe)
        projected = self.mlp(fused)  # (N, max_L, d_model)
        
        # Masked sum pooling: 仅对有效路径节点求和
        max_L = projected.size(1)
        mask = torch.arange(max_L, device=projected.device).unsqueeze(0)  # (1, max_L)
        mask = (mask < valid_lens.unsqueeze(1)).unsqueeze(-1)  # (N, max_L, 1)
        
        return (projected * mask).sum(dim=1)  # (N, d_model)
