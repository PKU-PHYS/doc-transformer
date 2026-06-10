import torch
import torch.nn as nn
from torch import Tensor
from typing import List

from .json_parser import LeafNode
from .frozen_lm import FrozenLM
from .value_encoder import ValueEncoder
from .path_encoder import GRUPathEncoder
from config import ModelConfig

class TokenEmbedding(nn.Module):
    """
    负责将所有解析出的 LeafNode 转化为 (N, d_model) 的最终 token embedding。
    
    最终嵌入 = 值编码 + 路径编码（不再包含组嵌入，组信息由 fork bias 在注意力层提供）
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.frozen_lm_dim = config.frozen_lm_dim
        
        self.value_encoder = ValueEncoder(
            d_model=config.d_model,
            frozen_lm_dim=config.frozen_lm_dim,
            n_fourier_feats=config.n_fourier_feats,
            fourier_learnable=config.fourier_learnable,
            exponent_min=config.exponent_min,
            exponent_max=config.exponent_max,
            numeric_path_beta=config.numeric_path_beta,
            numeric_path_film=config.numeric_path_film,
        )
        
        self.path_encoder = GRUPathEncoder(
            frozen_lm_dim=config.frozen_lm_dim,
            d_model=config.d_model,
        )

    def forward(self, leaves: List[LeafNode], frozen_lm: FrozenLM) -> Tensor:
        """
        leaves: Batch 内展平的叶子节点列表，长度为 N
        返回: (N, d_model) 的张量
        """
        N = len(leaves)
        device = self.value_encoder.mask_token.device
        if N == 0:
            return torch.zeros(0, self.d_model, device=device)
            
        # ── 第一步：收集所有需要 FrozenLM 编码的唯一文本 ──
        all_unique_texts = set()
        str_values = []
        str_indices = []
        
        for i, leaf in enumerate(leaves):
            if leaf.value_type == "string":
                all_unique_texts.add(leaf.value)
                str_values.append(leaf.value)
                str_indices.append(i)
            for p in leaf.path:
                all_unique_texts.add(p)
        
        # 一次性编码所有唯一文本，构建查找表
        if all_unique_texts:
            unique_list = list(all_unique_texts)
            unique_embs = frozen_lm.encode(unique_list)  # (K, lm_dim) — 单次调用
            text_to_idx = {t: j for j, t in enumerate(unique_list)}
        else:
            unique_embs = None
            text_to_idx = {}

        # ── 第二步：路径编码 — 全量批构建，零逐元素 CUDA 操作 ──
        lm_dim = frozen_lm.dim()
        max_path_len = max(len(l.path) for l in leaves) if leaves else 0
        
        # 在 Python 侧收集所有数据为普通列表，最后一次性创建张量
        # (a) path_types: 构建 (N, max_path_len) 的扁平列表
        types_data = []
        for leaf in leaves:
            L = len(leaf.path_types)
            types_data.extend(leaf.path_types)
            types_data.extend([0] * (max_path_len - L))  # padding
        
        path_types = torch.tensor(types_data, dtype=torch.long, device=device
                                  ).reshape(N, max_path_len)
        
        # (b) valid_lens
        lens_data = [len(leaf.path) for leaf in leaves]
        valid_lens = torch.tensor(lens_data, dtype=torch.long, device=device)
        
        # (c) path_text_embs: 用高级索引一次性填充
        #     收集所有 (row, col, text_idx) 三元组
        row_indices = []
        col_indices = []
        emb_indices = []
        
        for i, leaf in enumerate(leaves):
            for j, text in enumerate(leaf.path):
                row_indices.append(i)
                col_indices.append(j)
                emb_indices.append(text_to_idx[text])
        
        path_text_embs = torch.zeros(N, max_path_len, lm_dim, device=device)
        if row_indices:
            # 一次性高级索引赋值 → 1 次 CUDA 操作代替 N×L 次
            path_text_embs[row_indices, col_indices] = unique_embs[emb_indices]
        
        # path_depths 未被 GRU 实际使用，传零张量
        path_depths = torch.zeros(N, max_path_len, dtype=torch.long, device=device)
        
        path_embs = self.path_encoder(path_text_embs, path_depths, path_types, valid_lens)

        # ── 第三步：值编码 ──
        if str_values:
            lm_value_embs = unique_embs[[text_to_idx[s] for s in str_values]]
        else:
            lm_value_embs = None

        node_types = [l.value_type for l in leaves]
        raw_values = [l.value for l in leaves]
        val_embs = self.value_encoder(
            node_types,
            raw_values,
            lm_embeddings=lm_value_embs,
            path_context=path_embs
            if (self.value_encoder.numeric_path_beta or self.value_encoder.numeric_path_film)
            else None,
        )
        
        # ── 最终求和（值 + 路径，组信息由 fork bias 在注意力层提供）──
        return val_embs + path_embs
