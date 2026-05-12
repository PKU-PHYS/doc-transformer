import torch
import torch.nn as nn
from torch import Tensor
from typing import List

from .json_parser import LeafNode
from .frozen_lm import FrozenLM
from .value_encoder import ValueEncoder
from .path_encoder import GRUPathEncoder, DepthPathEncoder
from .group_embedding import generate_group_embeddings
from config import ModelConfig

class TokenEmbedding(nn.Module):
    """
    负责将所有解析出的 LeafNode 转化为 (N, d_model) 的最终 token embedding
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.frozen_lm_dim = config.frozen_lm_dim
        
        self.value_encoder = ValueEncoder(
            d_model=config.d_model,
            frozen_lm_dim=config.frozen_lm_dim,
            n_fourier_feats=config.n_fourier_feats,
            fourier_learnable=config.fourier_learnable
        )
        
        self.path_encoding_type = config.path_encoding
        if self.path_encoding_type == "gru":
            self.path_encoder = GRUPathEncoder(
                frozen_lm_dim=config.frozen_lm_dim,
                d_model=config.d_model
            )
        elif self.path_encoding_type == "depth":
            self.path_encoder = DepthPathEncoder(
                max_depth=config.max_depth,
                frozen_lm_dim=config.frozen_lm_dim,
                d_model=config.d_model
            )
        else:
            raise ValueError(f"Unknown path encoding: {self.path_encoding_type}")
            
        self.group_scale = config.group_scale

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
            text_lookup = {t: unique_embs[j] for j, t in enumerate(unique_list)}
        else:
            text_lookup = {}

        # ── 第二步：值编码 ──
        if str_values:
            lm_value_embs = torch.stack([text_lookup[s] for s in str_values])
        else:
            lm_value_embs = None

        node_types = [l.value_type for l in leaves]
        raw_values = [l.value for l in leaves]
        val_embs = self.value_encoder(node_types, raw_values, lm_embeddings=lm_value_embs)
        
        # ── 第三步：路径编码（通过查找表，不再逐叶子调用 encode） ──
        lm_dim = frozen_lm.dim()
        path_embs_list = []
        for leaf in leaves:
            if not leaf.path:
                path_embs_list.append(torch.zeros(0, lm_dim, device=device))
            else:
                path_embs_list.append(torch.stack([text_lookup[p] for p in leaf.path]))
                
        path_embs = self.path_encoder(path_embs_list) # (N, d_model)
        
        # ── 第四步：组嵌入 ──
        group_ids_list = [l.group_ids for l in leaves]
        group_embs = generate_group_embeddings(group_ids_list, self.d_model, self.group_scale, device=device)
        
        # ── 最终求和 ──
        return val_embs + path_embs + group_embs
