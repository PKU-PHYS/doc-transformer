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
            
        # 1. 准备 FrozenLM 需要计算的文本
        # 包括值编码中的 string 和 路径编码中的 key
        texts_to_encode = set()
        str_values = []
        str_indices = []
        
        for i, leaf in enumerate(leaves):
            if leaf.value_type == "string":
                texts_to_encode.add(leaf.value)
                str_values.append(leaf.value)
                str_indices.append(i)
            for p in leaf.path:
                texts_to_encode.add(p)
                
        # 批量使用 FrozenLM 获取文本特征
        # 由于 FrozenLM 有 cache，重复出现的其实不会重复算
        # 但我们为了给 ValueEncoder 传递 string 对应的值特征，需要显式算出 str_values 的
        if str_values:
            lm_value_embs = frozen_lm.encode(str_values) # (N_str, lm_dim)
        else:
            lm_value_embs = None

        # --- 值编码 ---
        node_types = [l.value_type for l in leaves]
        raw_values = [l.value for l in leaves]
        val_embs = self.value_encoder(node_types, raw_values, lm_embeddings=lm_value_embs)
        
        # --- 路径编码 ---
        path_embs_list = []
        for leaf in leaves:
            if not leaf.path:
                path_embs_list.append(torch.zeros(0, self.frozen_lm_dim, device=device))
            else:
                # encode 会返回 (len(leaf.path), lm_dim)
                p_emb = frozen_lm.encode(leaf.path)
                path_embs_list.append(p_emb)
                
        path_embs = self.path_encoder(path_embs_list) # (N, d_model)
        
        # --- 组嵌入 ---
        group_ids_list = [l.group_ids for l in leaves]
        group_embs = generate_group_embeddings(group_ids_list, self.d_model, self.group_scale, device=device)
        
        # --- 最终求和 ---
        return val_embs + path_embs + group_embs
