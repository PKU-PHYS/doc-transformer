import torch
import torch.nn as nn
from torch import Tensor


class GRUPathEncoder(nn.Module):
    """
    GRU 路径编码器 — 递归编码 JSON 路径。

    输入 = FrozenLM 文本特征 + 节点类型嵌入 (Dict Key / Array Instance)，
    通过 GRU 递归处理，取最终隐藏状态作为路径表示。

    优势：
    - 天然序列敏感：精确编码路径节点的顺序和依赖关系
    - 对短路径（Stage 0 深度 2）信号传播高效
    - node_type_emb 显式区分 Dict Key 与 Array Instance
    """
    def __init__(self, frozen_lm_dim: int, d_model: int):
        super().__init__()
        self.frozen_lm_dim = frozen_lm_dim

        # 节点类型嵌入 (0: Dict Key, 1: Array Instance)
        self.node_type_emb = nn.Embedding(2, frozen_lm_dim)

        # GRU 的输入 = 文本特征 + 类型嵌入 (frozen_lm_dim)
        # 隐藏状态 = 路径编码 (d_model)
        self.gru = nn.GRU(input_size=frozen_lm_dim, hidden_size=d_model, batch_first=True)
        # 初始隐藏状态：零向量
        self.h0 = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, path_text_embs: Tensor, path_depths: Tensor,
                path_types: Tensor, valid_lens: Tensor) -> Tensor:
        """
        Args:
            path_text_embs: (N, max_L, frozen_lm_dim) FrozenLM 编码的路径节点文本
            path_depths:    (N, max_L) 每个节点的深度索引 (未使用，保留接口兼容)
            path_types:     (N, max_L) 节点类型 (0=Dict Key, 1=Array Instance)
            valid_lens:     (N,) 每条路径的有效长度

        Returns:
            (N, d_model) 路径编码
        """
        N = path_text_embs.size(0)
        device = self.h0.device
        d_model = self.gru.hidden_size
        out = torch.zeros(N, d_model, device=device)

        if N == 0:
            return out

        # 融合节点类型信息到文本特征
        type_bias = self.node_type_emb(path_types)       # (N, max_L, frozen_lm_dim)
        fused_input = path_text_embs + type_bias          # (N, max_L, frozen_lm_dim)

        # 筛选有效路径（长度 > 0）
        valid_mask = valid_lens > 0
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]

        if len(valid_indices) == 0:
            return out

        # 抽取有效路径，构建 pack_sequence 所需的张量列表
        valid_seqs = []
        for idx in valid_indices:
            L = valid_lens[idx].item()
            valid_seqs.append(fused_input[idx, :L])  # (L, frozen_lm_dim)

        packed_input = nn.utils.rnn.pack_sequence(valid_seqs, enforce_sorted=False)
        h0 = self.h0.expand(1, len(valid_indices), -1).contiguous()

        _, hn = self.gru(packed_input, h0)  # hn: (1, n_valid, d_model)

        # 将结果填回对应位置
        for i, orig_idx in enumerate(valid_indices):
            out[orig_idx] = hn[0, i]

        return out
