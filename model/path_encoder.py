import torch
import torch.nn as nn
from torch import Tensor


class GRUPathEncoder(nn.Module):
    """
    GRU 路径编码器 — 递归编码 JSON 路径。

    输入 = path 文本特征 + 节点类型嵌入 (Dict Key / Array Instance)，
    通过 GRU 递归处理，取最终隐藏状态作为路径表示。

    优势：
    - 天然序列敏感：精确编码路径节点的顺序和依赖关系
    - 对短路径（Stage 0 深度 2）信号传播高效
    - node_type_emb 显式区分 Dict Key 与 Array Instance
    """
    def __init__(self, path_input_dim: int, d_model: int):
        super().__init__()
        self.path_input_dim = path_input_dim

        # 节点类型嵌入 (0: Dict Key, 1: Array Instance)
        self.node_type_emb = nn.Embedding(2, path_input_dim)

        # GRU 的输入 = 文本特征 + 类型嵌入 (path_input_dim)
        # 隐藏状态 = 路径编码 (d_model)
        self.gru = nn.GRU(input_size=path_input_dim, hidden_size=d_model, batch_first=True)
        # 初始隐藏状态：零向量
        self.h0 = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, path_text_embs: Tensor, path_depths: Tensor,
                path_types: Tensor, valid_lens: Tensor) -> Tensor:
        """
        Args:
            path_text_embs: (N, max_L, path_input_dim) 路径节点文本特征
            path_depths:    (N, max_L) 每个节点的深度索引 (未使用，保留接口兼容)
            path_types:     (N, max_L) 节点类型 (0=Dict Key, 1=Array Instance)
            valid_lens:     (N,) 每条路径的有效长度

        Returns:
            (N, d_model) 路径编码
        """
        N = path_text_embs.size(0)
        device = self.h0.device
        d_model = self.gru.hidden_size

        if N == 0:
            return torch.zeros(N, d_model, device=device)

        # 融合节点类型信息到文本特征
        type_bias = self.node_type_emb(path_types)       # (N, max_L, path_input_dim)
        fused_input = path_text_embs + type_bias          # (N, max_L, path_input_dim)

        # Padded GRU：直接在规整张量上跑，让 cuDNN 使用批量并行 kernel
        # （比 pack_sequence 的 backward 快 ~1000x）
        h0 = self.h0.expand(1, N, -1).contiguous()
        gru_out, _ = self.gru(fused_input, h0)  # (N, max_L, d_model)

        # 取每条路径最后一个 *真实* 位置的隐藏状态
        # GRU 是因果的，位置 t 只取决于 0..t，所以与 pack_sequence 结果完全一致
        last_idx = (valid_lens - 1).clamp(min=0)  # (N,)
        gather_idx = last_idx.unsqueeze(-1).unsqueeze(-1).expand(N, 1, d_model)
        out = gru_out.gather(1, gather_idx).squeeze(1)  # (N, d_model)

        # valid_lens=0 的路径输出零向量
        zero_mask = (valid_lens == 0)
        if zero_mask.any():
            out = out.masked_fill(zero_mask.unsqueeze(-1), 0.0)

        return out
