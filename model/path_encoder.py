import torch
import torch.nn as nn
from torch import Tensor

class GRUPathEncoder(nn.Module):
    """
    方案 A: 使用 GRU 递归编码路径
    """
    def __init__(self, frozen_lm_dim: int, d_model: int):
        super().__init__()
        # GRU 的输入是字段名的文本特征 (frozen_lm_dim)
        # 隐藏状态是路径编码 (d_model)
        self.gru = nn.GRU(input_size=frozen_lm_dim, hidden_size=d_model, batch_first=True)
        # 初始隐藏状态：0 向量
        self.h0 = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, path_embeddings_list: list[Tensor]) -> Tensor:
        """
        path_embeddings_list: 长度为 N 的列表。
        列表中的每个元素是一个 (L_i, frozen_lm_dim) 的张量，代表该节点路径上每一层的字段名文本特征。
        返回: (N, d_model) 的路径最终编码
        """
        N = len(path_embeddings_list)
        device = self.h0.device
        out = torch.zeros(N, self.gru.hidden_size, device=device)
        
        # 为了避免 for 循环过慢，可以将不同长度的路径 pad 起来一起输入 GRU
        # 此处使用 torch.nn.utils.rnn.pack_sequence 加速
        if N == 0:
            return out
            
        lengths = [emb.size(0) for emb in path_embeddings_list]
        
        # 将 length>0 的单独挑出来算
        valid_indices = [i for i, l in enumerate(lengths) if l > 0]
        if not valid_indices:
            # 所有人都是根节点（理论上不可能），直接返回 0
            return out
            
        valid_embs = [path_embeddings_list[i] for i in valid_indices]
        
        # 按照长度降序排序（pack_sequence 的要求，不过 enforce_sorted=False 也可以）
        packed_input = nn.utils.rnn.pack_sequence(valid_embs, enforce_sorted=False)
        
        h0 = self.h0.expand(1, len(valid_embs), -1).contiguous()
        
        # 前向传播
        _, hn = self.gru(packed_input, h0)
        # hn shape: (1, len(valid_embs), d_model)
        
        # 将结果填回对应的 index
        for idx_in_valid, orig_idx in enumerate(valid_indices):
            out[orig_idx] = hn[0, idx_in_valid]
            
        return out


class DepthPathEncoder(nn.Module):
    """
    方案 B: 使用绝对深度嵌入
    """
    def __init__(self, max_depth: int, frozen_lm_dim: int, d_model: int):
        super().__init__()
        # 深度嵌入矩阵 (Max_Depth, d_model)
        self.depth_embed = nn.Embedding(max_depth, d_model)
        # 将文本特征投影到 d_model
        self.text_proj = nn.Linear(frozen_lm_dim, d_model)
        self.max_depth = max_depth

    def forward(self, path_embeddings_list: list[Tensor]) -> Tensor:
        """
        path_embeddings_list: 长度为 N 的列表。
        列表中的每个元素是一个 (L_i, frozen_lm_dim) 的张量。
        返回: (N, d_model) 的路径最终编码
        """
        N = len(path_embeddings_list)
        device = self.text_proj.weight.device
        out = torch.zeros(N, self.text_proj.out_features, device=device)
        
        for i, embs in enumerate(path_embeddings_list):
            L = embs.size(0)
            if L == 0:
                continue
                
            # 限制深度不超过 max_depth
            actual_L = min(L, self.max_depth)
            embs = embs[:actual_L]
            
            # (L, d_model)
            text_features = self.text_proj(embs)
            
            # 获取 0 到 L-1 的深度编码
            depth_indices = torch.arange(actual_L, device=device)
            depth_features = self.depth_embed(depth_indices)
            
            # h_l = Depth_l + FrozenLM("field")
            # 路径编码 p = sum(h_l)
            path_repr = text_features + depth_features
            out[i] = path_repr.sum(dim=0)
            
        return out
