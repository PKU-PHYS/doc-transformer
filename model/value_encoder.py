import math
import torch
import torch.nn as nn
from torch import Tensor

class FourierFeatureEncoder(nn.Module):
    """
    傅里叶特征映射，消除标准 MLP 的谱偏见。
    将一维标量展开为高频正余弦特征，然后再投影回 d_model。
    """
    def __init__(self, n_feats: int, d_model: int, learnable: bool = True):
        super().__init__()
        if learnable:
            # 使用对数均匀分布初始化频率，覆盖从宏观趋势 (1e-4) 到微观细节 (1e1)
            freqs = 10.0 ** torch.empty(n_feats).uniform_(-4, 1)
            self.freqs = nn.Parameter(freqs) 
        else:
            # 固定的几何级数频率：从很小的频率开始
            freqs = 10.0 ** torch.linspace(-4, 1, n_feats)
            self.register_buffer('freqs', freqs)
            
        self.proj = nn.Linear(2 * n_feats, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (N,) -> (N, 1)
        angles = x.unsqueeze(-1) * self.freqs  # (N, 1) * (n_feats) -> (N, n_feats)
        fourier = torch.cat([angles.sin(), angles.cos()], dim=-1) # (N, 2*n_feats)
        return self.proj(fourier) # (N, d_model)


class ValueEncoder(nn.Module):
    """
    统一的值编码器入口。
    根据叶子节点的类型调用不同的子编码器。
    """
    def __init__(self, d_model: int, frozen_lm_dim: int, n_fourier_feats: int, fourier_learnable: bool):
        super().__init__()
        
        # 数值型编码
        self.num_encoder = FourierFeatureEncoder(
            n_feats=n_fourier_feats, 
            d_model=d_model, 
            learnable=fourier_learnable
        )
        
        # 文本型编码：先用 FrozenLM 得到特征，这里再做线性投影
        self.text_proj = nn.Linear(frozen_lm_dim, d_model)
        
        # 布尔型编码：简单的 0/1 标量投影
        self.bool_proj = nn.Linear(1, d_model)
        
        # 掩码向量
        self.mask_token = nn.Parameter(torch.randn(d_model))
        
    def forward(self, node_types: list[str], raw_values: list, lm_embeddings: Tensor = None) -> Tensor:
        """
        对一批混合类型的值进行编码。
        node_types: 长度为 N 的列表，如 ["number", "string", "mask", ...]
        raw_values: 长度为 N 的原始值列表，如 [2.1, "TiO2", "[MASK]", ...]
        lm_embeddings: 如果该 batch 中有 string，需在外部先用 FrozenLM 算出 (N_str, lm_dim) 传入。
                       如果没有 string，可传 None。
        返回: (N, d_model) 的特征表示
        """
        N = len(node_types)
        device = self.mask_token.device
        out = torch.zeros(N, self.mask_token.size(0), device=device)
        
        # 为了高效处理，将不同类型分组
        num_indices = []
        num_vals = []
        
        str_indices = []
        
        bool_indices = []
        bool_vals = []
        
        mask_indices = []
        
        for i, (t, v) in enumerate(zip(node_types, raw_values)):
            if t == "number":
                num_indices.append(i)
                num_vals.append(float(v))
            elif t == "string":
                str_indices.append(i)
            elif t == "boolean":
                bool_indices.append(i)
                bool_vals.append(1.0 if v else 0.0)
            elif t == "mask":
                mask_indices.append(i)
            else:
                raise ValueError(f"Unknown value type: {t}")
                
        # 批量处理每种类型
        if num_indices:
            num_tensor = torch.tensor(num_vals, dtype=torch.float32, device=device)
            out[num_indices] = self.num_encoder(num_tensor)
            
        if str_indices:
            if lm_embeddings is None:
                raise ValueError("String nodes found but lm_embeddings is None.")
            if lm_embeddings.shape[0] != len(str_indices):
                raise ValueError(f"lm_embeddings 数量 ({lm_embeddings.shape[0]}) 与字符串节点数量 ({len(str_indices)}) 不符")
            # 注意：lm_embeddings 是已经在 GPU 上的张量
            out[str_indices] = self.text_proj(lm_embeddings)
            
        if bool_indices:
            bool_tensor = torch.tensor(bool_vals, dtype=torch.float32, device=device).unsqueeze(1) # (N_bool, 1)
            out[bool_indices] = self.bool_proj(bool_tensor)
            
        if mask_indices:
            out[mask_indices] = self.mask_token.expand(len(mask_indices), -1)
            
        return out
