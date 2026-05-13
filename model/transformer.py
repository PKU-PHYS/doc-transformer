import math
import torch
import torch.nn as nn
from torch import Tensor
from config import ModelConfig


class ForkBiasEncoder(nn.Module):
    """
    正弦编码 + 可学习投影 → per-head 注意力偏置。
    
    将 fork level（整数）编码为每个注意力头的标量偏置值。
    支持任意嵌套深度（正弦编码无上限）。
    
    - fork_level=0：无分叉或同元素 → 偏置强制为 0
    - fork_level=N：第 N 层 group 分叉 → proj(sinusoidal(N))，可学习
    """
    def __init__(self, num_heads: int, encoding_dim: int = 32):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.num_heads = num_heads
        # 可学习投影：sinusoidal(level) → per-head scalar bias
        self.proj = nn.Linear(encoding_dim, num_heads)
        # 初始化为零 → 初始偏置 ≈ 0，训练稳定
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def _sinusoidal_encode(self, levels: Tensor) -> Tensor:
        """levels: 任意形状整数张量 → (..., encoding_dim)"""
        half = self.encoding_dim // 2
        div_term = torch.exp(
            torch.arange(half, device=levels.device, dtype=torch.float32)
            * (-math.log(10000.0) / half)
        )
        pos = levels.unsqueeze(-1).float()  # (..., 1)
        pe = torch.zeros(*levels.shape, self.encoding_dim, device=levels.device)
        pe[..., 0::2] = torch.sin(pos * div_term)
        pe[..., 1::2] = torch.cos(pos * div_term)
        return pe
    
    def forward(self, fork_levels: Tensor) -> Tensor:
        """
        fork_levels: [B, S, S] 整数张量（0=无分叉）
        Returns: [B*H, S, S] float 偏置张量，可直接作为 src_mask 传入 TransformerEncoder
        """
        pe = self._sinusoidal_encode(fork_levels)      # [B, S, S, encoding_dim]
        bias = self.proj(pe)                            # [B, S, S, num_heads]
        # level=0 强制归零（结构性保证，不仅靠初始化）
        bias = bias.masked_fill(fork_levels.unsqueeze(-1) == 0, 0.0)
        B, S, _, H = bias.shape
        # [B, S, S, H] → [B, H, S, S] → [B*H, S, S]
        return bias.permute(0, 3, 1, 2).reshape(B * H, S, S)


class GlobalTransformer(nn.Module):
    """
    包装标准 TransformerEncoder，处理 padding mask、fork bias 和 dropout 等。
    
    Fork bias 通过 src_mask (float additive) 注入注意力分数，
    利用 PyTorch 原生 3D mask 支持 (N*num_heads, S, S)，
    无需手写注意力层，保留 FlashAttention 优化路径。
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.emb_norm = nn.LayerNorm(config.d_model)
        
        # Fork bias 编码器
        self.fork_bias_encoder = ForkBiasEncoder(
            num_heads=config.n_heads,
            encoding_dim=config.fork_bias_encoding_dim,
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,   # (batch_size, seq_len, d_model)
            norm_first=True     # Pre-LN
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.n_layers,
            enable_nested_tensor=False
        )
        self.out_norm = nn.LayerNorm(config.d_model)

    def forward(self, x: Tensor, padding_mask: Tensor = None,
                fork_bias_indices: Tensor = None) -> Tensor:
        """
        x: (B, seq_len, d_model)
        padding_mask: (B, seq_len) Boolean Tensor, True 表示是 <PAD>，会被赋 -inf
        fork_bias_indices: (B, seq_len, seq_len) 整数张量，fork level 矩阵
        """
        x = self.emb_norm(x)
        
        B, S, _ = x.shape
        H = self.fork_bias_encoder.num_heads
        
        # 构建统一的 float additive mask (B*H, S, S)
        # fork bias + padding 都以 float 形式合并，避免类型不匹配 warning
        src_mask = torch.zeros(B * H, S, S, device=x.device, dtype=x.dtype)
        
        if fork_bias_indices is not None:
            src_mask = src_mask + self.fork_bias_encoder(fork_bias_indices)
        
        if padding_mask is not None:
            # padding_mask: (B, S) bool, True = PAD
            # 仅做列方向遮蔽：阻止任何 token 关注 PAD 位置
            # 不做行方向遮蔽：PAD token 可以自注意力，避免全 -inf 行导致 softmax NaN
            # （PAD 行的输出是无意义的，但必须保持数值稳定，否则 NaN 会通过残差连接传播）
            pad_mask_col = padding_mask.unsqueeze(1)  # (B, 1, S) — 列方向
            pad_bias = pad_mask_col.unsqueeze(1).float() * (-1e9)  # (B, 1, 1, S)
            pad_bias = pad_bias.expand(B, H, S, S).reshape(B * H, S, S)
            src_mask = src_mask + pad_bias
        
        out = self.encoder(x, mask=src_mask)
        out = self.out_norm(out)
        return out
