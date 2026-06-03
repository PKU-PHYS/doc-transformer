import math
import torch
import torch.nn as nn
from typing import Dict
from torch import Tensor
from config import ModelConfig


class StructuralBiasEncoder(nn.Module):
    """
    可扩展的 attention bias 编码器，支持任意数量的结构关系信号。

    每个信号要么是:
      - "category": 离散类别 → nn.Embedding(num_classes, num_heads) 查表
      - "continuous": 整数值 → sinusoidal + Linear(encoding_dim, num_heads) 投影

    所有信号的 per-head bias 独立计算后线性相加。

    扩展方式：在初始化时调用 register_category / register_continuous，
    新信号无需修改 forward，只需在调用方传入对应的 (B, T, T) 张量。
    """
    def __init__(self, num_heads: int, encoding_dim: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.encoding_dim = encoding_dim
        self._encoders = nn.ModuleDict()
        self._kinds: Dict[str, str] = {}          # name -> "category"/"continuous"
        self._enabled: Dict[str, bool] = {}       # 消融实验开关

    def register_category(self, name: str, num_classes: int):
        emb = nn.Embedding(num_classes, self.num_heads)
        nn.init.zeros_(emb.weight)                # 零初始化，训练初期 bias ≡ 0
        self._encoders[name] = emb
        self._kinds[name] = "category"
        self._enabled[name] = True

    def register_continuous(self, name: str):
        proj = nn.Linear(self.encoding_dim, self.num_heads)
        nn.init.zeros_(proj.weight)
        nn.init.zeros_(proj.bias)
        self._encoders[name] = proj
        self._kinds[name] = "continuous"
        self._enabled[name] = True

    def set_enabled(self, name: str, enabled: bool):
        """消融实验：运行时打开/关闭某个信号，无需重训。"""
        self._enabled[name] = enabled

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

    def forward(self, **signals: Tensor) -> Tensor:
        """
        Args:
            **signals: {name: (B, T, T) int/long tensor}，name 必须已注册
        Returns:
            bias: (B*H, T, T) float，全部启用信号的 per-head 贡献之和
        """
        total = None
        for name, value in signals.items():
            if not self._enabled.get(name, True):
                continue
            module = self._encoders[name]
            if self._kinds[name] == "category":
                contrib = module(value.long())                       # (B,T,T,H)
            else:
                contrib = module(self._sinusoidal_encode(value))     # (B,T,T,H)
            total = contrib if total is None else total + contrib
        if total is None:
            # 所有信号被禁用时返回零张量
            B, T = next(iter(signals.values())).shape[:2]
            device = next(iter(signals.values())).device
            total = torch.zeros(B, T, T, self.num_heads, device=device)
        B, T, _, H = total.shape
        # [B, T, T, H] → [B, H, T, T] → [B*H, T, T]
        return total.permute(0, 3, 1, 2).reshape(B * H, T, T)


class GlobalTransformer(nn.Module):
    """
    包装标准 TransformerEncoder，处理 padding mask、structural bias 和 dropout 等。
    
    Structural bias 通过 src_mask (float additive) 注入注意力分数，
    利用 PyTorch 原生 3D mask 支持 (N*num_heads, S, S)，
    无需手写注意力层，保留 FlashAttention 优化路径。
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.emb_norm = nn.LayerNorm(config.d_model)
        
        # Structural bias 编码器
        self.bias_encoder = StructuralBiasEncoder(
            num_heads=config.n_heads,
            encoding_dim=config.bias_encoding_dim,
        )
        self.bias_encoder.register_category("is_group_fork", num_classes=2)
        self.bias_encoder.register_continuous("first_diff")
        self.bias_encoder.register_continuous("tree_dist")
        # Apply config-driven enable flags
        self.bias_encoder.set_enabled("is_group_fork", config.bias_is_group_fork)
        self.bias_encoder.set_enabled("first_diff", config.bias_first_diff)
        self.bias_encoder.set_enabled("tree_dist", config.bias_tree_dist)
        
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
                bias_indices: Dict[str, Tensor] = None) -> Tensor:
        """
        x: (B, seq_len, d_model)
        padding_mask: (B, seq_len) Boolean Tensor, True 表示是 <PAD>，会被赋 -inf
        bias_indices: Dict[str, Tensor]，每个 value 为 (B, seq_len, seq_len) 整数张量
        """
        x = self.emb_norm(x)
        
        has_bias = (bias_indices is not None and len(bias_indices) > 0)
        
        B, S, _ = x.shape
        H = self.bias_encoder.num_heads
        
        # 构建统一的 float additive mask (B*H, S, S)
        src_mask = torch.zeros(B * H, S, S, device=x.device, dtype=x.dtype)
        
        if has_bias:
            src_mask = src_mask + self.bias_encoder(**bias_indices)
        
        if padding_mask is not None:
            pad_mask_col = padding_mask.unsqueeze(1)          # (B, 1, S)
            pad_bias = pad_mask_col.unsqueeze(1).float() * (-1e9)  # (B, 1, 1, S)
            pad_bias = pad_bias.expand(B, H, S, S).reshape(B * H, S, S)
            src_mask = src_mask + pad_bias
        
        out = self.encoder(x, mask=src_mask)
        out = self.out_norm(out)
        return out
