import torch
import torch.nn as nn
from torch import Tensor
from config import ModelConfig

class GlobalTransformer(nn.Module):
    """
    包装标准 TransformerEncoder，处理 padding mask 和 dropout 等
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.emb_norm = nn.LayerNorm(config.d_model)
        
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

    def forward(self, x: Tensor, padding_mask: Tensor = None) -> Tensor:
        """
        x: (B, seq_len, d_model)
        padding_mask: (B, seq_len) Boolean Tensor, True 表示是 <PAD>，会被赋 -inf
        """
        x = self.emb_norm(x)
        out = self.encoder(x, src_key_padding_mask=padding_mask)
        out = self.out_norm(out)
        return out
