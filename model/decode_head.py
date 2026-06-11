import torch
import torch.nn as nn
from torch import Tensor
from config import ModelConfig

class DecodeHead(nn.Module):
    """
    负责从 Transformer 输出的掩码位置提取预测值
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 数值预测：输出 1 维标量
        self.num_head = nn.Linear(config.d_model, 1)
        nn.init.normal_(self.num_head.weight, std=0.001)
        nn.init.constant_(self.num_head.bias, 0.0)
        
        # 布尔预测：输出 1 维 logits
        self.bool_head = nn.Linear(config.d_model, 1)
        nn.init.normal_(self.bool_head.weight, std=0.01)
        
        # 文本预测：将 d_model 投射回 LM embedding 空间，然后算 cosine similarity 或 InfoNCE
        self.text_head = nn.Linear(config.d_model, config.frozen_lm_dim)
        nn.init.normal_(self.text_head.weight, std=0.01)
        
        # 零值分类：预测 target 是否 ≈ 0
        self.zero_head = nn.Linear(config.d_model, 1)
        nn.init.normal_(self.zero_head.weight, std=0.01)
        nn.init.constant_(self.zero_head.bias, 0.0)
        
    def predict_number(self, x_mask: Tensor) -> Tensor:
        """
        x_mask: (N_mask, d_model)
        返回: (N_mask,)
        """
        return self.num_head(x_mask).squeeze(-1)
        
    def predict_boolean(self, x_mask: Tensor) -> Tensor:
        """
        返回: (N_mask,) logits
        """
        return self.bool_head(x_mask).squeeze(-1)
        
    def predict_string(self, x_mask: Tensor) -> Tensor:
        """
        返回: (N_mask, frozen_lm_dim)
        """
        return self.text_head(x_mask)
    
    def predict_is_zero(self, x_mask: Tensor) -> Tensor:
        """
        x_mask: (N_mask, d_model)
        返回: (N_mask,) logits
          > 0 表示预测为零值 (sigmoid > 0.5)
          < 0 表示预测为非零值
        """
        return self.zero_head(x_mask).squeeze(-1)
