from dataclasses import dataclass

@dataclass
class ModelConfig:
    d_model: int = 256          # 核心隐藏维度
    n_layers: int = 6           # Transformer 编码器层数
    n_heads: int = 8            # 注意力头数 (每个头 32 维)
    d_ff: int = 1024            # FFN 中间层维度
    dropout: float = 0.1
    max_depth: int = 10         # Depth Embedding 方案的最大深度支持
    max_tokens: int = 512       # 截断阈值，单样本最大 token 数
    
    # Frozen LM 配置
    frozen_lm_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    frozen_lm_dim: int = 384    
    
    # 路径编码选择: "gru" 递归 或 "depth" 绝对深度
    path_encoding: str = "gru"  
    
    # 傅里叶特征 (数值编码用)
    n_fourier_feats: int = 64   # 频率数 k，映射后维度为 2k = 128
    fourier_learnable: bool = True # 频率参数是否参与梯度更新
    
    # 组嵌入缩放
    group_scale: float = 1.0    # 随机向量 L2 归一化后的模长

@dataclass
class TrainConfig:
    batch_size: int = 32        # RTX 4000 24GB 显存充裕
    lr: float = 1e-4            # AdamW 学习率
    epochs: int = 100
    mask_ratio: float = 0.15    # 自监督掩码比例
    device: str = "cuda"
