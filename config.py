from dataclasses import dataclass

@dataclass
class ModelConfig:
    d_model: int = 192          # Small: 与 FT-Transformer 同级
    n_layers: int = 4           # Small: Transformer 编码器层数
    n_heads: int = 4            # Small: 注意力头数 (每个头 48 维)
    d_ff: int = 768             # Small: FFN 中间层维度
    dropout: float = 0.1
    max_tokens: int = 256       # 截断阈值，单样本最大 token 数
    
    # Frozen LM 配置
    frozen_lm_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    frozen_lm_dim: int = 384    
    
    # Fork Bias 编码维度 (正弦编码 → 可学习投影)
    fork_bias_encoding_dim: int = 32
    
    # 傅里叶特征 (数值编码用)
    n_fourier_feats: int = 64   # 频率数 k，映射后维度为 2k = 128
    fourier_learnable: bool = True # 频率参数是否参与梯度更新
    
    # 科学计数法解构 (Mantissa-Exponent Split)
    n_exponent_bins: int = 100  # 指数嵌入表大小 (覆盖 E = -50 到 +49)
    exponent_offset: int = 50   # 指数偏移 (E=0 映射到 index 50)
    


@dataclass
class TrainConfig:
    batch_size: int = 160
    lr: float = 1e-4            # AdamW 学习率
    weight_decay: float = 0.01  # AdamW 权重衰减
    betas: tuple = (0.9, 0.95)  # AdamW 动量参数 (β1, β2)
    mask_ratio: float = 0.15    # 自监督掩码比例
    device: str = "cuda"
    num_workers: int = 0        # DataLoader 进程数 (0=主进程，避免 fork COW 开销)
    
    checkpoint_dir: str = "checkpoints"
