from dataclasses import dataclass


@dataclass
class ModelConfig:
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
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
    device: str = "cuda"
    num_workers: int = 0        # DataLoader 进程数 (0=主进程，避免 fork COW 开销)
    
    checkpoint_dir: str = "checkpoints"


# ═══════════════════════════════════════════════════════════════
# 模型预设 — 通过 --model-size 一键切换
# ═══════════════════════════════════════════════════════════════
#   内存瓶颈是注意力的 O(S²)，与 d_model 无关，因此 batch_size 不需要随模型变
#
#   Size   | d_model | layers | heads | d_ff  | dropout | weight_decay
#   -------|---------|--------|-------|-------|---------|-------------
#   small  |   192   |   4    |   4   |  768  |  0.10   |    0.01
#   medium |   384   |   6    |   6   | 1536  |  0.10   |    0.01
#   large  |   512   |   8    |   8   | 2048  |  0.10   |    0.01
#   xl     |   768   |  12    |  12   | 3072  |  0.10   |    0.01

MODEL_PRESETS = {
    "small": {
        "model": dict(d_model=192, n_layers=4, n_heads=4, d_ff=768, dropout=0.10),
        "train": dict(weight_decay=0.01),
    },
    "medium": {
        "model": dict(d_model=384, n_layers=6, n_heads=6, d_ff=1536, dropout=0.10),
        "train": dict(weight_decay=0.01),
    },
    "large": {
        "model": dict(d_model=512, n_layers=8, n_heads=8, d_ff=2048, dropout=0.10),
        "train": dict(weight_decay=0.01),
    },
    "xl": {
        "model": dict(d_model=768, n_layers=12, n_heads=12, d_ff=3072, dropout=0.10),
        "train": dict(weight_decay=0.01),
    },
}


def get_configs(size: str = "large"):
    """根据预设名称返回 (ModelConfig, TrainConfig)。"""
    if size not in MODEL_PRESETS:
        raise ValueError(f"Unknown model size '{size}'. Choose from: {list(MODEL_PRESETS.keys())}")
    
    preset = MODEL_PRESETS[size]
    model_config = ModelConfig(**preset["model"])
    train_config = TrainConfig(**preset["train"])
    return model_config, train_config
