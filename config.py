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
    # 指数嵌入的覆盖范围 (bins = max - min + 1, offset = -min)
    exponent_min: int = -50     # 最小指数 → ~1e-15
    exponent_max: int = 49      # 最大指数 → ~5.6e14
    
    # ── Loss 尾数空间配置 ──
    # frexp 归一化时指数的 clamp 范围 (控制 scale = 2^E 的上下界)
    # 默认 min=max=0 → scale 恒为 1,等价关闭尾数归一化(loss 退化为 Huber on arcsinh)
    # 如需启用:E_min=0 → scale≥1 防小值/零值梯度被放大;E_max=4 → scale≤16 适度归一化大值
    loss_exponent_min: int = 0
    loss_exponent_max: int = 0
    # 量级补偿指数: loss *= scale^k
    # =0: 无补偿（大值梯度弱）; =1: 均匀梯度（对齐 MAE）; >1: 偏重大值
    loss_scale_power: float = 0.0
    # 额外压缩尺度: s·arcsinh(m/s)，=1 默认; >1 进一步放宽
    loss_compression_scale: float = 1.0
    
    # ── 零值分类头 ──
    use_zero_head: bool = False     # 是否启用零值分类头
    zero_threshold: float = 0.01   # |target| < 此值视为"零值"（训练标签）
    zero_neg_weight: float = 1.0    # 非零样本在 BCE 中的权重（>1 惩罚误杀，即把非零判为零）


@dataclass
class TrainConfig:
    batch_size: int = 160
    lr: float = 1e-4            # AdamW 学习率
    weight_decay: float = 0.01  # AdamW 权重衰减
    betas: tuple = (0.9, 0.95)  # AdamW 动量参数 (β1, β2)
    device: str = "cuda"
    num_workers: int = 0        # DataLoader 进程数 (0=主进程，避免 fork COW 开销)
    max_cpu_workers: int = 32   # 全局 CPU 密集型任务（如数据预处理）的最大进程数
    
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
