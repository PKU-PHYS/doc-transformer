from dataclasses import dataclass
from typing import Optional


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
    
    # Structural Bias 编码维度 (正弦编码 → 可学习投影)
    bias_encoding_dim: int = 32
    # Structural Bias 各信号开关 (训练/推理均生效)
    bias_is_group_fork: bool = True
    bias_first_diff: bool = True
    bias_tree_dist: bool = True
    bias_same_parent: bool = False
    bias_shared_group_depth: bool = False
    bias_same_path_template: bool = False
    
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
    # 压缩空间中的逐样本数值 loss: "huber" 保持历史默认, "l1" 直接对齐 MAE
    numeric_loss: str = "huber"
    # Huber loss 的 L1/L2 转折点；仅 numeric_loss="huber" 时生效
    numeric_huber_delta: float = 1.0
    # 数值输出约束：
    #   "linear": 无约束，适用于一般回归
    #   "softplus": 非负输出，适用于 band gap 等物理非负目标
    numeric_output: str = "linear"
    numeric_softplus_beta: float = 1.0
    # 数值预测校准: 先做 scale * raw + bias, 再做下界裁剪和近零阈值置零
    prediction_scale: float = 1.0
    prediction_bias: float = 0.0
    # 验证/推理阶段的数值下界；None 表示不裁剪，band gap 可显式设为 0.0
    prediction_min_value: Optional[float] = None
    # 可选的 train-only 分段 residual 校准映射，在 scale/bias/min 之后插值相加
    prediction_residual_centers: Optional[list] = None
    prediction_residual_corrections: Optional[list] = None
    prediction_zero_threshold: Optional[float] = None
    
    # ── 零值分类头 ──
    use_zero_head: bool = False     # 是否启用零值分类头
    zero_logit_threshold: float = 0.0
    zero_threshold: float = 0.01   # |target| < 此值视为"零值"（训练标签）
    zero_neg_weight: float = 1.0    # 非零样本在 BCE 中的权重（>1 惩罚误杀，即把非零判为零）


@dataclass
class TrainConfig:
    batch_size: int = 160
    lr: float = 1e-4            # AdamW 学习率
    weight_decay: float = 0.01  # AdamW 权重衰减
    betas: tuple = (0.9, 0.95)  # AdamW 动量参数 (β1, β2)
    # 结构 bias 编码器（整个 StructuralBiasEncoder）的 lr 倍率。
    # 各信号的基底已在编码器内归一化到 ‖b‖²=1（有效 lr 统一为 ~1×），此倍率是在该干净基线上
    # 主动选定的、统一施加的工作点——而非基底维数造成的偶然放大。
    # 动机：零初始化的 attention 偏置需赶在 backbone 锁死前长到决定性幅度（见赛跑/critical-period 假设）。
    # =1.0 关闭；恢复实验中 fork ≈20× 达到 0.188，故默认 20.0。挂在编码器角色上而非单个信号，保持 schema-agnostic。
    structural_bias_lr_mult: float = 20.0
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
