from dataclasses import dataclass

@dataclass
class ModelConfig:
    d_model: int = 1536         # XXXL: 核心隐藏维度
    n_layers: int = 16          # XXXL: Transformer 编码器层数
    n_heads: int = 16           # XXXL: 注意力头数 (每个头 96 维)
    d_ff: int = 6144            # XXXL: FFN 中间层维度
    dropout: float = 0.1
    max_tokens: int = 512       # 截断阈值，单样本最大 token 数
    
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
    batch_size: int = 16        # XXXL @ 512 tokens: 实测峰值 ~11-21G / 23.4G
    lr: float = 1e-4            # AdamW 学习率
    weight_decay: float = 0.01  # AdamW 权重衰减
    betas: tuple = (0.9, 0.95)  # AdamW 动量参数 (β1, β2)
    mask_ratio: float = 0.15    # 自监督掩码比例
    device: str = "cuda"
    
    # 课程学习配置 — 每阶段: max_epochs (上限) + patience (收敛判定)
    # 当连续 patience 个 epoch loss 不下降时，自动进入下一阶段
    stage0_max_epochs: int = 100
    stage0_patience: int = 15
    stage1_max_epochs: int = 100
    stage1_patience: int = 15
    stage2_max_epochs: int = 100
    stage2_patience: int = 15
    stage3_max_epochs: int = 100
    stage3_patience: int = 15        
    
    dataset_size: int = 10000   # 每个 epoch 的样本数 (464M 模型需要足够数据)
    
    stage0_target_tokens: int = 20    
    stage1_target_tokens: int = 60    
    stage2_target_tokens: int = 150   
    stage3_target_tokens: int = 200   
    
    stage0_distractor_level: int = 0  
    stage1_distractor_level: int = 0  
    stage2_distractor_level: int = 3  
    stage3_distractor_level: int = 1  
    
    checkpoint_dir: str = "checkpoints"
