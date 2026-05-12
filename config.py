from dataclasses import dataclass

@dataclass
class ModelConfig:
    d_model: int = 1536         # XXXL: 核心隐藏维度
    n_layers: int = 16          # XXXL: Transformer 编码器层数
    n_heads: int = 16           # XXXL: 注意力头数 (每个头 96 维)
    d_ff: int = 6144            # XXXL: FFN 中间层维度
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
    group_scale: float = 0.1    # 随机向量 L2 归一化后的模长 (降低以平衡 val/path 量级)

@dataclass
class TrainConfig:
    batch_size: int = 16        # XXXL @ 512 tokens: 实测峰值 ~11-21G / 23.4G
    lr: float = 1e-4            # AdamW 学习率
    mask_ratio: float = 0.15    # 自监督掩码比例
    device: str = "cuda"
    
    # 课程学习配置 — 每阶段: max_epochs (上限) + patience (收敛判定)
    # 当连续 patience 个 epoch loss 不下降时，自动进入下一阶段
    stage0_max_epochs: int = 30     
    stage0_patience: int = 5        
    stage1_max_epochs: int = 50     # Stage 1 最多跑 50 个 epoch
    stage1_patience: int = 8        # Stage 1 连续 8 epoch 无改善则切换
    stage2_max_epochs: int = 80     # Stage 2 最多跑 80 个 epoch
    stage2_patience: int = 10       # Stage 2 连续 10 epoch 无改善则切换
    stage3_max_epochs: int = 40     # Stage 3 最多跑 40 个 epoch
    stage3_patience: int = 8        # Stage 3 连续 8 epoch 无改善则结束
    
    dataset_size: int = 10000   # 每个 epoch 的样本数 (464M 模型需要足够数据)
    
    stage0_target_tokens: int = 20    
    stage1_target_tokens: int = 100   # Stage 1 目标序列长度
    stage2_target_tokens: int = 300   # Stage 2 目标序列长度
    stage3_target_tokens: int = 200   # Stage 3 目标序列长度
    
    stage0_distractor_level: int = 0  
    stage1_distractor_level: int = 1  # Stage 1 干扰强度 (0-3)
    stage2_distractor_level: int = 2  # Stage 2 干扰强度
    stage3_distractor_level: int = 3  # Stage 3 干扰强度
    
    checkpoint_dir: str = "checkpoints"
