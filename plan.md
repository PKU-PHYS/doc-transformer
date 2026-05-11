# Document Transformer 原型设计与详细实现计划

> 本文档是 Document Transformer 在项目根目录下的原型设计与实现规范。
> 上半部分描述了该模型处理 MongoDB 嵌套文档的理论架构与核心原理。
> 下半部分提供了伪代码级的详细实现规范，包含张量形状、算法逻辑与文件结构，确保任何人均可依据此文档写出完全相同的程序。

---

# 第一部分：理论架构与设计原理

## 一、从一个具体例子开始

假设我们的 MongoDB 中存了两条材料数据，以 JSONL 格式呈现：

```jsonl
{"formula": "Fe2O3", "band_gap": 2.1, "lattice": {"a": 5.04, "c": 13.77}, "sites": [{"element": "Fe", "z": 0.355}, {"element": "O", "z": 0.25}]}
{"formula": "TiO2", "band_gap": 3.2, "lattice": {"a": 4.59, "c": 2.96}, "sites": [{"element": "Ti", "z": 0.0}, {"element": "O", "z": 0.305}]}
```

**任务**：掩码 TiO2 的 `band_gap`，利用 TiO2 自身的其他字段以及 Fe2O3 的全部信息来预测它。

从模型的角度看，输入是一个 JSON 数组：

```json
[
  {"formula": "Fe2O3", "band_gap": 2.1,    "lattice": {...}, "sites": [...]},
  {"formula": "TiO2",  "band_gap": [MASK], "lattice": {...}, "sites": [...]}
]
```

这个 JSON 的结构是一棵树：

```
根 (array)                              ← 第 1 层数组："materials"
├── [0] (object)                        ← Fe2O3 文档
│   ├── "formula": "Fe2O3"              ← 叶子
│   ├── "band_gap": 2.1                 ← 叶子
│   ├── "lattice" (object)
│   │   ├── "a": 5.04                   ← 叶子
│   │   └── "c": 13.77                  ← 叶子
│   └── "sites" (array)                 ← 第 2 层数组
│       ├── [0] (object)                ← Fe 原子
│       │   ├── "element": "Fe"         ← 叶子
│       │   └── "z": 0.355             ← 叶子
│       └── [1] (object)                ← O 原子
│           ├── "element": "O"          ← 叶子
│           └── "z": 0.25              ← 叶子
├── [1] (object)                        ← TiO2 文档
│   ├── "formula": "TiO2"
│   ├── "band_gap": [MASK]             ← 要预测
│   ├── "lattice" (object)
│   │   ├── "a": 4.59
│   │   └── "c": 2.96
│   └── "sites" (array)
│       ├── [0]: {"element": "Ti", "z": 0.0}
│       └── [1]: {"element": "O",  "z": 0.305}
```

**核心问题**：如何把这棵树中的 16 个叶子变成 Transformer 能处理的 token 序列？

---

## 二、设计原理

### 2.1 核心思想

> **每个叶子节点 = 一个 token。**
> **每个 token 的嵌入 = 值编码 + 路径编码 + 组嵌入。**
> **然后用标准全局双向 Transformer 处理所有 token。**

这三个组成部分各自回答一个问题：

| 组成部分 | 回答的问题 | 类比 GPT |
|----------|-----------|---------|
| **值编码** | 这个值是什么？（"Fe2O3", 2.1, ...） | 词嵌入（"cat" 的含义） |
| **路径编码** | 这个值在 JSON 树的哪个位置？ | 位置编码（第 3 个词） |
| **组嵌入** | 这个值和谁属于同一组？ | BERT 的 segment embedding |

### 2.2 路径编码的处理规则

JSON 只有三种结构，每种的路径处理规则如下：

| JSON 类型 | 路径处理 | 组处理 |
|-----------|---------|--------|
| **标量** | 不增加路径层（它是叶子） | — |
| **对象** `{key: val}` | key **加入**路径 | — |
| **数组** `[item, ...]` | 数组的字段名**加入**路径，但索引**不加入** | 为同一元素的所有叶子分配**同一个组 ID** |

关键规则：**路径中遇到数组时，数组名加入路径（保留层次），数组索引不加入路径（用组嵌入代替，保持置换不变性）。**

### 2.3 对例子的解析

根据上述规则，顶层数组我们命名为 `"materials"`（或者用数据库 collection 名），解析 16 个叶子：

| # | 值 | 路径 | 组 L1 (materials) | 组 L2 (sites) |
|---|-----|------|:--:|:--:|
| 0 | "Fe2O3" | [materials, formula] | A | — |
| 1 | 2.1 | [materials, band_gap] | A | — |
| 2 | 5.04 | [materials, lattice, a] | A | — |
| 3 | 13.77 | [materials, lattice, c] | A | — |
| 4 | "Fe" | [materials, sites, element] | A | S0 |
| 5 | 0.355 | [materials, sites, z] | A | S0 |
| 6 | "O" | [materials, sites, element] | A | S1 |
| 7 | 0.25 | [materials, sites, z] | A | S1 |
| 8 | "TiO2" | [materials, formula] | B | — |
| 9 | **[MASK]** | [materials, band_gap] | B | — |
| 10 | 4.59 | [materials, lattice, a] | B | — |
| 11 | 2.96 | [materials, lattice, c] | B | — |
| 12 | "Ti" | [materials, sites, element] | B | S2 |
| 13 | 0.0 | [materials, sites, z] | B | S2 |
| 14 | "O" | [materials, sites, element] | B | S3 |
| 15 | 0.305 | [materials, sites, z] | B | S3 |

**观察这张表的模式：**

- token 0 和 token 8 路径完全一样（都是 `materials.formula`），但组 L1 不同（A vs B）。
  → 模型知道：它们是"不同材料的同一字段"。
- token 4 和 token 6 路径完全一样（`materials.sites.element`），组 L1 相同（A），但组 L2 不同（S0 vs S1）。
  → 模型知道：它们是"同一材料的不同原子的同一字段"。
- token 4 和 token 5 路径不同（`element` vs `z`），但组 L1 和 L2 都相同（A, S0）。
  → 模型知道：它们是"同一原子的不同属性"。

---

## 三、Token 嵌入的详细构成

每个 token 的最终嵌入（$d$ 维向量）由三项相加：

$$\mathbf{t}_i = \underbrace{\mathbf{v}_i}_{\text{值编码}} + \underbrace{\mathbf{p}_i}_{\text{路径编码}} + \underbrace{\sum_l \mathbf{g}_i^{(l)}}_{\text{各层组嵌入之和}}$$

### 3.1 值编码 $\mathbf{v}_i$

| 数据类型 | 编码方式 |
|----------|---------|
| 数值型 | **傅里叶特征映射**（见下文详述） |
| 文本型 | $\mathbf{v} = \mathbf{W}_\text{text} \cdot \text{FrozenLM}(x)$，冻结语言模型编码后投影 |
| 布尔型 | $\mathbf{v} = \mathbf{W}_\text{bool} \cdot [x]$，0/1 标量投影 |
| 被掩码 | $\mathbf{v} = \mathbf{m}$，一个可学习的掩码向量 |

#### 数值型编码：傅里叶特征映射（Fourier Feature Mapping）

如果将标量 $x$ 直接乘以权重向量 $\mathbf{W} \cdot x$，输出向量永远位于 $d$ 维空间中的**一条直线**上。这导致严重的**谱偏见（Spectral Bias）**——网络对高频细节（如 2.11 vs 2.12）极不敏感。

**解法：傅里叶特征映射。** 用不同频率的正弦/余弦基函数将标量"炸开"到高维空间：

$$\text{Fourier}(x) = \left[\sin(\omega_1 x),\, \cos(\omega_1 x),\, \sin(\omega_2 x),\, \cos(\omega_2 x),\, \ldots,\, \sin(\omega_k x),\, \cos(\omega_k x)\right]$$

最终数值编码：

$$\mathbf{v} = \mathbf{W}_\text{num} \cdot \text{Fourier}(x), \quad \mathbf{W}_\text{num} \in \mathbb{R}^{d \times 2k}$$

### 3.2 路径编码 $\mathbf{p}_i$（类比 GPT 的位置编码）

路径是一个字段名序列，如 `["materials", "lattice", "a"]`。有两种候选方案：

**方案 A：GRU 递归编码（有序，O(L)时间）**
```
  h₀ = 零向量 (d维)
  h₁ = GRU(h₀, FrozenLM("materials"))
  h₂ = GRU(h₁, FrozenLM("lattice"))
  h₃ = GRU(h₂, FrozenLM("a"))
  路径编码 p = h₃  (d维)
```

**方案 B：绝对深度嵌入（并行，O(1)时间）**
```
  h₁ = Depth_1 + FrozenLM("materials")
  h₂ = Depth_2 + FrozenLM("lattice")
  h₃ = Depth_3 + FrozenLM("a")
  路径编码 p = h₁ + h₂ + h₃  (d维)
```

### 3.3 组嵌入 $\mathbf{g}_i^{(l)}$

每遇到一层数组，就为该层生成一个组嵌入。同一数组元素内的所有叶子共享**同一个随机采样的 $d$ 维向量**。

为防止随机向量求内积时引发方差波动，采样的随机向量必须经过 L2 归一化：
`g = random_normal(d); g = g / ||g|| * scale`
这确保了同组内积为稳定常数 `scale^2`，异组在高维空间几近绝对正交，彻底消除注意力分数的随机噪声污染。

---

# 第二部分：代码实现规范与伪代码

## 四、文件结构概览

```
./
├── README.md                      # 子项目说明与运行指南
├── config.py                      # 全局超参数与配置 (ModelConfig, TrainConfig)
├── data/
│   └── synthetic.py               # 合成数据生成器与 Collate 函数
├── model/
│   ├── __init__.py
│   ├── json_parser.py             # JSON 递归解析算法 -> List[LeafNode]
│   ├── frozen_lm.py               # 冻结句子模型包装与缓存机制
│   ├── value_encoder.py           # 值编码器（傅里叶/文本/布尔/MASK）
│   ├── path_encoder.py            # 路径编码器（GRU/Depth Embedding 双方案）
│   ├── group_embedding.py         # 动态随机组嵌入生成与 L2 归一化
│   ├── token_embedding.py         # 最终 token 嵌入组装 (Value + Path + Group)
│   ├── transformer.py             # 标准双向 Transformer (含 Padding Mask)
│   ├── decode_head.py             # 类型特定解码头 (数值/布尔/文本)
│   └── document_transformer.py    # 顶层模型：端到端前向传播与 Loss 计算
├── tests/
│   ├── test_stage1_overfit.py     # 阶段 1：单样本无 Pad 过拟合测试 (Zero-Loss)
│   ├── test_stage2_copy.py        # 阶段 2：单样本寻址与复制逻辑测试 (Logic Copy)
│   └── test_stage3_padding.py     # 阶段 3：变长 Batch + Padding 阻断测试
├── train.py                       # 正式训练入口
├── inference.py                   # 推理入口
└── utils.py                       # 工具函数
```

---

## 五、全局超参数字典 (Config)

`config.py` 中定义所有的超参数，供全局调用：

```python
# config.py
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
```

---

## 六、数据流转核心结构

### 6.1 叶子节点数据结构与解析算法 (`model/json_parser.py`)

**伪代码**：
```python
from dataclasses import dataclass
from typing import Any, List

@dataclass
class LeafNode:
    value: Any              # 原始值 (如 2.1, "TiO2", True)
    value_type: str         # 枚举: "number", "string", "boolean", "mask"
    path: List[str]         # 根到叶子的字段名列表, e.g., ["materials", "lattice", "a"]
    group_ids: List[int]    # 祖先数组元素对应的全局唯一组 ID, e.g., [14, 52]

```python
class JSONParser:
    def __init__(self):
        # 实例级计数器，确保多进程 DataLoader (num_workers>0) 下不发生状态冲突
        self.group_counter = 0
        
    def parse(self, data: Any, current_path: List[str], current_groups: List[int]) -> List[LeafNode]:
        if isinstance(data, (int, float, str, bool)):
            val_type = "number" if isinstance(data, (int, float)) and not isinstance(data, bool) else type(data).__name__
            return [LeafNode(value=data, value_type=val_type, path=current_path, group_ids=current_groups)]
            
        elif isinstance(data, dict):
            leaves = []
            for key, val in data.items():
                new_path = current_path + [key]
                leaves.extend(self.parse(val, new_path, current_groups))
            return leaves
            
        elif isinstance(data, list):
            leaves = []
            for item in data:
                self.group_counter += 1
                new_groups = current_groups + [self.group_counter]
                leaves.extend(self.parse(item, current_path, new_groups))
            return leaves
```

---

## 七、网络模块伪代码

所有的嵌入模块最终都要将输入映射到 `d_model` (256维) 空间。

### 7.1 值编码器 (`model/value_encoder.py`)

**核心：数值型傅里叶编码 (FourierFeatureEncoder)**
```python
class FourierFeatureEncoder(nn.Module):
    def __init__(self, n_feats, d_model, learnable=True):
        super().__init__()
        if learnable:
            self.freqs = nn.Parameter(torch.randn(n_feats)) 
        else:
            self.register_buffer('freqs', 2 * math.pi * (2.0 ** torch.arange(n_feats)))
        self.proj = nn.Linear(2 * n_feats, d_model)

    def forward(self, x: Tensor): # x shape: (N,)
        angles = x.unsqueeze(-1) * self.freqs  # (N, 1) * (n_feats) -> (N, n_feats)
        fourier = torch.cat([angles.sin(), angles.cos()], dim=-1) # (N, 2*n_feats)
        return self.proj(fourier) # (N, d_model)
```

### 7.2 组嵌入 (`model/group_embedding.py`)

```python
def generate_group_embeddings(group_ids_list: List[List[int]], d_model: int, scale: float) -> Tensor:
    unique_ids = set(gid for gids in group_ids_list for gid in gids)
    
    random_vecs = {}
    for uid in unique_ids:
        vec = torch.randn(d_model)
        random_vecs[uid] = (vec / vec.norm(p=2)) * scale
        
    embeddings = []
    for gids in group_ids_list:
        if not gids:
            embeddings.append(torch.zeros(d_model))
        else:
            sum_vec = sum(random_vecs[gid] for gid in gids)
            embeddings.append(sum_vec)
            
    return torch.stack(embeddings) # shape: (N, d_model)
```

### 7.3 主干网络与解码头

**主干网络**直接调用 `torch.nn.TransformerEncoder`，设置 `norm_first=True`, `batch_first=True`，并传入 `src_key_padding_mask=Padding_Mask` 阻断 `<PAD>` 节点的注意力。

> [!WARNING]
> **PyTorch Padding Mask 布尔逻辑陷阱**：
> 在 `src_key_padding_mask` 中，**`True` 表示“这是垃圾填充，请忽略（赋 $-\infty$）”**，**`False` 表示“这是有效数据，请计算”**。这与普通直觉完全相反！在 `collate_fn` 生成该矩阵时切勿填反，否则网络只会对着全 `<PAD>` 的张量算 Attention，第一步 Loss 就会变成 `NaN`。

**解码头**仅提取被掩码节点的向量：
- **数值型预测**：`pred_val = Linear(d_model -> 1)(h_mask).squeeze()`，使用 `HuberLoss`。
- **布尔型预测**：使用 `BCEWithLogitsLoss`。
- **文本型预测**：投影到 384 维后，使用 InfoNCE 或 Cosine Embedding Loss 对齐 `FrozenLM(target_text)`。

---

## 八、验证计划 (Sanity Check)

**强制执行**以下三阶段测试，任何阶段失败禁止进入下一阶段：

### 阶段 1：单样本过拟合测试 (`test_stage1_overfit.py`)
- **操作**：死循环只喂 1 条固定字典，如 `{"val1": 1.0, "val2": 2.0, "pred": [MASK]}` (GT=3.0)。不经过 collate 的 batching，没有 pad。
- **断言判断**：100 step 内，预测误差 `|pred - 3.0| < 1e-4` 且 Loss 单调下降。

### 阶段 2：寻址与复制测试 (`test_stage2_copy.py`)
- **操作**：动态生成 N 条字典，如 `{"source": {"id": "Fe", "val": RANDOM}, "target": {"id": "Fe", "pred": [MASK]}}`。
- **任务**：模型学会根据同级的 `id` 相等，去 `source` 把 `val` 原封不动搬到 `pred`。
- **断言判断**：500 step 后，对全新生成的随机数值复制误差 `< 1e-3`。证明路径和组嵌入真正地引导了 Attention 构建起了正确的空间几何。

### 阶段 3：Padding & Batch 阻断测试 (`test_stage3_padding.py`)
- **操作**：将阶段 2 的任务加入到不同长度的数组里，用 `collate_fn` 打包成带 Pad 和 `src_key_padding_mask` 的 Batch Tensor。
- **断言判断**：Loss 同样必须收敛到趋于 0。证明 `-inf` 被正确应用，没有 Ghost Entanglement。

### 阶段 4：正式训练 (`train.py`)
在合成数据生成器 (带有些微物理相关性噪音) 的无限流下训练 100 Epochs，保存 Checkpoint，运行 `inference.py`。
