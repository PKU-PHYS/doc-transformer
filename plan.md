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
> **每个 token 的嵌入 = 值编码 + 路径编码。**
> **组关系通过 Group-Fork Relative Attention Bias 注入注意力层。**
> **然后用标准全局双向 Transformer 处理所有 token。**

这三个组成部分各自回答一个问题：

| 组成部分 | 回答的问题 | 类比 GPT |
|----------|-----------|---------|
| **值编码** | 这个值是什么？（"Fe2O3", 2.1, ...） | 词嵌入（"cat" 的含义） |
| **路径编码** | 这个值在 JSON 树的哪个位置？ | 位置编码（第 3 个词） |
| **Fork Bias** | 两个值的结构关系如何？（同组/跨组/哪层分叉） | T5 Relative Position Bias |

### 2.2 路径编码的处理规则

JSON 只有三种结构，每种的路径处理规则如下：

| JSON 类型 | 路径处理 | 组处理 |
|-----------|---------|--------|
| **标量** | 不增加路径层（它是叶子） | — |
| **对象** `{key: val}` | key 加入路径（type=0, id=hash(key)） | — |
| **数组** `[item, ...]` | 字段名已在路径中；为每个元素插入实例节点（type=1, id=自增） | 实例节点的 id 承载组信息 |

关键规则：**遍历数组时，为每个元素在路径中插入一个实例节点（text=父数组字段名, type=1）。同一元素的所有叶子共享该实例节点的 path_id，fork bias 自动检测分组关系。**

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

每个 token 的最终嵌入（$d$ 维向量）由两项相加：

$$\mathbf{t}_i = \underbrace{\mathbf{v}_i}_{\text{值编码}} + \underbrace{\mathbf{p}_i}_{\text{路径编码}}$$

组信息通过 **Group-Fork Relative Attention Bias** 直接注入注意力分数（见 §3.3）。

### 3.1 值编码 $\mathbf{v}_i$

| 数据类型 | 编码方式 |
|----------|---------|
| 数值型 | **Base-2 科学计数法解构** + 傅里叶特征映射（见下文详述） |
| 文本型 | $\mathbf{v} = \mathbf{W}_\text{text} \cdot \text{FrozenLM}(x)$，冻结语言模型编码后投影 |
| 布尔型 | $\mathbf{v} = \mathbf{W}_\text{bool} \cdot [x]$，0/1 标量投影 |
| 被掩码 | $\mathbf{v} = \mathbf{m}$，一个可学习的掩码向量 |

#### 数值型编码：Base-2 Mantissa-Exponent Split（xVal 机制）

如果将标量 $x$ 直接乘以权重向量 $\mathbf{W} \cdot x$，输出向量永远位于 $d$ 维空间中的**一条直线**上。这导致严重的**谱偏见（Spectral Bias）**——网络对高频细节（如 2.11 vs 2.12）极不敏感。

**解法：Base-2 科学计数法解构。** 使用 IEEE 754 的 `torch.frexp` 在 O(1) 时间内将任意标量分解为**尾数**（Mantissa）和**指数**（Exponent）两部分：

$$x = M \times 2^E$$

其中 $M \in [0.5, 1.0)$（正数）或 $M \in (-1.0, -0.5]$（负数），$x=0$ 时 $M=0$。$E$ 为整数指数。

**尾数编码：** 由于 $M$ 天然归一化在 $[-1, 1]$ 范围内，使用傅里叶特征映射将其展开为高频正余弦特征：

$$\text{Fourier}(M) = \left[\sin(\omega_1 M),\, \cos(\omega_1 M),\, \ldots,\, \sin(\omega_k M),\, \cos(\omega_k M)\right]$$

频率 $\omega$ 初始化为 $10^{[-1, 2]}$ 的对数均匀分布（因 $M$ 已归一化，无需覆盖 $10^{-4}$ 级低频）。

**指数编码：** $E$ 通常是 $-50$ 到 $+49$ 之间的整数，使用查表嵌入：

$$\mathbf{e} = \text{Embedding}(E + \text{offset}), \quad \text{Embedding} \in \mathbb{R}^{N_\text{bins} \times d}$$

**最终数值编码（两路相加）：**

$$\mathbf{v} = \underbrace{\mathbf{W}_\text{num} \cdot \text{Fourier}(M)}_{\text{精度编码（尾数）}} + \underbrace{\text{Embedding}(E)}_{\text{量级编码（指数）}}$$

这一设计使模型对数值的"精度"（0.50 vs 0.51）和"量级"（$10^3$ vs $10^{-3}$）获得了**完全解耦**的感知能力。


### 3.2 路径编码 $\mathbf{p}_i$：GRU 递归编码

路径现在包含字段名节点（type=0）和数组实例节点（type=1）。例如 `["doc"(0), "sites"(0), "sites"(1), "element"(0)]`。

采用 **GRU 递归编码** 方案——对每个路径节点，将 FrozenLM 文本特征与节点类型嵌入相加后，按路径顺序送入 GRU，取最终隐藏状态作为路径表示：

```
  GRU 初始隐藏状态 h0: 可学习零向量 (d_model)
  对路径中每个节点 l (按顺序):
    input_l = FrozenLM(text_l) + TypeEmb(type_l)   # (frozen_lm_dim,)
  路径编码 p = GRU([input_0, input_1, ...], h0)[-1]  # 取最终隐藏状态
```

**核心优势**：GRU 天然序列敏感，精确编码路径节点的顺序和依赖关系。对短路径（如 Stage 0 的深度 2 路径 `["doc", "value"]`）信号传播效率极高。`node_type_emb` 显式区分 Dict Key（type=0）与 Array Instance（type=1）。

### 3.3 Group-Fork Relative Attention Bias

组信息不再作为嵌入加法项，而是通过 **注意力偏置** 直接注入 Attention Score（类比 T5 的 Relative Position Bias）。

**原理**：对于任意两个叶子节点 (i, j)，比较它们的路径 `path_ids`。找到第一个不匹配的位置，如果该位置双方都是 Array Instance（type=1），则 `fork_level = 到该位置为止的累计 group 数`，否则 `fork_level = 0`。

- fork_level=0：同一元素内（Dict Key 分叉）或完全相同路径 → 无偏置
- fork_level=N：第 N 层 group 分叉 → 可学习偏置 `proj(sinusoidal(N))`

偏置编码采用 **正弦编码 + 可学习线性投影**，每个注意力头获得独立的标量偏置，支持任意嵌套深度。初始化为零，训练稳定。

> **方案 A（当前实现）**：fork bias 通过 `src_mask`（3D float tensor `[B*H, S, S]`）注入标准 `nn.TransformerEncoder`，零手写注意力代码。
> **方案 B（未来备选）**：使用 PyTorch 2.5+ 的 `FlexAttention` + `score_mod` 回调，在 kernel 内部 on-the-fly 计算偏置，避免实例化 `[B, S, S]` 矩阵，内存效率更高。需配合 `torch.compile`。

---

# 第二部分：代码实现规范与伪代码

## 四、文件结构概览

```
./
├── README.md                      # 子项目说明与运行指南
├── pixi.toml                      # Pixi 包管理配置
├── config.py                      # 全局超参数与配置 (ModelConfig, TrainConfig)
├── data/
│   ├── synthetic.py               # 顶层合成数据集与 Collate 函数
│   ├── in_context_generator.py    # In-Context Learning 任务生成器
│   ├── functions/                 # 数学函数注册表
│   │   ├── __init__.py            # 导入并触发所有函数模块注册
│   │   ├── registry.py            # 基类 MathRelation、FunctionRegistry 与工厂函数
│   │   ├── unary.py               # 单变量函数 (sin, cos, exp, log, sqrt, ...)
│   │   ├── binary.py              # 双变量函数 (add, sub, mul, div, pow, ...)
│   │   ├── multivar.py            # 多变量函数 (weighted_sum, dot_product, ...)
│   │   ├── implicit.py            # 扩展数学函数 (多项式/有理式等，支持隐式渲染)
│   │   └── composite.py           # 复合函数 (链式组合，如 sin(x+y))
│   ├── templates/                 # 结构模板引擎
│   │   ├── __init__.py
│   │   ├── engine.py              # TemplateEngine: ~60 种基础 JSON 模板 (扁平/嵌套/数组/复合，含动态变体达数十万)
│   │   └── distractors.py         # 干扰字段注入器 (inject_distractors)
│   └── text_tasks/                # 纯文本推理任务生成器
│       ├── __init__.py
│       └── generators.py          # TextTaskGenerator: 13 种跨模态桥接任务
├── model/
│   ├── __init__.py
│   ├── json_parser.py             # JSON 递归解析算法 -> List[LeafNode]
│   ├── frozen_lm.py               # 冻结句子模型包装与缓存机制
│   ├── value_encoder.py           # 值编码器（Base-2 frexp 尾数傅里叶 + 指数嵌入/文本/布尔/MASK）
│   ├── path_encoder.py            # GRU 路径编码器（text + type_emb -> GRU -> 最终隐藏状态）
│   ├── token_embedding.py         # 最终 token 嵌入组装 (Value + Path) + Batch 级去重
│   ├── transformer.py             # ForkBiasEncoder + 标准双向 Transformer (fork bias 通过 src_mask 注入)
│   ├── decode_head.py             # 类型特定解码头 (数值/布尔/文本，极小方差初始化)
│   └── document_transformer.py    # 顶层模型：端到端前向传播与 arcsinh Loss 计算
├── tests/
│   ├── test_stage1_overfit.py     # 阶段 1：单样本无 Pad 过拟合测试 (Zero-Loss)
│   ├── test_stage2_copy.py        # 阶段 2：单样本寻址与复制逻辑测试 (Logic Copy)
│   └── test_stage3_padding.py     # 阶段 3：变长 Batch + Padding 阻断测试
├── train.py                       # 正式训练入口 (四阶段课程学习 + TensorBoard)
└── inference.py                   # 推理入口 (支持 --checkpoint 参数或自动查找最新权重)
```

---

## 五、全局超参数字典 (Config)

`config.py` 中定义所有的超参数，供全局调用：

```python
# config.py
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
    
    # 傅里叶特征 (数值编码用，仅作用于 Base-2 尾数 M)
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
```

### 5.1 模型长度限制与截断策略说明

Document Transformer 中定义的 `max_tokens` (如 512) 是单条数据在被展平处理为叶子节点序列后的长度上限。需要特别注意：
1. **Token 的本质**：在这里，一个 token 代表 JSON 树中的一个叶子节点（例如一个数值或一个字符串），而非 NLP 传统意义上的一个子词。
2. **超长截断（Truncation）与致命隐患**：目前代码中采用了简单的顺序截断（超过 `max_tokens` 的尾部节点直接丢弃）。但这会带来一个**致命问题**：如果被截掉的尾部包含了逻辑推导的核心参数，而程序又恰好在剩下的节点里 MASK 了目标，这就会变成**无解任务（Unsolvable Task）**。
3. **智能截断策略（Smart Truncation，概念阶段）**：由于不能简单使用 NLP 的定长滑动窗口，我们在未来应该采用**基于树结构与 Mask 关联度的筛选机制**。
   - **先锁定目标**：在未经截断的完整树中，先选定要 Mask 的核心节点。
   - **计算亲缘度**：计算所有节点与该 Mask 节点的结构距离（如：共享多长的 `path` 前缀、是否属于同一个 `group_id` 的同级元素）。同组元素、兄弟节点保留优先级最高；不相干的分支（如扰动字段 `author`, `timestamp` 等）保留优先级最低。
   - **按优先级丢弃**：优先从低优先级的无关节点开始剔除，直到总节点数降至 `max_tokens`。
4. **显存与计算复杂度**：由于标准全局双向 Transformer 具有 $O(N^2)$ 的注意力计算复杂度，如果将 `max_tokens` 设得过大（例如应对极大 JSON），将导致显存爆炸。因此，对于非常冗长的文档数据，应当考虑预先过滤掉无用的嵌套字段，或在未来改进中引入稀疏注意力 (Sparse Attention) 机制。

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
    path: List[str]         # 根到叶子的文本标签序列（含实例节点）
    path_types: List[int]   # 每个路径节点的类型: 0=Dict Key, 1=Array Instance
    path_ids: List[int]     # 每个路径节点的唯一 ID (dict key: hash, array instance: 自增)
    group_ids: List[int]    # 祖先数组元素对应的全局唯一组 ID（兼容旧代码）

class JSONParser:
    def __init__(self):
        # 实例级计数器，确保多进程 DataLoader (num_workers>0) 下不发生状态冲突
        self.group_counter = 0
    
    @staticmethod
    def _key_hash(key: str) -> int:
        return (hash(key) & 0x7FFFFFFF) + 1_000_000  # 偏移避免与 instance ID 冲突
        
    def parse(self, data, current_path, current_path_types, current_path_ids, current_groups):
        if isinstance(data, bool):
            return [LeafNode(value=data, value_type="boolean", path=current_path,
                             path_types=current_path_types, path_ids=current_path_ids,
                             group_ids=current_groups)]
        # ... (int/float/str/None 同理)
            
        elif isinstance(data, dict):
            leaves = []
            for key, val in data.items():
                new_path = current_path + [str(key)]
                new_types = current_path_types + [0]            # Dict Key
                new_ids = current_path_ids + [self._key_hash(key)]
                leaves.extend(self.parse(val, new_path, new_types, new_ids, current_groups))
            return leaves
            
        elif isinstance(data, list):
            leaves = []
            parent_name = current_path[-1] if current_path else "array"
            for item in data:
                self.group_counter += 1
                new_path = current_path + [parent_name]         # 复用父数组字段名
                new_types = current_path_types + [1]            # Array Instance
                new_ids = current_path_ids + [self.group_counter]
                new_groups = current_groups + [self.group_counter]
                leaves.extend(self.parse(item, new_path, new_types, new_ids, new_groups))
            return leaves
```

### 6.2 合成数据生成流 (`data/synthetic.py`)

我们的合成数据不再只是简单的材料字典，而是包含了多类数学函数、推理任务和大量随机模板生成的丰富数据：

**生成流程核心组件**：
1. **FunctionRegistry / TextTaskGenerator**：随机采样数学关系或文本推理任务。其中，纯文本推理任务 (`TextTaskGenerator`) 并非简单的占位符，而是被精心设计为**三大类跨模态桥接任务**：
   - **数值 → 文本分类** (如根据数值大小判断 magnitude、正负号、象限)
   - **文本 → 文本推理** (如输出反函数名称、计算导数表达式配对)
   - **文本 → 数值检索** (如基于自然语言"archimedes_constant"输出 $3.141593$)
   这三类任务直接强制模型将 FrozenLM 的语义向量空间与傅里叶特征的高维数值空间进行深度对齐。
2. **FunctionRegistry 的 Tier 分级过滤**：每个注册的数学函数都带有 `tier` 属性（0=基础函数，含四则运算、基本三角/指数/对数/幂函数、比较函数等共 22 个；1=进阶函数，含反三角、双曲、隐函数、复合函数等）。在 Stage 0 中通过 `FunctionRegistry.set_filter(exact_tier=0)` 仅使用基础函数冷启动；其他阶段使用 `set_filter(max_tier=1)` 允许全部函数。
3. **TemplateEngine**：负责将抽象的关系渲染为千变万化的 JSON 树。它囊括了 4 大类（扁平、嵌套、数组、复合函数专用）约 60 种基础模板结构，更在每次生成后引入**动态扰动后处理 (Post-processing Perturbations)**。结合键名同义词池的随机化与后处理变换，有效变体数可达数十万。此外 `render_simple` 方法专为 Stage 0 设计，强制使用最简扁平模板。
4. **Distractor Injector**：随机向生成的 JSON 中插入完全无关的干扰分支（如 `timestamp`, `confidence` 等），迫使注意力机制学会在海量噪声中精准锁定有逻辑关联的有效节点。

**数据流伪代码**：
```python
def __getitem__(self, idx):
    # 0. 设定函数难度层级
    if current_mode == "simple":
        FunctionRegistry.set_filter(exact_tier=0)
    else:
        FunctionRegistry.set_filter(max_tier=1)
    
    # 1. 根据 train_mode 选择生成策略
    if current_mode == "simple":
        rel = FunctionRegistry.sample()
        doc = TemplateEngine.render_simple(rel)  # 极简扁平模板
    elif current_mode == "in_context":
        return generate_in_context_task(...)      # In-Context 专用生成器
    elif current_mode == "explicit_long" or (target_tokens is not None and target_tokens > 80):
        doc = generate_mixed_long_document(target_tokens)  # 混合长文档（内部已含 distractor 注入）
    else:  # "explicit" — 按概率分配
        nested_prob = [0.0, 0.2, 0.5, 0.8][min(distractor_level, 3)]
        r = random.random()
        if r < text_task_ratio:
            doc = TextTaskGenerator.generate()
        elif r < text_task_ratio + 0.3:
            doc = generate_compound_document(n_relations)   # 复合文档
        elif r < text_task_ratio + 0.5:
            doc = generate_array_document(n_items)          # 数组文档
        else:
            doc = generate_math_document()                  # 单关系文档
    
    # 2. 解析成叶子节点序列
    parser = JSONParser()
    leaves = parser.parse(doc, ["doc"], [0], [JSONParser._key_hash("doc")], [])
    
    # 3. 超长截断
    leaves = leaves[:self.max_tokens]
    
    # 4. "输出优先" Mask 策略（4 级 fallback）
    #    利用 TemplateEngine 返回的 safe_mask_keys 精准定位安全目标
    #    避免 mask 不可逆函数 (max/min/abs/sign) 的输入
    num_indices = [i for i, l in enumerate(leaves) if l.value_type == "number"]
    # P1: TemplateEngine 返回的精确安全键名（当前关系的所有输出同义词）
    candidates = [i for i in num_indices if leaves[i].path[-1] in safe_mask_keys]
    # P2 fallback: 全局输出键名集合
    if not candidates: candidates = [i for i in num_indices if leaves[i].path[-1] in _FALLBACK_OUTPUT_KEYS]
    # P3 fallback: 所有非装饰性数值节点
    if not candidates: candidates = [i for i in num_indices if leaves[i].path[-1] not in _DECORATION_KEYS]
    # P4 fallback: 最后一个数值节点
    if not candidates: candidates = [num_indices[-1]]
    mask_indices = random.sample(candidates, min(num_masks, len(candidates)))
    # 仍需更多 mask 时，从非数值节点补充
    
    return leaves, target_masks
```

---

## 七、网络模块伪代码

所有的嵌入模块最终都要将输入映射到 `d_model` 维空间。

### 7.1 值编码器 (`model/value_encoder.py`)

**核心：Base-2 科学计数法解构 (Mantissa-Exponent Split)**

数值编码分为两路：尾数编码器（傅里叶特征）和指数嵌入表。

```python
class FourierFeatureEncoder(nn.Module):
    """仅处理尾数 M ∈ [-1, 1]（由 torch.frexp 产生）"""
    def __init__(self, n_feats, d_model, learnable=True):
        super().__init__()
        if learnable:
            # 对数均匀分布初始化频率，为 M ∈ [-1, 1] 优化
            # [0.1, 100]: 最低频率捕捉宏观趋势，最高频率区分 0.50 和 0.51
            freqs = 10.0 ** torch.empty(n_feats).uniform_(-1, 2)
            self.freqs = nn.Parameter(freqs) 
        else:
            freqs = 10.0 ** torch.linspace(-1, 2, n_feats)
            self.register_buffer('freqs', freqs)
        self.proj = nn.Linear(2 * n_feats, d_model)

    def forward(self, x: Tensor): # x shape: (N,) — 尾数 M
        angles = x.unsqueeze(-1) * self.freqs
        fourier = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return self.proj(fourier) # (N, d_model)

class ValueEncoder(nn.Module):
    def __init__(self, d_model, frozen_lm_dim, n_fourier_feats, fourier_learnable,
                 n_exponent_bins=100, exponent_offset=50):
        super().__init__()
        # 数值型编码：Base-2 科学计数法解构
        self.mantissa_encoder = FourierFeatureEncoder(n_fourier_feats, d_model, fourier_learnable)
        self.exponent_embed = nn.Embedding(n_exponent_bins, d_model)  # E 查表
        # 文本、布尔、掩码编码器 (略)
    
    def forward(self, node_types, raw_values, lm_embeddings=None):
        # 数值型：Base-2 torch.frexp 向量化编码
        if num_indices:
            raw = torch.tensor(num_vals, dtype=torch.float32, device=device)
            m_tensor, e_tensor = torch.frexp(raw)
            # m_tensor ∈ [0.5, 1.0) 或 (-1.0, -0.5]，x=0 时 m=0
            # e_tensor 为整数指数，x = m * 2^e
            e_indices = (e_tensor + self.exponent_offset).clamp(0, self.n_exponent_bins - 1)
            m_emb = self.mantissa_encoder(m_tensor)   # (N_num, d_model) — 傅里叶编码尾数
            e_emb = self.exponent_embed(e_indices)     # (N_num, d_model) — 查表获取量级
            out[num_indices] = m_emb + e_emb  # 直接相加
```

### 7.2 Group-Fork Relative Attention Bias (`model/transformer.py`)

```python
class ForkBiasEncoder(nn.Module):
    """正弦编码 + 可学习投影 → per-head 注意力偏置，支持任意嵌套深度"""
    def __init__(self, num_heads, encoding_dim=32):
        self.proj = nn.Linear(encoding_dim, num_heads)
        nn.init.zeros_(self.proj.weight)  # 初始偏置 ≈ 0
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, fork_levels):  # [B, S, S] int
        pe = sinusoidal_encode(fork_levels)  # [B, S, S, encoding_dim]
        bias = self.proj(pe)                  # [B, S, S, num_heads]
        bias = bias.masked_fill(fork_levels.unsqueeze(-1) == 0, 0.0)  # level=0 → 0
        return bias.permute(0, 3, 1, 2).reshape(B * H, S, S)  # [B*H, S, S]
```

向量化分叉检测（在 `collate_fn` 中 CPU 计算）：
```python
def compute_fork_bias_indices(path_ids, is_group, valid_path_lens):
    # 1. 逐元素比对路径 → match_matrix [B, S, S, L]
    # 2. 找第一个不匹配位置 → first_mismatch_idx [B, S, S]
    # 3. 检查该位置是否双方均为 Array Instance → diverged_at_group
    # 4. 计算累计 group 数 → fork_level
    return where(~all_match & diverged_at_group, fork_level, 0)
```

### 7.3 主干网络与解码头

**主干网络**直接调用 `torch.nn.TransformerEncoder`，设置 `norm_first=True`, `batch_first=True`。Fork bias 和 padding mask 都以 float 形式合并为单一的 `src_mask`（`[B*H, S, S]`），通过 `encoder(x, mask=src_mask)` 传入。padding 位置赋 `-1e9`（等效 $-\infty$），fork bias 为可学习偏置值。

> [!WARNING]
> **Padding Mask 合并注意事项**：
> 不再单独传 `src_key_padding_mask`，而是将 padding 信息（bool → float × -1e9）与 fork bias 合并为统一的 float `src_mask`。这避免了 PyTorch 对 bool mask 和 float mask 类型不匹配的 warning。

**解码头**仅提取被掩码节点的向量：
- **数值型预测**：`pred_val = Linear(d_model -> 1)(h_mask).squeeze()`，在 **arcsinh 压缩空间**中使用 `HuberLoss`（即 `HuberLoss(arcsinh(pred), arcsinh(target))`）。arcsinh 变换消除了极端数值（如 $10^4$）带来的梯度方差爆炸，使模型对大数值和小数值同等敏感。
- **布尔型预测**：使用 `BCEWithLogitsLoss`。
- **文本型预测**：投影到 384 维后，使用 Cosine Embedding Loss 对齐 `FrozenLM(target_text)`。

### 7.4 Token 嵌入引擎的全局去重与缓存优化 (Global Deduplication & Caching)
为打破 `FrozenLM` 处理大规模深层 JSON 时带来的推理瓶颈（即同一 Batch 的不同节点内大量重复出现如 `"materials"`, `"formula"` 等短语），底层网络引入了两项超前性能优化：
1. **Batch 级全局词表去重 (Global Token Lookup)**：在每次前向传播的起始阶段 (`TokenEmbedding`)，引擎会主动提取当前 Batch 内所有叶子的字符串值和路径节点，放入 `set` 统一去重。提取出的独一无二的词汇表会被“一次性”送入语言模型，并建立 `text_lookup` 哈希表供后续路径编码和值编码查询，将庞大 Batch 下的大量冗余文本推理开销瞬间清零。
2. **跨步内存级缓存与防爆机制 (Eviction Policy)**：`FrozenLM` 内部封装了带状态的字典缓存 (`self._cache`)，使跨 Batch 间频繁出现的高频键名永远只需编码一次。同时加入了**自动驱逐清洗机制** (`if len(self._cache) > 10000: self._cache.clear()`)，完美杜绝了持续数万个 Epoch 的海量生僻随机词采样可能引发的 GPU 显存泄漏 (OOM) 崩溃问题。
3. **Batch 级展平优化 (Batch Flattening)**：`DocumentTransformer.forward()` 在嵌入阶段会先将整个 Batch 的 leaves 展平为一个大列表，单次调用 `TokenEmbedding` 处理所有 leaves（内部已按类型分组批量化），然后 scatter 回 `(B, max_len, d_model)`。嵌入阶段关闭 `autocast` 以避免手动张量赋值与 BF16 的不兼容问题。
4. **Fork Bias 计算**：`collate_fn` 在 CPU 上批量提取 `path_ids` 和 `path_types`，调用 `compute_fork_bias_indices()` 向量化计算 fork level 矩阵（O(B*S^2*L)），与叶子节点和 padding mask 一起返回。训练循环中移至 GPU 后传入 `model.forward()`。

## 八、训练策略与课程学习 (Curriculum Learning)

整个训练过程被设计为四个由易到难的课程阶段，逐步提升任务复杂度与干扰噪声：

### 阶段 0：极简预热训练 (Stage 0 - Simple)
- **目标**：在没有嵌套、干扰字段和数组等复杂 JSON 结构的理想环境下，使得刚初始化的冷启动模型快速学会最基础的数值傅里叶映射与加减乘除逻辑对应关系。
- **数据**：固定为全扁平键值对 (如 `{"a": 1, "b": 2, "func": "add", "result": 3}`)。
- **配置**：`target_tokens=20`, `distractor_level=0`。

### 阶段 1：复合与树状结构基础训练 (Stage 1 - Composite)
- **切换时机**：Stage 0 的 Loss 趋于收敛（patience 触发）。
- **数据形态**：单条或多条数学关系打包的 JSON 文档，包含嵌套结构（如 `{params: {...}, result: val}`）、复合文档（多条关系打包为子对象）、数组文档（多个同结构对象组成的 JSON 数组）和文本推理任务，内部包含显式的函数名称字段（如 `function: "sin"`）。
- **训练目的**：让模型在 Stage 0 的数值基础上，进一步学会理解树状结构、GRU 路径编码和 Fork Bias 引导的跨组注意力。
- **配置**：`train_mode="explicit"`, `target_tokens=60`, `distractor_level=0`。
- **函数过滤**：`FunctionRegistry.set_filter(max_tier=1)`，允许 Tier 0 和 Tier 1 函数。

### 阶段 2：抗噪训练 (Stage 2 - Anti-noise)
- **切换时机**：Stage 1 的 Loss 趋于收敛。
- **数据形态**：混合长文档，包含多条数学关系、文本任务和大量干扰字段（Distractors），通过 `generate_mixed_long_document` 循环追加内容直至接近 `target_tokens`。
- **训练目的**：迫使注意力机制学会在海量噪声中精准锁定有逻辑关联的有效节点，增强模型在真实噪音环境下的鲁棒性。
- **配置**：`train_mode="explicit_long"`, `target_tokens=150`, `distractor_level=3`（重度干扰）。

### 阶段 3：上下文规则归纳 (Stage 3 - In-Context Learning)
- **切换时机**：Stage 2 的 Loss 趋于收敛。
- **数据形态**：包含多个对象的纯 JSON 数组（Few-shot 演示），**移除显式的 `function` 字段**。多条 demonstrations 与 1 条 query 并列放入同一个数组中。
- **训练目的**：迫使模型激活多头注意力机制，跨越 JSON 组别观察前序数据的输入输出对，推断出隐含的数学规律，并应用到预测目标（`[MASK]`）上。
- **配置**：`train_mode="in_context"`, `target_tokens=200`, `distractor_level=1`（轻度干扰）。

**伪代码：In-Context 数据生成 (`data/in_context_generator.py`)**
```python
def generate_in_context_task(target_tokens=None, max_tokens=512, distractor_level=0):
    # 1. 拿到一个 generator 引用，多次调用获取同类函数的不同采样
    gen = FunctionRegistry.sample_generator()
    
    # 2. 动态计算 shots 数量（根据 target_tokens 自适应）
    if target_tokens is not None:
        tokens_per_shot = random.randint(8, 14)  # 实测平均值
        query_reserve = tokens_per_shot + 5       # 为 query 预留的 token 数
        available = max(target_tokens - query_reserve, tokens_per_shot)
        actual_shots = max(1, min(40, available // tokens_per_shot))
        actual_shots = max(1, min(40, int(actual_shots * random.uniform(0.7, 1.3))))  # ±30% 随机性
    else:
        actual_shots = random.randint(2, 6)
    
    # 3. 生成 demonstrations + query（全部使用 render_implicit 隐去函数名）
    context_jsons = []
    for _ in range(actual_shots):
        companion_rel = gen()
        doc = TemplateEngine.render_implicit(companion_rel)
        if distractor_level > 0:
            inject_distractors(doc, ...)
        context_jsons.append(doc)
    
    target_doc = TemplateEngine.render_implicit(gen())
    
    # 4. 纯数组并列结构（不使用 dict 包装）
    final_batch = context_jsons + [target_doc]
    
    # 5. 解析并 Mask 最后一个元素的数值/字符串节点
    parser = JSONParser()
    leaves = parser.parse(final_batch, ["records"], [0], [JSONParser._key_hash("records")], [])
    # 通过 group_ids 定位 query（数组最后一个元素的所有叶子共享同一 group_id）
    last_group_id = leaves[-1].group_ids[0] if leaves and leaves[-1].group_ids else None
    if last_group_id is not None:
        query_indices = [i for i, node in enumerate(leaves)
                         if node.group_ids and node.group_ids[0] == last_group_id
                         and node.value_type in ("number", "string")]
    else:
        query_indices = []
    
    if query_indices:
        mask_indices = random.sample(query_indices, min(len(query_indices), random.randint(1, 2)))
    else:
        # fallback: 当 query_indices 为空时，随机 mask 10% 节点
        num_masks = max(1, int(len(leaves) * 0.1))
        mask_indices = random.sample(range(len(leaves)), min(num_masks, len(leaves)))
    
    return leaves, target_masks
```

### 关于 Mixed 模式
`train_mode="mixed"` 作为一个可选的混合模式保留在 `SyntheticDataset` 中，当被选中时会按权重随机分配：40% explicit + 40% in_context + 20% explicit_long。可用于未来的混合鲁棒性训练阶段，防止灾难性遗忘。

### 8.4 大规模参数训练稳定性保障 (Scaling up to XXXL)
当模型规模扩大至大参数级别（例如 `d_model=1536, L=16`，约 4.6 亿参数）且使用 `norm_first=True` 时，底层网络的输出方差会极大，如果目标函数又是大数值范围（如指数运算），极易引发 Huber Loss 瞬间爆炸（> 500）与动量崩溃。

为保障大模型训练稳定性，实现中必须包含以下防线：
1. **Step 级学习率预热 (Warmup)**：绝对禁止大模型直接以 `1e-4` 等高学习率冷启动。使用 `get_cosine_schedule_with_warmup`，预热步数为 `min(625, total_steps // 10)`（约 10% 或固定上限 625 步，防止长阶段过度预热）。`scheduler.step()` 在每个 Batch 结束后执行，实现细粒度平滑过渡。
2. **极小方差初始化解码头**：数值预测头 `DecodeHead` (`nn.Linear(d_model, 1)`) 的权重必须用极小的方差（如 `std=0.001`）初始化，偏置设为 `0.0`。这迫使大模型在训练初期的盲目猜测阶段输出接近 `0` 的保守数值，避免巨大误差带来的毁灭性梯度惩罚，为主干网络争取建立注意力几何的时间。此外，为全面保障各类目标的预测稳定性，布尔分类头 (`bool_head`) 和文本匹配头 (`text_head`) 的权重也应当应用较小的方差（如 `std=0.01`）进行初始化约束。
3. **消除傅里叶特征的高频混叠**：由于 Base-2 frexp 解构后尾数 $M \in [-1, 1]$ 已天然归一化，傅里叶频率初始化为 $10^{[-1, 2]}$ 的对数均匀分布即可覆盖从宏观趋势（0.1）到微观精度（100）的全部特征尺度。量级信息由指数嵌入表 `exponent_embed` 独立处理，彻底消除了大数值混叠问题。
4. **Fork Bias 零初始化**：`ForkBiasEncoder` 的投影层权重和偏置初始化为零，确保训练初期注意力分数不受未学习偏置的干扰。随训练推进，模型逐渐学会对不同 fork level 施加恰当的偏置。
5. **嵌入层独立归一化 (Embedding LayerNorm)**：当主干 Transformer 使用 `norm_first=True` (Pre-LN) 时，第一层的 LayerNorm 是作用在注意力的分支上，而残差主干（Residual Stream）在初始阶段未受任何归一化约束。由于最终的 Token 嵌入是值和路径编码之和，存在初始方差累积。必须在传入 `TransformerEncoder` 前额外应用一个全局的 `nn.LayerNorm(d_model)`（代码中为 `GlobalTransformer.emb_norm`），截断残差流初始的方差膨胀。此外，Transformer 输出后还加了一层 `out_norm = nn.LayerNorm(d_model)` 用于稳定解码头的输入分布。
6. **全局梯度范数裁剪 (Gradient Clipping)**：在每个训练步的 `scaler.unscale_()` 之后、`scaler.step()` 之前，对全部参数执行 `clip_grad_norm_(max_norm=1.0)`。这是防止偶发的极端样本（如指数函数产生的大数值）导致单步梯度爆炸、破坏已学习注意力几何的最后一道防线。裁剪后的梯度范数同时被记录至 TensorBoard (`grad_norm`)，用于诊断训练是否处于稳定区间。

### 8.5 自动化课程学习调度与工程化加速
为保障四阶段课程学习连续平稳进行并极大化利用硬件算力，训练入口 (`train.py`) 需要部署以下工程组件：
1. **基于 Patience 的自动阶段流转**：摒弃单一的 `epochs` 设置。为每个阶段设置独立的目标词元长度 (`target_tokens`) 和干扰强度 (`distractor_level`)。利用 `patience` 机制监控阶段收敛，当验证 Loss 连续 `patience` 轮不再改善时，自动保存模型并无缝切换至下一阶段（例如从 Stage 1 进入 Stage 2），极大减少人工干预。
2. **混合精度训练 (BF16 AMP)**：当模型攀升至数亿参数，应当使用 `torch.amp.autocast('cuda', dtype=torch.bfloat16)` 进行混合精度训练。BFloat16 保留了与 FP32 相同的指数位，既有效杜绝了大规模数值计算时的下溢出，又能大幅降低显存开销并带来近 2 倍的吞吐量提升。**注意**：由于 BF16 与 FP32 指数范围相同，不存在 FP16 的下溢问题，因此 `GradScaler` 应被禁用（`enabled=False`），不需要动态 loss scaling。
3. **多维指标监控与诊断**：引入 TensorBoard 实时记录各阶段的训练动态（Loss, 学习率, GPU显存/峰值利用率, 梯度范数）。每 200 个 batch 调用 `_log_sample_case` 展示详细诊断输出，包括：将叶子节点按组关联分为 Useful Leaves（与 mask 同组）和 Distractor Leaves（无关干扰），自动推断函数上下文（通过 `FUNC_NAME_KEYS` 匹配函数名字段），并分类型显示预测值与真实值的对比（数值显示绝对/相对误差，布尔显示 logit，文本显示余弦相似度）。全部阶段完结后自动绘制跨阶段的 Loss 曲线图像供后续归纳总结。
4. **完全可复现的断点续训 (Deterministic Resume)**：每个 checkpoint 除模型权重外，还完整保存 optimizer state_dict、scheduler state_dict、GradScaler state_dict、当前 epoch/global_step、best_loss 以及 Python/PyTorch/CUDA 的全部随机数生成器状态 (`get_rng_states`)。恢复时先加载模型权重，再通过 `scheduler.load_state_dict()` 精确恢复学习率调度器状态（若 checkpoint 中缺少 scheduler 状态，则 fallback 为循环快进到对应 step），最后恢复 RNG 状态，确保数据生成和 dropout 等随机行为与连续训练完全一致。训练入口支持 `--resume <checkpoint.pth>` 命令行参数，自动识别 checkpoint 所属阶段并跳过已完成的前序阶段。

---

## 九、验证计划 (Sanity Check)

**强制执行**以下三阶段测试，任何阶段失败禁止进入下一阶段。

> [!NOTE]
> 三个测试均使用**缩小版模型** (`d_model=128, n_layers=2, n_heads=4, d_ff=256`) 以保证在数秒内完成。
> 通过阈值相应放宽，仅验证逻辑正确性（嵌入→注意力→解码通路是否打通），不验证大模型学习能力。

### 阶段 1：单样本过拟合测试 (`test_stage1_overfit.py`)
- **操作**：死循环只喂 1 条固定字典，如 `{"val1": 1.0, "val2": 2.0, "pred": [MASK]}` (GT=3.0)。不经过 collate 的 batching，没有 pad。
- **断言判断**：100 step 内，Loss < 1e-4（在 arcsinh 压缩空间下，等效于原始误差极小）。

### 阶段 2：寻址与复制测试 (`test_stage2_copy.py`)
- **操作**：动态生成 N 条字典，如 `{"source": {"id": "Fe", "val": RANDOM}, "target": {"id": "Fe", "pred": [MASK]}}`。
- **任务**：模型学会根据同级的 `id` 相等，去 `source` 把 `val` 原封不动搬到 `pred`。
- **断言判断**：2000 step 后，对全新随机数值的平均复制误差 `< 0.25`。证明路径和组嵌入引导了 Attention 构建起正确的空间几何。

### 阶段 3：Padding & Batch 阻断测试 (`test_stage3_padding.py`)
- **操作**：将阶段 2 的任务加入到不同长度的数组里，手动构造带 padding mask 的 Batch Tensor。
- **断言判断**：3000 step 后，带 pad 和不带 pad 的预测误差均 `< 0.4`。证明 `-inf` 被正确应用于行列双向，没有 Ghost Entanglement。

### 阶段 4：正式训练 (`train.py`)
在合成数据生成器的无限流下按四阶段课程学习训练：Stage 0 极简预热 → Stage 1 复合结构基础 → Stage 2 抗噪训练 → Stage 3 In-Context 归纳能力。每阶段基于 patience 自动切换，保存 Checkpoint，完成后运行 `inference.py`。

