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

**指数编码：** $E$ 通常是 $-50$ 到 $+49$ 之间的整数（由 `exponent_min` 和 `exponent_max` 控制），使用查表嵌入：

$$\mathbf{e} = \text{Embedding}(E + \text{offset}), \quad \text{Embedding} \in \mathbb{R}^{N_\text{bins} \times d}$$

其中 $N_\text{bins} = \text{exponent\_max} - \text{exponent\_min} + 1$（默认 100 bins），$\text{offset} = -\text{exponent\_min}$（默认 50），确保 $E=0$ 映射到 index 50。

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

**核心优势**：GRU 天然序列敏感，精确编码路径节点的顺序和依赖关系。对短路径（如深度 2 路径 `["doc", "value"]`）信号传播效率极高。`node_type_emb` 显式区分 Dict Key（type=0）与 Array Instance（type=1）。

**性能优化**：使用 **Padded GRU**（非 `pack_sequence`），让 cuDNN 使用批量并行 kernel，backward 速度提升约 1000 倍。通过 `valid_lens` + `gather` 取每条路径最后一个真实位置的隐藏状态，与 `pack_sequence` 的输出完全一致。

### 3.3 Group-Fork Relative Attention Bias

组信息不再作为嵌入加法项，而是通过 **注意力偏置** 直接注入 Attention Score（类比 T5 的 Relative Position Bias）。

**原理**：对于任意两个叶子节点 (i, j)，比较它们的路径 `path_ids`。找到第一个不匹配的位置，如果该位置双方都是 Array Instance（type=1），则 `fork_level = 到该位置为止的累计 group 数`，否则 `fork_level = 0`。

- fork_level=0：同一元素内（Dict Key 分叉）或完全相同路径 → 无偏置
- fork_level=N：第 N 层 group 分叉 → 可学习偏置 `proj(sinusoidal(N))`

偏置编码采用 **正弦编码 + 可学习线性投影**，每个注意力头获得独立的标量偏置，支持任意嵌套深度。初始化为零，训练稳定。

> **当前实现**：fork bias 通过 `src_mask`（3D float tensor `[B*H, S, S]`）注入标准 `nn.TransformerEncoder`，零手写注意力代码。Padding mask 与 fork bias 合并为统一的 float `src_mask`，避免 PyTorch 对 bool mask 和 float mask 类型不匹配的 warning。

---

# 第二部分：代码实现规范与伪代码

## 四、文件结构概览

```
./
├── README.md                      # 项目说明与运行指南
├── pixi.toml                      # Pixi 包管理配置
├── config.py                      # 全局超参数与配置 (ModelConfig, TrainConfig, MODEL_PRESETS)
├── data/
│   ├── __init__.py                # 统一导出 collate_fn, TabularDataset, TableLoader
│   ├── base.py                    # 通用 collate_fn + fork_bias 计算 (compute_fork_bias_indices, compute_single_fork_bias)
│   ├── tabular/                   # 表格数据管线
│   │   ├── __init__.py
│   │   ├── configs.py             # 各表格数据集训练配方 (StageConfig, DatasetConfig, DATASET_CONFIGS)
│   │   ├── loader.py              # TableLoader: 下载/缓存 + StandardScaler + BallTree 索引
│   │   ├── dataset.py             # TabularDataset: query-aware 多行样本生成 (预计算邻居、路径模板、fork_bias)
│   │   └── cache/                 # BallTree 磁盘缓存
│   └── matbench/                  # Matbench 晶体材料数据管线
│       ├── __init__.py
│       ├── configs.py             # Matbench 任务训练配方 + CLI 参数注册 (MatbenchTaskConfig, register_args, build_dataset_options)
│       ├── loader.py              # MatbenchLoader: matminer 加载 → pymatgen Structure → 嵌套 JSON (含 angles/bonds/ewald 等)
│       ├── dataset.py             # MatbenchDataset: 预解析 + fork_bias 缓存 + mask target
│       └── cache/                 # 转换结果 + 预解析磁盘缓存
├── model/
│   ├── __init__.py
│   ├── json_parser.py             # JSON 递归解析算法 -> List[LeafNode]
│   ├── frozen_lm.py               # 冻结句子模型包装与缓存机制 (all-MiniLM-L6-v2)
│   ├── value_encoder.py           # 值编码器（Base-2 frexp 尾数傅里叶 + 指数嵌入/文本/布尔/MASK）
│   ├── path_encoder.py            # GRU 路径编码器（text + type_emb -> GRU -> 最终隐藏状态）
│   ├── token_embedding.py         # 最终 token 嵌入组装 (Value + Path) + Batch 级去重
│   ├── transformer.py             # ForkBiasEncoder + 标准双向 Transformer (fork bias 通过 src_mask 注入)
│   ├── decode_head.py             # 类型特定解码头 (数值/布尔/文本/零值分类，极小方差初始化)
│   └── document_transformer.py    # 顶层模型：端到端前向传播与 Loss 计算
├── tests/                         # 测试目录（当前为空，预留位置）
├── train.py                       # 正式训练入口 (课程学习 + TensorBoard + 断点续训)
├── eval.py                        # 评估模块 (MAE / RMSE 指标计算)
├── inference.py                   # 推理入口 (支持 --checkpoint 参数或自动查找最新权重)
├── slides/                        # Slidev 演示文稿
│   ├── slides.md
│   └── public/
└── tmp/                           # 临时/分析脚本 (git ignored)
```

---

## 五、全局超参数字典 (Config)

`config.py` 中定义所有的超参数，供全局调用：

```python
# config.py
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
    
    # 傅里叶特征 (数值编码用，仅作用于 Base-2 尾数 M)
    n_fourier_feats: int = 64   # 频率数 k，映射后维度为 2k = 128
    fourier_learnable: bool = True # 频率参数是否参与梯度更新
    
    # 科学计数法解构 (Mantissa-Exponent Split)
    # 指数嵌入的覆盖范围 (bins = max - min + 1, offset = -min)
    exponent_min: int = -50     # 最小指数 → ~1e-15
    exponent_max: int = 49      # 最大指数 → ~5.6e14
    
    # ── Loss 尾数空间配置 ──
    # frexp 归一化时指数的 clamp 范围 (控制 scale = 2^E 的上下界)
    # E_min=0 → scale≥1，避免小值/零值梯度被放大
    # E_max=0 → scale=1，即不归一化（当前默认）
    loss_exponent_min: int = 0
    loss_exponent_max: int = 0
    # 量级补偿指数: loss *= scale^k
    # =0: 无补偿（大值梯度弱）; =1: 均匀梯度（对齐 MAE）; >1: 偏重大值
    loss_scale_power: float = 1.0
    # 额外压缩尺度: s·arcsinh(m/s)，=1 默认; >1 进一步放宽
    loss_compression_scale: float = 10.0
    
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
```

### 5.1 模型预设 (Model Presets)

通过 `--model-size` 一键切换模型规模。内存瓶颈是注意力的 O(S²)，与 d_model 无关，因此 batch_size 不需要随模型变。

```python
MODEL_PRESETS = {
    "small":  {"model": dict(d_model=192,  n_layers=4,  n_heads=4,  d_ff=768,  dropout=0.10)},
    "medium": {"model": dict(d_model=384,  n_layers=6,  n_heads=6,  d_ff=1536, dropout=0.10)},
    "large":  {"model": dict(d_model=512,  n_layers=8,  n_heads=8,  d_ff=2048, dropout=0.10)},
    "xl":     {"model": dict(d_model=768,  n_layers=12, n_heads=12, d_ff=3072, dropout=0.10)},
}
```

| Size | d_model | layers | heads | d_ff | dropout |
|------|---------|--------|-------|------|---------|
| small | 192 | 4 | 4 | 768 | 0.10 |
| medium | 384 | 6 | 6 | 1536 | 0.10 |
| large | 512 | 8 | 8 | 2048 | 0.10 |
| xl | 768 | 12 | 12 | 3072 | 0.10 |

### 5.2 模型长度限制与截断策略说明

Document Transformer 中定义的 `max_tokens` (如 256) 是单条数据在被展平处理为叶子节点序列后的长度上限。需要特别注意：
1. **Token 的本质**：在这里，一个 token 代表 JSON 树中的一个叶子节点（例如一个数值或一个字符串），而非 NLP 传统意义上的一个子词。
2. **超长截断（Truncation）**：代码中采用简单的顺序截断（超过 `max_tokens` 的尾部节点直接丢弃）。
3. **Matbench 动态 Token 预算**：`structure_to_json` 根据 `max_tokens` 动态计算 `max_sites`，优先为 bonds 预留预算，sites 用剩余预算。大结构先尝试 primitive cell 缩减，超限后按元素分层采样 (`_stratified_sample`)，保持化学组成比例。
4. **显存与计算复杂度**：标准全局双向 Transformer 具有 $O(N^2)$ 注意力计算复杂度，`max_tokens` 设得过大会导致显存爆炸。

---

## 六、数据流转核心结构

### 6.1 叶子节点数据结构与解析算法 (`model/json_parser.py`)

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
    group_ids: List[int]    # 祖先数组元素对应的全局唯一组 ID

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
        elif isinstance(data, (int, float)):
            return [LeafNode(value=data, value_type="number", ...)]
        elif isinstance(data, str):
            if data == "[MASK]":
                return [LeafNode(value=data, value_type="mask", ...)]
            return [LeafNode(value=data, value_type="string", ...)]
        elif data is None:
            return []  # 忽略 None 值
            
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
        
        else:
            # 对于不支持的类型，转为字符串处理
            return [LeafNode(value=str(data), value_type="string", ...)]
```

### 6.2 数据源概览

当前项目支持两种数据源，各有独立的加载器 (Loader)、数据集 (Dataset) 和训练配方 (Config)：

| 数据源 | 目录 | 输入格式 | 典型任务 |
|--------|------|----------|----------|
| **Tabular** | `data/tabular/` | CSV / sklearn 内置数据集 | California Housing, Diabetes, Wine Quality |
| **Matbench** | `data/matbench/` | matminer + pymatgen Structure | Band Gap, Formation Energy, Dielectric |

### 6.3 表格数据管线 (`data/tabular/`)

#### TableLoader (`data/tabular/loader.py`)

负责数据加载、标准化和近邻索引：

```python
class TableLoader:
    def __init__(self, df: pd.DataFrame, target_col=None, cache_dir="data/tabular/cache"):
        # 分离数值列和类别列
        # StandardScaler 标准化数值列
        # 构建/加载 BallTree 缓存
    
    def precompute_neighbors(self, k: int) -> np.ndarray:
        """批量预计算所有行的 K 近邻索引 (n_rows, k)"""
    
    def get_row_dict(self, idx: int) -> dict:
        """返回第 idx 行的 flat dict"""
    
    def split(self, test_ratio=0.2, seed=42) -> Tuple[TableLoader, TableLoader]:
        """标准 train/test 分割"""
    
    @classmethod
    def from_builtin(cls, name: str, max_rows=None, target_col=None):
        """从 sklearn 内置数据集创建 (california_housing, diabetes, wine_quality, covertype)"""
    
    @classmethod
    def from_csv(cls, path: str):
        """从 CSV 文件创建"""
```

支持的内置数据集：
- `california_housing`: 20640 行 × 9 列
- `diabetes`: 442 行 × 10 列
- `wine_quality`: 6497 行 × 12 列
- `covertype`: 581012 行 × 54 列（默认截取 50000）

#### TabularDataset (`data/tabular/dataset.py`)

将表格行转为 JSON 数组文档，mask target 列预测：

```python
class TabularDataset(Dataset):
    def __init__(self, loader: TableLoader, n_rows=5, max_tokens=512):
        # 1. 预计算所有行的 K 近邻 (一次性 BallTree 查询)
        # 2. 预构建路径模板 (所有样本共享相同的 key 结构)
        # 3. 预计算 fork_bias (所有样本路径相同，只需算一次)
    
    def __getitem__(self, idx):
        # 1. 取预计算的邻居行索引
        # 2. 打乱行顺序，组装 LeafNode（复用模板路径）
        # 3. Mask seed 行的 target 列
        return leaves, target_masks, self._fork_bias
```

**数据流**：
1. `__init__` 时预计算邻居 → 预构建路径模板 → 预计算 fork_bias
2. `__getitem__` 只做：取行数据 → 填模板 → mask target → 返回
3. fork_bias 为预计算的 `(T, T) int8` 张量，所有样本共享

#### 表格数据训练配方 (`data/tabular/configs.py`)

```python
DATASET_CONFIGS = {
    "california_housing": DatasetConfig(n_rows=1, stages=[StageConfig("train", 100, 15)]),
    "diabetes":           DatasetConfig(n_rows=1, stages=[StageConfig("train", 200, 20)]),
    "wine_quality":       DatasetConfig(n_rows=1, stages=[StageConfig("train", 100, 15)]),
    "covertype":          DatasetConfig(n_rows=1, stages=[StageConfig("train", 50, 10)], max_rows=50000),
}
```

### 6.4 Matbench 晶体材料数据管线 (`data/matbench/`)

#### MatbenchLoader (`data/matbench/loader.py`)

负责将 pymatgen Structure 转为精简嵌套 JSON：

```python
class MatbenchLoader:
    def __init__(self, task_name: str, max_tokens=256, test_ratio=0.2, seed=42,
                 dataset_options=None, max_cpu_workers=32):
        # 1. 通过 matminer 加载 Matbench 数据集
        # 2. 并行转换 Structure → JSON (joblib Parallel)
        # 3. Train/Test split
        # 4. 磁盘缓存
```

**`structure_to_json` 核心转换函数**：

将 pymatgen Structure 转为精简嵌套 JSON dict。支持丰富的可选特征：

| 选项 | CLI 参数 | 说明 |
|------|----------|------|
| `add_angles` | `--add-angles` | per-site 配位统计 (cn, avg_angle, min_angle) |
| `add_bonds` | `--add-bonds` | 全局 bonds 列表（去重原子对 + 距离） |
| `add_composition` | `--add-composition` | 元素比例数组 [{element, ratio}] |
| `add_ewald` | `--add-ewald` | per-site Ewald 静电能 |
| `add_element_props` | `--add-element-props` | per-element 物理属性 (en, ie, ea) |
| `add_comp_ewald` | `--add-comp-ewald` | per-element 平均 Ewald 能 |
| `add_comp_nn` | `--add-comp-nn` | per-element 平均最近邻距离 |
| `add_spacegroup` | `--add-spacegroup` | 空间群编号 + 晶系 |
| `add_density` | `--add-density` | 密度 + 体积/原子 |
| `add_nn_stats` | `--add-nn-stats` | 全局最近邻距离统计 (nn_min, nn_mean) |
| `drop_coords` | `--drop-coords` | 移除 per-site 绝对坐标 (x,y,z) |
| `add_comp_ewald_stats` | `--add-comp-ewald-stats` | per-element Ewald std/min/max |
| `add_comp_nn_stats` | `--add-comp-nn-stats` | per-element NN distance std/min/max |
| `add_mean_bonds` | `--add-mean-bonds` | per-element-pair 平均键距 |

**Token 预算动态分配**：
1. 计算 fixed_overhead（lattice 6 + target 1 + 可选特征）
2. Bonds 保底预留 `max_bonds × tokens_per_bond`
3. Sites 用剩余预算：`max_sites = (max_tokens - overhead - bonds_budget) / tokens_per_site`
4. 大结构先尝试 primitive cell 缩减，超限后按元素分层采样

**Ewald 静电能计算三层策略**：
1. sites ≤ 50 → BVAnalyzer（键价分析，物理最准）
2. reduced_atoms ≤ 30 → `oxi_state_guesses(max_sites=-1)`（组成穷举）
3. 都失败 → 跳过 Ewald 特征

**转换后的 JSON 结构示例**：
```json
{
  "lattice": {"a": 4.59, "b": 4.59, "c": 2.96, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
  "spacegroup": 136,
  "crystal_system": "tetragonal",
  "density": 4.25,
  "vol_per_atom": 15.88,
  "nn_min": 1.95,
  "nn_mean": 2.13,
  "sites": [
    {"element": "Ti", "x": 0.0, "y": 0.0, "z": 0.0, "cn": 6, "avg_angle": 90.0, "min_angle": 77.8, "ewald_energy": -12.45},
    {"element": "O",  "x": 0.3, "y": 0.3, "z": 0.0, "cn": 3, "avg_angle": 120.0, "min_angle": 102.5, "ewald_energy": 6.22}
  ],
  "bonds": [{"elements": ["Ti", "O"], "dist": 1.95}, ...],
  "composition": [{"element": "Ti", "ratio": 0.3333, "en": 1.54, "ewald": -12.45, "nn": 1.95}, ...],
  "target": 3.2
}
```

#### MatbenchDataset (`data/matbench/dataset.py`)

```python
class MatbenchDataset(Dataset):
    def __init__(self, docs: list, max_tokens=512, target_key="target", cache_tag=None):
        # 1. 预解析所有文档为 LeafNode 序列（根节点命名为 "crystal"）
        # 2. 预计算每个样本的 target 叶子索引
        # 3. 预计算每个样本的 fork_bias 矩阵 (int8 节省内存)
        # 4. 磁盘缓存（跳过后续加载的解析开销）
    
    def __getitem__(self, idx):
        # 1. 复制预解析的叶子（避免修改原数据）
        # 2. Mask target 叶子
        return leaves, target_masks, fork_bias
```

**与 TabularDataset 的核心区别**：
- 每个样本的结构不同（site 数量不同）→ fork_bias 每个样本独立
- 无 BallTree / 邻居：每个 structure 自包含全部信息
- 路径深度更深 (2-4 层 vs tabular 的 2 层)

#### Matbench 训练配方 (`data/matbench/configs.py`)

```python
MATBENCH_CONFIGS = {
    "matbench_dielectric":  MatbenchTaskConfig(stages=[StageConfig("train", 100, 15)]),
    "matbench_perovskites": MatbenchTaskConfig(stages=[StageConfig("train", 100, 15)]),
    "matbench_mp_gap":      MatbenchTaskConfig(stages=[StageConfig("train", 200, 15)]),
    "matbench_mp_e_form":   MatbenchTaskConfig(stages=[StageConfig("train", 50, 10)]),
}
```

CLI 选项通过 `register_args()` → `build_dataset_options()` → `build_cache_tag()` 三个函数与 train.py 交互。

### 6.5 通用 Collate 与 Fork Bias (`data/base.py`)

**Fork Bias 预计算** (`compute_single_fork_bias`)：

由于每个样本的路径在训练过程中不变，fork_bias 在数据集初始化时预计算并缓存为 `(T, T) int8` 张量：

```python
def compute_single_fork_bias(leaves: List[LeafNode]) -> torch.Tensor:
    """单个样本的 fork bias 矩阵 (T, T) int8"""
    # 构建 path_ids (1, T, D), is_group (1, T, D), valid_lens (1, T)
    # 调用 compute_fork_bias_indices 批量计算
    return result[0].to(torch.int8)  # 节省内存
```

**向量化分叉检测** (`compute_fork_bias_indices`)：

```python
def compute_fork_bias_indices(path_ids, is_group, valid_path_lens):
    # 1. 逐元素比对路径 → match_matrix [B, S, S, L]
    # 2. 找第一个不匹配位置 → first_mismatch_idx [B, S, S]
    # 3. 检查该位置是否为 group 层级 → diverged_at_group
    # 4. 输出 fork_level (0 = 无分叉或 dict key 分叉)
    return where(~all_match & diverged_at_group, fork_level, 0)
```

**Collate 函数** (`collate_fn`)：

```python
def collate_fn(batch, max_tokens=512):
    # 1. 双重截断保护（dataset 层 + collate 层）
    # 2. 构建 padding_mask (B, max_len)
    # 3. Pad + stack 预计算的 fork_bias → (B, max_len, max_len)
    return batched_leaves, batched_masks, padding_mask, fork_bias_indices
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
                 exponent_min=-50, exponent_max=49):
        super().__init__()
        # 数值型编码：Base-2 科学计数法解构
        self.mantissa_encoder = FourierFeatureEncoder(n_fourier_feats, d_model, fourier_learnable)
        # 从 exponent_min/max 派生 bins 数和 offset
        n_exponent_bins = exponent_max - exponent_min + 1  # 默认 100
        self.exponent_embed = nn.Embedding(n_exponent_bins, d_model)
        self.exponent_offset = -exponent_min  # E=0 映射到 index 50
        # 文本投影、布尔投影、掩码向量 (略)
    
    def forward(self, node_types, raw_values, lm_embeddings=None):
        # 数值型：Base-2 torch.frexp 向量化编码
        if num_indices:
            raw = torch.tensor(num_vals, dtype=torch.float32, device=device)
            raw = torch.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=-1e6)
            m_tensor, e_tensor = torch.frexp(raw)
            e_indices = (e_tensor + self.exponent_offset).clamp(0, self.n_exponent_bins - 1).long()
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

class GlobalTransformer(nn.Module):
    def __init__(self, config):
        self.emb_norm = nn.LayerNorm(config.d_model)  # 嵌入层独立归一化
        self.fork_bias_encoder = ForkBiasEncoder(config.n_heads, config.fork_bias_encoding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.n_heads,
            dim_feedforward=config.d_ff, dropout=config.dropout,
            activation="gelu", batch_first=True, norm_first=True  # Pre-LN
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers,
                                              enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(config.d_model)  # 输出归一化
    
    def forward(self, x, padding_mask=None, fork_bias_indices=None):
        x = self.emb_norm(x)
        # 构建统一的 float additive mask (B*H, S, S)
        src_mask = torch.zeros(B * H, S, S, ...)
        if fork_bias_indices is not None and fork_bias_indices.any():
            src_mask = src_mask + self.fork_bias_encoder(fork_bias_indices)
        if padding_mask is not None:
            pad_bias = padding_mask.float() * (-1e9)  # True → -inf
            src_mask = src_mask + pad_bias  # 广播到 (B*H, S, S)
        out = self.encoder(x, mask=src_mask)
        return self.out_norm(out)
```

### 7.3 解码头与 Loss 计算

**解码头** (`model/decode_head.py`)：

| 预测类型 | 线性层 | 初始化 |
|----------|--------|--------|
| 数值 | `Linear(d_model → 1)` | `std=0.001`, bias=0 |
| 布尔 | `Linear(d_model → 1)` | `std=0.01` |
| 文本 | `Linear(d_model → frozen_lm_dim)` | `std=0.01` |
| 零值分类 | `Linear(d_model → 1)` | `std=0.01`, bias=0 |

**尾数空间 Loss 计算** (`document_transformer.py: compute_loss`)：

数值预测的 Loss 在**尾数空间**中计算，通过真实值的量级 $2^E$ 进行归一化：

```python
# 1. 用真实值的指数做量级归一化
_, e_true = torch.frexp(targets_t)
e_clamped = e_true.clamp(loss_exponent_min, loss_exponent_max)
scale = torch.pow(2.0, e_clamped.float())

# 2. 归一化到尾数空间
m_pred = preds_t / scale
m_true = targets_t / scale

# 3. arcsinh 压缩 + Huber Loss
s = loss_compression_scale  # 默认 10.0
per_sample = F.huber_loss(
    torch.arcsinh(m_pred / s) * s,
    torch.arcsinh(m_true / s) * s,
    reduction='none')

# 4. 量级补偿
k = loss_scale_power  # =0 无补偿, =1 均匀梯度, >1 偏重大值
if k != 0:
    per_sample = per_sample * torch.pow(scale, k)
```

**Loss 空间配置参数说明**：
- `loss_exponent_min=0`: `scale ≥ 1`，避免小值/零值梯度被放大
- `loss_exponent_max=0`: `scale = 1`（当前默认不归一化，即直接在原始空间 arcsinh）
- `loss_scale_power=1.0`: `loss *= scale^k`，k=1 时梯度与量级无关，等效 MAE
- `loss_compression_scale=10.0`: `s·arcsinh(m/s)`，s 越大压缩越温和

**零值分类头** (可选，`use_zero_head=True`)：

```python
# 训练时：独立的 BCE loss（detach 阻断梯度回流 backbone）
zero_logit = self.decode_head.predict_is_zero(mask_repr.detach().unsqueeze(0))
zero_labels = (targets_t.abs() < zero_threshold).float()
sample_weights = where(zero_labels == 1, 1.0, zero_neg_weight)  # 非零加权
zero_loss = F.binary_cross_entropy_with_logits(zero_logits, zero_labels, weight=sample_weights)

# 推理时：logit > 0 (sigmoid > 0.5) → 直接输出 0；否则输出回归头预测
```

**布尔预测**：`BCEWithLogitsLoss`。
**文本预测**：投影到 384 维后，用 `CosineEmbeddingLoss` 对齐 `FrozenLM(target_text)`。

### 7.4 Token 嵌入引擎的全局去重与缓存优化 (Global Deduplication & Caching)

为打破 `FrozenLM` 处理大规模深层 JSON 时带来的推理瓶颈（即同一 Batch 的不同节点内大量重复出现如 `"element"`, `"lattice"` 等短语），底层网络引入了以下性能优化：

1. **Batch 级全局词表去重 (Global Token Lookup)**：在每次前向传播的起始阶段 (`TokenEmbedding`)，引擎会主动提取当前 Batch 内所有叶子的字符串值和路径节点，放入 `set` 统一去重。提取出的独一无二的词汇表会被"一次性"送入语言模型，并建立 `text_to_idx` 哈希表供后续路径编码和值编码查询，将庞大 Batch 下的大量冗余文本推理开销瞬间清零。

2. **跨步内存级缓存与防爆机制 (Eviction Policy)**：`FrozenLM` 内部封装了带状态的字典缓存 (`self._cache`)，使跨 Batch 间频繁出现的高频键名永远只需编码一次。同时加入了**自动驱逐清洗机制** (`if len(self._cache) > 10000: self._cache.clear()`)，完美杜绝了持续数万个 Epoch 的海量词采样可能引发的 GPU 显存泄漏 (OOM) 崩溃问题。

3. **Batch 级展平优化 (Batch Flattening)**：`DocumentTransformer.forward()` 在嵌入阶段会先将整个 Batch 的 leaves 展平为一个大列表，单次调用 `TokenEmbedding` 处理所有 leaves（内部已按类型分组批量化），然后 scatter 回 `(B, max_len, d_model)`。嵌入阶段关闭 `autocast` 以避免手动张量赋值与 BF16 的不兼容问题。

4. **路径编码零逐元素操作**：`TokenEmbedding` 在 Python 侧收集所有路径数据为普通列表，最后一次性创建张量。路径文本嵌入通过高级索引赋值 `path_text_embs[row_indices, col_indices] = unique_embs[emb_indices]` 完成，1 次 CUDA 操作代替 N×L 次。

---

## 八、训练策略 (`train.py`)

### 8.1 训练架构概览

训练入口 `train.py` 支持两种数据路径：

1. **Tabular 路径**：`--dataset california_housing` → `TableLoader` + `TabularDataset`
2. **Matbench 路径**：`--dataset matbench_mp_gap` → `MatbenchLoader` + `MatbenchDataset`

两种路径共享同一个 `train_stage()` 函数和 `DocumentTransformer` 模型。

### 8.2 课程学习 (Curriculum Learning)

每个数据集配方定义独立的训练阶段列表 (`stages`)，每阶段包含：
- `name`: 阶段名称
- `max_epochs`: 最大 epoch 数
- `patience`: 连续 N 个 epoch loss 不下降则视为收敛

```python
for stage_idx, stage_cfg in enumerate(stages):
    dataset = build_dataset(stage_cfg, ...)
    log, global_step = train_stage(
        model, dataset,
        max_epochs=stage_cfg.max_epochs,
        patience=stage_cfg.patience,
        ...
    )
```

### 8.3 训练稳定性保障

1. **Step 级学习率预热 (Warmup)**：使用 `get_cosine_schedule_with_warmup`，预热步数为 `min(625, total_steps // 10)`。`scheduler.step()` 在每个 Batch 结束后执行，实现细粒度平滑过渡。

2. **极小方差初始化解码头**：数值预测头 `DecodeHead` (`nn.Linear(d_model, 1)`) 的权重用极小方差（`std=0.001`）初始化，偏置设为 `0.0`。布尔分类头和文本匹配头的权重用 `std=0.01` 初始化。

3. **消除傅里叶特征的高频混叠**：Base-2 frexp 解构后尾数 $M \in [-1, 1]$ 天然归一化，傅里叶频率初始化为 $10^{[-1, 2]}$ 的对数均匀分布。量级信息由指数嵌入表独立处理。

4. **Fork Bias 零初始化**：`ForkBiasEncoder` 的投影层权重和偏置初始化为零。

5. **嵌入层独立归一化 (Embedding LayerNorm)**：`GlobalTransformer.emb_norm`（`nn.LayerNorm(d_model)`）在传入 `TransformerEncoder` 前截断残差流初始的方差膨胀。Transformer 输出后还有 `out_norm = nn.LayerNorm(d_model)` 稳定解码头的输入分布。

6. **全局梯度范数裁剪 (Gradient Clipping)**：`clip_grad_norm_(max_norm=1.0)`，防止极端样本导致单步梯度爆炸。裁剪后的梯度范数记录至 TensorBoard (`grad_norm`)。

### 8.4 工程化配置

1. **混合精度训练 (BF16 AMP)**：使用 `torch.amp.autocast('cuda', dtype=torch.bfloat16)`。BFloat16 保留了与 FP32 相同的指数位，不存在下溢问题，因此 `GradScaler` 被禁用（`enabled=False`）。

2. **多维指标监控与诊断**：TensorBoard 记录训练动态（batch_loss, epoch_avg_loss, lr, grad_norm, GPU 显存/峰值）。每个 epoch 末展示诊断样例：按 group_ids 分行显示叶子节点，分类型展示预测值与真实值的对比（数值显示绝对/相对误差，布尔显示 logit，文本显示余弦相似度）。阶段完结后自动绘制跨阶段 Loss 曲线图像。

3. **完全可复现的断点续训 (Deterministic Resume)**：每个 checkpoint 完整保存：model state_dict、optimizer state_dict、scheduler state_dict、scaler state_dict、当前 epoch/global_step、best_loss、随机数生成器状态（Python/PyTorch/CUDA）。支持 `--resume <checkpoint.pth>` 精确恢复。支持 `--warm-restart` 只加载模型权重，重新初始化 optimizer/scheduler。

4. **评估集成**：每个 epoch 末调用 `evaluate()` 在 train/test 集上计算指标（Matbench 用 MAE，Tabular 用 RMSE），记录至 TensorBoard 并打印 gap 分析。

---

## 九、评估与推理

### 9.1 评估模块 (`eval.py`)

```python
def evaluate(model, test_loader, device, metric="mae"):
    """
    在 test set 上评估模型。
    支持指标：mae (Mean Absolute Error), rmse (Root Mean Squared Error)。
    集成零值分类头：logit > 0 → 直接输出 0。
    """
```

### 9.2 推理入口 (`inference.py`)

```python
def predict(model, frozen_lm, doc, device, root_name="doc"):
    """
    对一个 JSON 文档执行推理，返回所有 [MASK] 位置的预测结果。
    1. JSONParser 解析文档
    2. 构建 padding_mask 和 fork_bias
    3. 前向推理
    4. 提取掩码位置的预测值
    """
```

支持参数：
- `--checkpoint <path>`: 指定权重路径
- `--model-size`: 模型预设（small/medium/large/xl）
- 若未指定 checkpoint，自动查找 `checkpoints/` 下最新的 `.pth` 文件

内置 3 个 Demo：
1. 单材料 band_gap 预测
2. 数学关系预测 (3 + 7 = ?)
3. In-Context 推理 (推断隐含规则 y=2x)


