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
│   ├── synthetic.py               # 顶层合成数据集与 Collate 函数
│   ├── functions/                 # 数学函数注册表 (unary, binary, multivar...)
│   ├── templates/                 # 结构模板引擎 (100+种JSON模板) 与 干扰字段生成器
│   └── text_tasks/                # 纯文本推理任务生成器
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
    stage1_max_epochs: int = 50     
    stage1_patience: int = 8        
    stage2_max_epochs: int = 80     
    stage2_patience: int = 10       
    stage3_max_epochs: int = 40     
    stage3_patience: int = 8        
    
    dataset_size: int = 10000   # 每个 epoch 的样本数 (464M 模型需要足够数据)
    
    stage1_target_tokens: int = 100   
    stage2_target_tokens: int = 300   
    stage3_target_tokens: int = 200   
    
    stage1_distractor_level: int = 1  
    stage2_distractor_level: int = 2  
    stage3_distractor_level: int = 3  
    
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
    path: List[str]         # 根到叶子的字段名列表, e.g., ["materials", "lattice", "a"]
    group_ids: List[int]    # 祖先数组元素对应的全局唯一组 ID, e.g., [14, 52]

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

### 6.2 合成数据生成流 (`data/synthetic.py`)

我们的合成数据不再只是简单的材料字典，而是包含了多类数学函数、推理任务和大量随机模板生成的丰富数据：

**生成流程核心组件**：
1. **FunctionRegistry / TextTaskGenerator**：随机采样数学关系或文本推理任务。其中，纯文本推理任务 (`TextTaskGenerator`) 并非简单的占位符，而是被精心设计为**三大类跨模态桥接任务**：
   - **数值 → 文本分类** (如根据数值大小判断 magnitude、正负号、象限)
   - **文本 → 文本推理** (如输出反函数名称、计算导数表达式配对)
   - **文本 → 数值检索** (如基于自然语言“archimedes_constant”输出 $3.141593$)
   这三类任务直接强制模型将 FrozenLM 的语义向量空间与傅里叶特征的高维数值空间进行深度对齐。
2. **TemplateEngine**：负责将抽象的关系渲染为千变万化的 JSON 树。它不仅囊括了 5 大类（扁平、嵌套、数组、复合链式等）逾百种模板结构，更在每次生成后引入**动态扰动后处理 (Post-processing Perturbations)**，如全随机打乱字典键的顺序 (`_shuffle_dict`) 或概率性展平单层嵌套，极大化结构多样性，彻底破坏模型对固定 JSON 格式产生“位置捷径过拟合”的可能。
3. **Distractor Injector**：随机向生成的 JSON 中插入完全无关的干扰分支（如 `timestamp`, `confidence` 等），迫使注意力机制学会在海量噪声中精准锁定有逻辑关联的有效节点。

**数据流伪代码**：
```python
def __getitem__(self, idx):
    # 1. 采样与渲染
    if random.random() < self.text_task_ratio:
        doc = TextTaskGenerator.generate()
        inject_distractors(doc)
    else:
        rel = FunctionRegistry.sample()
        doc = TemplateEngine.render(rel)
        inject_distractors(doc)
        
    # 2. 解析成叶子节点序列
    parser = JSONParser()
    leaves = parser.parse(doc, ["doc"], [])
    
    # 3. 超长截断与随机 Mask
    leaves = leaves[:self.max_tokens]
    mask_indices = random.sample(range(len(leaves)), num_masks)
    for i in mask_indices:
        leaves[i].value = "[MASK]"
        leaves[i].value_type = "mask"
        
    return leaves, target_masks
```

---

## 七、网络模块伪代码

所有的嵌入模块最终都要将输入映射到 `d_model` 维空间。

### 7.1 值编码器 (`model/value_encoder.py`)

**核心：数值型傅里叶编码 (FourierFeatureEncoder)**
```python
class FourierFeatureEncoder(nn.Module):
    def __init__(self, n_feats, d_model, learnable=True):
        super().__init__()
        if learnable:
            # 使用对数均匀分布初始化频率，覆盖从宏观趋势 (1e-4) 到微观细节 (1e1)
            freqs = 10.0 ** torch.empty(n_feats).uniform_(-4, 1)
            self.freqs = nn.Parameter(freqs) 
        else:
            freqs = 10.0 ** torch.linspace(-4, 1, n_feats)
            self.register_buffer('freqs', freqs)
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
        # 必须放大到 sqrt(d_model) 量级以保证方差不坍缩
        random_vecs[uid] = (vec / vec.norm(p=2)) * scale * (d_model ** 0.5)
        
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

### 7.4 Token 嵌入引擎的全局去重与缓存优化 (Global Deduplication & Caching)
为打破 `FrozenLM` 处理大规模深层 JSON 时带来的推理瓶颈（即同一 Batch 的不同节点内大量重复出现如 `"materials"`, `"formula"` 等短语），底层网络引入了两项超前性能优化：
1. **Batch 级全局词表去重 (Global Token Lookup)**：在每次前向传播的起始阶段 (`TokenEmbedding`)，引擎会主动提取当前 Batch 内所有叶子的字符串值和路径节点，放入 `set` 统一去重。提取出的独一无二的词汇表会被“一次性”送入语言模型，并建立 `text_lookup` 哈希表供后续路径编码和值编码查询，将庞大 Batch 下的大量冗余文本推理开销瞬间清零。
2. **跨步内存级缓存与防爆机制 (Eviction Policy)**：`FrozenLM` 内部封装了带状态的字典缓存 (`self._cache`)，使跨 Batch 间频繁出现的高频键名永远只需编码一次。同时加入了**自动驱逐清洗机制** (`if len(self._cache) > 10000: self._cache.clear()`)，完美杜绝了持续数万个 Epoch 的海量生僻随机词采样可能引发的 GPU 显存泄漏 (OOM) 崩溃问题。

---

## 八、训练策略与课程学习 (Curriculum Learning)

针对 Zero-shot / In-Context Learning (上下文学习) 规则推断的目标，必须采用**连续预训练（Continued Pre-training）**的课程学习策略，而不是简单的微调（SFT）或一开始就完全混合。这可以有效防止模型产生“捷径依赖（Shortcut Learning）”。

### Stage 1：结构与显式规则基础训练（当前阶段）
- **数据形态**：单条 JSON 或少量 JSON，内部包含显式的物理规律名称（如 `function: "band_gap_formula"`）或参数。
- **训练目的**：让模型学会理解树状结构、路径编码（Path Encoding）、组嵌入（Group Embedding），以及基础的傅里叶数值计算和跨节点复制寻址。
- **配置开关**：`train_mode = "explicit"`

### Stage 2：连续预训练（强制上下文规则归纳 In-Context Rule Induction）
- **切换时机**：Stage 1 的 Loss 趋于收敛且验证任务通过。
- **数据形态**：包含多个对象的 JSON 数组（Few-shot 演示），**移除显式的 `function` 字段**。
- **训练目的**：迫使模型激活多头注意力机制，跨越 JSON 组别观察前序数据的输入输出对，推断出隐含的数学规律，并应用到预测目标（`[MASK]`）上。

**伪代码：Stage 2 隐式上下文数据生成 (`data/in_context_generator.py`)**
```python
def generate_in_context_task(num_shots=3):
    # 1. 随机采样一个未知的数学关系（或物理公式）
    rel = FunctionRegistry.sample()
    
    # 2. 生成多条演示数据 (shots) + 1条目标数据
    context_jsons = []
    for _ in range(num_shots):
        # 渲染不带 function 名称的隐式推断 JSON
        doc = TemplateEngine.render_implicit(rel) 
        context_jsons.append(doc)
        
    target_doc = TemplateEngine.render_implicit(rel)
    
    # 3. 将它们放入一个数组中作为 In-Context Prompt
    final_batch = {
        "task_id": generate_uuid(),
        "demonstrations": context_jsons,
        "query": target_doc
    }
    
    # 解析并 Mask Target
    leaves = JSONParser().parse(final_batch, ["root"], [])
    # 找到 query 里的目标值并 MASK (伪代码)
    mask_target_node(leaves)
    return leaves
```

### Stage 3：混合鲁棒性训练（可选）
- **数据形态**：混合 Stage 1（显式指令）和 Stage 2（隐式上下文推断），并大量注入无意义的干扰字段（Distractors）。
- **训练目的**：防止灾难性遗忘（Catastrophic Forgetting），增强模型在真实噪音环境下的鲁棒性。

### 8.4 大规模参数训练稳定性保障 (Scaling up to XXXL)
当模型规模扩大至大参数级别（例如 `d_model=1536, L=16`，约 4.6 亿参数）且使用 `norm_first=True` 时，底层网络的输出方差会极大，如果目标函数又是大数值范围（如指数运算），极易引发 Huber Loss 瞬间爆炸（> 500）与动量崩溃。

为保障大模型训练稳定性，实现中必须包含以下防线：
1. **Step 级学习率预热 (Warmup)**：绝对禁止大模型直接以 `1e-4` 等高学习率冷启动。必须分配一定比例（如 5%）的训练步数用于线性预热（`get_cosine_schedule_with_warmup`），并将 `scheduler.step()` 移至每个 Batch 结束后执行，实现细粒度平滑过渡。
2. **极小方差初始化解码头**：数值预测头 `DecodeHead` (`nn.Linear(d_model, 1)`) 的权重必须用极小的方差（如 `std=0.001`）初始化，偏置设为 `0.0`。这迫使大模型在训练初期的盲目猜测阶段输出接近 `0` 的保守数值，避免巨大误差带来的毁灭性梯度惩罚，为主干网络争取建立注意力几何的时间。此外，为全面保障各类目标的预测稳定性，布尔分类头 (`bool_head`) 和文本匹配头 (`text_head`) 的权重也应当应用较小的方差（如 `std=0.01`）进行初始化约束。
3. **消除傅里叶特征的高频混叠**：如果 `freqs` 采用默认的正态分布初始化，模型只能捕获频率 1.0 附近的特征。当遇到物理公式中产生的巨大目标数值（如 `10000`）时，正弦波会发生严重的混叠（aliasing）与高频噪音。必须将 `freqs` 初始化为跨越多个量级的对数均匀分布（如 $10^{-4}$ 到 $10^1$），以确保模型能感知大数值的宏观差异。
4. **防止组嵌入（Group Embedding）方差坍缩**：为求内积稳定，强行将高维向量的 L2 范数归一化为 1.0 会导致其内部元素的方差坍缩至 $1/d_{model}$（接近 0）。能量过弱会使 Transformer 完全忽视结构组嵌入，导致数组内原本不同的元素无法被有效区分。必须在归一化后乘以 $\sqrt{d_{model}}$，将方差恢复至正常的 `1.0` 尺度。
5. **嵌入层独立归一化 (Embedding LayerNorm)**：当主干 Transformer 使用 `norm_first=True` (Pre-LN) 时，第一层的 LayerNorm 是作用在注意力的分支上，而残差主干（Residual Stream）在初始阶段未受任何归一化约束。由于最终的 Token 嵌入是值、路径、组嵌入之和，存在极大的初始方差累积。必须在传入 `TransformerEncoder` 前额外应用一个全局的 `nn.LayerNorm(d_model)`，彻底截断残差流初始的巨大方差膨胀，避免损失爆炸。

### 8.5 自动化课程学习调度与工程化加速
为保障三阶段课程学习连续平稳进行并极大化利用硬件算力，训练入口 (`train.py`) 需要部署以下工程组件：
1. **基于 Patience 的自动阶段流转**：摒弃单一的 `epochs` 设置。为每个阶段设置独立的目标词元长度 (`target_tokens`) 和干扰强度 (`distractor_level`)。利用 `patience` 机制监控阶段收敛，当验证 Loss 连续 `patience` 轮不再改善时，自动保存模型并无缝切换至下一阶段（例如从 Stage 1 进入 Stage 2），极大减少人工干预。
2. **混合精度训练 (BF16 AMP)**：当模型攀升至数亿参数，应当结合 `torch.amp.autocast('cuda', dtype=torch.bfloat16)` 与 `GradScaler`。BFloat16 保留了与 FP32 相同的指数位，既有效杜绝了大规模数值计算时的下溢出，又能大幅降低显存开销并带来近 2 倍的吞吐量提升。
3. **多维指标监控与诊断**：引入 TensorBoard 实时记录各阶段的训练动态（Loss, 学习率, GPU显存/峰值利用率）。不仅要求记录 Loss，还必须在每个阶段内定期提取真实样本（如每 200 个 batch）展示其输入、预测值与真实值的文本对照 (`_log_sample_case`)，并在全部阶段完结后自动绘制跨阶段的 Loss 曲线图像供后续归纳总结。

---

## 九、验证计划 (Sanity Check)

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
在合成数据生成器 (带有些微物理相关性噪音) 的无限流下训练：先经过 Stage 1 学习基础架构，之后通过 Stage 2 注入 In-Context 归纳能力，保存 Checkpoint，运行 `inference.py`。
