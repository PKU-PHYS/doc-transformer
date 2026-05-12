"""
合成数据生成器 — 支持课程学习的多样化训练数据。

核心流程：
  1. 根据 train_mode 选择生成策略
  2. 根据 target_tokens 控制文档复杂度
  3. 支持 compound (复合) / array (数组) / in_context (上下文推断) 等模式
  4. 用 JSONParser 解析为 LeafNode 列表
  5. 随机 Mask 部分叶子节点

外部接口：SyntheticDataset + collate_fn。
"""

import torch
import random
from typing import List, Dict, Any, Tuple, Optional
from torch.utils.data import Dataset
from model.json_parser import LeafNode, JSONParser

# 数据生成模块
from data.functions import FunctionRegistry
from data.templates.engine import TemplateEngine
from data.templates.distractors import inject_distractors
from data.text_tasks.generators import TextTaskGenerator
from data.in_context_generator import generate_in_context_task


# ═══════════════════════════════════════════════════
# 旧版生成器（保留为 fallback / 兼容）
# ═══════════════════════════════════════════════════

def generate_synthetic_document_old() -> Dict[str, Any]:
    """
    [旧版] 生成一条测试用的随机嵌套文档。
    包含材料的 formula，数值属性（band_gap, lattice），以及嵌套数组（sites）。
    """
    elements = ["Fe", "O", "Ti", "Si", "C", "H", "N", "S", "Cl", "Na"]

    doc = {
        "formula": "".join(random.sample(elements, 2)) + str(random.randint(1, 4)),
        "is_metal": random.choice([True, False]),
        "band_gap": random.uniform(0.0, 5.0),
        "lattice": {
            "a": random.uniform(2.0, 10.0),
            "b": random.uniform(2.0, 10.0),
            "c": random.uniform(2.0, 20.0),
            "alpha": random.uniform(60, 120),
            "beta": random.uniform(60, 120),
            "gamma": random.uniform(60, 120)
        },
        "sites": []
    }

    num_sites = random.randint(1, 5)
    for _ in range(num_sites):
        site = {
            "element": random.choice(elements),
            "x": random.uniform(0.0, 1.0),
            "y": random.uniform(0.0, 1.0),
            "z": random.uniform(0.0, 1.0),
            "occupancy": 1.0
        }
        doc["sites"].append(site)

    return doc


# ═══════════════════════════════════════════════════
# 新版数据生成 — 单关系
# ═══════════════════════════════════════════════════

def generate_math_document(nested_prob: float = 0.0) -> Dict[str, Any]:
    """
    从函数注册表采样一条数学关系，用模板引擎渲染为 JSON 文档。
    """
    rel = FunctionRegistry.sample()
    doc = TemplateEngine.render(rel)

    if isinstance(doc, dict):
        inject_distractors(doc, n_min=0, n_max=4, nested_prob=nested_prob)

    return doc


def generate_text_task_document(nested_prob: float = 0.0) -> Dict[str, Any]:
    """生成一条纯文本推理任务文档。"""
    doc = TextTaskGenerator.generate()

    if isinstance(doc, dict):
        inject_distractors(doc, n_min=0, n_max=3, nested_prob=nested_prob)

    return doc


# ═══════════════════════════════════════════════════
# 新版数据生成 — 复合文档 (多关系打包)
# ═══════════════════════════════════════════════════

_SECTION_KEYS = [
    "section", "block", "part", "group", "module", "component",
    "segment", "chunk", "unit", "entry", "record", "item",
    "experiment", "trial", "measurement", "observation",
]


def generate_compound_document(
    n_relations: int = 3,
    distractor_max: int = 8,
    nested_prob: float = 0.5,
) -> Dict[str, Any]:
    """
    生成包含多条独立数学关系的复合文档。
    每条关系渲染为一个嵌套子对象，打包进一个大 dict。

    典型输出 token 数: n_relations × 5-15 + distractors = 30-100+
    """
    doc = {}

    for i in range(n_relations):
        rel = FunctionRegistry.sample()
        section = TemplateEngine.render(rel)

        # 选择不重复的键名
        key_base = random.choice(_SECTION_KEYS)
        section_key = f"{key_base}_{i}"

        if isinstance(section, dict):
            doc[section_key] = section
        else:
            # 如果模板返回的是 list，包裹进对象
            doc[section_key] = {"data": section}

    # 顶层干扰
    inject_distractors(doc, n_min=2, n_max=distractor_max, nested_prob=nested_prob)

    return doc


def generate_array_document(
    n_items: int = 5,
    distractor_per_item: int = 2,
    nested_prob: float = 0.3,
) -> list:
    """
    生成一个包含多个同结构对象的大数组文档。
    所有对象基于同一数学关系但使用不同参数值。

    典型输出 token 数: n_items × 5-12 + distractors = 30-100+
    """
    gen = FunctionRegistry.sample_generator()
    items = []

    for _ in range(n_items):
        companion = gen()
        item = TemplateEngine.render(companion)
        if isinstance(item, dict):
            inject_distractors(item, n_min=0, n_max=distractor_per_item,
                               nested_prob=nested_prob)
        items.append(item)

    return items


def generate_mixed_long_document(target_tokens: int = 100) -> Dict[str, Any]:
    """
    生成一条混合长文档：包含多条数学关系 + 文本任务 + 大量干扰。
    通过循环追加内容来接近 target_tokens。

    典型输出 token 数: target_tokens ± 30%
    """
    doc = {}
    parser = JSONParser()
    section_idx = 0

    # 持续追加内容直到接近目标长度
    while True:
        current_leaves = parser.parse(doc, ["doc"], [])
        current_len = len(current_leaves)

        if current_len >= target_tokens * 0.8:
            break

        # 随机选择追加内容类型
        choice = random.random()
        if choice < 0.6:
            # 数学关系
            rel = FunctionRegistry.sample()
            section = TemplateEngine.render(rel)
        elif choice < 0.85:
            # 文本任务
            section = TextTaskGenerator.generate()
        else:
            # 小数组
            gen = FunctionRegistry.sample_generator()
            items = []
            for _ in range(random.randint(2, 4)):
                items.append(TemplateEngine.render(gen()))
            section = items

        key = f"{random.choice(_SECTION_KEYS)}_{section_idx}"
        doc[key] = section
        section_idx += 1

        # 重新创建 parser 以重置 group_counter
        parser = JSONParser()

    # 最后注入干扰
    inject_distractors(doc, n_min=3, n_max=10, nested_prob=0.7)

    return doc


# ═══════════════════════════════════════════════════
# Dataset & Collate
# ═══════════════════════════════════════════════════

class SyntheticDataset(Dataset):
    """
    合成数据集。

    Args:
        size: 数据集大小
        mask_ratio: 每个样本中被 mask 的叶子比例
        text_task_ratio: 文本任务占比 (0.0-1.0)
        max_tokens: 单样本最大叶子数，超过则截断。None 表示不限制。
        use_old_generator: 是否使用旧版生成器（用于兼容性测试）
        train_mode: 训练模式:
            - "explicit": Stage 1, 显式规则, 单/复合文档
            - "in_context": Stage 2, 隐式上下文推断
            - "mixed": Stage 3, 混合模式
        target_tokens: 目标序列长度。生成器会尽量生成接近此长度的文档。
                       None 表示不限制（使用默认生成策略）。
        distractor_level: 干扰强度 (0-3)。
            0=无嵌套干扰, 1=轻度, 2=中度, 3=重度
    """
    def __init__(self, size: int, mask_ratio: float = 0.15,
                 text_task_ratio: float = 0.2,
                 max_tokens: int = 512,
                 use_old_generator: bool = False,
                 train_mode: str = "explicit",
                 target_tokens: Optional[int] = None,
                 distractor_level: int = 1):
        self.size = size
        self.mask_ratio = mask_ratio
        self.text_task_ratio = text_task_ratio
        self.max_tokens = max_tokens
        self.use_old_generator = use_old_generator
        self.train_mode = train_mode
        self.target_tokens = target_tokens
        self.distractor_level = distractor_level

    def __len__(self):
        return self.size

    def __getitem__(self, idx) -> Tuple[List[LeafNode], Dict[int, Any]]:
        # 模式判定
        current_mode = self.train_mode
        if current_mode == "mixed":
            # Stage 3: 按权重随机选择
            r = random.random()
            if r < 0.4:
                current_mode = "explicit"
            elif r < 0.8:
                current_mode = "in_context"
            else:
                current_mode = "explicit_long"

        # 设定函数的难度层级
        if current_mode == "simple":
            FunctionRegistry.set_filter(exact_tier=0)
        else:
            FunctionRegistry.set_filter(max_tier=1)

        # ── Simple 模式 (Stage 0) ──
        if current_mode == "simple":
            rel = FunctionRegistry.sample()
            doc = TemplateEngine.render_simple(rel)
            parser = JSONParser()
            leaves = parser.parse(doc, ["doc"], [])
            
        # ── In-Context 模式 ──
        elif current_mode == "in_context":
            return generate_in_context_task(
                target_tokens=self.target_tokens,
                max_tokens=self.max_tokens,
                distractor_level=self.distractor_level,
            )

        # ── Explicit 模式 ──
        elif self.use_old_generator:
            doc1 = generate_synthetic_document_old()
            doc2 = generate_synthetic_document_old()
            parser = JSONParser()
            leaves = parser.parse([doc1, doc2], ["materials"], [])

        elif current_mode == "explicit_long" or (
            self.target_tokens is not None and self.target_tokens > 80
        ):
            # 长文档模式：使用混合长文档生成器
            target = self.target_tokens or 150
            doc = generate_mixed_long_document(target_tokens=target)
            parser = JSONParser()
            leaves = parser.parse(doc, ["doc"], [])

        else:
            # 标准 explicit 模式 — 按概率选择生成策略
            nested_prob = [0.0, 0.2, 0.5, 0.8][min(self.distractor_level, 3)]
            r = random.random()

            if r < self.text_task_ratio:
                # 文本任务
                doc = generate_text_task_document(nested_prob=nested_prob)
            elif r < self.text_task_ratio + 0.3:
                # 复合文档 (多关系)
                n_rel = random.randint(2, 5)
                doc = generate_compound_document(
                    n_relations=n_rel,
                    distractor_max=6,
                    nested_prob=nested_prob,
                )
            elif r < self.text_task_ratio + 0.5:
                # 大数组文档
                n_items = random.randint(3, 8)
                doc = generate_array_document(
                    n_items=n_items,
                    distractor_per_item=2,
                    nested_prob=nested_prob,
                )
            else:
                # 单关系文档 (经典)
                doc = generate_math_document(nested_prob=nested_prob)

            parser = JSONParser()
            if isinstance(doc, list):
                leaves = parser.parse(doc, ["records"], [])
            else:
                leaves = parser.parse(doc, ["doc"], [])

        # 确保至少有 1 个叶子节点
        if not leaves:
            parser = JSONParser()
            leaves = parser.parse({"value": random.uniform(-1, 1)}, ["doc"], [])

        # ── 截断：超过 max_tokens 的叶子直接丢弃 ──
        if self.max_tokens is not None and len(leaves) > self.max_tokens:
            leaves = leaves[:self.max_tokens]

        target_masks = {}
        # 优先 mask 输出变量（最后一个数值节点），保证被 mask 的值总是可从其余变量唯一确定。
        # 在所有生成器中，输出变量总是 variables dict 的最后一个 key，
        # flat_basic 模板保持此顺序，因此 parse 后最后一个数值叶子 = 输出变量。
        # 避免 mask 不可逆函数（max/min/abs/sign）的输入，消除不可确定的训练噪声。
        num_masks = max(1, int(len(leaves) * self.mask_ratio))
        num_indices = [i for i, l in enumerate(leaves) if l.value_type == "number"]
        other_indices = [i for i, l in enumerate(leaves) if l.value_type != "number"]
        
        if num_indices:
            # 输出变量 = 最后一个数值叶子，优先 mask
            output_idx = num_indices[-1]
            mask_indices = [output_idx]
            
            # 如果还需要更多 mask，从其余数值节点中选
            if num_masks > 1:
                remaining_num = [i for i in num_indices if i != output_idx]
                extra_needed = num_masks - 1
                if remaining_num and extra_needed > 0:
                    mask_indices += random.sample(remaining_num, min(extra_needed, len(remaining_num)))
                    extra_needed = num_masks - len(mask_indices)
                # 仍不够，从非数值节点补
                if other_indices and extra_needed > 0:
                    mask_indices += random.sample(other_indices, min(extra_needed, len(other_indices)))
        else:
            # 没有数值节点，从其他类型选
            mask_indices = random.sample(other_indices, min(num_masks, len(other_indices))) if other_indices else []

        for i in mask_indices:
            orig_node = leaves[i]
            target_masks[i] = (orig_node.value, orig_node.value_type)
            leaves[i] = LeafNode(
                value="[MASK]",
                value_type="mask",
                path=orig_node.path,
                group_ids=orig_node.group_ids
            )

        return leaves, target_masks


def collate_fn(batch: List[Tuple[List[LeafNode], Dict[int, Any]]],
               max_tokens: int = 512):
    """
    处理变长 Batch，生成 padding mask。

    Args:
        batch: DataLoader 传入的 (leaves, target_masks) 列表
        max_tokens: 安全上限，collate 时再次截断（双重保护）
    """
    batched_leaves = []
    batched_masks = []

    # 双重保护：collate 层再做一次截断
    max_len = min(max(len(leaves) for leaves, _ in batch), max_tokens)

    # src_key_padding_mask: True 表示 <PAD> 需要被阻断
    padding_mask = torch.ones((len(batch), max_len), dtype=torch.bool)

    for b, (leaves, masks) in enumerate(batch):
        # 截断叶子
        truncated_leaves = leaves[:max_len]
        # 丢弃超出范围的 mask target
        truncated_masks = {k: v for k, v in masks.items() if k < max_len}

        batched_leaves.append(truncated_leaves)
        batched_masks.append(truncated_masks)
        # 将有效部分置为 False
        padding_mask[b, :len(truncated_leaves)] = False

    return batched_leaves, batched_masks, padding_mask
