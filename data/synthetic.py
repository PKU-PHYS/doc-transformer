"""
合成数据生成器 — 基于数学函数的多样化训练数据。

核心流程：
  1. 从 FunctionRegistry 采样一条 MathRelation（或从 TextTaskGenerator 生成文本任务）
  2. 用 TemplateEngine 渲染为多样化的 JSON 文档
  3. 注入随机干扰字段
  4. 用 JSONParser 解析为 LeafNode 列表
  5. 随机 Mask 部分叶子节点

外部接口保持不变：SyntheticDataset + collate_fn。
"""

import torch
import random
from typing import List, Dict, Any, Tuple
from torch.utils.data import Dataset
from model.json_parser import LeafNode, JSONParser

# 新数据生成模块
from data.functions import FunctionRegistry
from data.templates.engine import TemplateEngine
from data.templates.distractors import inject_distractors
from data.text_tasks.generators import TextTaskGenerator


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
# 新版数据生成
# ═══════════════════════════════════════════════════

def generate_math_document() -> Dict[str, Any]:
    """
    从函数注册表采样一条数学关系，用模板引擎渲染为 JSON 文档，
    并注入随机干扰字段。
    """
    rel = FunctionRegistry.sample()
    doc = TemplateEngine.render(rel)

    # 对 dict 类型文档注入干扰字段
    if isinstance(doc, dict):
        inject_distractors(doc, n_min=0, n_max=4)

    return doc


def generate_text_task_document() -> Dict[str, Any]:
    """
    生成一条纯文本推理任务文档。
    """
    doc = TextTaskGenerator.generate()

    # 文本任务也可以注入干扰字段
    if isinstance(doc, dict):
        inject_distractors(doc, n_min=0, n_max=3)

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
    """
    def __init__(self, size: int, mask_ratio: float = 0.15,
                 text_task_ratio: float = 0.2,
                 max_tokens: int = 512,
                 use_old_generator: bool = False):
        self.size = size
        self.mask_ratio = mask_ratio
        self.text_task_ratio = text_task_ratio
        self.max_tokens = max_tokens
        self.use_old_generator = use_old_generator

    def __len__(self):
        return self.size

    def __getitem__(self, idx) -> Tuple[List[LeafNode], Dict[int, Any]]:
        if self.use_old_generator:
            # 旧版逻辑：生成两个材料文档
            doc1 = generate_synthetic_document_old()
            doc2 = generate_synthetic_document_old()
            parser = JSONParser()
            leaves = parser.parse([doc1, doc2], ["materials"], [])
        else:
            # 新版逻辑：按概率生成数学文档或文本任务
            if random.random() < self.text_task_ratio:
                doc = generate_text_task_document()
            else:
                doc = generate_math_document()

            parser = JSONParser()
            # 根据文档类型决定根路径
            if isinstance(doc, list):
                leaves = parser.parse(doc, ["records"], [])
            else:
                leaves = parser.parse(doc, ["doc"], [])

        # 确保至少有 1 个叶子节点
        if not leaves:
            # fallback: 生成最简单的文档
            leaves = parser.parse({"value": random.uniform(-1, 1)}, ["doc"], [])

        # ── 截断：超过 max_tokens 的叶子直接丢弃 ──
        if self.max_tokens is not None and len(leaves) > self.max_tokens:
            leaves = leaves[:self.max_tokens]

        target_masks = {}
        # 随机 Mask 一部分叶子节点（在截断后的范围内）
        num_masks = max(1, int(len(leaves) * self.mask_ratio))
        mask_indices = random.sample(range(len(leaves)), min(num_masks, len(leaves)))

        for i in mask_indices:
            orig_node = leaves[i]
            # 记录真实值和类型
            target_masks[i] = (orig_node.value, orig_node.value_type)
            # 替换为 [MASK]
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
