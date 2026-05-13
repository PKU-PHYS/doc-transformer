"""
合成数据生成器 — 支持课程学习的多样化训练数据。

数据生成管线：
  1. 根据 train_mode 选择文档生成策略
  2. 用 JSONParser 将 JSON 展平为 LeafNode 列表
  3. 使用 TemplateEngine 返回的 safe_mask_keys 精准 mask 输出变量
  4. collate_fn 构建 padding mask 和 fork bias 矩阵

外部接口：SyntheticDataset, collate_fn, compute_fork_bias_indices。
"""

import torch
import random
from typing import List, Dict, Any, Tuple, Optional
from torch.utils.data import Dataset
from model.json_parser import LeafNode, JSONParser

from data.functions import FunctionRegistry
from data.functions.registry import (
    UNARY_OUTPUT_KEYS, BINARY_OUTPUT_KEYS, FUNC_NAME_KEYS, CATEGORY_KEYS,
)
from data.templates.engine import TemplateEngine
from data.templates.distractors import inject_distractors
from data.text_tasks.generators import TextTaskGenerator
from data.in_context_generator import generate_in_context_task


# ═══════════════════════════════════════════════════════════════
# §1  常量
# ═══════════════════════════════════════════════════════════════

# Fallback 输出键名集合（仅当 safe_mask_keys=None 时使用，如纯文本任务）
_FALLBACK_OUTPUT_KEYS: set = set(UNARY_OUTPUT_KEYS) | set(BINARY_OUTPUT_KEYS) | {
    # multivar 专用输出键名
    "f_x", "D", "discriminant", "delta", "det", "disc",
    "z", "z_score", "standard_score", "z_val",
    "scaled", "normalized", "n_val",
    "avg", "weighted_avg", "mean", "w_mean",
    "dot", "dot_product", "inner", "scalar_result", "d_val",
    "cross", "cross_z", "signed_area_val", "wedge",
    "mag", "magnitude", "length", "norm", "abs_val",
    "dist", "distance", "d", "separation",
    "area", "A", "surface", "trap_area_val",
    "V", "volume", "capacity", "vol_val",
    "interpolated", "mixed", "blended",
    "clamped", "clipped", "bounded",
    "E", "energy", "KE", "kinetic", "e_val",
    "F", "force", "gravitational_force_val", "attraction",
    "c", "side_c", "opposite_side", "third_side",
}

# 不应被 mask 的装饰性键名（函数名、类别、干扰字段的 key）
_DECORATION_KEYS: set = set(FUNC_NAME_KEYS) | set(CATEGORY_KEYS) | {
    "id", "timestamp", "author", "confidence", "tags", "label",
    "description", "desc", "info", "summary", "note",
    "units", "precision", "status", "format", "encoding",
    "index", "batch_id", "seed", "verified", "is_exact",
    "deprecated", "converged", "cached", "is_valid",
    "equation", "record_type", "expression",
}

# 复合文档子对象的键名池
_SECTION_KEYS = [
    "section", "block", "part", "group", "module", "component",
    "segment", "chunk", "unit", "entry", "record", "item",
    "experiment", "trial", "measurement", "observation",
]


# ═══════════════════════════════════════════════════════════════
# §2  文档生成器
# ═══════════════════════════════════════════════════════════════

def generate_math_document(nested_prob: float = 0.0):
    """
    从函数注册表采样一条数学关系，用模板引擎渲染为 JSON 文档。

    Returns: (doc, safe_mask_keys)
        safe_mask_keys: 可安全 mask 的叶子键名集合
    """
    rel = FunctionRegistry.sample()
    doc, safe_keys, _ = TemplateEngine.render(rel)
    if isinstance(doc, dict):
        inject_distractors(doc, n_min=0, n_max=4, nested_prob=nested_prob)
    return doc, safe_keys


def generate_text_task_document(nested_prob: float = 0.0):
    """
    生成一条纯文本推理任务文档。

    Returns: (doc, safe_mask_keys=None)
        文本任务无可逆性约束，safe_mask_keys 返回 None。
    """
    doc = TextTaskGenerator.generate()
    if isinstance(doc, dict):
        inject_distractors(doc, n_min=0, n_max=3, nested_prob=nested_prob)
    return doc, None


def generate_compound_document(n_relations: int = 3, distractor_max: int = 8,
                               nested_prob: float = 0.5):
    """
    生成包含多条独立数学关系的复合文档。

    Returns: (doc, safe_mask_keys)
    """
    doc = {}
    all_safe_keys: set = set()
    for i in range(n_relations):
        rel = FunctionRegistry.sample()
        section, safe_keys, _ = TemplateEngine.render(rel)
        all_safe_keys.update(safe_keys)
        key = f"{random.choice(_SECTION_KEYS)}_{i}"
        doc[key] = section if isinstance(section, dict) else {"data": section}
    inject_distractors(doc, n_min=2, n_max=distractor_max, nested_prob=nested_prob)
    return doc, all_safe_keys


def generate_array_document(n_items: int = 5, distractor_per_item: int = 2,
                            nested_prob: float = 0.3):
    """
    生成包含多个同结构对象的数组文档。

    Returns: (doc_list, safe_mask_keys)
    """
    gen = FunctionRegistry.sample_generator()
    items = []
    all_safe_keys: set = set()
    for _ in range(n_items):
        item, safe_keys, _ = TemplateEngine.render(gen())
        all_safe_keys.update(safe_keys)
        if isinstance(item, dict):
            inject_distractors(item, n_min=0, n_max=distractor_per_item,
                               nested_prob=nested_prob)
        items.append(item)
    return items, all_safe_keys


def generate_mixed_long_document(target_tokens: int = 100):
    """
    循环追加多种内容块直到接近 target_tokens，生成混合长文档。

    Returns: (doc, safe_mask_keys)
    """
    doc = {}
    parser = JSONParser()
    section_idx = 0
    all_safe_keys: set = set()

    while True:
        current_leaves = parser.parse(
            doc, ["doc"], [0], [JSONParser._key_hash("doc")], []
        )
        if len(current_leaves) >= target_tokens * 0.8:
            break

        choice = random.random()
        if choice < 0.6:
            rel = FunctionRegistry.sample()
            section, safe_keys, _ = TemplateEngine.render(rel)
            all_safe_keys.update(safe_keys)
        elif choice < 0.85:
            section = TextTaskGenerator.generate()
        else:
            gen = FunctionRegistry.sample_generator()
            items = []
            for _ in range(random.randint(2, 4)):
                item, safe_keys, _ = TemplateEngine.render(gen())
                all_safe_keys.update(safe_keys)
                items.append(item)
            section = items

        doc[f"{random.choice(_SECTION_KEYS)}_{section_idx}"] = section
        section_idx += 1
        parser = JSONParser()

    inject_distractors(doc, n_min=3, n_max=10, nested_prob=0.7)
    return doc, all_safe_keys


# ═══════════════════════════════════════════════════════════════
# §3  SyntheticDataset
# ═══════════════════════════════════════════════════════════════

class SyntheticDataset(Dataset):
    """
    合成数据集 — 按 train_mode 生成不同复杂度的训练样本。

    Args:
        size:             数据集大小（每 epoch 样本数）
        mask_ratio:       每个样本中被 mask 的叶子比例
        text_task_ratio:  文本推理任务占比 (0.0–1.0)
        max_tokens:       单样本最大叶子数，超过则截断
        train_mode:       训练模式
            - "simple":        Stage 0, 极简扁平 KV, Tier 0 函数冷启动
            - "explicit":      Stage 1, 显式规则, 单/复合/数组/文本文档
            - "explicit_long": Stage 2, 混合长文档 + 重度干扰
            - "in_context":    Stage 3, 隐式上下文推断
            - "mixed":         混合模式 (40% explicit + 40% in_context + 20% explicit_long)
        target_tokens:    目标序列长度（生成器会尽量接近此值）
        distractor_level: 干扰强度 (0=无, 1=轻度, 2=中度, 3=重度)
    """

    def __init__(self, size: int, mask_ratio: float = 0.15,
                 text_task_ratio: float = 0.2,
                 max_tokens: int = 512,
                 train_mode: str = "explicit",
                 target_tokens: Optional[int] = None,
                 distractor_level: int = 1):
        self.size = size
        self.mask_ratio = mask_ratio
        self.text_task_ratio = text_task_ratio
        self.max_tokens = max_tokens
        self.train_mode = train_mode
        self.target_tokens = target_tokens
        self.distractor_level = distractor_level

    def __len__(self):
        return self.size

    def __getitem__(self, idx) -> Tuple[List[LeafNode], Dict[int, Any]]:
        # ── Step 1: 确定当前模式 & 函数难度 ──
        mode = self._resolve_mode()
        self._apply_tier_filter(mode)

        # ── Step 2: In-Context 有独立的生成+mask 流程，直接返回 ──
        if mode == "in_context":
            return generate_in_context_task(
                target_tokens=self.target_tokens,
                max_tokens=self.max_tokens,
                distractor_level=self.distractor_level,
            )

        # ── Step 3: 生成文档 → 解析为叶子列表 → 截断 ──
        leaves, safe_keys = self._generate_and_parse(mode)

        if self.max_tokens is not None and len(leaves) > self.max_tokens:
            leaves = leaves[:self.max_tokens]

        if not leaves:
            parser = JSONParser()
            leaves = parser.parse(
                {"value": random.uniform(-1, 1)},
                ["doc"], [0], [JSONParser._key_hash("doc")], [],
            )
            safe_keys = None

        # ── Step 4: 应用 mask ──
        return self._apply_masks(leaves, safe_keys)

    # ──────────────── 内部方法 ────────────────

    def _resolve_mode(self) -> str:
        """解析 mixed 模式为具体模式，其余直接返回。"""
        if self.train_mode != "mixed":
            return self.train_mode
        r = random.random()
        if r < 0.4:
            return "explicit"
        elif r < 0.8:
            return "in_context"
        else:
            return "explicit_long"

    @staticmethod
    def _apply_tier_filter(mode: str):
        """根据模式设定函数难度过滤器。"""
        if mode == "simple":
            FunctionRegistry.set_filter(exact_tier=0)
        else:
            FunctionRegistry.set_filter(max_tier=1)

    def _generate_and_parse(self, mode: str) -> Tuple[List[LeafNode], Optional[set]]:
        """
        按模式生成 JSON 文档，解析为 LeafNode 列表。

        Returns: (leaves, safe_mask_keys)
        """
        # ── Simple (Stage 0): 极简扁平 KV ──
        if mode == "simple":
            rel = FunctionRegistry.sample()
            doc, safe_keys, _ = TemplateEngine.render_simple(rel)

        # ── Explicit Long (Stage 2) 或 target_tokens 较大时 ──
        elif mode == "explicit_long" or (
            self.target_tokens is not None and self.target_tokens > 80
        ):
            doc, safe_keys = generate_mixed_long_document(
                target_tokens=self.target_tokens or 150
            )

        # ── Explicit (Stage 1): 按概率分配生成策略 ──
        else:
            doc, safe_keys = self._generate_explicit()

        # 解析为叶子节点
        parser = JSONParser()
        root = "records" if isinstance(doc, list) else "doc"
        leaves = parser.parse(
            doc, [root], [0], [JSONParser._key_hash(root)], []
        )
        return leaves, safe_keys

    def _generate_explicit(self) -> Tuple[Any, Optional[set]]:
        """Stage 1 的生成策略：按概率在文本任务/复合文档/数组文档/单关系间分配。"""
        nested_prob = [0.0, 0.2, 0.5, 0.8][min(self.distractor_level, 3)]
        r = random.random()

        if r < self.text_task_ratio:
            return generate_text_task_document(nested_prob=nested_prob)

        elif r < self.text_task_ratio + 0.3:
            return generate_compound_document(
                n_relations=random.randint(2, 5),
                distractor_max=6,
                nested_prob=nested_prob,
            )

        elif r < self.text_task_ratio + 0.5:
            return generate_array_document(
                n_items=random.randint(3, 8),
                distractor_per_item=2,
                nested_prob=nested_prob,
            )

        else:
            return generate_math_document(nested_prob=nested_prob)

    def _apply_masks(self, leaves: List[LeafNode],
                     safe_keys: Optional[set]) -> Tuple[List[LeafNode], Dict[int, Any]]:
        """
        对叶子列表应用掩码，返回 (masked_leaves, target_masks)。

        Mask 选择优先级：
          1. safe_keys (TemplateEngine 返回的当前关系精确安全键名集合)
          2. _FALLBACK_OUTPUT_KEYS (全局输出键名)
          3. 非装饰性数值节点
          4. 最后一个数值节点
        """
        num_masks = max(1, int(len(leaves) * self.mask_ratio))
        num_indices = [i for i, l in enumerate(leaves) if l.value_type == "number"]
        other_indices = [i for i, l in enumerate(leaves) if l.value_type != "number"]

        if num_indices:
            mask_indices = self._select_num_targets(
                leaves, num_indices, num_masks, safe_keys
            )
            # 仍需更多 mask 时，从非数值节点补充
            extra = num_masks - len(mask_indices)
            if extra > 0 and other_indices:
                mask_indices += random.sample(
                    other_indices, min(extra, len(other_indices))
                )
        else:
            mask_indices = (
                random.sample(other_indices, min(num_masks, len(other_indices)))
                if other_indices else []
            )

        # 执行 mask 替换
        target_masks = {}
        for i in mask_indices:
            orig = leaves[i]
            target_masks[i] = (orig.value, orig.value_type)
            leaves[i] = LeafNode(
                value="[MASK]", value_type="mask",
                path=orig.path, path_types=orig.path_types,
                path_ids=orig.path_ids, group_ids=orig.group_ids,
            )
        return leaves, target_masks

    @staticmethod
    def _select_num_targets(leaves: List[LeafNode], num_indices: List[int],
                            num_masks: int, safe_keys: Optional[set]) -> List[int]:
        """
        从数值节点中选出最安全的 mask 目标。

        逻辑：
          1. 若 safe_keys 可用，优先 mask path[-1] ∈ safe_keys 的节点
          2. Fallback 到全局 _FALLBACK_OUTPUT_KEYS
          3. 再 fallback 到非装饰性数值节点
          4. 最后兜底取最后一个数值节点
        """
        def _match(key_set):
            return [
                i for i in num_indices
                if leaves[i].path and leaves[i].path[-1] in key_set
            ]

        # 优先使用 TemplateEngine 返回的精确安全键名
        candidates = _match(safe_keys) if safe_keys else []

        # Fallback: 全局输出键名
        if not candidates:
            candidates = _match(_FALLBACK_OUTPUT_KEYS)

        # Fallback: 排除装饰性字段后的所有数值节点
        if not candidates:
            candidates = [
                i for i in num_indices
                if not leaves[i].path or leaves[i].path[-1] not in _DECORATION_KEYS
            ]

        # 兜底: 至少 mask 最后一个数值节点
        if not candidates:
            candidates = [num_indices[-1]]

        return random.sample(candidates, min(num_masks, len(candidates)))


# ═══════════════════════════════════════════════════════════════
# §4  Fork Bias 计算
# ═══════════════════════════════════════════════════════════════

def compute_fork_bias_indices(path_ids: torch.Tensor, is_group: torch.Tensor,
                              valid_path_lens: torch.Tensor) -> torch.Tensor:
    """
    向量化分叉检测：计算每对 token 的 fork level。

    对于任意两个叶子节点 (i, j)，找到它们路径中第一个不匹配的位置。
    如果该位置双方都是 Array Instance (is_group=1)，则 fork_level = 该位置的累计 group 数。
    否则 fork_level = 0（Dict Key 分叉或完全匹配）。

    Args:
        path_ids:        [B, S, L] 每个 path 节点的唯一 ID
        is_group:        [B, S, L] 1=Array Instance, 0=Dict Key/PAD
        valid_path_lens: [B, S]    每个 token 的实际路径长度
    Returns:
        [B, S, S] fork level 索引（0=无分叉/同元素）
    """
    B, S, L = path_ids.shape

    # 有效区域 mask：padding 视为匹配
    depth_idx = torch.arange(L).view(1, 1, L)
    valid_A = depth_idx < valid_path_lens.unsqueeze(2)          # [B, S, L]
    joint_valid = valid_A.unsqueeze(2) & valid_A.unsqueeze(1)   # [B, S, S, L]

    ids_A = path_ids.unsqueeze(2)  # [B, S, 1, L]
    ids_B = path_ids.unsqueeze(1)  # [B, 1, S, L]
    match_matrix = (ids_A == ids_B) | ~joint_valid  # padding 视为匹配
    all_match = match_matrix.all(dim=-1)            # [B, S, S]

    # 第一个不匹配的索引位置
    first_mismatch_idx = (~match_matrix).long().argmax(dim=-1)  # [B, S, S]

    # 检查分叉处是否双方均为 Array Instance
    is_grp_A = is_group.unsqueeze(2).expand(B, S, S, L)
    is_grp_B = is_group.unsqueeze(1).expand(B, S, S, L)
    fork_grp_A = torch.gather(is_grp_A, 3, first_mismatch_idx.unsqueeze(-1)).squeeze(-1)
    fork_grp_B = torch.gather(is_grp_B, 3, first_mismatch_idx.unsqueeze(-1)).squeeze(-1)
    diverged_at_group = (fork_grp_A == 1) & (fork_grp_B == 1)

    # Fork level = 到分叉处为止的 group 累计数
    group_depth = torch.cumsum(is_group, dim=-1)                # [B, S, L]
    group_depth_A = group_depth.unsqueeze(2).expand(B, S, S, L)
    fork_level = torch.gather(group_depth_A, 3, first_mismatch_idx.unsqueeze(-1)).squeeze(-1)

    return torch.where(~all_match & diverged_at_group, fork_level,
                       torch.zeros_like(fork_level))


# ═══════════════════════════════════════════════════════════════
# §5  Collate 函数
# ═══════════════════════════════════════════════════════════════

def collate_fn(batch: List[Tuple[List[LeafNode], Dict[int, Any]]],
               max_tokens: int = 512):
    """
    处理变长 Batch，生成 padding mask 和 fork bias 矩阵。

    Returns:
        (batched_leaves, batched_masks, padding_mask, fork_bias_indices)
    """
    batched_leaves = []
    batched_masks = []

    # 双重保护：collate 层再做一次截断
    max_len = min(max(len(leaves) for leaves, _ in batch), max_tokens)
    B = len(batch)

    padding_mask = torch.ones((B, max_len), dtype=torch.bool)

    for b, (leaves, masks) in enumerate(batch):
        truncated_leaves = leaves[:max_len]
        truncated_masks = {k: v for k, v in masks.items() if k < max_len}
        batched_leaves.append(truncated_leaves)
        batched_masks.append(truncated_masks)
        padding_mask[b, :len(truncated_leaves)] = False

    # ── 计算 fork bias 矩阵 ──
    max_path_len = 1
    for leaves_list in batched_leaves:
        for leaf in leaves_list:
            max_path_len = max(max_path_len, len(leaf.path_ids))

    path_ids_t = torch.zeros(B, max_len, max_path_len, dtype=torch.long)
    is_group_t = torch.zeros(B, max_len, max_path_len, dtype=torch.long)
    valid_path_lens_t = torch.zeros(B, max_len, dtype=torch.long)

    for b, leaves_list in enumerate(batched_leaves):
        for i, leaf in enumerate(leaves_list):
            L = len(leaf.path_ids)
            valid_path_lens_t[b, i] = L
            if L > 0:
                path_ids_t[b, i, :L] = torch.tensor(leaf.path_ids, dtype=torch.long)
                is_group_t[b, i, :L] = torch.tensor(leaf.path_types, dtype=torch.long)

    fork_bias_indices = compute_fork_bias_indices(path_ids_t, is_group_t, valid_path_lens_t)

    return batched_leaves, batched_masks, padding_mask, fork_bias_indices
