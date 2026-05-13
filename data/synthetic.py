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
from data.functions.registry import (
    UNARY_OUTPUT_KEYS, BINARY_OUTPUT_KEYS, FUNC_NAME_KEYS, CATEGORY_KEYS,
)
from data.templates.engine import TemplateEngine
from data.templates.distractors import inject_distractors
from data.text_tasks.generators import TextTaskGenerator
from data.in_context_generator import generate_in_context_task

# 所有可能的输出变量键名集合（合并 unary/binary/multivar 的输出同义词池）
# 用于在模板打乱键序后，依然能精准定位输出变量，避免 mask 不可逆函数的输入
_OUTPUT_KEY_NAMES: set = set(UNARY_OUTPUT_KEYS) | set(BINARY_OUTPUT_KEYS) | {
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

# 不应被 mask 的装饰性键名（非数学变量）
_DECORATION_KEY_NAMES: set = set(FUNC_NAME_KEYS) | set(CATEGORY_KEYS) | {
    "id", "timestamp", "author", "confidence", "tags", "label",
    "description", "desc", "info", "summary", "note",
    "units", "precision", "status", "format", "encoding",
    "index", "batch_id", "seed", "verified", "is_exact",
    "deprecated", "converged", "cached", "is_valid",
    "equation", "record_type", "expression",
}

# 不可逆函数的所有同义词（用于 group 级后过滤）
_NON_INVERTIBLE_FUNC_NAMES: set = set()

def _build_non_invertible_set():
    """从注册表动态构建不可逆函数同义词集合。必须在所有函数注册后调用。"""
    global _NON_INVERTIBLE_FUNC_NAMES
    for gen in FunctionRegistry._generators:
        rel = gen()
        if not rel.invertible:
            _NON_INVERTIBLE_FUNC_NAMES.update(rel.func_synonyms)


def _is_group_safe(leaf, group_func_map: dict, output_key_set: set) -> bool:
    """
    检查一个叶子节点是否可安全 mask（group 级别）。

    在复合文档中，即使键名通过了 safe/unsafe 集合过滤，
    仍需检查同 group 的函数是否不可逆：
      - 如果叶子的 key 是已知输出键名 → 安全（mask 输出永远可解）
      - 如果同 group 的函数是不可逆的且叶子 key 不是输出 → 不安全
    """
    key = leaf.path[-1] if leaf.path else ""
    # 输出键名总是安全的
    if key in output_key_set:
        return True
    # 检查同 group 的函数是否不可逆
    grp_key = tuple(leaf.group_ids) if leaf.group_ids else ()
    func_name = group_func_map.get(grp_key)
    if func_name and func_name in _NON_INVERTIBLE_FUNC_NAMES:
        return False
    return True

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

def generate_math_document(nested_prob: float = 0.0):
    """
    从函数注册表采样一条数学关系，用模板引擎渲染为 JSON 文档。
    Returns: (doc, safe_mask_keys)
    """
    rel = FunctionRegistry.sample()
    doc, safe_keys, unsafe_keys = TemplateEngine.render(rel)

    if isinstance(doc, dict):
        inject_distractors(doc, n_min=0, n_max=4, nested_prob=nested_prob)

    return doc, safe_keys, unsafe_keys


def generate_text_task_document(nested_prob: float = 0.0):
    """生成一条纯文本推理任务文档。Returns: (doc, safe_mask_keys)"""
    doc = TextTaskGenerator.generate()

    if isinstance(doc, dict):
        inject_distractors(doc, n_min=0, n_max=3, nested_prob=nested_prob)

    # 文本任务没有可逆/不可逆之分，所有键都安全
    return doc, None, None


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
):
    """
    生成包含多条独立数学关系的复合文档。
    Returns: (doc, safe_mask_keys)
    """
    doc = {}
    all_safe_keys: set = set()
    all_unsafe_keys: set = set()

    for i in range(n_relations):
        rel = FunctionRegistry.sample()
        section, safe_keys, unsafe_keys = TemplateEngine.render(rel)
        all_safe_keys.update(safe_keys)
        all_unsafe_keys.update(unsafe_keys)

        key_base = random.choice(_SECTION_KEYS)
        section_key = f"{key_base}_{i}"

        if isinstance(section, dict):
            doc[section_key] = section
        else:
            doc[section_key] = {"data": section}

    inject_distractors(doc, n_min=2, n_max=distractor_max, nested_prob=nested_prob)

    return doc, all_safe_keys, all_unsafe_keys


def generate_array_document(
    n_items: int = 5,
    distractor_per_item: int = 2,
    nested_prob: float = 0.3,
):
    """
    生成一个包含多个同结构对象的大数组文档。
    Returns: (doc_list, safe_mask_keys)
    """
    gen = FunctionRegistry.sample_generator()
    items = []
    all_safe_keys: set = set()
    all_unsafe_keys: set = set()

    for _ in range(n_items):
        companion = gen()
        item, safe_keys, unsafe_keys = TemplateEngine.render(companion)
        all_safe_keys.update(safe_keys)
        all_unsafe_keys.update(unsafe_keys)
        if isinstance(item, dict):
            inject_distractors(item, n_min=0, n_max=distractor_per_item,
                               nested_prob=nested_prob)
        items.append(item)

    return items, all_safe_keys, all_unsafe_keys


def generate_mixed_long_document(target_tokens: int = 100):
    """
    生成一条混合长文档。
    Returns: (doc, safe_mask_keys)
    """
    doc = {}
    parser = JSONParser()
    section_idx = 0
    all_safe_keys: set = set()
    all_unsafe_keys: set = set()

    while True:
        current_leaves = parser.parse(doc, ["doc"], [0], [JSONParser._key_hash("doc")], [])
        current_len = len(current_leaves)

        if current_len >= target_tokens * 0.8:
            break

        choice = random.random()
        if choice < 0.6:
            rel = FunctionRegistry.sample()
            section, safe_keys, unsafe_keys = TemplateEngine.render(rel)
            all_safe_keys.update(safe_keys)
            all_unsafe_keys.update(unsafe_keys)
        elif choice < 0.85:
            section = TextTaskGenerator.generate()
        else:
            gen = FunctionRegistry.sample_generator()
            items = []
            for _ in range(random.randint(2, 4)):
                item, safe_keys, unsafe_keys = TemplateEngine.render(gen())
                all_safe_keys.update(safe_keys)
                all_unsafe_keys.update(unsafe_keys)
                items.append(item)
            section = items

        key = f"{random.choice(_SECTION_KEYS)}_{section_idx}"
        doc[key] = section
        section_idx += 1

        parser = JSONParser()

    inject_distractors(doc, n_min=3, n_max=10, nested_prob=0.7)

    return doc, all_safe_keys, all_unsafe_keys


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
            - "simple": Stage 0, 极简扁平 KV，Tier 0 函数冷启动
            - "explicit": Stage 1, 显式规则, 单/复合文档
            - "explicit_long": Stage 2, 混合长文档 + 重度干扰
            - "in_context": Stage 3, 隐式上下文推断
            - "mixed": 混合模式 (40% explicit + 40% in_context + 20% explicit_long)
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
        # 延迟构建不可逆函数同义词集合（确保所有函数模块已加载）
        if not _NON_INVERTIBLE_FUNC_NAMES:
            _build_non_invertible_set()

    def __len__(self):
        return self.size

    def __getitem__(self, idx) -> Tuple[List[LeafNode], Dict[int, Any]]:
        # 模式判定
        current_mode = self.train_mode
        if current_mode == "mixed":
            # 备用混合模式：按权重随机选择（当前未被任何 Stage 使用）
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
            doc, safe_mask_keys, unsafe_mask_keys = TemplateEngine.render_simple(rel)
            parser = JSONParser()
            leaves = parser.parse(doc, ["doc"], [0], [JSONParser._key_hash("doc")], [])
            
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
            leaves = parser.parse([doc1, doc2], ["materials"], [0], [JSONParser._key_hash("materials")], [])
            safe_mask_keys = None  # 旧版生成器不追踪可逆性
            unsafe_mask_keys = None

        elif current_mode == "explicit_long" or (
            self.target_tokens is not None and self.target_tokens > 80
        ):
            # 长文档模式：使用混合长文档生成器
            target = self.target_tokens or 150
            doc, safe_mask_keys, unsafe_mask_keys = generate_mixed_long_document(target_tokens=target)
            parser = JSONParser()
            leaves = parser.parse(doc, ["doc"], [0], [JSONParser._key_hash("doc")], [])

        else:
            # 标准 explicit 模式 — 按概率选择生成策略
            nested_prob = [0.0, 0.2, 0.5, 0.8][min(self.distractor_level, 3)]
            r = random.random()

            if r < self.text_task_ratio:
                # 文本任务
                doc, safe_mask_keys, unsafe_mask_keys = generate_text_task_document(nested_prob=nested_prob)
            elif r < self.text_task_ratio + 0.3:
                # 复合文档 (多关系)
                n_rel = random.randint(2, 5)
                doc, safe_mask_keys, unsafe_mask_keys = generate_compound_document(
                    n_relations=n_rel,
                    distractor_max=6,
                    nested_prob=nested_prob,
                )
            elif r < self.text_task_ratio + 0.5:
                # 大数组文档
                n_items = random.randint(3, 8)
                doc, safe_mask_keys, unsafe_mask_keys = generate_array_document(
                    n_items=n_items,
                    distractor_per_item=2,
                    nested_prob=nested_prob,
                )
            else:
                # 单关系文档 (经典)
                doc, safe_mask_keys, unsafe_mask_keys = generate_math_document(nested_prob=nested_prob)

            parser = JSONParser()
            if isinstance(doc, list):
                leaves = parser.parse(doc, ["records"], [0], [JSONParser._key_hash("records")], [])
            else:
                leaves = parser.parse(doc, ["doc"], [0], [JSONParser._key_hash("doc")], [])

        # 确保至少有 1 个叶子节点
        if not leaves:
            parser = JSONParser()
            leaves = parser.parse({"value": random.uniform(-1, 1)}, ["doc"], [0], [JSONParser._key_hash("doc")], [])

        # ── 截断：超过 max_tokens 的叶子直接丢弃 ──
        if self.max_tokens is not None and len(leaves) > self.max_tokens:
            leaves = leaves[:self.max_tokens]

        target_masks = {}
        # ── 可解性保证的 Mask 策略 ──
        # safe_mask_keys: 从 TemplateEngine 传递的安全键名集合
        #   - 可逆函数: 包含所有变量键名（输入+输出都可 mask）
        #   - 不可逆函数: 只包含输出键名（只 mask 输出）
        #   - None: 无约束（文本任务等）
        num_masks = max(1, int(len(leaves) * self.mask_ratio))
        num_indices = [i for i, l in enumerate(leaves) if l.value_type == "number"]
        other_indices = [i for i, l in enumerate(leaves) if l.value_type != "number"]
        
        if num_indices:
            # ── 只 mask 输出键名 — 100% 保证可解性 ──
            # 在复合文档中，无法在叶子级别可靠区分可逆/不可逆函数的输入，
            # 因此统一只 mask path[-1] 匹配输出键名的节点。
            # 这保证被 mask 的值总是某个函数的输出，可由输入唯一确定。
            safe_candidates = [
                i for i in num_indices
                if leaves[i].path and leaves[i].path[-1] in _OUTPUT_KEY_NAMES
            ]

            if not safe_candidates:
                # Fallback: 文本任务或特殊模板无输出键名匹配，
                # 退回到非装饰性数值节点
                safe_candidates = [
                    i for i in num_indices
                    if not leaves[i].path or leaves[i].path[-1] not in _DECORATION_KEY_NAMES
                ]

            if safe_candidates:
                n = min(num_masks, len(safe_candidates))
                mask_indices = random.sample(safe_candidates, n)
            else:
                mask_indices = [num_indices[-1]]
            
            # 仍需更多 mask 时，从非数值节点补充
            extra_needed = num_masks - len(mask_indices)
            if extra_needed > 0 and other_indices:
                n = min(extra_needed, len(other_indices))
                mask_indices += random.sample(other_indices, n)
        else:
            mask_indices = random.sample(other_indices, min(num_masks, len(other_indices))) if other_indices else []

        for i in mask_indices:
            orig_node = leaves[i]
            target_masks[i] = (orig_node.value, orig_node.value_type)
            leaves[i] = LeafNode(
                value="[MASK]",
                value_type="mask",
                path=orig_node.path,
                path_types=orig_node.path_types,
                path_ids=orig_node.path_ids,
                group_ids=orig_node.group_ids
            )

        return leaves, target_masks


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


def collate_fn(batch: List[Tuple[List[LeafNode], Dict[int, Any]]],
               max_tokens: int = 512):
    """
    处理变长 Batch，生成 padding mask 和 fork bias 矩阵。

    Args:
        batch: DataLoader 传入的 (leaves, target_masks) 列表
        max_tokens: 安全上限，collate 时再次截断（双重保护）
    
    Returns:
        (batched_leaves, batched_masks, padding_mask, fork_bias_indices)
    """
    batched_leaves = []
    batched_masks = []

    # 双重保护：collate 层再做一次截断
    max_len = min(max(len(leaves) for leaves, _ in batch), max_tokens)
    B = len(batch)

    # src_key_padding_mask: True 表示 <PAD> 需要被阻断
    padding_mask = torch.ones((B, max_len), dtype=torch.bool)

    for b, (leaves, masks) in enumerate(batch):
        # 截断叶子
        truncated_leaves = leaves[:max_len]
        # 丢弃超出范围的 mask target
        truncated_masks = {k: v for k, v in masks.items() if k < max_len}

        batched_leaves.append(truncated_leaves)
        batched_masks.append(truncated_masks)
        # 将有效部分置为 False
        padding_mask[b, :len(truncated_leaves)] = False

    # ── 计算 fork bias 矩阵 ──
    # 提取所有 token 的 path_ids 和 path_types
    max_path_len = 1  # 至少为 1 防止空张量
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
