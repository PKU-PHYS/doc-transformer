"""
数据管线通用组件 — collate_fn + structural_bias 计算。

这些函数与数据源无关，只处理 LeafNode[] → Tensor 的通用逻辑。
所有 Dataset 的 __getitem__ 返回 (List[LeafNode], Dict[int, Any], Dict[str, Tensor])，
其中第三个元素是预计算的 structural_bias 字典。
"""

import torch
from typing import List, Dict, Any, Tuple
from model.json_parser import LeafNode


# ═══════════════════════════════════════════════════════════════
# §1  Structural Bias 计算
# ═══════════════════════════════════════════════════════════════

def compute_structural_bias_indices(path_ids: torch.Tensor,
                                    is_group: torch.Tensor,
                                    valid_path_lens: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    计算结构关系信号，衡量两个叶子在 JSON 树中的拓扑关系。

    对于每对叶子 (i, j)：
      - is_group_fork: 分叉点是否在 group（数组 instance）层
      - first_diff:    路径首次不同的位置索引（共同前缀长度）
      - tree_dist:     两叶子到 LCA 的步数之和
      - same_parent:   是否为同一个 JSON object/list instance 下的 sibling leaf
      - shared_group_depth: 共享的数组 instance 祖先数量
      - same_path_template: 忽略数组实例 ID 后是否为同一 JSON 路径模板

    Args:
        path_ids:        (B, T, D) 每个叶子的路径 ID 序列
        is_group:        (B, T, D) 每个层级是否为 group (数组/列表)
        valid_path_lens: (B, T)    每个叶子路径的有效长度

    Returns:
        Dict[str, Tensor]，每个 value 为 (B, T, T) int8:
          "is_group_fork": 分叉点是否在 group 层（布尔→0/1）
          "first_diff":    路径首次不同的位置索引
          "tree_dist":     两叶子到 LCA 的步数之和
          "same_parent":   同父 leaf（布尔→0/1）
          "shared_group_depth": 共同数组 instance 祖先数量
          "same_path_template": 同路径模板 leaf（布尔→0/1）
    """
    B, T, D = path_ids.shape

    ids_i = path_ids.unsqueeze(2).expand(B, T, T, D)
    ids_j = path_ids.unsqueeze(1).expand(B, T, T, D)

    match = (ids_i == ids_j)

    depth_idx = torch.arange(D, device=path_ids.device).view(1, 1, 1, D)
    len_i = valid_path_lens.unsqueeze(2).unsqueeze(3)
    len_j = valid_path_lens.unsqueeze(1).unsqueeze(3)
    valid_mask = (depth_idx < len_i) & (depth_idx < len_j)

    effective_match = match | ~valid_mask

    all_match = effective_match.all(dim=-1)

    first_diff_raw = effective_match.long().argmin(dim=-1)

    # ── Bias 1: is_group_fork ──
    # 分叉点是否在 group 层。对角线/完全匹配 → 0
    is_group_pairs = is_group.unsqueeze(2).expand(B, T, T, D)
    diverged_at_group = is_group_pairs.gather(3, first_diff_raw.unsqueeze(-1)).squeeze(-1).bool()
    is_group_fork = (~all_match & diverged_at_group).to(torch.int8)

    # ── Bias 2: first_diff ──
    # 对角线/完全匹配时 argmin 返回 0，需修正为 min(L_i, L_j)
    len_i_2d = valid_path_lens.unsqueeze(2)   # (B, T, 1)
    len_j_2d = valid_path_lens.unsqueeze(1)   # (B, 1, T)
    min_len = torch.minimum(len_i_2d, len_j_2d)
    first_diff = torch.where(all_match, min_len, first_diff_raw).to(torch.int8)

    # ── Bias 3: tree_dist ──
    # tree_dist = (L_i - first_diff) + (L_j - first_diff) = L_i + L_j - 2 * first_diff
    tree_dist = (len_i_2d + len_j_2d - 2 * first_diff.long()).to(torch.int8)

    # ── Bias 4: same_parent ──
    # 同一个父 object / array instance 下的 sibling leaf。对角线不算 sibling。
    same_len = len_i_2d == len_j_2d
    same_parent = (
        ~all_match
        & same_len
        & (len_i_2d > 1)
        & (first_diff.long() == (len_i_2d - 1))
    ).to(torch.int8)

    # ── Bias 5: shared_group_depth ──
    # 共享的数组 instance 祖先数量：同一 table row / composition item / site
    # 会得到更高值；无数组祖先的普通 dict 字段为 0。
    group_mask = is_group.bool()
    group_pos = group_mask.long().cumsum(dim=-1) - 1
    compact_groups = torch.zeros_like(path_ids)
    compact_groups.scatter_add_(
        2,
        group_pos.clamp_min(0),
        torch.where(group_mask, path_ids, torch.zeros_like(path_ids)),
    )
    group_lens = group_mask.long().sum(dim=-1)
    groups_i = compact_groups.unsqueeze(2).expand(B, T, T, D)
    groups_j = compact_groups.unsqueeze(1).expand(B, T, T, D)
    group_depth_idx = torch.arange(D, device=path_ids.device).view(1, 1, 1, D)
    group_len_i = group_lens.unsqueeze(2).unsqueeze(3)
    group_len_j = group_lens.unsqueeze(1).unsqueeze(3)
    group_min_len = torch.minimum(group_len_i, group_len_j)
    group_valid = group_depth_idx < group_min_len
    group_effective_match = (groups_i == groups_j) | ~group_valid
    group_all_match = group_effective_match.all(dim=-1)
    group_first_diff = group_effective_match.long().argmin(dim=-1)
    shared_group_depth = torch.where(
        group_all_match,
        group_min_len.squeeze(-1),
        group_first_diff,
    ).to(torch.int8)

    # ── Bias 6: same_path_template ──
    # 忽略数组实例 ID，只比较 dict-key 序列是否完全一致。不同数组元素中的
    # 同名字段（如 composition[*].ratio）会得到 1；不同数组字段或不同 leaf key 为 0。
    template_ids = torch.where(is_group.bool(), torch.zeros_like(path_ids), path_ids)
    template_i = template_ids.unsqueeze(2).expand(B, T, T, D)
    template_j = template_ids.unsqueeze(1).expand(B, T, T, D)
    template_match = (template_i == template_j) | ~valid_mask
    same_path_template = (
        ~all_match
        & (valid_path_lens.unsqueeze(2) == valid_path_lens.unsqueeze(1))
        & template_match.all(dim=-1)
    ).to(torch.int8)

    return {
        "is_group_fork": is_group_fork,
        "first_diff": first_diff,
        "tree_dist": tree_dist,
        "same_parent": same_parent,
        "shared_group_depth": shared_group_depth,
        "same_path_template": same_path_template,
    }


def compute_single_structural_bias(leaves: List[LeafNode]) -> Dict[str, torch.Tensor]:
    """
    计算单个样本的 structural bias 字典。

    由于每个样本的路径在训练过程中不变，可在预解析阶段调用一次并缓存，
    避免 collate_fn 中每 batch 重复计算 O(B*T²*D) 的开销。

    Args:
        leaves: 单个样本的叶子节点列表

    Returns:
        Dict[str, Tensor]，每个 value 为 (T, T) int8
    """
    T = len(leaves)
    if T == 0:
        return {
            "is_group_fork": torch.zeros(0, 0, dtype=torch.int8),
            "first_diff": torch.zeros(0, 0, dtype=torch.int8),
            "tree_dist": torch.zeros(0, 0, dtype=torch.int8),
            "same_parent": torch.zeros(0, 0, dtype=torch.int8),
            "shared_group_depth": torch.zeros(0, 0, dtype=torch.int8),
            "same_path_template": torch.zeros(0, 0, dtype=torch.int8),
        }

    max_path_len = max(len(l.path_ids) for l in leaves)
    if max_path_len == 0:
        return {
            "is_group_fork": torch.zeros(T, T, dtype=torch.int8),
            "first_diff": torch.zeros(T, T, dtype=torch.int8),
            "tree_dist": torch.zeros(T, T, dtype=torch.int8),
            "same_parent": torch.zeros(T, T, dtype=torch.int8),
            "shared_group_depth": torch.zeros(T, T, dtype=torch.int8),
            "same_path_template": torch.zeros(T, T, dtype=torch.int8),
        }

    path_ids = torch.zeros(1, T, max_path_len, dtype=torch.long)
    is_group = torch.zeros(1, T, max_path_len, dtype=torch.long)
    valid_lens = torch.zeros(1, T, dtype=torch.long)

    for i, leaf in enumerate(leaves):
        L = len(leaf.path_ids)
        valid_lens[0, i] = L
        if L > 0:
            path_ids[0, i, :L] = torch.tensor(leaf.path_ids, dtype=torch.long)
            is_group[0, i, :L] = torch.tensor(leaf.path_types, dtype=torch.long)

    result = compute_structural_bias_indices(path_ids, is_group, valid_lens)
    return {k: v[0] for k, v in result.items()}


# ═══════════════════════════════════════════════════════════════
# §2  Collate 函数
# ═══════════════════════════════════════════════════════════════

def collate_fn(batch, max_tokens: int = 512):
    """
    处理变长 Batch，生成 padding mask 并组装预计算的 structural bias 矩阵。

    Args:
        batch: List of (leaves, target_masks, structural_bias) from Dataset.__getitem__
        max_tokens: 截断阈值

    Returns:
        (batched_leaves, batched_masks, padding_mask, structural_bias_indices)
        structural_bias_indices: Dict[str, Tensor]，每个 value 为 (B, max_len, max_len)
    """
    batched_leaves = []
    batched_masks = []
    bias_list = []

    # 约定:此处的 max_tokens 必须 >= dataset 的 max_tokens。
    # 若 dataset 已按自己的 max_tokens 截断并保护过 target leaf,collate 不应再砍。
    # 一旦违反约定,会破坏 dataset 阶段对 target leaf 的保护,故显式断言而非静默二次截断。
    batch_max = max(len(item[0]) for item in batch)
    if batch_max > max_tokens:
        raise ValueError(
            f"collate_fn received leaves of length {batch_max} > max_tokens={max_tokens}. "
            f"This likely means collate_fn's max_tokens is smaller than the dataset's max_tokens, "
            f"which would silently break target leaf protection."
        )
    max_len = min(batch_max, max_tokens)
    B = len(batch)

    padding_mask = torch.ones((B, max_len), dtype=torch.bool)

    for b, (leaves, masks, structural_bias) in enumerate(batch):
        truncated_leaves = leaves[:max_len]
        truncated_masks = {k: v for k, v in masks.items() if k < max_len}
        batched_leaves.append(truncated_leaves)
        batched_masks.append(truncated_masks)
        padding_mask[b, :len(truncated_leaves)] = False
        bias_list.append(structural_bias)

    # ── 组装 structural bias 矩阵：每个信号独立 pad + stack ──
    signal_names = list(bias_list[0].keys())
    structural_bias_indices = {}
    for name in signal_names:
        stacked = torch.zeros(B, max_len, max_len, dtype=torch.long)
        for b, sb in enumerate(bias_list):
            t = min(sb[name].size(0), max_len)
            stacked[b, :t, :t] = sb[name][:t, :t].long()
        structural_bias_indices[name] = stacked

    return batched_leaves, batched_masks, padding_mask, structural_bias_indices
