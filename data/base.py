"""
数据管线通用组件 — collate_fn + fork_bias 计算。

这些函数与数据源无关，只处理 LeafNode[] → Tensor 的通用逻辑。
任何 Dataset 只要 __getitem__ 返回 (List[LeafNode], Dict[int, Any])，
就可以使用这里的 collate_fn 构建 batch。
"""

import torch
from typing import List, Dict, Any, Tuple
from model.json_parser import LeafNode


# ═══════════════════════════════════════════════════════════════
# §1  Fork Bias 计算
# ═══════════════════════════════════════════════════════════════

def compute_fork_bias_indices(path_ids: torch.Tensor,
                              is_group: torch.Tensor,
                              valid_path_lens: torch.Tensor) -> torch.Tensor:
    """
    计算 fork bias 矩阵：衡量两个叶子在 JSON 树中的结构距离。

    对于每对叶子 (i, j)：
      - 找到它们路径 ID 首次不同的层级 k（fork point）
      - 如果 fork 发生在 group 层级（数组），编码为 k
      - 如果完全相同或没有 group 层级的分叉，编码为 0

    Args:
        path_ids:        (B, T, D) 每个叶子的路径 ID 序列
        is_group:        (B, T, D) 每个层级是否为 group (数组/列表)
        valid_path_lens: (B, T)    每个叶子路径的有效长度

    Returns:
        (B, T, T) fork bias 索引矩阵
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

    first_diff = effective_match.long().argmin(dim=-1)

    fork_is_group = is_group.unsqueeze(2).expand(B, T, T, D)
    fork_level = fork_is_group.gather(3, first_diff.unsqueeze(-1)).squeeze(-1)

    diverged_at_group = fork_is_group.gather(3, first_diff.unsqueeze(-1)).squeeze(-1).bool()

    return torch.where(~all_match & diverged_at_group, fork_level,
                       torch.zeros_like(fork_level))


# ═══════════════════════════════════════════════════════════════
# §2  Collate 函数
# ═══════════════════════════════════════════════════════════════

def collate_fn(batch: List[Tuple[List[LeafNode], Dict[int, Any]]],
               max_tokens: int = 512):
    """
    处理变长 Batch，生成 padding mask 和 fork bias 矩阵。

    Args:
        batch: List of (leaves, target_masks) from Dataset.__getitem__
        max_tokens: 截断阈值

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
