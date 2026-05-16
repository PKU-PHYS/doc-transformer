"""
数据管线通用组件 — collate_fn + fork_bias 计算。

这些函数与数据源无关，只处理 LeafNode[] → Tensor 的通用逻辑。
所有 Dataset 的 __getitem__ 返回 (List[LeafNode], Dict[int, Any], Tensor)，
其中第三个元素是预计算的 fork_bias 矩阵。
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


def compute_single_fork_bias(leaves: List[LeafNode]) -> torch.Tensor:
    """
    计算单个样本的 fork bias 矩阵。

    由于每个样本的路径在训练过程中不变，可在预解析阶段调用一次并缓存，
    避免 collate_fn 中每 batch 重复计算 O(B*T²*D) 的开销。

    Args:
        leaves: 单个样本的叶子节点列表

    Returns:
        (T, T) int8 张量 — fork bias 索引矩阵
    """
    T = len(leaves)
    if T == 0:
        return torch.zeros(0, 0, dtype=torch.int8)

    max_path_len = max(len(l.path_ids) for l in leaves)
    if max_path_len == 0:
        return torch.zeros(T, T, dtype=torch.int8)

    path_ids = torch.zeros(1, T, max_path_len, dtype=torch.long)
    is_group = torch.zeros(1, T, max_path_len, dtype=torch.long)
    valid_lens = torch.zeros(1, T, dtype=torch.long)

    for i, leaf in enumerate(leaves):
        L = len(leaf.path_ids)
        valid_lens[0, i] = L
        if L > 0:
            path_ids[0, i, :L] = torch.tensor(leaf.path_ids, dtype=torch.long)
            is_group[0, i, :L] = torch.tensor(leaf.path_types, dtype=torch.long)

    result = compute_fork_bias_indices(path_ids, is_group, valid_lens)
    return result[0].to(torch.int8)


# ═══════════════════════════════════════════════════════════════
# §2  Collate 函数
# ═══════════════════════════════════════════════════════════════

def collate_fn(batch, max_tokens: int = 512):
    """
    处理变长 Batch，生成 padding mask 并组装预计算的 fork bias 矩阵。

    Args:
        batch: List of (leaves, target_masks, fork_bias) from Dataset.__getitem__
        max_tokens: 截断阈值

    Returns:
        (batched_leaves, batched_masks, padding_mask, fork_bias_indices)
    """
    batched_leaves = []
    batched_masks = []
    fork_bias_list = []

    # 双重保护：collate 层再做一次截断
    max_len = min(max(len(item[0]) for item in batch), max_tokens)
    B = len(batch)

    padding_mask = torch.ones((B, max_len), dtype=torch.bool)

    for b, (leaves, masks, fork_bias) in enumerate(batch):
        truncated_leaves = leaves[:max_len]
        truncated_masks = {k: v for k, v in masks.items() if k < max_len}
        batched_leaves.append(truncated_leaves)
        batched_masks.append(truncated_masks)
        padding_mask[b, :len(truncated_leaves)] = False
        fork_bias_list.append(fork_bias)

    # ── 组装 fork bias 矩阵：pad + stack 预计算结果 ──
    fork_bias_indices = torch.zeros(B, max_len, max_len, dtype=torch.long)
    for b, fb in enumerate(fork_bias_list):
        t = min(fb.size(0), max_len)
        fork_bias_indices[b, :t, :t] = fb[:t, :t].long()

    return batched_leaves, batched_masks, padding_mask, fork_bias_indices
