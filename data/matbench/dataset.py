"""
Matbench 数据集 — 嵌套 JSON 晶体结构的 mask-predict。

优化：
  - 初始化时预解析所有文档为 LeafNode 序列（避免 __getitem__ 重复解析）
  - 预计算 target 叶子索引
  - 预计算 fork_bias 矩阵（路径在训练中不变，无需每 batch 重算）
  - 支持磁盘缓存（跳过首次解析和 fork_bias 计算开销）

与 TabularDataset 的核心区别：
  - 每个样本的结构不同（site 数量不同）
  - 无 BallTree / 邻居：每个 structure 自包含全部信息
  - 路径深度更深 (2-4 层 vs tabular 的 2 层)
  - fork bias 天然区分不同 site（同一 sites 数组的不同元素）
"""

import os
import warnings
import pickle
import pathlib
import random
import torch
from typing import List, Dict, Any, Tuple
from torch.utils.data import Dataset
from model.json_parser import LeafNode, JSONParser
from data.base import compute_single_fork_bias


class MatbenchDataset(Dataset):
    """
    Matbench 晶体结构数据集（预解析版）。

    初始化时一次性将所有 JSON doc 解析为 LeafNode 序列，
    并预计算每个样本的 fork_bias 矩阵。
    __getitem__ 只做 mask + 返回，速度极快。

    支持缓存：解析结果和 fork_bias 存到磁盘，下次直接加载。
    """

    def __init__(self, docs: list, max_tokens: int = 512,
                 target_key: str = "target",
                 cache_tag: str = None):
        """
        Args:
            docs: JSON dict 列表
            max_tokens: 最大 token 数
            target_key: 预测目标的 key 名
            cache_tag: 缓存标识 (e.g., "matbench_mp_gap_train")，
                       设为 None 则不缓存
        """
        self.max_tokens = max_tokens
        self.target_key = target_key

        # ── 尝试加载预解析缓存 ──
        cache_path = self._cache_path(cache_tag) if cache_tag else None

        if cache_path and cache_path.exists():
            print(f"    💾 Loading pre-parsed cache: {cache_path.name}")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            self._all_leaves = cached["all_leaves"]
            self._target_indices = cached["target_indices"]
            self._all_fork_bias = cached["all_fork_bias"]
            print(f"    ✅ {len(self._all_leaves)} samples loaded from cache")
            return

        # ── 预解析所有文档 ──
        print(f"    ⏳ Pre-parsing {len(docs)} documents...")
        self._all_leaves = []
        self._target_indices = []

        for i, doc in enumerate(docs):
            parser = JSONParser()
            leaves = parser.parse(
                doc, ["crystal"], [0], [JSONParser._key_hash("crystal")], []
            )

            # 截断
            if len(leaves) > max_tokens:
                warnings.warn(
                    f"Document {i} has {len(leaves)} tokens, exceeding "
                    f"max_tokens={max_tokens}. Truncating to {max_tokens}.",
                    stacklevel=2,
                )
                leaves = leaves[:max_tokens]

            # 找 target 叶子索引
            target_idx = -1
            for j, leaf in enumerate(leaves):
                if leaf.path and leaf.path[-1] == target_key:
                    target_idx = j
                    break

            self._all_leaves.append(leaves)
            self._target_indices.append(target_idx)

            if (i + 1) % 10000 == 0:
                print(f"      ... {i+1}/{len(docs)} parsed")

        # ── 汇总报告 ──
        n_truncated = sum(1 for leaves in self._all_leaves if len(leaves) >= max_tokens)
        n_target_lost = sum(1 for t in self._target_indices if t < 0)
        print(f"    ✅ Pre-parsed {len(self._all_leaves)} documents")
        if n_truncated > 0:
            warnings.warn(
                f"{n_truncated}/{len(self._all_leaves)} documents were truncated "
                f"to max_tokens={max_tokens}.",
                stacklevel=2,
            )
        if n_target_lost > 0:
            warnings.warn(
                f"{n_target_lost}/{len(self._all_leaves)} documents lost their "
                f"target leaf after truncation! These samples will use fallback "
                f"masking (random numeric field), which corrupts training.",
                stacklevel=2,
            )

        # ── 预计算 fork_bias ──
        self._precompute_fork_bias()

        # ── 保存缓存 ──
        if cache_path:
            self._save_cache(cache_path)

    def _precompute_fork_bias(self):
        """预计算所有样本的 fork_bias 矩阵，存为 int8 节省内存。"""
        n = len(self._all_leaves)
        print(f"    ⏳ Precomputing fork bias for {n} samples...")
        self._all_fork_bias = []
        for i, leaves in enumerate(self._all_leaves):
            fb = compute_single_fork_bias(leaves)
            self._all_fork_bias.append(fb)
            if (i + 1) % 10000 == 0:
                print(f"      ... {i+1}/{n}")
        # 估算内存占用
        total_bytes = sum(fb.nelement() * fb.element_size() for fb in self._all_fork_bias)
        print(f"    ✅ Fork bias precomputed ({total_bytes / 1024 / 1024:.1f} MB in memory)")

    def _save_cache(self, cache_path):
        """保存解析结果和 fork_bias 到磁盘缓存。"""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "all_leaves": self._all_leaves,
            "target_indices": self._target_indices,
            "all_fork_bias": self._all_fork_bias,
        }
        with open(cache_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = cache_path.stat().st_size / 1024 / 1024
        print(f"    💾 Cached: {cache_path.name} ({size_mb:.1f} MB)")

    @staticmethod
    def _cache_path(cache_tag: str):
        if not cache_tag:
            return None
        cache_dir = pathlib.Path(__file__).parent / "cache"
        return cache_dir / f"{cache_tag}_parsed.pkl"

    def __len__(self):
        return len(self._all_leaves)

    def __getitem__(self, idx) -> Tuple[List[LeafNode], Dict[int, Any], torch.Tensor]:
        # ── 复制预解析的叶子（避免修改原数据）──
        leaves = list(self._all_leaves[idx])
        target_idx = self._target_indices[idx]
        fork_bias = self._all_fork_bias[idx]

        # ── Mask target ──
        target_masks = {}

        if target_idx >= 0 and target_idx < len(leaves):
            orig = leaves[target_idx]
            target_masks[target_idx] = (orig.value, orig.value_type)
            leaves[target_idx] = LeafNode(
                value="[MASK]", value_type="mask",
                path=orig.path, path_types=orig.path_types,
                path_ids=orig.path_ids, group_ids=orig.group_ids,
            )
        else:
            # 兜底：随机 mask 一个数值字段
            num_indices = [i for i, l in enumerate(leaves)
                          if l.value_type == "number"]
            if num_indices:
                i = random.choice(num_indices)
                orig = leaves[i]
                target_masks[i] = (orig.value, orig.value_type)
                leaves[i] = LeafNode(
                    value="[MASK]", value_type="mask",
                    path=orig.path, path_types=orig.path_types,
                    path_ids=orig.path_ids, group_ids=orig.group_ids,
                )

        return leaves, target_masks, fork_bias
