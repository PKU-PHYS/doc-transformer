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
            self._n_truncated = cached["n_truncated"]
            print(f"    ✅ {len(self._all_leaves)} samples loaded from cache")
        else:
            # ── 预解析所有文档 ──
            print(f"    ⏳ Pre-parsing {len(docs)} documents...")
            self._all_leaves = []
            self._target_indices = []
            self._n_truncated = 0

            for i, doc in enumerate(docs):
                parser = JSONParser()
                leaves = parser.parse(
                    doc, ["crystal"], [0], [JSONParser._key_hash("crystal")], []
                )

                # 截断前先定位 target leaf(structure_to_json 把 target 放在最后一个 key,
                # 朴素截断必砍掉它)
                orig_target_idx = -1
                for j, leaf in enumerate(leaves):
                    # target 一定在根级: path == ["crystal", target_key],长度为 2。
                    # 加深度约束跳过任何同名嵌套字段,避免取错位置。
                    if len(leaf.path) == 2 and leaf.path[-1] == target_key:
                        orig_target_idx = j
                        break

                # 截断:保护 target leaf 不被砍掉
                if len(leaves) > max_tokens:
                    self._n_truncated += 1
                    warnings.warn(
                        f"Document {i} has {len(leaves)} tokens, exceeding "
                        f"max_tokens={max_tokens}. Truncating to {max_tokens} "
                        f"(target leaf preserved if present).",
                        stacklevel=2,
                    )
                    if orig_target_idx < 0:
                        # doc 本身没有 target,普通截断
                        leaves = leaves[:max_tokens]
                        target_idx = -1
                    elif orig_target_idx < max_tokens:
                        # target 落在窗口内,普通截断已不会丢它
                        leaves = leaves[:max_tokens]
                        target_idx = orig_target_idx
                    else:
                        # target 落在窗口外:取前 max_tokens-1 个其他叶子,把 target 挪到末尾
                        target_leaf = leaves[orig_target_idx]
                        leaves = leaves[:max_tokens - 1] + [target_leaf]
                        target_idx = max_tokens - 1
                else:
                    target_idx = orig_target_idx

                self._all_leaves.append(leaves)
                self._target_indices.append(target_idx)

                if (i + 1) % 10000 == 0:
                    print(f"      ... {i+1}/{len(docs)} parsed")

            print(f"    ✅ Pre-parsed {len(self._all_leaves)} documents")

            # ── 预计算 fork_bias ──
            self._precompute_fork_bias()

            # ── 保存缓存 ──
            if cache_path:
                self._save_cache(cache_path)

        # ── 统一汇总报告(cache hit / miss 两条路径均走这里)──
        n_target_missing = sum(1 for t in self._target_indices if t < 0)
        if self._n_truncated > 0:
            warnings.warn(
                f"{self._n_truncated}/{len(self._all_leaves)} documents were truncated "
                f"to max_tokens={max_tokens} (target leaf preserved, other fields dropped).",
                stacklevel=2,
            )
        if n_target_missing > 0:
            warnings.warn(
                f"{n_target_missing}/{len(self._all_leaves)} documents have no "
                f"'{target_key}' key — these samples will raise at __getitem__.",
                stacklevel=2,
            )

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
            "n_truncated": self._n_truncated,
        }
        with open(cache_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = cache_path.stat().st_size / 1024 / 1024
        print(f"    💾 Cached: {cache_path.name} ({size_mb:.1f} MB)")

    def _cache_path(self, cache_tag: str):
        if not cache_tag:
            return None
        cache_dir = pathlib.Path(__file__).parent / "cache"
        return cache_dir / f"{cache_tag}_t{self.max_tokens}_parsed.pkl"

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
            raise RuntimeError(
                f"Sample {idx}: target leaf not found (target_idx={target_idx}). "
                f"This document is missing the '{self.target_key}' key entirely "
                f"(截断逻辑会保护 target leaf,所以这里不再是截断引起的)。"
                f"Check the upstream loader (structure_to_json) for documents without targets."
            )

        return leaves, target_masks, fork_bias
