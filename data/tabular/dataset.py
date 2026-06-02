"""
表格数据集 — query-aware 多行样本生成。

数据流：
  1. 初始化时通过 loader 预计算所有行的 K 近邻
  2. 预构建 JSON 路径模板（所有样本共享相同的 key 结构）
  3. __getitem__ 只做：取行数据 → 填模板 → mask target → 返回
"""

import random
import torch
from typing import List, Dict, Any, Tuple
from torch.utils.data import Dataset
from model.json_parser import LeafNode, JSONParser
from data.tabular.loader import TableLoader
from data.base import compute_single_structural_bias


class TabularDataset(Dataset):
    """
    表格数据集 — 将表格行转为 JSON 数组文档，mask target 列预测。

    优化点：
      - 预计算全部邻居索引 → __getitem__ 零 BallTree 查询
      - 预构建路径模板 → __getitem__ 零 hash 计算
    """

    def __init__(self, loader: TableLoader,
                 n_rows: int = 5,
                 max_tokens: int = 512,
                 **kwargs):
        self.loader = loader
        self.n_rows = n_rows
        self.max_tokens = max_tokens

        # ── 预计算所有行的邻居 ──
        k = n_rows - 1
        if k > 0:
            print(f"  ⏳ Precomputing {k}-NN for {loader.n_rows} rows...")
            self._all_neighbors = loader.precompute_neighbors(k)
            print(f"  ✅ Neighbor index precomputed: {self._all_neighbors.shape}")
        else:
            self._all_neighbors = None

        # ── 预构建路径模板 ──
        # 用一个 dummy 样本解析出路径结构，之后复用
        dummy_rows = [loader.get_row_dict(0)] * n_rows
        parser = JSONParser()
        self._template_leaves = parser.parse(
            dummy_rows, ["records"], [0], [JSONParser._key_hash("records")], []
        )
        if len(self._template_leaves) > max_tokens:
            self._template_leaves = self._template_leaves[:max_tokens]

        # 预计算每个 leaf 的 (行位置, 列名)
        self._leaf_meta = []
        n_cols = loader.n_cols
        for i, leaf in enumerate(self._template_leaves):
            row_pos = i // n_cols
            col_name = leaf.path[-1] if leaf.path else ""
            self._leaf_meta.append((row_pos, col_name))

        # target 列对应的 leaf 索引（按行）
        self._target_leaf_by_row = {}
        target_col = loader.target_col
        for i, (row_pos, col_name) in enumerate(self._leaf_meta):
            if col_name == target_col:
                self._target_leaf_by_row[row_pos] = i

        # ── 预计算 structural_bias（所有样本路径相同，只需算一次）──
        self._structural_bias = compute_single_structural_bias(self._template_leaves)

    def __len__(self):
        return self.loader.n_rows

    def __getitem__(self, idx) -> Tuple[List[LeafNode], Dict[int, Any]]:
        seed_idx = idx % self.loader.n_rows

        # ── Step 1: 取预计算的邻居 ──
        if self._all_neighbors is not None:
            neighbor_indices = self._all_neighbors[seed_idx]
            row_indices = [seed_idx] + list(neighbor_indices)
        else:
            row_indices = [seed_idx]

        # 打乱行顺序
        random.shuffle(row_indices)
        seed_position = row_indices.index(seed_idx)

        # ── Step 2: 组装 LeafNode（复用模板路径） ──
        rows_data = [self.loader.get_row_dict(i) for i in row_indices]
        leaves = []

        for i, tmpl in enumerate(self._template_leaves):
            row_pos, col_name = self._leaf_meta[i]
            if row_pos < len(rows_data) and col_name in rows_data[row_pos]:
                value = rows_data[row_pos][col_name]
                vtype = "number" if isinstance(value, (int, float)) else "string"
            else:
                value = tmpl.value
                vtype = tmpl.value_type

            leaves.append(LeafNode(
                value=value, value_type=vtype,
                path=tmpl.path, path_types=tmpl.path_types,
                path_ids=tmpl.path_ids, group_ids=tmpl.group_ids,
            ))

        # ── Step 3: Mask seed 行的 target 列 ──
        target_masks = {}

        if seed_position in self._target_leaf_by_row:
            i = self._target_leaf_by_row[seed_position]
            if i < len(leaves):
                target_masks[i] = (leaves[i].value, leaves[i].value_type)
                orig = leaves[i]
                leaves[i] = LeafNode(
                    value="[MASK]", value_type="mask",
                    path=orig.path, path_types=orig.path_types,
                    path_ids=orig.path_ids, group_ids=orig.group_ids,
                )

        if not target_masks:
            raise RuntimeError(
                f"Sample {idx}: target leaf not found for seed row "
                f"at position {seed_position}. "
                f"Check template structure and target_col='{self.loader.target_col}'."
            )

        return leaves, target_masks, self._structural_bias
