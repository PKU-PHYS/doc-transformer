"""
Matbench 数据集 — 嵌套 JSON 晶体结构的 mask-predict。

与 TabularDataset 的核心区别：
  - 每个样本的结构不同（site 数量不同），必须逐样本解析
  - 无 BallTree / 邻居：每个 structure 自包含全部信息
  - 路径深度更深 (2-4 层 vs tabular 的 2 层)
  - fork bias 天然区分不同 site（同一 sites 数组的不同元素）
"""

from typing import List, Dict, Any, Tuple
from torch.utils.data import Dataset
from model.json_parser import LeafNode, JSONParser


class MatbenchDataset(Dataset):
    """
    Matbench 晶体结构数据集。

    每个样本是一个嵌套 JSON 文档：
      { "lattice": {...}, "sites": [...], "target": value }

    __getitem__ 流程：
      1. 取预转换的 JSON doc
      2. JSONParser.parse() 展平为 LeafNode 序列
      3. Mask target 字段
      4. 返回 (leaves, target_masks)
    """

    def __init__(self, docs: list, max_tokens: int = 512,
                 target_key: str = "target"):
        self.docs = docs
        self.max_tokens = max_tokens
        self.target_key = target_key

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx) -> Tuple[List[LeafNode], Dict[int, Any]]:
        doc = self.docs[idx]

        # ── Step 1: 解析为 LeafNode 序列 ──
        # 每次新建 parser，避免 group_counter 状态冲突 (多 worker)
        parser = JSONParser()
        leaves = parser.parse(
            doc, ["crystal"], [0], [JSONParser._key_hash("crystal")], []
        )

        # 截断到 max_tokens
        if len(leaves) > self.max_tokens:
            leaves = leaves[:self.max_tokens]

        # ── Step 2: 找到并 mask target 字段 ──
        target_masks = {}
        target_idx = None

        for i, leaf in enumerate(leaves):
            if leaf.path and leaf.path[-1] == self.target_key:
                target_idx = i
                break

        if target_idx is not None:
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
                import random
                i = random.choice(num_indices)
                orig = leaves[i]
                target_masks[i] = (orig.value, orig.value_type)
                leaves[i] = LeafNode(
                    value="[MASK]", value_type="mask",
                    path=orig.path, path_types=orig.path_types,
                    path_ids=orig.path_ids, group_ids=orig.group_ids,
                )

        return leaves, target_masks
