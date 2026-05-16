"""
Matbench 数据加载器 — 将 pymatgen Structure 转为嵌套 JSON。

数据流：
  1. 通过 matminer 加载 Matbench 任务数据
  2. pymatgen Structure → 精简嵌套 JSON dict
  3. 大结构截断：primitive cell + 按元素分层采样
"""

import math
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple


# ═══════════════════════════════════════════════════════════════
# §1  Structure → JSON 转换
# ═══════════════════════════════════════════════════════════════

def structure_to_json(structure, target_val: Optional[float] = None,
                      max_sites: int = 126) -> dict:
    """
    将 pymatgen Structure 转为精简嵌套 JSON。

    精简策略：只保留对属性预测有用的字段。
    截断策略：primitive cell → 按元素分层采样。

    Args:
        structure:  pymatgen Structure 对象
        target_val: 预测目标值 (训练时提供，测试时为 None)
        max_sites:  最大 site 数量 (max_tokens=512 → 126)

    Returns:
        嵌套 JSON dict
    """
    # ── Step 1: 尝试缩到 primitive cell ──
    if len(structure) > max_sites:
        try:
            structure = structure.get_primitive_structure()
        except Exception:
            pass  # 部分结构无法缩，保持原样

    # ── Step 2: 按元素分层采样 ──
    sites_data = []
    for site in structure:
        element = str(site.specie)
        frac = site.frac_coords
        sites_data.append({
            "element": element,
            "x": round(float(frac[0]), 6),
            "y": round(float(frac[1]), 6),
            "z": round(float(frac[2]), 6),
        })

    if len(sites_data) > max_sites:
        sites_data = _stratified_sample(sites_data, max_sites)

    # ── 组装 JSON ──
    lattice = structure.lattice
    doc = {
        "lattice": {
            "a": round(float(lattice.a), 4),
            "b": round(float(lattice.b), 4),
            "c": round(float(lattice.c), 4),
            "alpha": round(float(lattice.alpha), 2),
            "beta": round(float(lattice.beta), 2),
            "gamma": round(float(lattice.gamma), 2),
        },
        "sites": sites_data,
    }

    if target_val is not None:
        doc["target"] = round(float(target_val), 6)

    return doc


def _stratified_sample(sites: List[dict], max_n: int) -> List[dict]:
    """按元素类型分层采样，保持化学组成比例。"""
    # 按元素分组
    by_element: Dict[str, List[dict]] = {}
    for s in sites:
        by_element.setdefault(s["element"], []).append(s)

    total = len(sites)
    sampled = []

    for elem, group in by_element.items():
        # 按比例分配名额，至少保留 1 个
        quota = max(1, round(len(group) / total * max_n))
        if len(group) <= quota:
            sampled.extend(group)
        else:
            sampled.extend(random.sample(group, quota))

    # 微调到恰好 max_n
    if len(sampled) > max_n:
        sampled = sampled[:max_n]

    return sampled


# ═══════════════════════════════════════════════════════════════
# §2  MatbenchLoader
# ═══════════════════════════════════════════════════════════════

class MatbenchLoader:
    """
    Matbench 任务数据加载器。

    加载 matminer 数据 → 转为嵌套 JSON docs 列表。
    支持 train/test split。

    Attributes:
        task_name:   Matbench 任务名 (e.g., "matbench_dielectric")
        train_docs:  List[dict] 训练集 JSON 文档
        test_docs:   List[dict] 测试集 JSON 文档
        target_key:  预测目标的 key 名 ("target")
    """

    def __init__(self, task_name: str, max_sites: int = 126,
                 test_ratio: float = 0.2, seed: int = 42):
        self.task_name = task_name
        self.max_sites = max_sites
        self.target_key = "target"

        # ── 加载数据 ──
        print(f"  📂 Loading Matbench task: {task_name}")
        df = self._load_task(task_name)
        print(f"  ✅ {len(df)} samples loaded")

        # ── Structure → JSON ──
        print(f"  ⏳ Converting structures to JSON (max_sites={max_sites})...")
        structure_col = self._find_structure_col(df)
        target_col = self._find_target_col(df, structure_col)

        docs = []
        n_truncated = 0
        for idx in range(len(df)):
            structure = df.iloc[idx][structure_col]
            target = df.iloc[idx][target_col]

            orig_n = len(structure)
            doc = structure_to_json(structure, target_val=target,
                                    max_sites=max_sites)
            if len(doc["sites"]) < orig_n:
                n_truncated += 1
            docs.append(doc)

        if n_truncated > 0:
            print(f"  ⚠️  {n_truncated}/{len(docs)} structures truncated "
                  f"(>{max_sites} sites after primitive cell)")

        # ── Train/Test split ──
        from sklearn.model_selection import train_test_split
        train_docs, test_docs = train_test_split(
            docs, test_size=test_ratio, random_state=seed
        )
        self.train_docs = train_docs
        self.test_docs = test_docs
        print(f"  📊 Split: {len(train_docs)} train / {len(test_docs)} test")

        # ── 统计 ──
        n_sites = [len(d["sites"]) for d in docs]
        print(f"  📏 Sites per structure: "
              f"min={min(n_sites)}, max={max(n_sites)}, "
              f"mean={np.mean(n_sites):.1f}, median={np.median(n_sites):.0f}")

    @staticmethod
    def _load_task(task_name: str) -> pd.DataFrame:
        """通过 matminer 加载 Matbench 数据集。"""
        from matminer.datasets import load_dataset
        return load_dataset(task_name)

    @staticmethod
    def _find_structure_col(df: pd.DataFrame) -> str:
        """找到包含 pymatgen Structure 的列。"""
        for col in df.columns:
            sample = df.iloc[0][col]
            if hasattr(sample, 'lattice') and hasattr(sample, 'sites'):
                return col
        raise ValueError(
            f"No structure column found. Columns: {list(df.columns)}"
        )

    @staticmethod
    def _find_target_col(df: pd.DataFrame, structure_col: str) -> str:
        """找到目标列（非 structure 的列）。"""
        for col in df.columns:
            if col != structure_col:
                return col
        raise ValueError("No target column found")
