"""
Matbench 数据加载器 — 将 pymatgen Structure 转为嵌套 JSON。

数据流：
  1. 通过 matminer 加载 Matbench 任务数据
  2. pymatgen Structure → 精简嵌套 JSON dict
  3. 大结构截断：primitive cell + 按元素分层采样
  4. (可选) 添加最近邻距离信息
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
                      max_sites: int = 126,
                      add_nn_distances: bool = False,
                      n_neighbors: int = 2) -> dict:
    """
    将 pymatgen Structure 转为精简嵌套 JSON。

    精简策略：只保留对属性预测有用的字段。
    截断策略：primitive cell → 按元素分层采样。

    Args:
        structure:  pymatgen Structure 对象
        target_val: 预测目标值 (训练时提供，测试时为 None)
        max_sites:  最大 site 数量
        add_nn_distances: 是否添加最近邻距离和元素信息
        n_neighbors: 添加几个最近邻 (默认 2)

    Returns:
        嵌套 JSON dict
    """
    # ── Step 1: 尝试缩到 primitive cell ──
    if len(structure) > max_sites:
        try:
            structure = structure.get_primitive_structure()
        except Exception:
            pass  # 部分结构无法缩，保持原样

    # ── Step 2: 构建 sites 数据 ──
    # 如果需要近邻距离，提前计算全部距离矩阵
    nn_info = None
    if add_nn_distances and len(structure) > 1:
        nn_info = _compute_nn_info(structure, n_neighbors)

    sites_data = []
    for site_idx, site in enumerate(structure):
        element = str(site.specie)
        frac = site.frac_coords
        site_dict = {
            "element": element,
            "x": round(float(frac[0]), 6),
            "y": round(float(frac[1]), 6),
            "z": round(float(frac[2]), 6),
        }

        # 添加最近邻信息
        if nn_info is not None and site_idx in nn_info:
            for k, (nn_elem, nn_dist) in enumerate(nn_info[site_idx]):
                site_dict[f"nn{k+1}_dist"] = round(float(nn_dist), 4)
                site_dict[f"nn{k+1}_elem"] = nn_elem

        sites_data.append(site_dict)

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


def _compute_nn_info(structure, n_neighbors: int) -> Dict[int, List[Tuple[str, float]]]:
    """
    计算每个 site 的 k 个最近邻距离和元素。

    Returns:
        {site_idx: [(elem, dist), ...]}
    """
    result = {}
    n_sites = len(structure)
    k = min(n_neighbors, n_sites - 1)
    if k <= 0:
        return result

    # 用 structure.get_all_neighbors 批量计算
    # cutoff 设为 10 Å，覆盖绝大多数晶体的近邻
    all_neighbors = structure.get_all_neighbors(r=10.0)

    for site_idx in range(n_sites):
        neighbors = all_neighbors[site_idx]
        if not neighbors:
            continue
        # 按距离排序，取前 k 个
        neighbors_sorted = sorted(neighbors, key=lambda x: x.nn_distance)[:k]
        result[site_idx] = [
            (str(nn.specie), nn.nn_distance)
            for nn in neighbors_sorted
        ]

    return result


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
    支持 train/test split + 磁盘缓存。

    Attributes:
        task_name:   Matbench 任务名 (e.g., "matbench_dielectric")
        train_docs:  List[dict] 训练集 JSON 文档
        test_docs:   List[dict] 测试集 JSON 文档
        target_key:  预测目标的 key 名 ("target")
    """

    def __init__(self, task_name: str, max_sites: int = 126,
                 test_ratio: float = 0.2, seed: int = 42,
                 dataset_options: dict = None):
        self.task_name = task_name
        self.max_sites = max_sites
        self.target_key = "target"

        # 解析 dataset_options
        opts = dataset_options or {}
        add_nn = opts.get("add_nn_distances", False)
        n_nn = opts.get("n_neighbors", 2)

        # ── 尝试加载缓存 ──
        cache_path = self._cache_path(task_name, max_sites, add_nn, n_nn, seed)
        if cache_path.exists():
            print(f"  💾 Loading cached data: {cache_path}")
            import pickle
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            self.train_docs = cached["train_docs"]
            self.test_docs = cached["test_docs"]
            print(f"  ✅ Loaded {len(self.train_docs)} train / "
                  f"{len(self.test_docs)} test from cache")
            return

        # ── 加载数据 ──
        print(f"  📂 Loading Matbench task: {task_name}")
        df = self._load_task(task_name)
        print(f"  ✅ {len(df)} samples loaded")

        # ── Structure → JSON ──
        nn_msg = f", add_nn_distances={add_nn}" if add_nn else ""
        print(f"  ⏳ Converting structures to JSON "
              f"(max_sites={max_sites}{nn_msg})...")
        structure_col = self._find_structure_col(df)
        target_col = self._find_target_col(df, structure_col)

        docs = []
        n_truncated = 0
        for idx in range(len(df)):
            structure = df.iloc[idx][structure_col]
            target = df.iloc[idx][target_col]

            orig_n = len(structure)
            doc = structure_to_json(
                structure, target_val=target, max_sites=max_sites,
                add_nn_distances=add_nn, n_neighbors=n_nn,
            )
            if len(doc["sites"]) < orig_n:
                n_truncated += 1
            docs.append(doc)

            # 进度显示
            if (idx + 1) % 5000 == 0:
                print(f"    ... {idx+1}/{len(df)} converted")

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

        # ── 保存缓存 ──
        self._save_cache(cache_path)

    def _cache_path(self, task_name, max_sites, add_nn, n_nn, seed):
        """生成缓存文件路径（包含所有影响数据内容的参数）。"""
        import pathlib
        cache_dir = pathlib.Path(__file__).parent / "cache"
        nn_tag = f"_nn{n_nn}" if add_nn else ""
        return cache_dir / f"{task_name}_s{max_sites}{nn_tag}_seed{seed}.pkl"

    def _save_cache(self, cache_path):
        """将转换好的数据保存到磁盘。"""
        import pickle
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "train_docs": self.train_docs,
            "test_docs": self.test_docs,
        }
        with open(cache_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = cache_path.stat().st_size / 1024 / 1024
        print(f"  💾 Cached to {cache_path} ({size_mb:.1f} MB)")

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
