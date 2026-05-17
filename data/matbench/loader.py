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
import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple


# ═══════════════════════════════════════════════════════════════
# §1  Structure → JSON 转换
# ═══════════════════════════════════════════════════════════════

def structure_to_json(structure, target_val: Optional[float] = None,
                      max_sites: int = 126,
                      add_angles: bool = False,
                      add_bonds: bool = False,
                      max_bonds: int = 50,
                      add_composition: bool = False,
                      no_sites: bool = False) -> dict:
    """
    将 pymatgen Structure 转为精简嵌套 JSON。

    精简策略：只保留对属性预测有用的字段。
    截断策略：primitive cell → 按元素分层采样。

    Args:
        structure:  pymatgen Structure 对象
        target_val: 预测目标值 (训练时提供，测试时为 None)
        max_sites:  最大 site 数量
        add_angles: 是否添加 per-site 配位统计 (cn, avg_angle, min_angle)
        add_bonds:  是否添加全局 bonds 列表 (去重的原子对 + 距离)
        max_bonds:  最大 bond 数量 (超过则取最短的)
        add_composition: 是否添加元素比例数组 [{element, ratio}]
        no_sites: 是否移除 sites 字段（绝对坐标），仅保留相对结构信息

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
    angle_info = None
    if add_angles and len(structure) > 1:
        angle_info = _compute_angle_info(structure)

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

        # 添加配位统计
        if angle_info is not None and site_idx in angle_info:
            info = angle_info[site_idx]
            site_dict["cn"] = info["cn"]
            site_dict["avg_angle"] = info["avg_angle"]
            site_dict["min_angle"] = info["min_angle"]

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
    }

    if not no_sites:
        doc["sites"] = sites_data

    # 全局 bonds 列表
    if add_bonds and len(structure) > 1:
        doc["bonds"] = _compute_bonds(structure, max_bonds)

    # 元素比例数组 — 用数组格式使元素名作为叶子值而非 path key
    if add_composition:
        from collections import Counter
        elem_counts = Counter(str(site.specie) for site in structure)
        total = sum(elem_counts.values())
        doc["composition"] = [
            {"element": elem, "ratio": round(count / total, 4)}
            for elem, count in sorted(elem_counts.items())
        ]

    if target_val is not None:
        doc["target"] = round(float(target_val), 6)

    return doc


def _compute_angle_info(structure, cutoff_factor: float = 1.3) -> Dict[int, dict]:
    """
    计算每个 site 的配位统计：配位数、平均键角、最小键角。

    使用自适应 cutoff：最近邻距离 × cutoff_factor 定义配位壳层。
    配位数 + 平均角 + 最小角 足以区分常见配位几何：
      - cn=4, avg≈109° → 四面体
      - cn=6, avg≈90°  → 八面体
      - min_angle 远低于 avg → 严重畸变

    Args:
        structure:      pymatgen Structure
        cutoff_factor:  配位壳层 = 最近邻距离 × 此因子 (默认 1.5)

    Returns:
        {site_idx: {"cn": int, "avg_angle": float, "min_angle": float}}
    """
    n_sites = len(structure)
    if n_sites <= 1:
        return {}

    result = {}

    # 批量获取 5 Å 内的所有邻居
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 静默 pymatgen 的结构警告
        all_neighbors = structure.get_all_neighbors(r=5.0)

    for site_idx in range(n_sites):
        neighbors = all_neighbors[site_idx]
        if not neighbors:
            continue

        # 按距离排序
        neighbors_sorted = sorted(neighbors, key=lambda x: x.nn_distance)

        # 自适应 cutoff：最近邻距离 × factor
        min_dist = neighbors_sorted[0].nn_distance
        cutoff = min_dist * cutoff_factor
        coord_neighbors = [n for n in neighbors_sorted if n.nn_distance <= cutoff]
        cn = len(coord_neighbors)

        if cn < 2:
            # 配位数 < 2 无法计算角度
            result[site_idx] = {"cn": cn, "avg_angle": 0.0, "min_angle": 0.0}
            continue

        # 计算所有配位近邻对之间的键角
        center_coords = structure[site_idx].coords
        angles = []
        for i in range(len(coord_neighbors)):
            for j in range(i + 1, len(coord_neighbors)):
                vec_i = coord_neighbors[i].coords - center_coords
                vec_j = coord_neighbors[j].coords - center_coords
                norm_i = np.linalg.norm(vec_i)
                norm_j = np.linalg.norm(vec_j)
                if norm_i < 1e-10 or norm_j < 1e-10:
                    continue
                cos_angle = np.dot(vec_i, vec_j) / (norm_i * norm_j)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angles.append(np.degrees(np.arccos(cos_angle)))

        if angles:
            result[site_idx] = {
                "cn": cn,
                "avg_angle": round(float(np.mean(angles)), 1),
                "min_angle": round(float(np.min(angles)), 1),
            }
        else:
            result[site_idx] = {"cn": cn, "avg_angle": 0.0, "min_angle": 0.0}

    return result


def _compute_bonds(structure, max_bonds: int) -> List[dict]:
    """
    计算晶体中最短的 max_bonds 个原子对。

    每个 bond = {"elements": ["Ti", "O"], "dist": 1.94}
    全局去重：每个 (i, j) 对只出现一次 (i < j)。

    Returns:
        List[dict] 按距离排序
    """
    n_sites = len(structure)
    # pymatgen 距离矩阵（考虑周期性）
    dist_matrix = structure.distance_matrix  # (N, N)

    # 收集上三角的所有 pair (i < j)
    pairs = []
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            pairs.append((i, j, dist_matrix[i, j]))

    # 按距离排序，取前 max_bonds 个
    pairs.sort(key=lambda x: x[2])
    pairs = pairs[:max_bonds]

    bonds = []
    for i, j, dist in pairs:
        bonds.append({
            "elements": [str(structure[i].specie), str(structure[j].specie)],
            "dist": round(float(dist), 4),
        })

    return bonds


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
        add_angles = opts.get("add_angles", False)
        add_bonds = opts.get("add_bonds", False)
        max_bonds = opts.get("max_bonds", 50)
        add_composition = opts.get("add_composition", False)
        no_sites = opts.get("no_sites", False)

        # ── 尝试加载缓存 ──
        cache_path = self._cache_path(task_name, max_sites, add_angles,
                                      add_bonds, max_bonds,
                                      add_composition, no_sites, seed)
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
        extras = []
        if add_angles:
            extras.append("add_angles")
        if add_bonds:
            extras.append(f"add_bonds (max={max_bonds})")
        if add_composition:
            extras.append("add_composition")
        if no_sites:
            extras.append("no_sites")
        extras_msg = f", {', '.join(extras)}" if extras else ""
        print(f"  ⏳ Converting structures to JSON "
              f"(max_sites={max_sites}{extras_msg})...")
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
                add_angles=add_angles,
                add_bonds=add_bonds, max_bonds=max_bonds,
                add_composition=add_composition,
                no_sites=no_sites,
            )
            if "sites" in doc and len(doc["sites"]) < orig_n:
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
        if not no_sites:
            n_sites = [len(d["sites"]) for d in docs]
            print(f"  📏 Sites per structure: "
                  f"min={min(n_sites)}, max={max(n_sites)}, "
                  f"mean={np.mean(n_sites):.1f}, median={np.median(n_sites):.0f}")
        else:
            print(f"  📏 Sites removed (no_sites=True)")

        # ── 保存缓存 ──
        self._save_cache(cache_path)

    def _cache_path(self, task_name, max_sites, add_angles,
                    add_bonds, max_bonds, add_composition, no_sites, seed):
        """生成缓存文件路径（包含所有影响数据内容的参数）。"""
        import pathlib
        cache_dir = pathlib.Path(__file__).parent / "cache"
        angles_tag = "_angles" if add_angles else ""
        bonds_tag = f"_bonds{max_bonds}" if add_bonds else ""
        comp_tag = "_comp" if add_composition else ""
        nosites_tag = "_nosites" if no_sites else ""
        return cache_dir / f"{task_name}_s{max_sites}{angles_tag}{bonds_tag}{comp_tag}{nosites_tag}_seed{seed}.pkl"

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
