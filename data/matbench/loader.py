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
                      max_tokens: int = 256,
                      add_angles: bool = False,
                      add_bonds: bool = False,
                      max_bonds: int = 32,
                      add_composition: bool = False,
                      no_coords: bool = False,
                      add_ewald: bool = False,
                      add_element_props: bool = False,
                      add_comp_ewald: bool = False,
                      add_spacegroup: bool = False,
                      add_density: bool = False,
                      add_nn_stats: bool = False,
                      add_comp_nn: bool = False) -> dict:
    """
    将 pymatgen Structure 转为精简嵌套 JSON。

    精简策略：只保留对属性预测有用的字段。
    截断策略：primitive cell → 按元素分层采样。
    Token 预算：sites 优先占用，bonds 用剩余预算。

    Args:
        structure:  pymatgen Structure 对象
        target_val: 预测目标值 (训练时提供，测试时为 None)
        max_tokens: Token 预算上限（动态计算 max_sites）
        add_angles: 是否添加 per-site 配位统计 (cn, avg_angle, min_angle)
        add_bonds:  是否添加全局 bonds 列表 (去重的原子对 + 距离)
        max_bonds:  最大 bond 数量 (超过则取最短的)
        add_composition: 是否添加元素比例数组 [{element, ratio}]
        no_coords: 移除 per-site 绝对坐标 (x,y,z)。若 site 还有其他字段 (如 angles) 则保留 sites；
                   若只剩 element 则整个 sites 删除
        add_ewald: 是否添加 per-site Ewald 静电能 (ewald_energy)
        add_element_props: 是否添加 per-element 物理属性 (electronegativity, ionization_energy, electron_affinity)
        add_spacegroup: 是否添加空间群编号和晶系 (spacegroup, crystal_system)
        add_density: 是否添加密度和体积/原子 (density, vol_per_atom)
        add_nn_stats: 是否添加全局最近邻距离统计 (nn_min, nn_mean)
        add_comp_nn: 是否添加 per-element 平均最近邻距离到 composition (nn)
        add_comp_ewald: 是否添加 per-element 平均 Ewald 静电能到 composition (ewald)

    Returns:
        嵌套 JSON dict
    """
    # ── Step 0: 动态计算 token 预算（bonds 保底，sites 用剩余）──
    # 对大结构：bonds 的距离信息密度 > sites 的坐标信息密度
    # 对小结构：两者都能装下，不受影响
    tokens_per_site = 1  # element
    if not no_coords:
        tokens_per_site += 3  # x, y, z
    if add_angles:
        tokens_per_site += 3  # cn, avg_angle, min_angle
    if add_ewald:
        tokens_per_site += 1  # ewald_energy

    tokens_per_bond = 3  # elements(2) + dist(1)

    fixed_overhead = 7  # lattice(6) + target(1)
    if add_spacegroup:
        fixed_overhead += 2  # spacegroup(1) + crystal_system(1)
    if add_density:
        fixed_overhead += 2  # density(1) + vol_per_atom(1)
    if add_nn_stats:
        fixed_overhead += 2  # nn_min(1) + nn_mean(1)
    n_unique = len(set(str(site.specie) for site in structure))
    if add_composition or add_element_props or add_comp_ewald or add_comp_nn:
        tokens_per_elem = 1  # element
        if add_composition:
            tokens_per_elem += 1  # ratio
        if add_element_props:
            tokens_per_elem += 3  # en + ie + ea
        if add_comp_ewald:
            tokens_per_elem += 1  # ewald
        if add_comp_nn:
            tokens_per_elem += 1  # nn
        fixed_overhead += n_unique * tokens_per_elem

    # Bonds 保底：先预留 bonds 预算
    bonds_budget = max_bonds * tokens_per_bond if add_bonds else 0

    # Sites 用剩余预算
    max_sites = max(1, (max_tokens - fixed_overhead - bonds_budget) // tokens_per_site)

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

    ewald_energies = None
    if add_ewald:
        ewald_energies = _compute_ewald_site_energies(structure)

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

        # 添加 Ewald 静电能
        if ewald_energies is not None and site_idx < len(ewald_energies):
            site_dict["ewald_energy"] = ewald_energies[site_idx]

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

    # 空间群 + 晶系（对称性信息，零计算成本）
    if add_spacegroup:
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            sga = SpacegroupAnalyzer(structure)
            doc["spacegroup"] = sga.get_space_group_number()
            doc["crystal_system"] = sga.get_crystal_system()
        except Exception:
            pass  # 极少数畸变结构无法确定空间群

    # 密度 + 体积/原子（基本物理量，零计算成本）
    if add_density:
        doc["density"] = round(float(structure.density), 4)
        doc["vol_per_atom"] = round(float(structure.volume / len(structure)), 4)

    # 全局最近邻距离统计（轨道重叠的直接代理量，与 band gap 相关 r=-0.46）
    if add_nn_stats and len(structure) > 1:
        import numpy as np
        dm = structure.distance_matrix  # (N, N) 考虑周期性
        np.fill_diagonal(dm, np.inf)
        nn_dists = dm.min(axis=1)  # 每个 site 的最近邻距离
        doc["nn_min"] = round(float(nn_dists.min()), 4)
        doc["nn_mean"] = round(float(nn_dists.mean()), 4)

    # ── 处理 no_coords：条件性删除绝对坐标 ──
    if no_coords:
        for site_dict in sites_data:
            site_dict.pop("x", None)
            site_dict.pop("y", None)
            site_dict.pop("z", None)
        # 如果 site 只剩 element，和 composition 冗余 → 不加 sites
        has_extras = any(len(s) > 1 for s in sites_data)
        if has_extras:
            doc["sites"] = sites_data
    else:
        doc["sites"] = sites_data

    # Bonds：用保底预算 + 小结构释放的 site 名额
    if add_bonds and len(structure) > 1:
        tokens_used = fixed_overhead + len(sites_data) * tokens_per_site
        remaining = max_tokens - tokens_used
        effective_max_bonds = min(max_bonds, max(0, remaining // tokens_per_bond))
        if effective_max_bonds > 0:
            doc["bonds"] = _compute_bonds(structure, effective_max_bonds)

    # 组成信息 — per-unique-element（比例 + 物理属性 + Ewald 均值 + NN 均值 合并）
    if add_composition or add_element_props or add_comp_ewald or add_comp_nn:
        from collections import Counter, defaultdict
        elem_counts = Counter(str(site.specie) for site in structure)
        total = sum(elem_counts.values())

        # per-element 平均 Ewald
        ewald_by_elem = {}
        if add_comp_ewald:
            all_ewald = _compute_ewald_site_energies(structure)
            if all_ewald is not None:
                sums = defaultdict(float)
                counts = defaultdict(int)
                for site_idx, site in enumerate(structure):
                    sym = str(site.specie)
                    sums[sym] += all_ewald[site_idx]
                    counts[sym] += 1
                ewald_by_elem = {
                    sym: round(sums[sym] / counts[sym], 4)
                    for sym in sums
                }

        # per-element 平均最近邻距离
        nn_by_elem = {}
        if add_comp_nn and len(structure) > 1:
            import numpy as _np
            dm = structure.distance_matrix
            _np.fill_diagonal(dm, _np.inf)
            nn_dists = dm.min(axis=1)  # 每个 site 的最近邻距离
            from collections import defaultdict as _dd
            nn_sums = _dd(float)
            nn_counts = _dd(int)
            for site_idx, site in enumerate(structure):
                sym = str(site.specie)
                nn_sums[sym] += nn_dists[site_idx]
                nn_counts[sym] += 1
            nn_by_elem = {
                sym: round(nn_sums[sym] / nn_counts[sym], 4)
                for sym in nn_sums
            }

        comp_list = []
        for elem, count in sorted(elem_counts.items()):
            entry = {"element": elem}
            if add_composition:
                entry["ratio"] = round(count / total, 4)
            if add_element_props:
                from pymatgen.core import Element as _Element
                el = _Element(elem)
                if el.X is not None:
                    entry["en"] = round(float(el.X), 2)
                if el.ionization_energy is not None:
                    entry["ie"] = round(float(el.ionization_energy), 2)
                if el.electron_affinity is not None:
                    entry["ea"] = round(float(el.electron_affinity), 2)
            if add_comp_ewald and elem in ewald_by_elem:
                entry["ewald"] = ewald_by_elem[elem]
            if add_comp_nn and elem in nn_by_elem:
                entry["nn"] = nn_by_elem[elem]
            comp_list.append(entry)
        doc["composition"] = comp_list

    if target_val is not None:
        doc["target"] = round(float(target_val), 6)

    return doc


def _compute_ewald_site_energies(structure,
                                  _bva_max_sites: int = 50,
                                  _guess_max_reduced: int = 30,
                                  ) -> Optional[List[float]]:
    """
    计算每个 site 的 Ewald 静电能（行和形式），三层策略保证不卡死。

    策略：
      1. sites ≤ _bva_max_sites  → BVAnalyzer（键价分析，物理最准）
      2. reduced_atoms ≤ _guess_max_reduced → oxi_state_guesses(max_sites=-1)
         （组成穷举，保证电荷中性，先 reduce 到最简再搜索）
      3. 都失败或结构太大 → 返回 None（跳过 Ewald 特征）

    Args:
        structure: pymatgen Structure
        _bva_max_sites: BVAnalyzer 层的 site 数阈值（默认 50）
        _guess_max_reduced: guess 层的 reduced 原子数阈值（默认 30）

    Returns:
        List[float] 长度 = len(structure)，或 None（计算失败时）
    """
    if len(structure) < 1:
        return None

    s_oxi = None

    # ── 层 1: BVAnalyzer（小结构优先，物理最准）──
    if len(structure) <= _bva_max_sites:
        try:
            import os, sys
            from pymatgen.core.bond_valence import BVAnalyzer
            s_oxi = structure.copy()
            bva = BVAnalyzer()
            # 屏蔽 spglib 的 C 层 stderr 警告
            stderr_fd = sys.stderr.fileno()
            old_stderr = os.dup(stderr_fd)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, stderr_fd)
            try:
                valences = bva.get_valences(s_oxi)
            finally:
                os.dup2(old_stderr, stderr_fd)
                os.close(old_stderr)
                os.close(devnull)
            s_oxi.add_oxidation_state_by_site(valences)
        except Exception:
            s_oxi = None

    # ── 层 2: oxi_state_guesses（预检 reduced atoms 防卡死）──
    if s_oxi is None:
        reduced_n = int(structure.composition.reduced_composition.num_atoms)
        if reduced_n <= _guess_max_reduced:
            try:
                s_oxi = structure.copy()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    s_oxi.add_oxidation_state_by_guess(max_sites=-1)
            except Exception:
                s_oxi = None

    # ── 层 3: 都失败 → 跳过 ──
    if s_oxi is None:
        return None

    # ── Ewald 求和 ──
    try:
        from pymatgen.analysis.ewald import EwaldSummation
        ewald = EwaldSummation(s_oxi)
        site_energies = ewald.total_energy_matrix.sum(axis=1)
        return [round(float(e), 4) for e in site_energies]
    except Exception:
        return None


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

    def __init__(self, task_name: str, max_tokens: int = 256,
                 test_ratio: float = 0.2, seed: int = 42,
                 dataset_options: dict = None,
                 max_cpu_workers: int = 32):
        self.task_name = task_name
        self.max_tokens = max_tokens
        self.target_key = "target"
        self.max_cpu_workers = max_cpu_workers

        # 解析 dataset_options
        opts = dataset_options or {}
        add_angles = opts.get("add_angles", False)
        add_bonds = opts.get("add_bonds", False)
        max_bonds = opts.get("max_bonds", 32)
        add_composition = opts.get("add_composition", False)
        no_coords = opts.get("no_coords", False)
        add_ewald = opts.get("add_ewald", False)
        add_element_props = opts.get("add_element_props", False)
        add_comp_ewald = opts.get("add_comp_ewald", False)
        add_spacegroup = opts.get("add_spacegroup", False)
        add_density = opts.get("add_density", False)
        add_nn_stats = opts.get("add_nn_stats", False)
        add_comp_nn = opts.get("add_comp_nn", False)

        # ── 尝试加载缓存 ──
        cache_path = self._cache_path(task_name, max_tokens, add_angles,
                                      add_bonds, max_bonds,
                                      add_composition, no_coords,
                                      add_ewald, add_element_props,
                                      add_comp_ewald, add_spacegroup, add_nn_stats,
                                      add_density, add_comp_nn, seed)
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
        if no_coords:
            extras.append("no_coords")
        if add_ewald:
            extras.append("add_ewald")
        if add_element_props:
            extras.append("add_element_props")
        if add_comp_ewald:
            extras.append("add_comp_ewald")
        if add_spacegroup:
            extras.append("add_spacegroup")
        if add_density:
            extras.append("add_density")
        if add_nn_stats:
            extras.append("add_nn_stats")
        if add_comp_nn:
            extras.append("add_comp_nn")
        extras_msg = f", {', '.join(extras)}" if extras else ""
        print(f"  ⏳ Converting structures to JSON "
              f"(max_tokens={max_tokens}{extras_msg})...")
        structure_col = self._find_structure_col(df)
        target_col = self._find_target_col(df, structure_col)

        # ── 并行转换 ──
        from joblib import Parallel, delayed
        from tqdm import tqdm

        structures = [df.iloc[i][structure_col] for i in range(len(df))]
        targets = [df.iloc[i][target_col] for i in range(len(df))]
        orig_sizes = [len(s) for s in structures]

        def _convert_one(s, t):
            return structure_to_json(
                s, target_val=t, max_tokens=max_tokens,
                add_angles=add_angles,
                add_bonds=add_bonds, max_bonds=max_bonds,
                add_composition=add_composition,
                no_coords=no_coords,
                add_ewald=add_ewald,
                add_element_props=add_element_props,
                add_comp_ewald=add_comp_ewald,
                add_spacegroup=add_spacegroup,
                add_density=add_density,
                add_nn_stats=add_nn_stats,
                add_comp_nn=add_comp_nn,
            )

        import os
        n_workers = min(self.max_cpu_workers, os.cpu_count() or 1)
        docs = Parallel(n_jobs=n_workers, backend="loky")(
            delayed(_convert_one)(s, t)
            for s, t in tqdm(zip(structures, targets),
                             total=len(structures),
                             desc="  🔄 Converting")
        )

        n_truncated = sum(
            1 for doc, orig_n in zip(docs, orig_sizes)
            if "sites" in doc and len(doc["sites"]) < orig_n
        )
        if n_truncated > 0:
            print(f"  ⚠️  {n_truncated}/{len(docs)} structures truncated "
                  f"(dynamic max_sites, token budget={max_tokens})")

        # Ewald 覆盖率统计
        if add_ewald:
            n_ewald = sum(
                1 for doc in docs
                if "sites" in doc and any(
                    "ewald_energy" in s for s in doc["sites"]
                )
            )
            print(f"  ⚡ Ewald coverage: {n_ewald}/{len(docs)} "
                  f"({100*n_ewald/len(docs):.1f}%)")

        # ── Train/Test split ──
        from sklearn.model_selection import train_test_split
        train_docs, test_docs = train_test_split(
            docs, test_size=test_ratio, random_state=seed
        )
        self.train_docs = train_docs
        self.test_docs = test_docs
        print(f"  📊 Split: {len(train_docs)} train / {len(test_docs)} test")

        # ── 统计 ──
        n_sites = [len(d["sites"]) for d in docs if "sites" in d]
        if n_sites:
            print(f"  📏 Sites per structure: "
                  f"min={min(n_sites)}, max={max(n_sites)}, "
                  f"mean={np.mean(n_sites):.1f}, median={np.median(n_sites):.0f}")
        elif no_coords:
            print(f"  📏 Sites removed (no per-site extras)")

        # ── 保存缓存 ──
        self._save_cache(cache_path)

    def _cache_path(self, task_name, max_tokens, add_angles,
                    add_bonds, max_bonds, add_composition, no_coords,
                    add_ewald, add_element_props, add_comp_ewald,
                    add_spacegroup, add_nn_stats, add_density,
                    add_comp_nn, seed):
        """生成缓存文件路径（包含所有影响数据内容的参数）。"""
        import pathlib
        cache_dir = pathlib.Path(__file__).parent / "cache"
        angles_tag = "_angles" if add_angles else ""
        bonds_tag = f"_bonds{max_bonds}" if add_bonds else ""
        comp_tag = "_comp" if add_composition else ""
        nocoords_tag = "_nocoords" if no_coords else ""
        ewald_tag = "_ewald" if add_ewald else ""
        elprops_tag = "_elprops" if add_element_props else ""
        comp_ewald_tag = "_cewald" if add_comp_ewald else ""
        sg_tag = "_sg" if add_spacegroup else ""
        dens_tag = "_dens" if add_density else ""
        nn_tag = "_nn" if add_nn_stats else ""
        cnn_tag = "_cnn" if add_comp_nn else ""
        return cache_dir / f"{task_name}_t{max_tokens}{angles_tag}{bonds_tag}{comp_tag}{nocoords_tag}{ewald_tag}{elprops_tag}{comp_ewald_tag}{sg_tag}{dens_tag}{nn_tag}{cnn_tag}_seed{seed}.pkl"

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
