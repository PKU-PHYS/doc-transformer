"""
Matbench 数据加载器 — 将 pymatgen Structure 转为嵌套 JSON。

数据流：
  1. 通过 matminer 加载 Matbench 任务数据
  2. pymatgen Structure → 精简嵌套 JSON dict
  3. 大结构截断：primitive cell + 按元素分层采样
  4. (可选) 添加最近邻距离信息
"""

import hashlib
import json
import math
import random
import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Sequence, Tuple

from data.matbench.splits import (
    coerce_ids_to_index_type,
    official_fold_ids,
    split_internal_train_val,
)


# ═══════════════════════════════════════════════════════════════
# §1  Structure → JSON 转换
# ═══════════════════════════════════════════════════════════════

def structure_to_json(structure, target_val: Optional[float] = None,
                      max_tokens: int = 256,
                      add_angles: bool = False,
                      add_bonds: bool = False,
                      max_bonds: int = 32,
                      add_composition: bool = False,
                      drop_coords: bool = False,
                      add_ewald: bool = False,
                      add_element_props: bool = False,
                      add_comp_ewald: bool = False,
                      add_spacegroup: bool = False,
                      add_density: bool = False,
                      add_nn_stats: bool = False,
                      add_comp_nn: bool = False,
                      add_comp_ewald_stats: bool = False,
                      add_comp_nn_stats: bool = False,
                      add_mean_bonds: bool = False) -> dict:
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
        drop_coords: 移除 per-site 绝对坐标 (x,y,z)。若 site 还有其他字段 (如 angles) 则保留 sites；
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
    if not drop_coords:
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
        if add_comp_ewald_stats:
            tokens_per_elem += 3  # ewald_std + ewald_min + ewald_max
        if add_comp_nn:
            tokens_per_elem += 1  # nn
        if add_comp_nn_stats:
            tokens_per_elem += 3  # nn_std + nn_min + nn_max
        fixed_overhead += n_unique * tokens_per_elem
    if add_mean_bonds:
        n_pairs = n_unique * (n_unique + 1) // 2
        fixed_overhead += n_pairs * 3  # elements(2) + dist(1) per pair

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
    if add_ewald or add_comp_ewald:
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

        # 添加 Ewald 静电能（仅 add_ewald 时写入 per-site，comp_ewald 只需聚合）
        if add_ewald and ewald_energies is not None and site_idx < len(ewald_energies):
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
    # 提前计算 nn_dists，供 add_nn_stats 和 add_comp_nn 共用
    nn_dists = None
    if (add_nn_stats or add_comp_nn) and len(structure) > 1:
        dm = structure.distance_matrix  # (N, N) 考虑周期性
        np.fill_diagonal(dm, np.inf)
        nn_dists = dm.min(axis=1)  # 每个 site 的最近邻距离

    if add_nn_stats and nn_dists is not None:
        doc["nn_min"] = round(float(nn_dists.min()), 4)
        doc["nn_mean"] = round(float(nn_dists.mean()), 4)

    # ── 处理 drop_coords：条件性删除绝对坐标 ──
    if drop_coords:
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

        # per-element Ewald（复用已计算的 ewald_energies）
        ewald_by_elem = {}
        if add_comp_ewald and ewald_energies is not None:
            if add_comp_ewald_stats:
                # 需要统计量：用 list 收集
                vals_by_elem = defaultdict(list)
                for site_idx, site in enumerate(structure):
                    sym = str(site.specie)
                    vals_by_elem[sym].append(ewald_energies[site_idx])
                ewald_by_elem = {
                    sym: {
                        "mean": round(float(np.mean(vals)), 4),
                        "std":  round(float(np.std(vals)), 4),
                        "min":  round(float(np.min(vals)), 4),
                        "max":  round(float(np.max(vals)), 4),
                    }
                    for sym, vals in vals_by_elem.items()
                }
            else:
                # 只需均值
                sums = defaultdict(float)
                counts = defaultdict(int)
                for site_idx, site in enumerate(structure):
                    sym = str(site.specie)
                    sums[sym] += ewald_energies[site_idx]
                    counts[sym] += 1
                ewald_by_elem = {
                    sym: round(sums[sym] / counts[sym], 4)
                    for sym in sums
                }

        # per-element 最近邻距离（复用已计算的 nn_dists）
        nn_by_elem = {}
        if add_comp_nn and nn_dists is not None:
            if add_comp_nn_stats:
                nn_vals_by_elem = defaultdict(list)
                for site_idx, site in enumerate(structure):
                    sym = str(site.specie)
                    nn_vals_by_elem[sym].append(nn_dists[site_idx])
                nn_by_elem = {
                    sym: {
                        "mean": round(float(np.mean(vals)), 4),
                        "std":  round(float(np.std(vals)), 4),
                        "min":  round(float(np.min(vals)), 4),
                        "max":  round(float(np.max(vals)), 4),
                    }
                    for sym, vals in nn_vals_by_elem.items()
                }
            else:
                nn_sums = defaultdict(float)
                nn_counts = defaultdict(int)
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
                if add_comp_ewald_stats:
                    stats = ewald_by_elem[elem]
                    entry["ewald"] = stats["mean"]
                    entry["ewald_std"] = stats["std"]
                    entry["ewald_min"] = stats["min"]
                    entry["ewald_max"] = stats["max"]
                else:
                    entry["ewald"] = ewald_by_elem[elem]
            if add_comp_nn and elem in nn_by_elem:
                if add_comp_nn_stats:
                    stats = nn_by_elem[elem]
                    entry["nn"] = stats["mean"]
                    entry["nn_std"] = stats["std"]
                    entry["nn_min"] = stats["min"]
                    entry["nn_max"] = stats["max"]
                else:
                    entry["nn"] = nn_by_elem[elem]
            comp_list.append(entry)
        doc["composition"] = comp_list

    # 元素对平均键距
    if add_mean_bonds and len(structure) > 1:
        doc["bonds_stats"] = _compute_mean_bonds(structure)

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
        cutoff_factor:  配位壳层 = 最近邻距离 × 此因子 (默认 1.3)

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


def _compute_mean_bonds(structure, cutoff_factor: float = 1.3) -> List[dict]:
    """
    计算每对元素之间的平均配位键距。

    只统计配位壳层内的真实化学键（最近邻距离 × cutoff_factor），
    按元素对分组取平均距离，输出格式与 _compute_bonds 一致。

    Args:
        structure:      pymatgen Structure
        cutoff_factor:  配位壳层 = 最近邻距离 × 此因子 (默认 1.3)

    Returns:
        List[dict] — [{"elements": ["O", "Ti"], "dist": 1.95}, ...] 按距离排序
    """
    from collections import defaultdict
    n_sites = len(structure)
    if n_sites <= 1:
        return []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        all_neighbors = structure.get_all_neighbors(r=5.0)

    pair_dists = defaultdict(list)  # (elem1, elem2) sorted → [dist, ...]

    for site_idx in range(n_sites):
        neighbors = all_neighbors[site_idx]
        if not neighbors:
            continue

        center_elem = str(structure[site_idx].specie)
        neighbors_sorted = sorted(neighbors, key=lambda x: x.nn_distance)

        min_dist = neighbors_sorted[0].nn_distance
        cutoff = min_dist * cutoff_factor
        coord_neighbors = [n for n in neighbors_sorted if n.nn_distance <= cutoff]

        for n in coord_neighbors:
            n_elem = str(n.specie)
            pair_key = tuple(sorted([center_elem, n_elem]))
            pair_dists[pair_key].append(n.nn_distance)

    # 按平均距离排序输出
    result = []
    for (e1, e2), dists in pair_dists.items():
        result.append({
            "elements": [e1, e2],
            "dist": round(float(np.mean(dists)), 4),
        })
    result.sort(key=lambda x: x["dist"])
    return result


def _stable_sample_rng(sites: List[dict], max_n: int) -> random.Random:
    """从站点内容派生确定性 RNG，避免依赖进程级 random 状态。"""
    payload = json.dumps(
        {"max_n": max_n, "sites": sites},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _stratified_sample(sites: List[dict], max_n: int) -> List[dict]:
    """按元素类型分层采样，保持化学组成比例且结果可复现。"""
    if max_n <= 0:
        return []
    if len(sites) <= max_n:
        return list(sites)

    # 按元素分组
    by_element: Dict[str, List[dict]] = {}
    for s in sites:
        by_element.setdefault(s["element"], []).append(s)

    rng = _stable_sample_rng(sites, max_n)
    elements = sorted(by_element)

    # 元素种类比预算还多时，不再按插入顺序硬截，改为确定性随机抽元素。
    if len(elements) > max_n:
        selected_elements = set(rng.sample(elements, max_n))
        quotas = {elem: 1 for elem in elements if elem in selected_elements}
    else:
        total = len(sites)
        quotas = {elem: 1 for elem in elements}
        remaining = max_n - len(elements)
        ideal_quota = {
            elem: len(by_element[elem]) / total * max_n
            for elem in elements
        }

        # 逐个名额补给"当前配额低于理想比例最多"的元素,同时尊重每组容量。
        while remaining > 0:
            candidates = [
                elem for elem in elements
                if quotas[elem] < len(by_element[elem])
            ]
            if not candidates:
                break
            elem = max(candidates, key=lambda e: (ideal_quota[e] - quotas[e], e))
            quotas[elem] += 1
            remaining -= 1

    selected_indices = set()
    for elem, quota in quotas.items():
        indices = [i for i, site in enumerate(sites) if site["element"] == elem]
        if len(indices) <= quota:
            chosen_indices = indices
        else:
            chosen_indices = rng.sample(indices, quota)
        selected_indices.update(chosen_indices)

    return [site for i, site in enumerate(sites) if i in selected_indices]


# ═══════════════════════════════════════════════════════════════
# §2  MatbenchLoader
# ═══════════════════════════════════════════════════════════════

class MatbenchLoader:
    """
    Matbench 任务数据加载器。

    加载 matminer 数据 → 转为嵌套 JSON docs 列表。
    支持 official fold 的 train/val/test split + 磁盘缓存。

    Attributes:
        task_name:   Matbench 任务名 (e.g., "matbench_dielectric")
        train_docs:  List[dict] 训练集 JSON 文档
        val_docs:    List[dict] 验证集 JSON 文档（仅来自 official train+val）
        test_docs:   List[dict] 测试集 JSON 文档
        target_key:  预测目标的 key 名 ("target")
    """

    def __init__(self, task_name: str, max_tokens: int = 256,
                 test_ratio: float = 0.2, seed: int = 42,
                 dataset_options: dict = None,
                 max_cpu_workers: int = 32,
                 split_strategy: str = "official",
                 fold: int = 0,
                 val_ratio: float = 0.1,
                 include_test_targets: bool = False):
        self.task_name = task_name
        self.max_tokens = max_tokens
        self.target_key = "target"
        self.max_cpu_workers = max_cpu_workers
        self.split_strategy = split_strategy
        self.fold = fold
        self.val_ratio = val_ratio
        self.include_test_targets = include_test_targets

        # 解析 dataset_options
        opts = dataset_options or {}
        add_angles = opts.get("add_angles", False)
        add_bonds = opts.get("add_bonds", False)
        max_bonds = opts.get("max_bonds", 32)
        add_composition = opts.get("add_composition", False)
        drop_coords = opts.get("drop_coords", False)
        add_ewald = opts.get("add_ewald", False)
        add_element_props = opts.get("add_element_props", False)
        add_comp_ewald = opts.get("add_comp_ewald", False)
        add_spacegroup = opts.get("add_spacegroup", False)
        add_density = opts.get("add_density", False)
        add_nn_stats = opts.get("add_nn_stats", False)
        add_comp_nn = opts.get("add_comp_nn", False)
        add_comp_ewald_stats = opts.get("add_comp_ewald_stats", False)
        add_comp_nn_stats = opts.get("add_comp_nn_stats", False)
        add_mean_bonds = opts.get("add_mean_bonds", False)
        add_target_prior = opts.get("add_target_prior", False)

        # ── 尝试加载缓存 ──
        cache_path = self._cache_path(task_name, max_tokens, add_angles,
                                      add_bonds, max_bonds,
                                      add_composition, drop_coords,
                                      add_ewald, add_element_props,
                                      add_comp_ewald, add_spacegroup, add_nn_stats,
                                      add_density, add_comp_nn,
                                      add_comp_ewald_stats, add_comp_nn_stats,
                                      add_mean_bonds, add_target_prior, seed,
                                      split_strategy, fold, val_ratio, test_ratio,
                                      include_test_targets)
        if cache_path.exists():
            print(f"  💾 Loading cached data: {cache_path}")
            import pickle
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            self.train_docs = cached["train_docs"]
            self.val_docs = cached.get("val_docs", [])
            self.test_docs = cached["test_docs"]
            self.split_metadata = cached.get("split_metadata", {})
            print(f"  ✅ Loaded {len(self.train_docs)} train / "
                  f"{len(self.val_docs)} val / {len(self.test_docs)} test from cache")
            return
        if add_target_prior:
            base_cache_path = self._cache_path(
                task_name, max_tokens, add_angles,
                add_bonds, max_bonds,
                add_composition, drop_coords,
                add_ewald, add_element_props,
                add_comp_ewald, add_spacegroup, add_nn_stats,
                add_density, add_comp_nn,
                add_comp_ewald_stats, add_comp_nn_stats,
                add_mean_bonds, False, seed,
                split_strategy, fold, val_ratio, test_ratio,
                include_test_targets,
            )
            if base_cache_path.exists():
                print(f"  💾 Loading base cached data: {base_cache_path}")
                import pickle
                with open(base_cache_path, "rb") as f:
                    cached = pickle.load(f)
                self.train_docs = cached["train_docs"]
                self.val_docs = cached.get("val_docs", [])
                self.test_docs = cached["test_docs"]
                self.split_metadata = cached.get("split_metadata", {})
                self._add_target_priors(seed=seed)
                print(f"  ✅ Loaded {len(self.train_docs)} train / "
                      f"{len(self.val_docs)} val / {len(self.test_docs)} test "
                      "from base cache + target_prior")
                self._save_cache(cache_path)
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
        if drop_coords:
            extras.append("drop_coords")
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
        if add_comp_ewald_stats:
            extras.append("add_comp_ewald_stats")
        if add_comp_nn_stats:
            extras.append("add_comp_nn_stats")
        if add_mean_bonds:
            extras.append("add_mean_bonds")
        if add_target_prior:
            extras.append("add_target_prior")
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
                drop_coords=drop_coords,
                add_ewald=add_ewald,
                add_element_props=add_element_props,
                add_comp_ewald=add_comp_ewald,
                add_spacegroup=add_spacegroup,
                add_density=add_density,
                add_nn_stats=add_nn_stats,
                add_comp_nn=add_comp_nn,
                add_comp_ewald_stats=add_comp_ewald_stats,
                add_comp_nn_stats=add_comp_nn_stats,
                add_mean_bonds=add_mean_bonds,
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

        # ── Train/Val/Test split ──
        self._split_docs(
            df=df,
            docs=docs,
            target_col=target_col,
            test_ratio=test_ratio,
            seed=seed,
        )
        if add_target_prior:
            self._add_target_priors(seed=seed)
        print(f"  📊 Split: {len(self.train_docs)} train / "
              f"{len(self.val_docs)} val / {len(self.test_docs)} test")

        # ── 统计 ──
        n_sites = [len(d["sites"]) for d in docs if "sites" in d]
        if n_sites:
            print(f"  📏 Sites per structure: "
                  f"min={min(n_sites)}, max={max(n_sites)}, "
                  f"mean={np.mean(n_sites):.1f}, median={np.median(n_sites):.0f}")
        elif drop_coords:
            print(f"  📏 Sites removed (no per-site extras)")

        # ── 保存缓存 ──
        self._save_cache(cache_path)

    def _cache_path(self, task_name, max_tokens, add_angles,
                    add_bonds, max_bonds, add_composition, drop_coords,
                    add_ewald, add_element_props, add_comp_ewald,
                    add_spacegroup, add_nn_stats, add_density,
                    add_comp_nn,
                    add_comp_ewald_stats, add_comp_nn_stats,
                    add_mean_bonds, add_target_prior, seed,
                    split_strategy, fold, val_ratio, test_ratio,
                    include_test_targets):
        """生成缓存文件路径（包含所有影响数据内容的参数）。"""
        import pathlib
        cache_dir = pathlib.Path(__file__).parent / "cache"
        angles_tag = "_angles" if add_angles else ""
        bonds_tag = f"_bonds{max_bonds}" if add_bonds else ""
        comp_tag = "_comp" if add_composition else ""
        dropcoords_tag = "_dropcoords" if drop_coords else ""
        ewald_tag = "_ewald" if add_ewald else ""
        elprops_tag = "_elprops" if add_element_props else ""
        comp_ewald_tag = "_cewald" if add_comp_ewald else ""
        sg_tag = "_sg" if add_spacegroup else ""
        dens_tag = "_dens" if add_density else ""
        nn_tag = "_nn" if add_nn_stats else ""
        cnn_tag = "_cnn" if add_comp_nn else ""
        cewalds_tag = "_cewalds" if add_comp_ewald_stats else ""
        cnns_tag = "_cnns" if add_comp_nn_stats else ""
        mbonds_tag = "_mbonds" if add_mean_bonds else ""
        tprior_tag = "_tprior" if add_target_prior else ""
        if split_strategy == "official":
            split_tag = f"_official_f{fold}_val{val_ratio:g}"
        elif split_strategy == "random":
            split_tag = f"_random_test{test_ratio:g}"
        else:
            raise ValueError(f"Unknown split_strategy={split_strategy!r}")
        test_target_tag = "_testtargets" if include_test_targets else "_blindtest"
        return cache_dir / f"{task_name}_t{max_tokens}{angles_tag}{bonds_tag}{comp_tag}{dropcoords_tag}{ewald_tag}{elprops_tag}{comp_ewald_tag}{sg_tag}{dens_tag}{nn_tag}{cnn_tag}{cewalds_tag}{cnns_tag}{mbonds_tag}{tprior_tag}{split_tag}{test_target_tag}_seed{seed}.pkl"

    def _split_docs(self, df, docs, target_col, test_ratio: float, seed: int):
        """Populate train_docs/val_docs/test_docs with no test leakage."""
        doc_by_id = dict(zip(df.index, docs))
        target_by_id = df[target_col]

        if self.split_strategy == "official":
            train_val_ids, test_ids, fold_key = official_fold_ids(
                self.task_name,
                self.fold,
            )
            train_val_ids = coerce_ids_to_index_type(train_val_ids, df.index)
            test_ids = coerce_ids_to_index_type(test_ids, df.index)
            train_ids, val_ids = split_internal_train_val(
                train_val_ids,
                targets_by_id=target_by_id,
                val_ratio=self.val_ratio,
                seed=seed,
            )
            self.split_metadata = {
                "strategy": "official",
                "fold": self.fold,
                "fold_key": fold_key,
                "val_ratio": self.val_ratio,
                "seed": seed,
            }
        elif self.split_strategy == "random":
            from sklearn.model_selection import train_test_split

            train_val_ids, test_ids = train_test_split(
                list(df.index), test_size=test_ratio, random_state=seed
            )
            train_ids, val_ids = split_internal_train_val(
                train_val_ids,
                targets_by_id=target_by_id,
                val_ratio=self.val_ratio,
                seed=seed,
            )
            self.split_metadata = {
                "strategy": "random",
                "test_ratio": test_ratio,
                "val_ratio": self.val_ratio,
                "seed": seed,
            }
        else:
            raise ValueError(f"Unknown split_strategy={self.split_strategy!r}")

        overlap_train_val = set(train_ids) & set(val_ids)
        overlap_train_test = set(train_ids) & set(test_ids)
        overlap_val_test = set(val_ids) & set(test_ids)
        if overlap_train_val or overlap_train_test or overlap_val_test:
            raise RuntimeError(
                "Split leakage detected: "
                f"train∩val={len(overlap_train_val)}, "
                f"train∩test={len(overlap_train_test)}, "
                f"val∩test={len(overlap_val_test)}"
            )

        self.train_docs = [doc_by_id[i] for i in train_ids]
        self.val_docs = [doc_by_id[i] for i in val_ids]
        self.test_docs = [doc_by_id[i] for i in test_ids]
        if not self.include_test_targets:
            self.test_docs = [
                {k: v for k, v in doc.items() if k != self.target_key}
                for doc in self.test_docs
            ]
            self.split_metadata["test_targets"] = "omitted"
        else:
            self.split_metadata["test_targets"] = "included"

    @staticmethod
    def _target_prior_elements(train_docs: Sequence[dict]) -> List[str]:
        """Element vocabulary for the train-only target-prior model."""
        return sorted({
            entry["element"]
            for doc in train_docs
            for entry in doc.get("composition", [])
            if "element" in entry
        })

    @staticmethod
    def _target_prior_features(docs: Sequence[dict],
                               elements: Sequence[str]) -> np.ndarray:
        """Build compact composition/lattice features without reading targets."""
        elem_to_idx = {elem: i for i, elem in enumerate(elements)}
        p = len(elements)
        # ratio, ratio^2, ratio*ewald, ratio*nn, present, plus globals.
        n_global = 24
        x = np.zeros((len(docs), p * 5 + n_global), dtype=np.float32)

        for row, doc in enumerate(docs):
            ratios = []
            ewalds = []
            nns = []
            for entry in doc.get("composition", []):
                elem = entry.get("element")
                idx = elem_to_idx.get(elem)
                if idx is None:
                    continue
                ratio = float(entry.get("ratio", 0.0))
                ewald = float(entry.get("ewald", 0.0))
                nn = float(entry.get("nn", 0.0))
                x[row, idx] = ratio
                x[row, p + idx] = ratio * ratio
                x[row, 2 * p + idx] = ratio * ewald
                x[row, 3 * p + idx] = ratio * nn
                x[row, 4 * p + idx] = 1.0
                ratios.append(ratio)
                ewalds.append(ewald)
                nns.append(nn)

            off = p * 5
            lattice = doc.get("lattice", {})
            a = float(lattice.get("a", 0.0))
            b = float(lattice.get("b", 0.0))
            c = float(lattice.get("c", 0.0))
            alpha = float(lattice.get("alpha", 0.0))
            beta = float(lattice.get("beta", 0.0))
            gamma = float(lattice.get("gamma", 0.0))
            x[row, off:off + 6] = [a, b, c, alpha, beta, gamma]
            positive_lengths = [v for v in (a, b, c) if v > 0.0]
            if len(positive_lengths) == 3:
                x[row, off + 6] = a * b * c
                x[row, off + 7] = max(positive_lengths) / min(positive_lengths)

            if ratios:
                ratios_arr = np.asarray(ratios, dtype=np.float32)
                ewalds_arr = np.asarray(ewalds, dtype=np.float32)
                nns_arr = np.asarray(nns, dtype=np.float32)
                x[row, off + 8] = len(ratios_arr)
                x[row, off + 9] = float(np.sum(ratios_arr * ratios_arr))
                x[row, off + 10] = float(np.max(ratios_arr))
                x[row, off + 11] = float(
                    -np.sum(ratios_arr * np.log(np.maximum(ratios_arr, 1e-12)))
                )
                for base, vals in ((12, ewalds_arr), (16, nns_arr)):
                    x[row, off + base] = float(np.sum(ratios_arr * vals))
                    x[row, off + base + 1] = float(np.mean(vals))
                    x[row, off + base + 2] = float(np.min(vals))
                    x[row, off + base + 3] = float(np.max(vals))

            x[row, off + 20] = float(doc.get("density", 0.0) or 0.0)
            x[row, off + 21] = float(doc.get("vol_per_atom", 0.0) or 0.0)
            x[row, off + 22] = float(doc.get("nn_min", 0.0) or 0.0)
            x[row, off + 23] = float(doc.get("nn_mean", 0.0) or 0.0)

        return x

    @staticmethod
    def _make_target_prior_model(seed: int):
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=160,
            learning_rate=0.06,
            l2_regularization=0.1,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            random_state=seed,
            validation_fraction=None,
            early_stopping=False,
        )

    @staticmethod
    def _with_target_prior(docs: Sequence[dict],
                           predictions: Sequence[float]) -> List[dict]:
        updated = []
        for doc, pred in zip(docs, predictions):
            target_prior = round(max(0.0, float(pred)), 6)
            copied = {}
            for key, value in doc.items():
                if key == "target":
                    copied["target_prior"] = target_prior
                copied[key] = value
            if "target" not in copied:
                copied["target_prior"] = target_prior
            updated.append(copied)
        return updated

    def _add_target_priors(self, seed: int):
        """Attach a train-only target-prior token without target leakage.

        Train docs receive out-of-fold predictions, so a sample's own target is
        never used to build its prior.  Validation/test docs receive predictions
        from a model fit on the full training split only.
        """
        if not self.train_docs:
            return
        if any(self.target_key not in doc for doc in self.train_docs):
            raise RuntimeError("Cannot fit target_prior: train targets are missing")

        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_absolute_error

        y_train = np.asarray(
            [float(doc[self.target_key]) for doc in self.train_docs],
            dtype=np.float32,
        )
        elements = self._target_prior_elements(self.train_docs)
        x_train = self._target_prior_features(self.train_docs, elements)
        n_splits = min(5, len(self.train_docs))
        oof = np.zeros(len(self.train_docs), dtype=np.float32)

        if n_splits >= 2:
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for fold_idx, (fit_idx, pred_idx) in enumerate(kfold.split(x_train)):
                model = self._make_target_prior_model(seed + fold_idx)
                model.fit(x_train[fit_idx], y_train[fit_idx])
                oof[pred_idx] = model.predict(x_train[pred_idx])
        else:
            oof.fill(float(np.mean(y_train)))

        final_model = self._make_target_prior_model(seed + 1000)
        final_model.fit(x_train, y_train)

        x_val = self._target_prior_features(self.val_docs, elements)
        x_test = self._target_prior_features(self.test_docs, elements)
        val_prior = final_model.predict(x_val) if len(self.val_docs) else []
        test_prior = final_model.predict(x_test) if len(self.test_docs) else []

        self.train_docs = self._with_target_prior(self.train_docs, oof)
        self.val_docs = self._with_target_prior(self.val_docs, val_prior)
        self.test_docs = self._with_target_prior(self.test_docs, test_prior)

        train_oof_mae = mean_absolute_error(y_train, np.maximum(0.0, oof))
        val_mae = None
        if self.val_docs and all(self.target_key in doc for doc in self.val_docs):
            y_val = np.asarray(
                [float(doc[self.target_key]) for doc in self.val_docs],
                dtype=np.float32,
            )
            val_mae = mean_absolute_error(y_val, np.maximum(0.0, val_prior))

        self.split_metadata["target_prior"] = {
            "method": "HistGradientBoostingRegressor composition/lattice features",
            "train_predictions": f"{n_splits}-fold out-of-fold",
            "val_test_predictions": "fit on train split only",
            "n_train": len(self.train_docs),
            "n_elements": len(elements),
            "train_oof_mae": float(train_oof_mae),
            "val_mae": None if val_mae is None else float(val_mae),
            "test_targets": self.split_metadata.get("test_targets", "unknown"),
        }
        msg = f"  🎚️  Target prior: train OOF MAE={train_oof_mae:.4f}"
        if val_mae is not None:
            msg += f", val MAE={val_mae:.4f}"
        print(msg)

    def _save_cache(self, cache_path):
        """将转换好的数据保存到磁盘。"""
        import pickle
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "train_docs": self.train_docs,
            "val_docs": self.val_docs,
            "test_docs": self.test_docs,
            "split_metadata": getattr(self, "split_metadata", {}),
        }
        with open(cache_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = cache_path.stat().st_size / 1024 / 1024
        print(f"  💾 Cached to {cache_path} ({size_mb:.1f} MB)")

    @staticmethod
    def _load_task(task_name: str) -> pd.DataFrame:
        """通过 matminer 加载 Matbench 数据集，并对齐官方 Matbench mbid index。"""
        from matminer.datasets import load_dataset
        df = load_dataset(task_name)
        id_n_zeros = math.floor(math.log(df.shape[0], 10)) + 1
        prefix = task_name.replace("matbench", "mb").replace("_", "-")
        df = df.copy()
        df["mbid"] = [f"{prefix}-{i + 1:0{id_n_zeros}d}" for i in df.index]
        return df.set_index("mbid")

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
