"""
Matbench 任务训练配方。

每个任务配方包含训练超参、元信息和数据集选项。
dataset_options 是通用的 dict，可传递给 MatbenchLoader 和 MatbenchDataset，
不同数据集可定义自己特有的参数。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class StageConfig:
    name: str = "train"
    max_epochs: int = 100
    patience: int = 15


@dataclass
class MatbenchTaskConfig:
    """单个 Matbench 任务的训练配方。"""
    stages: List[StageConfig] = field(default_factory=lambda: [StageConfig()])
    task_type: str = "regression"
    description: str = ""
    metric: str = "mae"   # Matbench 标准评估指标

    # 数据集特有选项 — 传递给 MatbenchLoader/MatbenchDataset
    # 例如: {"add_angles": True, "add_bonds": True}
    dataset_options: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# 各任务配方
# ═══════════════════════════════════════════════════════════════

MATBENCH_CONFIGS = {
    "matbench_dielectric": MatbenchTaskConfig(
        stages=[StageConfig(name="train", max_epochs=100, patience=15)],
        description="Refractive Index (n), 4764 samples",
    ),

    "matbench_perovskites": MatbenchTaskConfig(
        stages=[StageConfig(name="train", max_epochs=100, patience=15)],
        description="Formation Energy (eV), 18928 samples",
    ),

    "matbench_mp_gap": MatbenchTaskConfig(
        stages=[StageConfig(name="train", max_epochs=200, patience=15)],
        description="Band Gap (eV), 106113 samples",
    ),

    "matbench_mp_e_form": MatbenchTaskConfig(
        stages=[StageConfig(name="train", max_epochs=50, patience=10)],
        description="Formation Energy (eV/atom), 132752 samples",
    ),
}


def get_matbench_config(task_name: str) -> MatbenchTaskConfig:
    """获取 Matbench 任务训练配方。"""
    if task_name in MATBENCH_CONFIGS:
        return MATBENCH_CONFIGS[task_name]
    # 未知任务 → 默认配置
    return MatbenchTaskConfig(description=f"Unknown task: {task_name}")


# ═══════════════════════════════════════════════════════════════
# CLI 接口 — train.py 通过这三个函数与 Matbench 选项交互
# ═══════════════════════════════════════════════════════════════

# Matbench 支持的所有 dataset_options 及其 CLI 映射
# (cli_flag, cli_kwargs, option_key)
_MATBENCH_OPTIONS = [
    ("--add-angles",      {"action": "store_true", "default": False,
                           "help": "[Matbench] Add per-site coordination stats (cn, avg_angle, min_angle)"},
     "add_angles"),
    ("--add-bonds",       {"action": "store_true", "default": False,
                           "help": "[Matbench] Add global bonds list (unique atom pairs + distances)"},
     "add_bonds"),
    ("--add-composition", {"action": "store_true", "default": False,
                           "help": "[Matbench] Add element ratio array [{element, ratio}]"},
     "add_composition"),
    ("--no-coords",       {"action": "store_true", "default": False,
                           "help": "[Matbench] Remove per-site absolute coords (x,y,z); keep sites if angles present"},
     "no_coords"),
]

# option_key → cache_tag 后缀
_OPTION_CACHE_TAGS = {
    "add_angles": "_angles",
    "add_bonds": "_bonds",
    "add_composition": "_comp",
    "no_coords": "_nocoords",
}


def register_args(parser):
    """在 argparse parser 上注册 Matbench 特有的 CLI 参数。"""
    for flag, kwargs, _ in _MATBENCH_OPTIONS:
        parser.add_argument(flag, **kwargs)


def build_dataset_options(args, mb_config: MatbenchTaskConfig) -> dict:
    """将 CLI 参数与配方默认值合并为 dataset_options dict。"""
    opts = dict(mb_config.dataset_options)
    for _, _, option_key in _MATBENCH_OPTIONS:
        attr_name = option_key  # argparse 把 '-' 转为 '_'
        if getattr(args, attr_name, False):
            opts[option_key] = True
    return opts


def build_cache_tag(dataset_options: dict) -> str:
    """从 dataset_options 构建缓存标签后缀（用于 MatbenchDataset）。"""
    tag = ""
    for option_key, suffix in _OPTION_CACHE_TAGS.items():
        if dataset_options.get(option_key):
            tag += suffix
    return tag

