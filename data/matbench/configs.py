"""
Matbench 任务训练配方。

每个任务配方包含训练超参、元信息和数据集选项。
dataset_options 是通用的 dict，可传递给 MatbenchLoader 和 MatbenchDataset，
不同数据集可定义自己特有的参数。
"""

import argparse
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
    ("--add-angles",      {"action": argparse.BooleanOptionalAction, "default": None,
                           "help": "[Matbench] Add per-site coordination stats (cn, avg_angle, min_angle)"},
     "add_angles"),
    ("--add-bonds",       {"action": argparse.BooleanOptionalAction, "default": None,
                           "help": "[Matbench] Add global bonds list (unique atom pairs + distances)"},
     "add_bonds"),
    ("--add-composition", {"action": argparse.BooleanOptionalAction, "default": None,
                           "help": "[Matbench] Add element ratio array [{element, ratio}]"},
     "add_composition"),
    ("--add-ewald",       {"action": argparse.BooleanOptionalAction, "default": None,
                           "help": "[Matbench] Add per-site Ewald electrostatic energy (ewald_energy)"},
     "add_ewald"),
    ("--add-element-props", {"action": argparse.BooleanOptionalAction, "default": None,
                             "help": "[Matbench] Add per-element physical properties (en, ie, ea)"},
     "add_element_props"),
    ("--add-comp-ewald",   {"action": argparse.BooleanOptionalAction, "default": None,
                            "help": "[Matbench] Add per-element averaged Ewald energy to composition"},
     "add_comp_ewald"),
    ("--add-spacegroup",   {"action": argparse.BooleanOptionalAction, "default": None,
                            "help": "[Matbench] Add space group number and crystal system"},
     "add_spacegroup"),
    ("--add-density",      {"action": argparse.BooleanOptionalAction, "default": None,
                            "help": "[Matbench] Add density (g/cm3) and volume per atom (A3/atom)"},
     "add_density"),
    ("--add-nn-stats",     {"action": argparse.BooleanOptionalAction, "default": None,
                            "help": "[Matbench] Add global nearest-neighbor distance stats (nn_min, nn_mean)"},
     "add_nn_stats"),
    ("--add-comp-nn",      {"action": argparse.BooleanOptionalAction, "default": None,
                            "help": "[Matbench] Add per-element averaged nearest-neighbor distance to composition"},
     "add_comp_nn"),
    ("--drop-coords",     {"action": argparse.BooleanOptionalAction, "default": None,
                           "help": "[Matbench] Remove per-site absolute coords (x,y,z); keep sites if angles present"},
     "drop_coords"),
    ("--add-comp-ewald-stats", {"action": argparse.BooleanOptionalAction, "default": None,
                                "help": "[Matbench] Add per-element Ewald std/min/max to composition (requires --add-comp-ewald)"},
     "add_comp_ewald_stats"),
    ("--add-comp-nn-stats",    {"action": argparse.BooleanOptionalAction, "default": None,
                                "help": "[Matbench] Add per-element NN distance std/min/max to composition (requires --add-comp-nn)"},
     "add_comp_nn_stats"),
    ("--add-mean-bonds",   {"action": argparse.BooleanOptionalAction, "default": None,
                            "help": "[Matbench] Add per-element-pair mean bond distance (bonds_stats)"},
     "add_mean_bonds"),
]

# option_key → cache_tag 后缀
_OPTION_CACHE_TAGS = {
    "add_angles": "_angles",
    "add_bonds": "_bonds",
    "add_composition": "_comp",
    "add_ewald": "_ewald",
    "add_element_props": "_elprops",
    "add_comp_ewald": "_cewald",
    "add_spacegroup": "_sg",
    "add_density": "_dens",
    "add_nn_stats": "_nn",
    "add_comp_nn": "_cnn",
    "drop_coords": "_dropcoords",
    "add_comp_ewald_stats": "_cewalds",
    "add_comp_nn_stats": "_cnns",
    "add_mean_bonds": "_mbonds",
}


# stats 类选项依赖的 base 选项 (option_key → 必需的 base option_key)
# 单传 stats 而不开 base 时，structure_to_json 里整块逻辑会被跳过 → 静默 no-op
_OPTION_DEPENDENCIES = {
    "add_comp_ewald_stats": "add_comp_ewald",
    "add_comp_nn_stats": "add_comp_nn",
}


def register_args(parser):
    """在 argparse parser 上注册 Matbench 特有的 CLI 参数。"""
    for flag, kwargs, _ in _MATBENCH_OPTIONS:
        parser.add_argument(flag, **kwargs)


def validate_args(args, parser):
    """校验 stats 类选项必须伴随其 base 选项，不满足则 parser.error 退出。"""
    for option_key, required_key in _OPTION_DEPENDENCIES.items():
        if getattr(args, option_key, False) and not getattr(args, required_key, False):
            flag = "--" + option_key.replace("_", "-")
            required_flag = "--" + required_key.replace("_", "-")
            parser.error(f"{flag} requires {required_flag}")


def build_dataset_options(args, mb_config: MatbenchTaskConfig) -> dict:
    """将 CLI 参数与配方默认值合并为 dataset_options dict。

    CLI 三态语义:不传 → None → 沿用 task default;`--X` → True;`--no-X` → False(覆盖)。
    """
    opts = dict(mb_config.dataset_options)
    for _, _, option_key in _MATBENCH_OPTIONS:
        val = getattr(args, option_key, None)
        if val is not None:
            opts[option_key] = val
    return opts


def build_cache_tag(dataset_options: dict) -> str:
    """从 dataset_options 构建缓存标签后缀（用于 MatbenchDataset）。"""
    tag = ""
    for option_key, suffix in _OPTION_CACHE_TAGS.items():
        if dataset_options.get(option_key):
            tag += suffix
    return tag

