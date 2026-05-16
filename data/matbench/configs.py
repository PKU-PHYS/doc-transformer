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
    # 例如: {"add_nn_distances": True, "n_neighbors": 2}
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
