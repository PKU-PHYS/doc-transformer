"""
Matbench 任务训练配方。

每个任务配方包含训练超参和元信息。
"""

from dataclasses import dataclass, field
from typing import List, Optional


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
        stages=[StageConfig(name="train", max_epochs=50, patience=10)],
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
