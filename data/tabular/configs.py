"""
各数据集的训练配方 — 定义每个 benchmark 的训练方式。

每个配方包含：
  - n_rows:    每个样本包含多少行（1=单行，与标准 ML 对齐）
  - stages:    训练阶段列表 [{name, max_epochs, patience}]
  - target_col: 目标列名（可选，默认自动推断）
  - task_type:  regression / classification
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StageConfig:
    name: str = "train"
    max_epochs: int = 100
    patience: int = 15


@dataclass
class DatasetConfig:
    """单个数据集的完整训练配方。"""
    n_rows: int = 1
    stages: List[StageConfig] = field(default_factory=lambda: [StageConfig()])
    target_col: Optional[str] = None
    task_type: str = "regression"  # "regression" or "classification"
    max_rows: Optional[int] = None  # 截取最大行数


# ═══════════════════════════════════════════════════════════════
# 各数据集配方
# ═══════════════════════════════════════════════════════════════

DATASET_CONFIGS = {
    "california_housing": DatasetConfig(
        n_rows=1,
        stages=[StageConfig(name="train", max_epochs=100, patience=15)],
        task_type="regression",
    ),

    "diabetes": DatasetConfig(
        n_rows=1,
        stages=[StageConfig(name="train", max_epochs=200, patience=20)],
        task_type="regression",
    ),

    "wine_quality": DatasetConfig(
        n_rows=1,
        stages=[StageConfig(name="train", max_epochs=100, patience=15)],
        task_type="regression",
    ),

    "covertype": DatasetConfig(
        n_rows=1,
        stages=[StageConfig(name="train", max_epochs=50, patience=10)],
        task_type="classification",
        max_rows=50000,
    ),
}


def get_dataset_config(name: str) -> DatasetConfig:
    """获取数据集训练配方，未知数据集返回默认配置。"""
    if name in DATASET_CONFIGS:
        return DATASET_CONFIGS[name]
    # CSV 或未知数据集 → 默认单行单阶段
    return DatasetConfig()
