"""
数据模块统一入口。

提供 collate_fn 和各数据集类型：
  - TabularDataset / TableLoader: 通用表格回归（依赖 pandas/sklearn）
  - MatbenchDataset / MatbenchLoader: 晶体性质预测

注意：tabular 相关导出走 lazy 加载（PEP 562 __getattr__），
避免仅 `from data.base import ...` 的 matbench 用户被迫导入 pandas/sklearn。
"""

from data.base import collate_fn

__all__ = ["collate_fn", "TabularDataset", "TableLoader"]


def __getattr__(name):
    if name in ("TabularDataset", "TableLoader"):
        from data.tabular import TabularDataset, TableLoader
        return {"TabularDataset": TabularDataset, "TableLoader": TableLoader}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
