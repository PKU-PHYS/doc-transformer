"""
数据模块统一入口。

提供 collate_fn 和各数据集类型：
  - TabularDataset / TableLoader: 通用表格回归
  - MatbenchDataset / MatbenchLoader: 晶体性质预测
"""

from data.base import collate_fn
from data.tabular import TabularDataset, TableLoader
