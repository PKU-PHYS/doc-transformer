"""
数据模块统一入口。

用法：
    from data import get_dataset, collate_fn

    dataset = get_dataset("california_housing", n_rows=5)
    loader = DataLoader(dataset, collate_fn=lambda b: collate_fn(b))
"""

from data.base import collate_fn
from data.tabular import TabularDataset, TableLoader


def get_dataset(dataset_name: str, n_rows: int = 5,
                mask_ratio: float = 0.15, max_tokens: int = 512,
                mask_target_only: bool = False,
                max_rows: int = None,
                csv_path: str = None,
                **kwargs) -> TabularDataset:
    """
    统一数据集工厂函数。

    Args:
        dataset_name:     内置数据集名 (california_housing, diabetes, etc.)
                          或 "csv" 表示从文件加载
        n_rows:           每样本行数
        mask_ratio:       mask 比例
        max_tokens:       最大 token 数
        mask_target_only: 是否只 mask target 列
        max_rows:         数据集最大行数 (截取)
        csv_path:         CSV 文件路径 (仅 dataset_name="csv" 时使用)
    """
    if dataset_name == "csv":
        if csv_path is None:
            raise ValueError("csv_path required when dataset_name='csv'")
        loader = TableLoader.from_csv(csv_path, **kwargs)
    else:
        loader = TableLoader.from_builtin(dataset_name, max_rows=max_rows, **kwargs)

    return TabularDataset(
        loader=loader,
        n_rows=n_rows,
        mask_ratio=mask_ratio,
        max_tokens=max_tokens,
        mask_target_only=mask_target_only,
    )
