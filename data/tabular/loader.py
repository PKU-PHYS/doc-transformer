"""
表格数据加载器 — 下载/缓存 + 标准化 + BallTree 索引。

支持的数据集：
  - california_housing: 20640 行 × 9 列 (回归)
  - diabetes:           442 行 × 10 列 (回归)
  - wine_quality:       6497 行 × 12 列 (回归)
  - covertype:          581012 行 × 54 列 (分类, 取子集)

也可通过 from_csv() 加载任意 CSV 文件。
"""

import os
import hashlib
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler


# ═══════════════════════════════════════════════════════════════
# §1  内置数据集下载器
# ═══════════════════════════════════════════════════════════════

def _load_builtin(name: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """下载内置数据集，返回 DataFrame (含 target 列)。"""
    if name == "california_housing":
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing(as_frame=True)
        df = data.frame  # 包含 target 列 'MedHouseVal'

    elif name == "diabetes":
        from sklearn.datasets import load_diabetes
        data = load_diabetes(as_frame=True)
        df = data.frame

    elif name == "wine_quality":
        from sklearn.datasets import fetch_openml
        data = fetch_openml(data_id=287, as_frame=True, parser="auto")
        df = data.frame

    elif name == "covertype":
        from sklearn.datasets import fetch_covtype
        data = fetch_covtype(as_frame=True)
        df = data.frame
        # Covertype 太大，默认截取
        if max_rows is None:
            max_rows = 50000

    else:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available: california_housing, diabetes, wine_quality, covertype. "
            f"Or use TableLoader.from_csv()."
        )

    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    return df


# ═══════════════════════════════════════════════════════════════
# §2  TableLoader
# ═══════════════════════════════════════════════════════════════

class TableLoader:
    """
    表格数据加载器，提供 query-aware 邻居查询。

    核心功能：
      1. 加载数据 → 分离数值/类别列
      2. 标准化数值列 → 构建 BallTree
      3. precompute_neighbors(): 批量预计算所有行的 K 近邻

    Attributes:
        df:           原始 DataFrame
        numeric_cols: 数值列名列表
        cat_cols:     类别列名列表
        target_col:   目标列名 (可选)
    """

    def __init__(self, df: pd.DataFrame, target_col: Optional[str] = None,
                 cache_dir: str = "data/tabular/cache"):
        self.df = df.reset_index(drop=True)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 分离数值列和类别列
        self.numeric_cols: List[str] = []
        self.cat_cols: List[str] = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                self.numeric_cols.append(col)
            else:
                self.cat_cols.append(col)

        # 推断 target 列
        if target_col:
            self.target_col = target_col
        elif "target" in df.columns:
            self.target_col = "target"
        else:
            # 默认最后一列
            self.target_col = df.columns[-1]

        # 标准化数值列 → BallTree
        self._scaler = StandardScaler()
        self._numeric_matrix = self._scaler.fit_transform(
            df[self.numeric_cols].fillna(0).values
        )

        # 构建/加载 BallTree 缓存
        self._tree = self._build_or_load_tree()

    def _cache_key(self) -> str:
        """基于数据内容的 hash，用于缓存 BallTree。"""
        h = hashlib.md5(
            self._numeric_matrix.tobytes()[:10000]  # 取前 10KB 做 hash
        ).hexdigest()[:12]
        return f"tree_{len(self.df)}x{len(self.numeric_cols)}_{h}"

    def _build_or_load_tree(self) -> BallTree:
        """构建或从缓存加载 BallTree。"""
        cache_path = self.cache_dir / f"{self._cache_key()}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                print(f"  📦 BallTree loaded from cache: {cache_path}")
                return pickle.load(f)

        print(f"  🔨 Building BallTree for {len(self.df)} rows × {len(self.numeric_cols)} cols...")
        tree = BallTree(self._numeric_matrix, leaf_size=40)

        with open(cache_path, "wb") as f:
            pickle.dump(tree, f)
        print(f"  💾 BallTree cached: {cache_path}")
        return tree

    def precompute_neighbors(self, k: int) -> np.ndarray:
        """
        批量预计算所有行的 K 近邻索引。

        Args:
            k: 每行的邻居数量

        Returns:
            np.ndarray of shape (n_rows, k)，每行是 K 个邻居的行索引
        """
        dists, indices = self._tree.query(self._numeric_matrix, k=k + 1)
        # indices 包含自己，去掉
        result = np.zeros((self.n_rows, k), dtype=np.int64)
        for i in range(self.n_rows):
            neigh = indices[i][indices[i] != i][:k]
            result[i] = neigh
        return result

    def get_row_dict(self, idx: int) -> dict:
        """返回第 idx 行的 flat dict，数值和类别都包含。"""
        row = self.df.iloc[idx]
        return {col: _convert_value(row[col]) for col in self.df.columns}

    @property
    def n_rows(self) -> int:
        return len(self.df)

    @property
    def n_cols(self) -> int:
        return len(self.df.columns)

    @property
    def column_names(self) -> List[str]:
        return list(self.df.columns)

    @classmethod
    def from_builtin(cls, name: str, max_rows: Optional[int] = None,
                     target_col: Optional[str] = None,
                     **kwargs) -> "TableLoader":
        """从内置数据集创建 loader。"""
        df = _load_builtin(name, max_rows=max_rows)
        return cls(df, target_col=target_col, **kwargs)

    @classmethod
    def from_csv(cls, path: str, **kwargs) -> "TableLoader":
        """从 CSV 文件创建 loader。"""
        df = pd.read_csv(path)
        return cls(df, **kwargs)

    def split(self, test_ratio: float = 0.2, seed: int = 42
              ) -> Tuple["TableLoader", "TableLoader"]:
        """
        标准 train/test 分割。

        Args:
            test_ratio: 测试集比例 (default 0.2 = 80/20 split)
            seed:       随机种子 (default 42, 与 sklearn benchmark 对齐)

        Returns:
            (train_loader, test_loader)
        """
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            self.df, test_size=test_ratio, random_state=seed
        )
        print(f"  📊 Split: {len(train_df)} train / {len(test_df)} test "
              f"(ratio={test_ratio}, seed={seed})")
        train_loader = TableLoader(
            train_df, target_col=self.target_col, cache_dir=str(self.cache_dir)
        )
        test_loader = TableLoader(
            test_df, target_col=self.target_col, cache_dir=str(self.cache_dir)
        )
        return train_loader, test_loader


def _convert_value(v):
    """将 pandas 值转为 Python 原生类型。"""
    if pd.isna(v):
        return 0.0
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 6)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return str(v)
