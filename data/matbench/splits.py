"""Matbench split helpers.

This module keeps official Matbench fold handling separate from feature
conversion.  During experiments we use one official outer fold, then carve an
internal validation set only from that fold's train+val ids.
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import urllib.request
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


MATBENCH_V01_VALIDATION_URL = (
    "https://raw.githubusercontent.com/materialsproject/matbench/"
    "main/matbench/matbench_v0.1_validation.json"
)


def default_validation_cache_path() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "cache" / "matbench_v0.1_validation.json"


def load_validation_splits(
    cache_path: str | os.PathLike | None = None,
    url: str = MATBENCH_V01_VALIDATION_URL,
) -> Dict[str, Any]:
    """Load the official Matbench v0.1 validation split JSON.

    The file is large, so we cache it under data/matbench/cache and only fetch
    it when missing.  The returned object is the top-level JSON dict.
    """
    path = pathlib.Path(cache_path) if cache_path else default_validation_cache_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
        os.close(fd)
        tmp_path = pathlib.Path(tmp_name)
        try:
            urllib.request.urlretrieve(url, tmp_path)
            tmp_path.replace(path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    with open(path, "r") as f:
        data = json.load(f)
    if "splits" not in data:
        raise ValueError(f"Invalid Matbench validation file: missing 'splits' in {path}")
    return data


def official_fold_ids(
    task_name: str,
    fold: int,
    validation_data: Dict[str, Any] | None = None,
    cache_path: str | os.PathLike | None = None,
) -> Tuple[List[Any], List[Any], str]:
    """Return official (train+val ids, test ids, fold key) for one fold."""
    data = validation_data or load_validation_splits(cache_path=cache_path)
    task_splits = data["splits"].get(task_name)
    if task_splits is None:
        raise ValueError(f"Task {task_name!r} not found in official Matbench splits")

    fold_key = f"fold_{fold}"
    if fold_key not in task_splits:
        keys = list(task_splits.keys())
        raise ValueError(f"Fold {fold} not found for {task_name!r}; available: {keys}")

    split = task_splits[fold_key]
    return list(split["train"]), list(split["test"]), fold_key


def coerce_ids_to_index_type(ids: Sequence[Any], index: Iterable[Any]) -> List[Any]:
    """Coerce official JSON ids to match a pandas index's scalar type."""
    sample = next(iter(index), None)
    if sample is None:
        return list(ids)

    if isinstance(sample, (int, np.integer)):
        return [int(x) for x in ids]
    if isinstance(sample, str):
        return [str(x) for x in ids]
    return list(ids)


def split_internal_train_val(
    train_val_ids: Sequence[Any],
    targets_by_id,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Any], List[Any]]:
    """Split official train+val ids into experiment train and validation ids.

    The official test ids are not involved.  For regression stability we try a
    quantile-binned stratification over the target, falling back to plain random
    splitting when bins are degenerate.
    """
    ids = list(train_val_ids)
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1); got {val_ratio}")
    if val_ratio == 0.0:
        return ids, []

    from sklearn.model_selection import train_test_split

    y = np.asarray([float(targets_by_id.loc[i]) for i in ids], dtype=np.float64)
    stratify = None
    n_bins = min(20, max(2, len(ids) // 5000))
    try:
        import pandas as pd

        bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
        bins = np.asarray(bins, dtype=np.int64)
        counts = np.bincount(bins)
        if len(counts) >= 2 and counts.min() >= 2:
            stratify = bins
    except Exception:
        stratify = None

    train_ids, val_ids = train_test_split(
        ids,
        test_size=val_ratio,
        random_state=seed,
        stratify=stratify,
    )
    return list(train_ids), list(val_ids)
