"""Matbench split helper tests."""

import unittest

import pandas as pd

from data.matbench.loader import MatbenchLoader
from data.matbench.splits import official_fold_ids, split_internal_train_val


class MatbenchSplitTests(unittest.TestCase):
    def test_official_fold_ids_reads_one_fold(self):
        validation = {
            "splits": {
                "matbench_mp_gap": {
                    "fold_0": {
                        "train": ["mb-mp-gap-000001", "mb-mp-gap-000002"],
                        "test": ["mb-mp-gap-000003"],
                    }
                }
            }
        }

        train_ids, test_ids, fold_key = official_fold_ids(
            "matbench_mp_gap",
            0,
            validation_data=validation,
        )

        self.assertEqual(fold_key, "fold_0")
        self.assertEqual(train_ids, ["mb-mp-gap-000001", "mb-mp-gap-000002"])
        self.assertEqual(test_ids, ["mb-mp-gap-000003"])

    def test_internal_val_split_uses_only_train_val_ids(self):
        ids = [f"mb-mp-gap-{i:06d}" for i in range(1, 101)]
        targets = pd.Series(
            [float(i % 10) for i in range(100)],
            index=ids,
        )

        train_ids, val_ids = split_internal_train_val(
            ids,
            targets_by_id=targets,
            val_ratio=0.2,
            seed=42,
        )

        self.assertEqual(len(train_ids), 80)
        self.assertEqual(len(val_ids), 20)
        self.assertFalse(set(train_ids) & set(val_ids))
        self.assertEqual(set(train_ids) | set(val_ids), set(ids))

    def test_loader_split_can_omit_test_targets(self):
        ids = [f"mb-mp-gap-{i:06d}" for i in range(1, 11)]
        df = pd.DataFrame({"gap pbe": [float(i) for i in range(10)]}, index=ids)
        docs = [{"target": float(i), "x": i} for i in range(10)]

        loader = MatbenchLoader.__new__(MatbenchLoader)
        loader.task_name = "matbench_mp_gap"
        loader.target_key = "target"
        loader.split_strategy = "random"
        loader.fold = 0
        loader.val_ratio = 0.25
        loader.include_test_targets = False

        loader._split_docs(df, docs, target_col="gap pbe", test_ratio=0.2, seed=42)

        self.assertTrue(loader.train_docs)
        self.assertTrue(loader.val_docs)
        self.assertTrue(loader.test_docs)
        self.assertTrue(all("target" in d for d in loader.train_docs))
        self.assertTrue(all("target" in d for d in loader.val_docs))
        self.assertTrue(all("target" not in d for d in loader.test_docs))
        self.assertEqual(loader.split_metadata["test_targets"], "omitted")


if __name__ == "__main__":
    unittest.main()
