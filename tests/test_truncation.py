"""文档截断保护 target leaf 测试 — 覆盖 BUGS.md #6。

回归点:structure_to_json 把 target 放在最后一个 key,朴素截断必砍掉它。
截断逻辑应把超窗口的 target leaf 挪到 leaves[max_tokens-1] 并保留其值;
缺 target key 的 doc 在 __getitem__ 处应 raise RuntimeError(而非静默错误)。
"""

import unittest
import warnings

from data.matbench.dataset import MatbenchDataset


class TargetPreservationTests(unittest.TestCase):
    def _make_doc(self, n_sites, target_val=1.23):
        # target 放最后一个 key,模拟 structure_to_json 的输出顺序
        return {
            "sites": [{"x": float(i)} for i in range(n_sites)],
            "target": target_val,
        }

    def test_target_preserved_when_beyond_window(self):
        doc = self._make_doc(n_sites=30)  # 31 leaves > max_tokens=10
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = MatbenchDataset([doc], max_tokens=10, target_key="target", cache_tag=None)

        self.assertEqual(ds._target_indices[0], 9)  # 挪到 max_tokens-1

        leaves, masks, _ = ds[0]
        self.assertEqual(len(leaves), 10)
        self.assertIn(9, masks)
        self.assertEqual(masks[9][1], "number")
        self.assertEqual(masks[9][0], 1.23)
        self.assertEqual(leaves[9].value_type, "mask")

    def test_target_within_window_keeps_index(self):
        doc = self._make_doc(n_sites=3)  # 4 leaves <= max_tokens=10
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = MatbenchDataset([doc], max_tokens=10, target_key="target", cache_tag=None)

        # target 是最后一个叶子,索引 3
        self.assertEqual(ds._target_indices[0], 3)
        leaves, masks, _ = ds[0]
        self.assertIn(3, masks)
        self.assertEqual(masks[3][0], 1.23)

    def test_missing_target_raises_at_getitem(self):
        doc = {"sites": [{"x": 1.0}]}  # 没有 target key
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = MatbenchDataset([doc], max_tokens=10, target_key="target", cache_tag=None)

        self.assertEqual(ds._target_indices[0], -1)
        with self.assertRaises(RuntimeError):
            _ = ds[0]


if __name__ == "__main__":
    unittest.main()
