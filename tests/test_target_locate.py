"""target 叶子定位测试。

回归点:target 一定在根级(path 长度 2),定位时必须加深度约束,
否则任何同名嵌套字段(leaf 顺序更靠前)会被先匹配,导致 mask 错位置。
"""

import unittest
import warnings

from data.matbench.dataset import MatbenchDataset


class TargetLocateTests(unittest.TestCase):
    def test_nested_same_name_field_is_not_mistaken_for_target(self):
        # sites 内嵌套字段也叫 "target"(99.0),且 leaf 顺序在根级 target 之前
        doc = {"sites": [{"target": 99.0}], "target": 1.23}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = MatbenchDataset([doc], max_tokens=512, target_key="target", cache_tag=None)

        _, masks, _ = ds[0]

        # 只 mask 根级 target,其值应为 1.23 而非嵌套的 99.0
        self.assertEqual(len(masks), 1)
        true_val, val_type = next(iter(masks.values()))
        self.assertEqual(true_val, 1.23)
        self.assertNotEqual(true_val, 99.0)
        self.assertEqual(val_type, "number")

    def test_root_target_located_in_plain_doc(self):
        doc = {"sites": [{"x": 1.0}], "target": 2.5}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = MatbenchDataset([doc], max_tokens=512, target_key="target", cache_tag=None)

        _, masks, _ = ds[0]
        self.assertEqual(len(masks), 1)
        self.assertEqual(next(iter(masks.values()))[0], 2.5)


if __name__ == "__main__":
    unittest.main()
