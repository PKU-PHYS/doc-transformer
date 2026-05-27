"""JSONParser 边界用例测试 — 覆盖解析基本契约。

回归点:numpy 标量(np.float32/np.int64/np.bool_)必须被识别为 number/boolean,
而不是落入 else 分支被 str() 当字符串编码。
"""

import unittest

import numpy as np

from model.json_parser import JSONParser


def _parse(doc):
    parser = JSONParser()
    return parser.parse(doc, ["root"], [0], [JSONParser._key_hash("root")], [])


class NumpyScalarTests(unittest.TestCase):
    def test_numpy_float_is_number(self):
        for val in (np.float32(2.5), np.float64(2.5)):
            leaves = _parse({"x": val})
            self.assertEqual(len(leaves), 1)
            self.assertEqual(leaves[0].value_type, "number")
            self.assertIsInstance(leaves[0].value, float)
            self.assertAlmostEqual(leaves[0].value, 2.5, places=5)

    def test_numpy_int_is_number(self):
        for val in (np.int32(7), np.int64(7)):
            leaves = _parse({"x": val})
            self.assertEqual(len(leaves), 1)
            self.assertEqual(leaves[0].value_type, "number")
            self.assertIsInstance(leaves[0].value, int)
            self.assertEqual(leaves[0].value, 7)

    def test_numpy_bool_is_boolean(self):
        leaves = _parse({"flag": np.bool_(True)})
        self.assertEqual(len(leaves), 1)
        self.assertEqual(leaves[0].value_type, "boolean")
        self.assertIsInstance(leaves[0].value, bool)
        self.assertTrue(leaves[0].value)

    def test_python_bool_not_number(self):
        # bool 必须先于 int 判定,否则 True 会被当作数字
        leaves = _parse({"flag": True})
        self.assertEqual(leaves[0].value_type, "boolean")


class ParserContractTests(unittest.TestCase):
    def test_mask_token(self):
        leaves = _parse({"x": "[MASK]"})
        self.assertEqual(leaves[0].value_type, "mask")

    def test_plain_string(self):
        leaves = _parse({"name": "TiO2"})
        self.assertEqual(leaves[0].value_type, "string")
        self.assertEqual(leaves[0].value, "TiO2")

    def test_none_is_dropped(self):
        leaves = _parse({"x": 1.0, "y": None})
        self.assertEqual(len(leaves), 1)
        self.assertEqual(leaves[0].path[-1], "x")

    def test_nested_structure_leaf_count(self):
        doc = {"sites": [{"coord": 1.0}, {"coord": 2.0}, {"coord": 3.0}]}
        leaves = _parse(doc)
        self.assertEqual(len(leaves), 3)

    def test_array_instance_path_types(self):
        # 数组元素会在 path 中插入 type=1 的实例节点
        leaves = _parse({"sites": [{"coord": 1.0}]})
        self.assertIn(1, leaves[0].path_types)


class KeyHashTests(unittest.TestCase):
    def test_key_hash_above_offset(self):
        self.assertGreaterEqual(JSONParser._key_hash("crystal"), 1_000_000)

    def test_key_hash_deterministic_in_process(self):
        self.assertEqual(JSONParser._key_hash("target"), JSONParser._key_hash("target"))


if __name__ == "__main__":
    unittest.main()
