"""Fork bias 层级测试。

回归点:fork_level 必须由累计 group 计数(cumsum)得到,而非直接取 is_group(0/1)。
修复前同一 material 内不同 site(应为 level 2)与不同 material(level 1)会压成同值。
"""

import unittest

import torch

from model.json_parser import JSONParser
from data.base import compute_single_fork_bias


def _parse(doc):
    parser = JSONParser()
    return parser.parse(doc, ["crystal"], [0], [JSONParser._key_hash("crystal")], [])


class ForkBiasLevelTests(unittest.TestCase):
    def setUp(self):
        # materials -> sites -> coord,2x2 共 4 个叶子
        doc = {
            "materials": [
                {"sites": [{"coord": 1.0}, {"coord": 2.0}]},
                {"sites": [{"coord": 3.0}, {"coord": 4.0}]},
            ]
        }
        self.leaves = _parse(doc)
        self.fb = compute_single_fork_bias(self.leaves)

    def test_four_leaves_parsed(self):
        self.assertEqual(len(self.leaves), 4)

    def test_fork_levels_are_multivalued(self):
        # 修复前会退化成 {0, 1};修复后应出现深层 group 分叉 level 2
        unique = set(self.fb.flatten().tolist())
        self.assertEqual(unique, {0, 1, 2})

    def test_same_material_different_site_is_level_2(self):
        # leaf0=(mat0,site0), leaf1=(mat0,site1):同 material 内不同 site → 第 2 层 group 分叉
        self.assertEqual(self.fb[0, 1].item(), 2)
        self.assertEqual(self.fb[2, 3].item(), 2)

    def test_different_material_is_level_1(self):
        # leaf0=(mat0,*), leaf2=(mat1,*):不同 material → 第 1 层 group 分叉
        self.assertEqual(self.fb[0, 2].item(), 1)
        self.assertEqual(self.fb[0, 3].item(), 1)
        self.assertEqual(self.fb[1, 2].item(), 1)

    def test_diagonal_is_zero(self):
        self.assertTrue(torch.all(torch.diagonal(self.fb) == 0))

    def test_symmetric(self):
        self.assertTrue(torch.equal(self.fb, self.fb.t()))


class ForkBiasDictKeyTests(unittest.TestCase):
    def test_dict_key_fork_stays_zero(self):
        # 同一 instance 内不同 dict-key 分叉不应赋 group bias
        doc = {"a": 1.0, "b": 2.0}
        leaves = _parse(doc)
        fb = compute_single_fork_bias(leaves)
        self.assertEqual(len(leaves), 2)
        self.assertEqual(fb[0, 1].item(), 0)


if __name__ == "__main__":
    unittest.main()
