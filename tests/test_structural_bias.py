"""Structural bias 测试 — 验证结构关系信号系统的正确性。

覆盖:
  - bias.md §3.1 对照表的取值验证
  - 对角线/对称性等基本性质
  - StructuralBiasEncoder 的注册/消融功能
  - 向后兼容回归测试（原 test_fork_bias.py 的语义）
"""

import unittest

import torch

from model.json_parser import JSONParser
from model.transformer import StructuralBiasEncoder
from data.base import compute_single_structural_bias


def _parse(doc, root="crystal"):
    parser = JSONParser()
    return parser.parse(doc, [root], [0], [JSONParser._key_hash(root)], [])


# ═══════════════════════════════════════════════════════════════
# §1  数据计算正确性测试
# ═══════════════════════════════════════════════════════════════

class StructuralBiasValueTests(unittest.TestCase):
    """验证三个 bias 的取值符合 bias.md §3.1 对照表。"""

    def setUp(self):
        # 构建一个覆盖多种关系的 Matbench 风格文档:
        #   crystal.lattice.a   (idx 0)  path_ids: [crystal, lattice, a]
        #   crystal.lattice.b   (idx 1)  path_ids: [crystal, lattice, b]
        #   crystal.density     (idx 2)  path_ids: [crystal, density]
        doc = {
            "lattice": {"a": 1.0, "b": 2.0},
            "density": 3.0,
        }
        self.leaves = _parse(doc)
        self.bias = compute_single_structural_bias(self.leaves)

    def test_three_leaves_parsed(self):
        self.assertEqual(len(self.leaves), 3)

    def test_same_parent_dict_keys(self):
        """lattice.a vs lattice.b → is_group_fork=0, first_diff=2, tree_dist=2"""
        b = self.bias
        # a (idx 0) vs b (idx 1)
        self.assertEqual(b["is_group_fork"][0, 1].item(), 0)
        self.assertEqual(b["first_diff"][0, 1].item(), 2)
        self.assertEqual(b["tree_dist"][0, 1].item(), 2)
        self.assertEqual(b["same_parent"][0, 1].item(), 1)
        self.assertEqual(b["shared_group_depth"][0, 1].item(), 0)

    def test_different_parent_dict_keys(self):
        """lattice.a vs density → is_group_fork=0, first_diff=1, tree_dist=3"""
        b = self.bias
        # a (idx 0) vs density (idx 2)
        self.assertEqual(b["is_group_fork"][0, 2].item(), 0)
        self.assertEqual(b["first_diff"][0, 2].item(), 1)
        self.assertEqual(b["tree_dist"][0, 2].item(), 3)
        self.assertEqual(b["same_parent"][0, 2].item(), 0)

    def test_tree_dist_distinguishes_same_vs_different_parent(self):
        """Bias 3 能区分 first_diff 相同但拓扑不同的 pair。
        
        lattice.a vs lattice.b: tree_dist=2 (近)
        lattice.a vs density:   tree_dist=3 (远)
        """
        b = self.bias
        self.assertNotEqual(b["tree_dist"][0, 1].item(), b["tree_dist"][0, 2].item())


class StructuralBiasGroupTests(unittest.TestCase):
    """验证 group（数组 instance）相关的 bias 取值。"""

    def setUp(self):
        # 两层嵌套 group: materials[i].sites[j].coord
        doc = {
            "materials": [
                {"sites": [{"coord": 1.0}, {"coord": 2.0}]},
                {"sites": [{"coord": 3.0}, {"coord": 4.0}]},
            ]
        }
        self.leaves = _parse(doc)
        self.bias = compute_single_structural_bias(self.leaves)

    def test_four_leaves_parsed(self):
        self.assertEqual(len(self.leaves), 4)

    def test_same_material_different_site(self):
        """mat0.site0 vs mat0.site1 → is_group_fork=1 (分叉在 site group 层)"""
        self.assertEqual(self.bias["is_group_fork"][0, 1].item(), 1)
        self.assertEqual(self.bias["is_group_fork"][2, 3].item(), 1)

    def test_different_material(self):
        """mat0 vs mat1 → is_group_fork=1 (分叉在 material group 层)"""
        self.assertEqual(self.bias["is_group_fork"][0, 2].item(), 1)
        self.assertEqual(self.bias["is_group_fork"][0, 3].item(), 1)
        self.assertEqual(self.bias["is_group_fork"][1, 2].item(), 1)

    def test_tree_dist_distinguishes_depth(self):
        """同 material 内 (近) vs 跨 material (远) 的 tree_dist 不同。"""
        same_mat = self.bias["tree_dist"][0, 1].item()    # 同 material 不同 site
        diff_mat = self.bias["tree_dist"][0, 2].item()    # 不同 material
        self.assertLess(same_mat, diff_mat)

    def test_first_diff_multivalued(self):
        """first_diff 应有多个不同取值（回归：修复前可能退化）。"""
        unique = set(self.bias["first_diff"].flatten().tolist())
        self.assertGreater(len(unique), 2)

    def test_shared_group_depth_tracks_common_array_instances(self):
        """同 material 不同 site 共享 1 层 group；跨 material 为 0。"""
        self.assertEqual(self.bias["shared_group_depth"][0, 1].item(), 1)
        self.assertEqual(self.bias["shared_group_depth"][0, 2].item(), 0)

    def test_different_site_is_not_same_parent(self):
        """不同 array instance 下的同名 leaf 不是同父 sibling。"""
        self.assertEqual(self.bias["same_parent"][0, 1].item(), 0)
        self.assertEqual(self.bias["same_parent"][0, 2].item(), 0)


class StructuralBiasNestedGroupTests(unittest.TestCase):
    """验证同一嵌套 array instance 内多个字段的关系。"""

    def setUp(self):
        doc = {
            "materials": [
                {"sites": [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]},
                {"sites": [{"x": 5.0, "y": 6.0}]},
            ]
        }
        self.leaves = _parse(doc)
        self.bias = compute_single_structural_bias(self.leaves)

    def test_same_site_fields_share_parent_and_two_groups(self):
        self.assertEqual(self.bias["same_parent"][0, 1].item(), 1)
        self.assertEqual(self.bias["shared_group_depth"][0, 1].item(), 2)

    def test_same_material_different_sites_share_one_group(self):
        self.assertEqual(self.bias["same_parent"][0, 2].item(), 0)
        self.assertEqual(self.bias["shared_group_depth"][0, 2].item(), 1)

    def test_different_materials_share_no_group(self):
        self.assertEqual(self.bias["shared_group_depth"][0, 4].item(), 0)


class StructuralBiasSamePathTemplateTests(unittest.TestCase):
    """验证忽略 array instance 后的同路径模板关系。"""

    def setUp(self):
        doc = {
            "composition": [
                {"element": "Na", "ratio": 0.5},
                {"element": "Cl", "ratio": 0.5},
            ],
            "target": 0.0,
        }
        self.leaves = _parse(doc)
        self.bias = compute_single_structural_bias(self.leaves)

    def test_same_leaf_key_across_array_instances_matches(self):
        # composition[0].element vs composition[1].element
        self.assertEqual(self.bias["same_path_template"][0, 2].item(), 1)
        # composition[0].ratio vs composition[1].ratio
        self.assertEqual(self.bias["same_path_template"][1, 3].item(), 1)

    def test_different_leaf_keys_do_not_match(self):
        # composition[*].element and composition[*].ratio are different templates.
        self.assertEqual(self.bias["same_path_template"][0, 1].item(), 0)
        self.assertEqual(self.bias["same_path_template"][0, 3].item(), 0)

    def test_target_does_not_match_composition_fields(self):
        self.assertEqual(self.bias["same_path_template"][1, 4].item(), 0)
        self.assertEqual(self.bias["same_path_template"][3, 4].item(), 0)


class StructuralBiasDiagonalTests(unittest.TestCase):
    """对角线和基本性质测试。"""

    def setUp(self):
        doc = {"a": 1.0, "b": 2.0, "c": 3.0}
        self.leaves = _parse(doc)
        self.bias = compute_single_structural_bias(self.leaves)

    def test_diagonal_is_group_fork_zero(self):
        """对角线 → is_group_fork=0"""
        diag = torch.diagonal(self.bias["is_group_fork"])
        self.assertTrue(torch.all(diag == 0))

    def test_diagonal_tree_dist_zero(self):
        """对角线 → tree_dist=0"""
        diag = torch.diagonal(self.bias["tree_dist"])
        self.assertTrue(torch.all(diag == 0))

    def test_diagonal_same_parent_zero(self):
        """对角线不是 sibling pair。"""
        diag = torch.diagonal(self.bias["same_parent"])
        self.assertTrue(torch.all(diag == 0))

    def test_diagonal_first_diff_equals_path_len(self):
        """对角线 → first_diff = path length of that leaf"""
        for i, leaf in enumerate(self.leaves):
            expected_len = len(leaf.path_ids)
            self.assertEqual(self.bias["first_diff"][i, i].item(), expected_len)

    def test_all_signals_symmetric(self):
        """所有信号矩阵关于主对角线对称。"""
        for name, tensor in self.bias.items():
            self.assertTrue(torch.equal(tensor, tensor.t()),
                            f"{name} is not symmetric")


class StructuralBiasDictKeyTests(unittest.TestCase):
    """Dict-key 分叉不应赋 group bias（回归测试）。"""

    def test_dict_key_fork_is_not_group(self):
        doc = {"a": 1.0, "b": 2.0}
        leaves = _parse(doc)
        bias = compute_single_structural_bias(leaves)
        self.assertEqual(len(leaves), 2)
        self.assertEqual(bias["is_group_fork"][0, 1].item(), 0)


class StructuralBiasEmptyTests(unittest.TestCase):
    """边界情况：空输入。"""

    def test_empty_leaves(self):
        bias = compute_single_structural_bias([])
        for name, tensor in bias.items():
            self.assertEqual(tensor.shape, (0, 0), f"{name} has wrong shape for empty input")


# ═══════════════════════════════════════════════════════════════
# §2  StructuralBiasEncoder 功能测试
# ═══════════════════════════════════════════════════════════════

class StructuralBiasEncoderTests(unittest.TestCase):
    """验证 encoder 的注册、前向传播和消融开关。"""

    def setUp(self):
        self.encoder = StructuralBiasEncoder(num_heads=4, encoding_dim=16)
        self.encoder.register_category("is_group_fork", num_classes=2)
        self.encoder.register_continuous("first_diff")
        self.encoder.register_continuous("tree_dist")

    def test_zero_init_produces_zero_bias(self):
        """全零初始化 → 初始输出全零。"""
        B, T = 2, 5
        signals = {
            "is_group_fork": torch.zeros(B, T, T, dtype=torch.long),
            "first_diff": torch.ones(B, T, T, dtype=torch.long),
            "tree_dist": torch.ones(B, T, T, dtype=torch.long) * 2,
        }
        out = self.encoder(**signals)
        self.assertEqual(out.shape, (B * 4, T, T))
        self.assertTrue(torch.allclose(out, torch.zeros_like(out), atol=1e-6))

    def test_output_shape(self):
        """输出形状为 (B*H, T, T)。"""
        B, T, H = 3, 7, 4
        signals = {
            "is_group_fork": torch.zeros(B, T, T, dtype=torch.long),
            "first_diff": torch.zeros(B, T, T, dtype=torch.long),
            "tree_dist": torch.zeros(B, T, T, dtype=torch.long),
        }
        out = self.encoder(**signals)
        self.assertEqual(out.shape, (B * H, T, T))

    def test_set_enabled_disables_signal(self):
        """set_enabled(name, False) → 该信号不参与计算。"""
        B, T = 1, 3
        signals = {
            "is_group_fork": torch.ones(B, T, T, dtype=torch.long),
            "first_diff": torch.zeros(B, T, T, dtype=torch.long),
            "tree_dist": torch.zeros(B, T, T, dtype=torch.long),
        }
        
        # 手动设置非零权重使 is_group_fork 有贡献
        with torch.no_grad():
            self.encoder._encoders["is_group_fork"].weight.fill_(1.0)
        
        out_enabled = self.encoder(**signals).clone()
        
        self.encoder.set_enabled("is_group_fork", False)
        out_disabled = self.encoder(**signals)
        
        # 禁用后输出应不同（因为 is_group_fork 的贡献被移除）
        self.assertFalse(torch.allclose(out_enabled, out_disabled))
        
        # 恢复
        self.encoder.set_enabled("is_group_fork", True)

    def test_all_disabled_returns_zero(self):
        """所有信号禁用 → 返回零张量。"""
        B, T = 1, 3
        signals = {
            "is_group_fork": torch.ones(B, T, T, dtype=torch.long),
            "first_diff": torch.ones(B, T, T, dtype=torch.long),
            "tree_dist": torch.ones(B, T, T, dtype=torch.long),
        }
        self.encoder.set_enabled("is_group_fork", False)
        self.encoder.set_enabled("first_diff", False)
        self.encoder.set_enabled("tree_dist", False)
        
        out = self.encoder(**signals)
        self.assertTrue(torch.allclose(out, torch.zeros_like(out)))
        
        # 恢复
        self.encoder.set_enabled("is_group_fork", True)
        self.encoder.set_enabled("first_diff", True)
        self.encoder.set_enabled("tree_dist", True)

    def test_unregistered_signal_is_ignored(self):
        """缓存里多出的新信号不应破坏旧/消融配置。"""
        B, T = 1, 3
        signals = {
            "is_group_fork": torch.zeros(B, T, T, dtype=torch.long),
            "first_diff": torch.zeros(B, T, T, dtype=torch.long),
            "tree_dist": torch.zeros(B, T, T, dtype=torch.long),
            "same_parent": torch.ones(B, T, T, dtype=torch.long),
        }
        out = self.encoder(**signals)
        self.assertEqual(out.shape, (B * 4, T, T))


if __name__ == "__main__":
    unittest.main()
