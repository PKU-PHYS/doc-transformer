"""evaluate() 训练模式还原测试 — 覆盖 BUGS.md #15。

回归点:evaluate() 退出时应还原入场时的 training 模式,而非无条件 model.train()。
否则在 inference 脚本里调 evaluate() 会被意外切回 train 模式(带 dropout)。
"""

import unittest

import torch
import torch.nn as nn

from eval import evaluate


class _DummyHead(nn.Module):
    def predict_number(self, x):
        return torch.zeros(1)

    def predict_is_zero(self, x):
        return torch.zeros(1)


class _DummyConfig:
    use_zero_head = False


class _DummyModel(nn.Module):
    """最小模型:eval()/train() 会递归设置 self.training,forward 返回可索引张量。"""

    def __init__(self):
        super().__init__()
        self.decode_head = _DummyHead()
        self.config = _DummyConfig()

    def forward(self, batched_leaves, padding_mask, fork_bias_indices=None):
        return torch.zeros(len(batched_leaves), 4, 2)


def _make_loader():
    batched_leaves = [[0]]                       # 1 个样本
    batched_masks = [{0: (1.0, "number")}]       # mask 在位置 0
    padding_mask = torch.zeros(1, 4, dtype=torch.bool)
    fork_bias = torch.zeros(1, 4, 4, dtype=torch.long)
    return [(batched_leaves, batched_masks, padding_mask, fork_bias)]


class EvalModeRestoreTests(unittest.TestCase):
    def test_restores_train_mode(self):
        model = _DummyModel()
        model.train()
        self.assertTrue(model.training)

        evaluate(model, _make_loader(), "cpu")

        self.assertTrue(model.training)

    def test_restores_eval_mode(self):
        model = _DummyModel()
        model.eval()
        self.assertFalse(model.training)

        evaluate(model, _make_loader(), "cpu")

        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
