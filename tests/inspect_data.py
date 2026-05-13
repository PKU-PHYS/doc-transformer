"""
数据生成诊断脚本 — 打印各模式的样本，检查 mask 策略是否正确。

用法: pixi run python tests/inspect_data.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import json
from data.synthetic import SyntheticDataset

random.seed(42)

MODES = [
    ("simple",        20,  0, "Stage 0: 极简预热"),
    ("explicit",      60,  0, "Stage 1: 复合结构"),
    ("explicit_long", 150, 3, "Stage 2: 抗噪长文档"),
    ("in_context",    200, 1, "Stage 3: In-Context"),
]

SAMPLES_PER_MODE = 5


def fmt_leaf(i, leaf, is_masked, mask_truth=None):
    """格式化一个叶子节点的显示。"""
    path_str = ".".join(leaf.path)
    if leaf.value_type == "number":
        val_str = f"{leaf.value:.6g}"
    elif leaf.value_type == "string":
        val_str = f'"{leaf.value}"'
    else:
        val_str = str(leaf.value)

    marker = ""
    if is_masked:
        truth_val, truth_type = mask_truth
        if truth_type == "number":
            marker = f"  🎯 MASKED (truth={truth_val:.6g})"
        else:
            marker = f'  🎯 MASKED (truth="{truth_val}")'

    grp = leaf.group_ids if leaf.group_ids else "—"
    return f"  [{i:3d}] {path_str:45s} = {val_str:20s} ({leaf.value_type:7s}) grp={grp}{marker}"


def inspect_sample(leaves, masks, sample_id, mode_label):
    """打印一个样本的完整信息。"""
    print(f"\n{'─'*80}")
    print(f"  {mode_label} | Sample #{sample_id} | {len(leaves)} leaves | {len(masks)} masks")
    print(f"{'─'*80}")

    for i, leaf in enumerate(leaves):
        is_masked = i in masks
        mask_truth = masks.get(i, None)
        print(fmt_leaf(i, leaf, is_masked, mask_truth))

    # 检查 mask 质量
    issues = []
    for idx, (truth_val, truth_type) in masks.items():
        leaf = leaves[idx]
        if leaf.value_type != "mask":
            issues.append(f"  ❌ mask[{idx}] 的 leaf.value_type 不是 'mask'，是 '{leaf.value_type}'")
        if leaf.path:
            key = leaf.path[-1]
            # 检查是否意外 mask 了函数名或类别字段
            from data.functions.registry import FUNC_NAME_KEYS, CATEGORY_KEYS
            if key in FUNC_NAME_KEYS:
                issues.append(f"  ⚠️  mask[{idx}] 的 path[-1]='{key}' 是函数名键！")
            if key in CATEGORY_KEYS:
                issues.append(f"  ⚠️  mask[{idx}] 的 path[-1]='{key}' 是类别键！")

    if issues:
        print("\n  🔍 质量检查:")
        for iss in issues:
            print(iss)
    else:
        print("\n  ✅ 质量检查通过")


def main():
    output_lines = []

    for mode, target_tokens, distractor_level, label in MODES:
        print(f"\n{'═'*80}")
        print(f"  {label}")
        print(f"  mode={mode}, target_tokens={target_tokens}, distractor_level={distractor_level}")
        print(f"{'═'*80}")

        ds = SyntheticDataset(
            size=SAMPLES_PER_MODE,
            mask_ratio=0.15,
            max_tokens=512,
            train_mode=mode,
            target_tokens=target_tokens,
            distractor_level=distractor_level,
        )

        for i in range(SAMPLES_PER_MODE):
            leaves, masks = ds[i]
            inspect_sample(leaves, masks, i, label)


if __name__ == "__main__":
    main()
