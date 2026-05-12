"""
生成 In-Context Learning 任务的生成器。
迫使模型在没有明确 function 名称的情况下，通过前序数据的输入输出对，推断出隐含的数学规律，并应用到目标上。

支持动态 shots 数量：当指定 target_tokens 时，自动计算需要多少 shots 来接近目标长度。
"""

import uuid
import random
from typing import List, Tuple, Dict, Any, Optional

from data.functions.registry import FunctionRegistry
from data.templates.engine import TemplateEngine
from data.templates.distractors import inject_distractors
from model.json_parser import JSONParser, LeafNode


def _estimate_tokens_per_shot():
    """估算每条 demonstration 的平均 token 数（用于动态计算 shots）"""
    # 基于实测：单条 render_implicit 平均 5-8 tokens，加干扰后 8-15
    return random.randint(8, 14)


def generate_in_context_task(
    num_shots: Optional[int] = None,
    target_tokens: Optional[int] = None,
    max_tokens: int = 512,
    distractor_level: int = 0,
) -> Tuple[List[LeafNode], Dict[int, Any]]:
    """
    生成一条 In-Context Learning 任务。

    Args:
        num_shots: 固定 shots 数。如果为 None 则由 target_tokens 动态决定。
        target_tokens: 目标 token 数。设置后自动计算 shots 数以接近此长度。
        max_tokens: 截断上限。
        distractor_level: 干扰强度 (0=无, 1=轻度, 2=中度, 3=重度)。
    """
    # 1. 拿到一个 generator 引用，多次调用获取正确数据
    gen = FunctionRegistry.sample_generator()

    # 2. 决定 shots 数量
    if num_shots is not None:
        actual_shots = num_shots
    elif target_tokens is not None:
        # 动态计算：(target - query预留) / 每条估算长度
        tokens_per_shot = _estimate_tokens_per_shot()
        query_reserve = tokens_per_shot + 5  # query + task_id
        available = max(target_tokens - query_reserve, tokens_per_shot)
        actual_shots = max(1, min(40, available // tokens_per_shot))
        # 加一点随机性 (±30%)
        jitter = random.uniform(0.7, 1.3)
        actual_shots = max(1, min(40, int(actual_shots * jitter)))
    else:
        actual_shots = random.randint(2, 6)

    # 3. 生成 demonstrations + query
    context_jsons = []
    for _ in range(actual_shots):
        companion_rel = gen()
        doc = TemplateEngine.render_implicit(companion_rel)
        # 根据干扰等级注入干扰
        if isinstance(doc, dict) and distractor_level > 0:
            n_max = [0, 2, 5, 8][min(distractor_level, 3)]
            nested_prob = [0.0, 0.0, 0.3, 0.6][min(distractor_level, 3)]
            inject_distractors(doc, n_min=0, n_max=n_max, nested_prob=nested_prob)
        context_jsons.append(doc)

    target_rel = gen()
    target_doc = TemplateEngine.render_implicit(target_rel)

    # 4. 构造纯数组并列结构 final_batch
    final_batch = context_jsons + [target_doc]

    # 5. 解析为叶子节点
    parser = JSONParser()
    leaves = parser.parse(final_batch, ["records"], [])

    if max_tokens is not None and len(leaves) > max_tokens:
        leaves = leaves[:max_tokens]

    # 6. 找到数组最后一个元素（相当于原来的 query）里的数值或文本节点并 MASK
    target_masks = {}
    last_idx_str = f"[{len(final_batch) - 1}]"
    query_indices = [
        i for i, node in enumerate(leaves)
        if len(node.path) > 1 and node.path[1] == last_idx_str and node.value_type in ("number", "string") and node.value_type != "mask"
    ]

    if query_indices:
        num_masks = min(len(query_indices), random.randint(1, 2))
        mask_indices = random.sample(query_indices, num_masks)
    else:
        # fallback
        num_masks = max(1, int(len(leaves) * 0.1))
        mask_indices = random.sample(range(len(leaves)), min(num_masks, len(leaves)))

    for i in mask_indices:
        orig_node = leaves[i]
        target_masks[i] = (orig_node.value, orig_node.value_type)
        leaves[i] = LeafNode(
            value="[MASK]",
            value_type="mask",
            path=orig_node.path,
            group_ids=orig_node.group_ids
        )

    return leaves, target_masks
