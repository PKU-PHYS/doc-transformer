"""
生成 In-Context Learning 任务的生成器。
迫使模型在没有明确 function 名称的情况下，通过前序数据的输入输出对，推断出隐含的数学规律，并应用到目标上。

核心保证：Query-First 构建策略 — 先确定 query 的 token 占用，再往前填充 demonstrations，
确保截断后 query 永远完整存在。
"""

import random
from typing import List, Tuple, Dict, Any, Optional

from data.functions.registry import FunctionRegistry
from data.templates.engine import TemplateEngine
from data.templates.distractors import inject_distractors
from model.json_parser import JSONParser, LeafNode


def generate_in_context_task(
    num_shots: Optional[int] = None,
    target_tokens: Optional[int] = None,
    max_tokens: int = 512,
    distractor_level: int = 0,
) -> Tuple[List[LeafNode], Dict[int, Any]]:
    """
    生成一条 In-Context Learning 任务。

    Query-First 构建策略：
      1. 先生成并解析 query，确定其 token 占用
      2. 逐条追加 demonstrations，直到接近 token 预算
      3. 截断只可能丢弃靠后的 demonstrations，query 始终完整

    Args:
        num_shots: 固定 shots 数。如果为 None 则由 target_tokens 动态决定。
        target_tokens: 目标 token 数。设置后自动计算 shots 数以接近此长度。
        max_tokens: 截断上限。
        distractor_level: 干扰强度 (0=无, 1=轻度, 2=中度, 3=重度)。
    """
    # 1. 拿到一个 generator 引用，多次调用获取同类函数的不同采样
    gen = FunctionRegistry.sample_generator()

    # 2. 先生成 query 文档并独立解析，确定其 token 占用
    target_rel = gen()
    target_doc, _, _ = TemplateEngine.render_implicit(target_rel)

    # 用临时 parser 估算 query token 数
    _tmp_parser = JSONParser()
    query_leaves_estimate = _tmp_parser.parse(
        [target_doc], ["records"], [0], [JSONParser._key_hash("records")], []
    )
    query_token_count = len(query_leaves_estimate)

    # 3. 计算可用于 demonstrations 的 token 预算
    token_budget = (target_tokens or max_tokens) - query_token_count
    if token_budget < 0:
        token_budget = 0

    # 4. 决定最大 shots 数
    if num_shots is not None:
        max_shots = num_shots
    elif target_tokens is not None:
        avg_tokens_per_shot = max(query_token_count, 5)  # 用 query 自身长度作为更精确的估算
        max_shots = max(1, min(40, token_budget // avg_tokens_per_shot))
        # 加一点随机性 (±30%)
        jitter = random.uniform(0.7, 1.3)
        max_shots = max(1, min(40, int(max_shots * jitter)))
    else:
        max_shots = random.randint(2, 6)

    # 5. 逐条生成 demonstrations，按 token 预算动态截止
    context_jsons = []
    accumulated_tokens = 0
    for _ in range(max_shots):
        companion_rel = gen()
        doc, _, _ = TemplateEngine.render_implicit(companion_rel)
        # 根据干扰等级注入干扰
        if isinstance(doc, dict) and distractor_level > 0:
            n_max = [0, 2, 5, 8][min(distractor_level, 3)]
            nested_prob = [0.0, 0.0, 0.3, 0.6][min(distractor_level, 3)]
            inject_distractors(doc, n_min=0, n_max=n_max, nested_prob=nested_prob)

        # 估算这条 demonstration 的 token 数
        _est_parser = JSONParser()
        est_leaves = _est_parser.parse(
            [doc], ["records"], [0], [JSONParser._key_hash("records")], []
        )
        shot_tokens = len(est_leaves)

        # 超出 max_tokens 预算则停止追加
        if max_tokens is not None and (accumulated_tokens + shot_tokens + query_token_count) > max_tokens:
            break

        context_jsons.append(doc)
        accumulated_tokens += shot_tokens

    # 保底：至少要有 1 条 demonstration
    if not context_jsons:
        companion_rel = gen()
        doc, _, _ = TemplateEngine.render_implicit(companion_rel)
        context_jsons.append(doc)

    # 6. 构造最终的纯数组并列结构：demonstrations + query
    final_batch = context_jsons + [target_doc]

    # 7. 用同一个 parser 解析完整批次
    parser = JSONParser()
    leaves = parser.parse(final_batch, ["records"], [0], [JSONParser._key_hash("records")], [])

    # 安全截断（理论上不应触发，因为我们已按预算控制）
    if max_tokens is not None and len(leaves) > max_tokens:
        leaves = leaves[:max_tokens]

    # 8. 定位 query 组并应用 MASK
    #    query 是数组最后一个元素，通过 group_ids[0] 定位。
    #    由于 Query-First 策略保证 query 完整，leaves[-1] 一定属于 query 组。
    target_masks = {}
    last_group_id = leaves[-1].group_ids[0] if (leaves and leaves[-1].group_ids) else None
    if last_group_id is not None:
        query_indices = [
            i for i, node in enumerate(leaves)
            if node.group_ids and node.group_ids[0] == last_group_id
            and node.value_type in ("number", "string")
        ]
    else:
        query_indices = []

    if query_indices:
        num_masks = min(len(query_indices), random.randint(1, 2))
        mask_indices = random.sample(query_indices, num_masks)
    else:
        # fallback: 当 query_indices 为空时，随机 mask 10% 节点
        num_masks = max(1, int(len(leaves) * 0.1))
        mask_indices = random.sample(range(len(leaves)), min(num_masks, len(leaves)))

    for i in mask_indices:
        orig_node = leaves[i]
        target_masks[i] = (orig_node.value, orig_node.value_type)
        leaves[i] = LeafNode(
            value="[MASK]",
            value_type="mask",
            path=orig_node.path,
            path_types=orig_node.path_types,
            path_ids=orig_node.path_ids,
            group_ids=orig_node.group_ids
        )

    return leaves, target_masks
