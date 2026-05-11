"""
生成 In-Context Learning 任务的生成器。
迫使模型在没有明确 function 名称的情况下，通过前序数据的输入输出对，推断出隐含的数学规律，并应用到目标上。
"""

import uuid
import random
from typing import List, Tuple, Dict, Any

from data.functions.registry import FunctionRegistry
from data.templates.engine import TemplateEngine, resample_relation
from model.json_parser import JSONParser, LeafNode

def generate_in_context_task(num_shots=3, max_tokens=512) -> Tuple[List[LeafNode], Dict[int, Any]]:
    # 1. 随机采样一个数学关系（或物理公式）
    rel = FunctionRegistry.sample()
    
    # 2. 生成多条演示数据 (shots) + 1条目标数据
    context_jsons = []
    for _ in range(num_shots):
        # resample 以得到同一关系下的不同输入输出值
        companion_rel = resample_relation(rel)
        doc = TemplateEngine.render_implicit(companion_rel)
        context_jsons.append(doc)
        
    target_rel = resample_relation(rel)
    target_doc = TemplateEngine.render_implicit(target_rel)
    
    # 3. 构造 final_batch 作为一个包含 demonstrations 和 query 的大字典
    final_batch = {
        "task_id": str(uuid.uuid4()),
        "demonstrations": context_jsons,
        "query": target_doc
    }
    
    # 4. 解析为叶子节点
    parser = JSONParser()
    leaves = parser.parse(final_batch, ["root"], [])
    
    if max_tokens is not None and len(leaves) > max_tokens:
        leaves = leaves[:max_tokens]
        
    # 5. 找到 query 里的数值或文本节点并 MASK
    target_masks = {}
    query_indices = [
        i for i, node in enumerate(leaves)
        if "query" in node.path and node.value_type in ("number", "string") and node.value_type != "mask"
    ]
    
    # 在 target_doc 中至少 mask 一个值（如果有 query 的话）
    if query_indices:
        # Mask 1 到 2 个元素作为预测目标
        num_masks = min(len(query_indices), random.randint(1, 2))
        mask_indices = random.sample(query_indices, num_masks)
    else:
        # fallback 如果不知为何 query 被截断没了
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
