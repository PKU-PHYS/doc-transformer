import hashlib
from dataclasses import dataclass, field
from typing import Any, List

import numpy as np


_KEY_HASH_OFFSET = 1_000_000
_KEY_HASH_MODULUS = (1 << 63) - _KEY_HASH_OFFSET - 1

@dataclass
class LeafNode:
    value: Any              # 原始值 (如 2.1, "TiO2", True)
    value_type: str         # 枚举: "number", "string", "boolean", "mask"
    path: List[str]         # 根到叶子的文本标签序列（含实例节点），用于 FrozenLM 编码
    path_types: List[int]   # 每个路径节点的类型: 0=Dict Key, 1=Array Instance
    path_ids: List[int]     # 每个路径节点的唯一 ID (dict key: 稳定哈希, array instance: 自增)
    group_ids: List[int]    # 祖先数组元素对应的全局唯一组 ID（兼容旧代码）

class JSONParser:
    def __init__(self):
        # 实例级计数器，确保多进程 DataLoader (num_workers>0) 下不发生状态冲突
        self.group_counter = 0
    
    @staticmethod
    def _key_hash(key: str) -> int:
        """对 dict key 生成稳定整数 ID，加偏移避免与自增 instance ID 冲突"""
        digest = hashlib.blake2b(str(key).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % _KEY_HASH_MODULUS + _KEY_HASH_OFFSET
        
    def parse(self, data: Any, current_path: List[str],
              current_path_types: List[int], current_path_ids: List[int],
              current_groups: List[int]) -> List[LeafNode]:
        """
        递归解析 JSON 数据并将其展平为叶子节点列表。
        
        与旧版的核心区别：遍历数组时，为每个元素在 path 中插入一个
        实例节点（text=父数组字段名, type=1, id=自增ID），用于 fork bias 检测。
        """
        if isinstance(data, (bool, np.bool_)):
            return [LeafNode(value=bool(data), value_type="boolean", path=current_path,
                             path_types=current_path_types, path_ids=current_path_ids,
                             group_ids=current_groups)]
        elif isinstance(data, (int, float, np.integer, np.floating)):
            # 统一转 Python 原生类型，避免 numpy 标量落入磁盘缓存
            return [LeafNode(value=float(data) if isinstance(data, (float, np.floating)) else int(data),
                             value_type="number", path=current_path,
                             path_types=current_path_types, path_ids=current_path_ids,
                             group_ids=current_groups)]
        elif isinstance(data, str):
            if data == "[MASK]":
                return [LeafNode(value=data, value_type="mask", path=current_path,
                                 path_types=current_path_types, path_ids=current_path_ids,
                                 group_ids=current_groups)]
            return [LeafNode(value=data, value_type="string", path=current_path,
                             path_types=current_path_types, path_ids=current_path_ids,
                             group_ids=current_groups)]
        elif data is None:
            return []  # 忽略 None 值
            
        elif isinstance(data, dict):
            leaves = []
            for key, val in data.items():
                new_path = current_path + [str(key)]
                new_types = current_path_types + [0]  # Dict Key
                new_ids = current_path_ids + [self._key_hash(key)]
                leaves.extend(self.parse(val, new_path, new_types, new_ids, current_groups))
            return leaves
            
        elif isinstance(data, list):
            leaves = []
            # 复用父数组的字段名作为实例节点文本
            parent_name = current_path[-1] if current_path else "array"
            for item in data:
                self.group_counter += 1
                # 插入数组实例节点到路径
                new_path = current_path + [parent_name]
                new_types = current_path_types + [1]  # Array Instance
                new_ids = current_path_ids + [self.group_counter]
                new_groups = current_groups + [self.group_counter]
                leaves.extend(self.parse(item, new_path, new_types, new_ids, new_groups))
            return leaves
            
        else:
            # 对于不支持的类型，转为字符串处理
            return [LeafNode(value=str(data), value_type="string", path=current_path,
                             path_types=current_path_types, path_ids=current_path_ids,
                             group_ids=current_groups)]
