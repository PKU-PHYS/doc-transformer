from dataclasses import dataclass
from typing import Any, List

@dataclass
class LeafNode:
    value: Any              # 原始值 (如 2.1, "TiO2", True)
    value_type: str         # 枚举: "number", "string", "boolean", "mask"
    path: List[str]         # 根到叶子的字段名列表, e.g., ["materials", "lattice", "a"]
    group_ids: List[int]    # 祖先数组元素对应的全局唯一组 ID, e.g., [14, 52]

class JSONParser:
    def __init__(self):
        # 实例级计数器，确保多进程 DataLoader (num_workers>0) 下不发生状态冲突
        self.group_counter = 0
        
    def parse(self, data: Any, current_path: List[str], current_groups: List[int]) -> List[LeafNode]:
        """
        递归解析 JSON 数据并将其展平为叶子节点列表
        """
        if isinstance(data, bool):
            return [LeafNode(value=data, value_type="boolean", path=current_path, group_ids=current_groups)]
        elif isinstance(data, (int, float)):
            return [LeafNode(value=data, value_type="number", path=current_path, group_ids=current_groups)]
        elif isinstance(data, str):
            if data == "[MASK]":
                return [LeafNode(value=data, value_type="mask", path=current_path, group_ids=current_groups)]
            return [LeafNode(value=data, value_type="string", path=current_path, group_ids=current_groups)]
        elif data is None:
            return []  # 忽略 None 值
            
        elif isinstance(data, dict):
            leaves = []
            for key, val in data.items():
                new_path = current_path + [str(key)]
                leaves.extend(self.parse(val, new_path, current_groups))
            return leaves
            
        elif isinstance(data, list):
            leaves = []
            for item in data:
                self.group_counter += 1
                new_groups = current_groups + [self.group_counter]
                leaves.extend(self.parse(item, current_path, new_groups))
            return leaves
            
        else:
            # 对于不支持的类型，转为字符串处理
            return [LeafNode(value=str(data), value_type="string", path=current_path, group_ids=current_groups)]
