import torch
from torch import Tensor

def generate_group_embeddings(group_ids_list: list[list[int]], d_model: int, scale: float, device: str) -> Tensor:
    """
    根据组 ID 列表，动态生成 L2 归一化的随机向量并求和。
    
    group_ids_list: 长度为 N 的列表。每个元素是一个该叶子节点所在的所有嵌套数组的 group_id 列表。
                    例如: [[14], [14, 52], [], ...]
    d_model: 向量维度
    scale: 缩放因子
    返回: (N, d_model) 的张量
    """
    # 找到所有出现过的唯一 ID
    unique_ids = set()
    for gids in group_ids_list:
        for gid in gids:
            unique_ids.add(gid)
            
    # 为每个唯一 ID 生成一个随机向量并归一化
    random_vecs = {}
    for uid in unique_ids:
        vec = torch.randn(d_model, device=device)
        # L2 归一化并缩放
        norm = vec.norm(p=2)
        if norm > 0:
            vec = (vec / norm) * scale
        random_vecs[uid] = vec
        
    # 组装每个 token 的组嵌入
    embeddings = []
    for gids in group_ids_list:
        if not gids:
            # 不在任何数组中
            embeddings.append(torch.zeros(d_model, device=device))
        else:
            # 对应层级的组嵌入求和
            sum_vec = sum(random_vecs[gid] for gid in gids)
            embeddings.append(sum_vec)
            
    return torch.stack(embeddings) # shape: (N, d_model)
