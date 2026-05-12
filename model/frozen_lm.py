import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class FrozenLM:
    """
    包装 SentenceTransformer 模型，提供冻结的文本编码能力。
    自带简单的缓存机制，避免同一进程内对相同字符串反复编码。
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cuda"):
        # 如果当前环境没有 GPU，自动回退到 CPU
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            
        self.device = device
        # 显式使用 CPU/GPU 加载，且不计算梯度
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
        self._cache = {}
        
    def encode(self, texts: list[str]) -> torch.Tensor:
        """
        对一批文本进行编码，返回 (N, dim) 张量。
        如果全在缓存里，直接拼接返回；否则将没见过的文本送入模型。
        """
        if not texts:
            return torch.empty((0, self.model.get_sentence_embedding_dimension()), device=self.device)

        # 区分命中与未命中的
        miss_indices = []
        miss_texts = []
        out_list = [None] * len(texts)
        
        for i, text in enumerate(texts):
            if text in self._cache:
                out_list[i] = self._cache[text]
            else:
                miss_indices.append(i)
                miss_texts.append(text)
                
        # 批量处理未命中的
        if miss_texts:
            # 防止缓存无限膨胀导致 GPU 显存泄漏
            if len(self._cache) > 10000:
                self._cache.clear()
            with torch.no_grad():
                # encode 返回的是 numpy array 或是 tensor 取决于 convert_to_tensor
                embs = self.model.encode(miss_texts, convert_to_tensor=True, show_progress_bar=False)
                # 存入 cache，将其保持在 model 的 device 上
                for j, text in enumerate(miss_texts):
                    emb = embs[j]
                    self._cache[text] = emb
                    out_list[miss_indices[j]] = emb
                    
        return torch.stack(out_list)
        
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
