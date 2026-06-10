import torch
import torch.nn as nn
from torch import Tensor


class FourierFeatureEncoder(nn.Module):
    """
    傅里叶特征映射，消除标准 MLP 的谱偏见。
    将一维标量展开为高频正余弦特征，然后再投影回 d_model。
    
    在 Base-2 Mantissa-Exponent 架构中，此编码器仅处理尾数 M ∈ [-1, 1]（由 torch.frexp 产生）。
    """
    def __init__(self, n_feats: int, d_model: int, learnable: bool = True):
        super().__init__()
        if learnable:
            # 对数均匀分布初始化频率，为 M ∈ [-1, 1] 优化
            # [0.1, 100]: 最低频率捕捉宏观趋势，最高频率区分 0.50 和 0.51 的微小差异
            freqs = 10.0 ** torch.empty(n_feats).uniform_(-1, 2)
            self.freqs = nn.Parameter(freqs) 
        else:
            freqs = 10.0 ** torch.linspace(-1, 2, n_feats)
            self.register_buffer('freqs', freqs)
            
        self.proj = nn.Linear(2 * n_feats, d_model)

        # ── 有效学习率归一化 ──
        # fourier 特征 ‖b‖² = n_feats（每个频率 sin²+cos²=1）。固定基底 + 线性投影下，
        # 输出空间有效 lr ∝ ‖b‖²+1，使尾数相对最小参数化(Embedding)被隐式放大 ~(n_feats+1)×。
        # 把基底缩放到 ‖b‖²=1，使其有效 lr 与指数 Embedding 同量级，消除该隐藏自由度。
        self.feat_scale = n_feats ** -0.5
        # 同步把 proj 权重 init 放大 1/feat_scale，使初始输出幅值不变 → 只改学习动力学，不改 init 分布。
        with torch.no_grad():
            self.proj.weight.mul_(1.0 / self.feat_scale)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (N,) -> (N, 1)
        angles = x.unsqueeze(-1) * self.freqs  # (N, 1) * (n_feats) -> (N, n_feats)
        fourier = torch.cat([angles.sin(), angles.cos()], dim=-1) # (N, 2*n_feats)
        fourier = fourier * self.feat_scale  # 归一化到 ‖b‖²=1（见 __init__）
        return self.proj(fourier) # (N, d_model)


class ValueEncoder(nn.Module):
    """
    统一的值编码器入口。
    根据叶子节点的类型调用不同的子编码器。
    
    数值编码采用 Base-2 Mantissa-Exponent Split (xVal 机制)：
      x = M × 2^E  (通过 torch.frexp 在 O(1) 时间分解)
      M ∈ [0.5, 1.0) 或 (-1.0, -0.5]，x=0 时 M=0
      V_num = Fourier(M) + Embedding(E)
    """
    def __init__(self, d_model: int, frozen_lm_dim: int, n_fourier_feats: int, fourier_learnable: bool,
                 exponent_min: int = -50, exponent_max: int = 49,
                 numeric_path_film: bool = False):
        super().__init__()
        self.numeric_path_film = numeric_path_film
        
        # ── 数值型编码：Base-2 科学计数法解构 ──
        # 尾数编码器（傅里叶特征，M ∈ [-1, 1] 天然归一化）
        self.mantissa_encoder = FourierFeatureEncoder(
            n_feats=n_fourier_feats, 
            d_model=d_model, 
            learnable=fourier_learnable
        )
        # 指数嵌入表：从 exponent_min/max 派生 bins 数和 offset
        n_exponent_bins = exponent_max - exponent_min + 1
        self.exponent_embed = nn.Embedding(n_exponent_bins, d_model)
        self.n_exponent_bins = n_exponent_bins
        self.exponent_offset = -exponent_min  # E=0 映射到 index -exponent_min
        if numeric_path_film:
            self.numeric_path_film_proj = nn.Linear(d_model, 2 * d_model)
            nn.init.zeros_(self.numeric_path_film_proj.weight)
            nn.init.zeros_(self.numeric_path_film_proj.bias)
        else:
            self.numeric_path_film_proj = None
        
        # 文本型编码：先用 FrozenLM 得到特征，这里再做线性投影
        self.text_proj = nn.Linear(frozen_lm_dim, d_model)
        
        # 布尔型编码：简单的 0/1 标量投影
        self.bool_proj = nn.Linear(1, d_model)
        
        # 掩码向量
        self.mask_token = nn.Parameter(torch.randn(d_model))
        
    def forward(self, node_types: list[str], raw_values: list, lm_embeddings: Tensor = None,
                path_context: Tensor = None) -> Tensor:
        """
        对一批混合类型的值进行编码。
        node_types: 长度为 N 的列表，如 ["number", "string", "mask", ...]
        raw_values: 长度为 N 的原始值列表，如 [2.1, "TiO2", "[MASK]", ...]
        lm_embeddings: 如果该 batch 中有 string，需在外部先用 FrozenLM 算出 (N_str, lm_dim) 传入。
                       如果没有 string，可传 None。
        path_context: 可选 (N, d_model) 路径上下文；启用 numeric_path_film 时只调制 number token。
        返回: (N, d_model) 的特征表示
        """
        N = len(node_types)
        device = self.mask_token.device
        out = torch.zeros(N, self.mask_token.size(0), device=device)
        
        # 为了高效处理，将不同类型分组
        num_indices = []
        num_vals = []
        
        str_indices = []
        
        bool_indices = []
        bool_vals = []
        
        mask_indices = []
        
        for i, (t, v) in enumerate(zip(node_types, raw_values)):
            if t == "number":
                num_indices.append(i)
                num_vals.append(float(v))
            elif t == "string":
                str_indices.append(i)
            elif t == "boolean":
                bool_indices.append(i)
                bool_vals.append(1.0 if v else 0.0)
            elif t == "mask":
                mask_indices.append(i)
            else:
                raise ValueError(f"Unknown value type: {t}")
                
        # ── 数值型：Base-2 torch.frexp 向量化编码 ──
        if num_indices:
            raw = torch.tensor(num_vals, dtype=torch.float32, device=device)
            raw = torch.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=-1e6)
            m_tensor, e_tensor = torch.frexp(raw)
            # m_tensor ∈ [0.5, 1.0) 或 (-1.0, -0.5]，x=0 时 m=0
            # e_tensor 为整数指数，x = m * 2^e
            e_indices = (e_tensor + self.exponent_offset).clamp(0, self.n_exponent_bins - 1).long()
            
            m_emb = self.mantissa_encoder(m_tensor)   # (N_num, d_model) — 傅里叶编码尾数
            e_emb = self.exponent_embed(e_indices)     # (N_num, d_model) — 查表获取量级
            num_emb = m_emb + e_emb  # 直接相加，与 pos+token embedding 范式一致
            if self.numeric_path_film:
                if path_context is None:
                    raise ValueError("numeric_path_film=True requires path_context.")
                gamma_beta = self.numeric_path_film_proj(path_context[num_indices])
                gamma, beta = gamma_beta.chunk(2, dim=-1)
                num_emb = num_emb * (1.0 + torch.tanh(gamma)) + beta
            out[num_indices] = num_emb
            
        if str_indices:
            if lm_embeddings is None:
                raise ValueError("String nodes found but lm_embeddings is None.")
            if lm_embeddings.shape[0] != len(str_indices):
                raise ValueError(f"lm_embeddings 数量 ({lm_embeddings.shape[0]}) 与字符串节点数量 ({len(str_indices)}) 不符")
            # 注意：lm_embeddings 是已经在 GPU 上的张量
            out[str_indices] = self.text_proj(lm_embeddings)
            
        if bool_indices:
            bool_tensor = torch.tensor(bool_vals, dtype=torch.float32, device=device).unsqueeze(1) # (N_bool, 1)
            out[bool_indices] = self.bool_proj(bool_tensor)
            
        if mask_indices:
            out[mask_indices] = self.mask_token.expand(len(mask_indices), -1)
            
        return out
