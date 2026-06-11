import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


# ============================================================
# Normalization（与 FATE_modules 相同）
# ============================================================

def normalization(data):
    """Min-max normalization, NaN-aware."""
    _min = np.nanmin(data, axis=0)
    _max = np.nanmax(data, axis=0)
    _den = _max - _min
    _den[_den == 0] = 1e-6
    norm_data = (data - _min) / _den
    norm_parameters = {"min": _min, "max": _max, "den": _den}
    return norm_data, norm_parameters


def normalization_with_parameter(data, norm_parameters):
    """Use training-time min/max/den to normalize new data."""
    return (data - norm_parameters["min"]) / norm_parameters["den"]


def renormalization(norm_data, norm_parameters):
    """Map normalized data back to original scale."""
    return norm_data * norm_parameters["den"] + norm_parameters["min"]


# ============================================================
# Random observed masking（与 FATE_modules 相同）
# ============================================================

def sample_observed_mask(mask, mask_rate):
    """
    从 observed 位置（mask=1）随机遮住一部分，制造有答案的补全训练题。

    Returns:
        train_mask:  模型训练时可见的位置（1=可见，0=隐藏）
        target_mask: 被人为遮住、用于计算 reconstruction loss 的位置
    """
    random_mask = (torch.rand_like(mask) < mask_rate).float()
    target_mask = mask * random_mask
    train_mask = mask * (1.0 - random_mask)
    return train_mask, target_mask


# ============================================================
# IPS 计算（DARN 特有）
# ============================================================

def compute_ips_simple(mask):
    """
    简单全局 IPS：每列用整列观测率计算，同列所有行共用同一 IPS 值。

    Args:
        mask: np.ndarray [n, d]，1=observed，0=missing

    Returns:
        ips: np.ndarray [n, d]，observed 位置有 IPS 值，missing 位置为 0

    原理：
        p_j    = mean(M_j)          ← 第 j 列的观测率
        IPS_j  = 1 / p_j            ← 逆倾向分数
        缺失位置（M=0）的 IPS 设为 0，因为它们不作为 attention key。
    """
    obs_rate = mask.mean(axis=0)                     # [d]
    obs_rate = np.clip(obs_rate, 1e-2, 1.0)
    ips_col = 1.0 / obs_rate                          # [d]
    ips = np.tile(ips_col, (len(mask), 1))            # [n, d]，广播到每行
    ips = ips * mask                                   # missing 位置 IPS = 0
    return ips.astype(np.float32)


def compute_ips_logistic(data, mask):
    """
    进阶逐格 IPS：对每一列，用逻辑回归估计每一行该格子缺失的概率。

    Args:
        data: np.ndarray [n, d]，NaN 已填 0 的归一化数据
        mask: np.ndarray [n, d]，1=observed，0=missing

    Returns:
        ips: np.ndarray [n, d]，每格独立 IPS，missing 位置为 0

    原理：
        对第 j 列，训练逻辑回归：
            输入 X = 所有列的值（缺失位置已用 0 或 fill_values 填充）
            标签 y = 1 - M_j（1=缺失，0=observed）
        预测出 P(M_ij = 0 | X_i)，即该格子缺失的概率。
        IPS_ij = 1 / (1 - P(M_ij = 0)) = 1 / P(M_ij = 1)

    注：
        随着渐进式自补全更新 data（填入更好的估计），
        再次调用此函数会得到更准确的 IPS。
    """
    from sklearn.linear_model import LogisticRegression

    n, d = data.shape
    p_miss = np.zeros((n, d), dtype=np.float32)

    for j in range(d):
        y = 1.0 - mask[:, j]           # 1=缺失，0=observed
        unique_vals = np.unique(y)
        if len(unique_vals) < 2:
            # 该列全部 observed 或全部 missing，用全局均值代替
            p_miss[:, j] = float(y.mean())
            continue
        try:
            lr = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs")
            lr.fit(data, y)
            p_miss[:, j] = lr.predict_proba(data)[:, 1]
        except Exception:
            p_miss[:, j] = float(y.mean())

    p_miss = np.clip(p_miss, 0.0, 0.95)
    ips = 1.0 / (1.0 - p_miss + 1e-8)
    ips = ips * mask                    # missing 位置 IPS = 0
    return ips.astype(np.float32)


# ============================================================
# 网络组件
# ============================================================

class ContinuousFeatureEmbedding(nn.Module):
    """
    把每个连续特征的标量值映射成 embedding_dim 维向量。

    每列使用独立的小 MLP（对应原论文 simple_MLP）：
        x[:, j] → Linear(1, emb) → ReLU → Linear(emb, emb) → emb 维向量

    输入:  [B, num_features]
    输出:  [B, num_features, embedding_dim]
    """

    def __init__(self, num_features, embedding_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim),
            )
            for _ in range(num_features)
        ])
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        outputs = []
        for i, emb in enumerate(self.embeddings):
            outputs.append(emb(x[:, i:i + 1]).unsqueeze(1))
        return torch.cat(outputs, dim=1)  # [B, d, emb]


class MissingAwareAttention(nn.Module):
    """
    支持两个额外操作的 self-attention（DARN 特有）：

    1. key_padding_mask：屏蔽缺失/被遮住的 key（与 FATE 相同）。
    2. ips_weights：对 key 位置的 attention score 乘以 IPS 权重。
       观测率越低的特征，IPS 越大，在 attention 中被放大。

    因为 nn.MultiheadAttention 不暴露原始 score 供修改，
    这里手动实现 attention 计算（使用 einsum）。
    """

    def __init__(self, embedding_dim, heads, dropout=0.0):
        super().__init__()
        assert embedding_dim % heads == 0
        self.heads = heads
        self.head_dim = embedding_dim // heads
        self.scale = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.to_out = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.to_qkv.weight)
        nn.init.xavier_uniform_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x, key_padding_mask=None, ips_weights=None):
        """
        x:               [B, T, emb]
        key_padding_mask:[B, T]，True 表示该 key 位置被忽略
        ips_weights:     [B, T]，每个 key 位置的 IPS 权重（None 表示不使用）
        """
        B, T, _ = x.shape
        h, d = self.heads, self.head_dim

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, h, d).transpose(1, 2)   # [B, h, T, d]
        k = k.view(B, T, h, d).transpose(1, 2)
        v = v.view(B, T, h, d).transpose(1, 2)

        # Attention score 矩阵
        sim = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale  # [B, h, T, T]

        # Step 1：IPS 加权（乘在 key 维度上）
        # ips_weights: [B, T] → [B, 1, 1, T]，广播到所有 query 和 head
        if ips_weights is not None:
            sim = sim * ips_weights.unsqueeze(1).unsqueeze(2)

        # Step 2：屏蔽缺失/被遮住的 key
        if key_padding_mask is not None:
            sim = sim.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            # 如果某行所有 key 都被屏蔽（全缺失行），避免 softmax 产生 NaN
            all_masked = key_padding_mask.all(dim=1)  # [B]
            if all_masked.any():
                sim = sim.clone()
                sim[all_masked] = 0.0

        attn = sim.softmax(dim=-1).to(sim.dtype)
        attn = self.dropout(attn)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.to_out(out)


class MissingAwareTransformerBlock(nn.Module):
    """
    第 0 层 Transformer block：missing-aware attention + 可选 IPS 加权。

    对应原论文 model.py 中 n_layer==0 时使用的 First_Attention。
    在 DARN 中额外支持 ips_weights，通过 MissingAwareAttention 实现。
    """

    def __init__(self, embedding_dim, heads, dropout):
        super().__init__()
        self.attention = MissingAwareAttention(embedding_dim, heads, dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.feed_forward:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, mask, ips_weights=None):
        """
        x:           [B, 2d, emb]（2d 个 token）
        mask:        [B, 2d]，1=可见，0=被屏蔽
        ips_weights: [B, 2d] 或 None
        """
        key_padding_mask = (mask == 0)
        attn_out = self.attention(x, key_padding_mask=key_padding_mask, ips_weights=ips_weights)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


class RegularTransformerBlock(nn.Module):
    """
    第 1 到 depth-1 层 Transformer block：普通 attention，无 mask，无 IPS。

    对应原论文 model.py 中后续层的普通 Attention。
    """

    def __init__(self, embedding_dim, heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.feed_forward:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


class DARNImputerNet(nn.Module):
    """
    DARN-inspired mask-aware Transformer imputer。

    在 FATE-imputer 结构基础上增加：
    1. IPS 加权 attention（第 0 层，通过 ips_weights 参数传入）。
    2. 缺失分布预测头 mask_prediction_head（可选，use_prob_head=True 时启用）。

    结构（与原论文对齐的部分和 FATEImputerNet 相同）：
        value_embedding   → 数值 token（per-feature MLP）
        mask_embedding    → mask token（0/1 mask 值的 per-feature MLP）
        missing_embeddings→ 缺失位置的可学习替换向量
        feature_embeddings→ 列身份 embedding（2d 个）
        first_block       → MissingAwareTransformerBlock（支持 IPS）
        rest_blocks       → RegularTransformerBlock（普通 attention）
        reconstruction_head   → value token → 预测值
        mask_prediction_head  → value token → 缺失概率（可选）
    """

    def __init__(
        self,
        num_features,
        embedding_dim=32,
        depth=3,
        heads=4,
        dropout=0.1,
        use_prob_head=False,
    ):
        super().__init__()
        assert embedding_dim % heads == 0, "embedding_dim must be divisible by heads."
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.use_prob_head = use_prob_head

        self.value_embedding = ContinuousFeatureEmbedding(num_features, embedding_dim)
        self.mask_embedding = ContinuousFeatureEmbedding(num_features, embedding_dim)
        self.missing_embeddings = nn.Parameter(
            torch.randn(num_features, embedding_dim) * 0.02
        )
        self.feature_embeddings = nn.Parameter(
            torch.randn(num_features * 2, embedding_dim) * 0.02
        )

        self.first_block = MissingAwareTransformerBlock(embedding_dim, heads, dropout)
        self.rest_blocks = nn.ModuleList([
            RegularTransformerBlock(embedding_dim, heads, dropout)
            for _ in range(depth - 1)
        ])

        self.reconstruction_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid(),
        )

        if use_prob_head:
            self.mask_prediction_head = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 1),
                nn.Sigmoid(),
            )

        self._init_weights()

    def _init_weights(self):
        heads_to_init = [self.reconstruction_head]
        if self.use_prob_head:
            heads_to_init.append(self.mask_prediction_head)
        for head in heads_to_init:
            for m in head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x, mask, ips_weights=None):
        """
        x:           [B, d]，缺失位置已填 0（或 fill_values）
        mask:        [B, d]，1=visible，0=missing/hidden
        ips_weights: [B, d] 或 None，value token 的 IPS 权重

        Returns（use_prob_head=False）:
            x_hat: [B, d]

        Returns（use_prob_head=True）:
            (x_hat, missing_prob): 各 [B, d]
        """
        B = x.size(0)

        # --- Value tokens（前 d 个 token）---
        value_embs = self.value_embedding(x)                                    # [B, d, emb]
        missing_embs = self.missing_embeddings.unsqueeze(0).expand(B, -1, -1)  # [B, d, emb]
        value_tokens = (
            mask.unsqueeze(-1) * value_embs
            + (1.0 - mask.unsqueeze(-1)) * missing_embs
        )

        # --- Mask tokens（后 d 个 token）---
        # 把 mask 的 0/1 数值 embedding 后作为额外 d 个 token（对应原论文 run.py:292）
        mask_tokens = self.mask_embedding(mask)                                  # [B, d, emb]

        # --- 拼接 2d tokens，加列身份 embedding ---
        all_tokens = torch.cat([value_tokens, mask_tokens], dim=1)              # [B, 2d, emb]
        all_tokens = all_tokens + self.feature_embeddings.unsqueeze(0)

        # --- 构造 2d tokens 的 attention mask ---
        # value tokens 按 mask 决定可见性；mask tokens 永远可见
        mask_tokens_visible = torch.ones(B, self.num_features, device=x.device)
        full_mask = torch.cat([mask, mask_tokens_visible], dim=1)               # [B, 2d]

        # --- 构造 2d tokens 的 IPS 权重 ---
        # value tokens 使用传入的 ips_weights；mask tokens IPS = 1（不放大也不缩小）
        if ips_weights is not None:
            mask_token_ips = torch.ones(B, self.num_features, device=x.device)
            full_ips = torch.cat([ips_weights, mask_token_ips], dim=1)         # [B, 2d]
        else:
            full_ips = None

        # --- 第 0 层：missing-aware + IPS attention ---
        hidden = self.first_block(all_tokens, full_mask, ips_weights=full_ips)

        # --- 第 1 到 depth-1 层：普通 attention ---
        for block in self.rest_blocks:
            hidden = block(hidden)

        # --- 从 value tokens（前 d 个）输出预测 ---
        value_hidden = hidden[:, :self.num_features, :]                         # [B, d, emb]
        x_hat = self.reconstruction_head(value_hidden).squeeze(-1)              # [B, d]

        if self.use_prob_head:
            # 从同一个 value_hidden 预测每格缺失的概率
            missing_prob = self.mask_prediction_head(value_hidden).squeeze(-1)  # [B, d]
            return x_hat, missing_prob

        return x_hat


# ============================================================
# 训练循环
# ============================================================

def train_darn_algorithm(model, data_x, mask, params, device, ips=None, ips_method="simple"):
    """
    训练 DARNImputerNet，集成四个层次：

    层次 1（基础）：
        random observed masking + MAE/MSE reconstruction loss

    层次 2（渐进式自补全）：
        每隔 progressive_interval 个 epoch，用当前模型预测 missing 位置的值，
        更新 fill_values。
        后续 epoch 的训练 loss 额外加入 pseudo-label loss：
            L_pseudo = MSE(x_hat[missing], fill_values[missing])
        fill_values 越来越准，pseudo-label 监督越来越可靠。
        如果 ips_method="logistic"，还会用 fill_values 更新 data 后重新计算 IPS。

    层次 3（IPS 加权 attention）：
        把传入的 ips（[n, d]）在 batch 维度切片后传给 model.forward，
        作用于第 0 层 attention 的 key 维度。

    层次 4（缺失分布预测 L_prob）：
        当 model.use_prob_head=True 时，model 额外输出 missing_prob，
        用 BCE loss 监督：L_prob = BCE(missing_prob, 1 - m_mb)。

    Total loss = L_rec + gamma * L_pseudo + beta * L_prob

    Args:
        model:      DARNImputerNet
        data_x:     [n, d]，归一化后 NaN 已填 0 的数据
        mask:       [n, d]，1=observed，0=missing
        params:     训练超参数字典
        device:     torch device
        ips:        [n, d] 或 None，预计算的 IPS 权重
        ips_method: "simple" 或 "logistic"，用于渐进更新 IPS 时选择方法
    """
    n, _ = data_x.shape
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    batch_size = params["batch_size"]
    mask_rate = params["mask_rate"]
    loss_type = params.get("loss_type", "mae")
    use_progressive = params.get("use_progressive", True)
    progressive_interval = params.get("progressive_interval", 10)
    gamma = params.get("gamma", 0.1)   # pseudo-label loss 权重
    beta = params.get("beta", 0.1)    # L_prob loss 权重

    # IPS 转 tensor
    ips_tensor = (
        torch.tensor(ips, dtype=torch.float32).to(device)
        if ips is not None else None
    )

    # 渐进式自补全：fill_values 初始化为全 0（冷启动）
    fill_values = np.zeros_like(data_x, dtype=np.float32)
    fill_initialized = False  # 第一次更新后变为 True，才激活 pseudo-label loss

    model.train()
    pbar = tqdm(range(params["epoch"]), desc="DARN Training")

    for epoch in pbar:

        # ── 渐进式自补全：每隔若干 epoch 更新 fill_values ──────────────────
        if use_progressive and epoch > 0 and epoch % progressive_interval == 0:
            model.eval()
            with torch.no_grad():
                all_x = torch.tensor(data_x, dtype=torch.float32).to(device)
                all_m = torch.tensor(mask, dtype=torch.float32).to(device)
                out = model(all_x, all_m, ips_weights=ips_tensor)
                pred = out[0].cpu().numpy() if model.use_prob_head else out.cpu().numpy()

            # 只更新 missing 位置；observed 位置保留原始值
            fill_values = (1.0 - mask) * pred + mask * data_x
            fill_initialized = True

            # 如果使用 logistic IPS，用更新后的 fill_values 重新估计缺失概率
            if ips is not None and ips_method == "logistic":
                x_for_ips = mask * data_x + (1.0 - mask) * fill_values
                ips_updated = compute_ips_logistic(x_for_ips, mask)
                ips_tensor = torch.tensor(ips_updated, dtype=torch.float32).to(device)

            model.train()

        # ── Batch 训练 ───────────────────────────────────────────────────────
        indices = np.random.permutation(n)
        current_loss = 0.0

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            if not len(batch_idx):
                continue

            x_mb = torch.tensor(data_x[batch_idx], dtype=torch.float32).to(device)
            m_mb = torch.tensor(mask[batch_idx], dtype=torch.float32).to(device)
            ips_mb = ips_tensor[batch_idx] if ips_tensor is not None else None

            # 随机遮住 observed 位置，制造训练题
            train_mask, target_mask = sample_observed_mask(m_mb, mask_rate)
            if torch.sum(target_mask) <= 0:
                continue

            x_input = train_mask * x_mb

            # 前向传播
            out = model(x_input, train_mask, ips_weights=ips_mb)
            if model.use_prob_head:
                x_hat, missing_prob = out
            else:
                x_hat = out

            # ── 层次 1：reconstruction loss（只在 target_mask=1 的位置算）──
            if loss_type == "mae":
                l_rec = (
                    torch.sum(target_mask * torch.abs(x_mb - x_hat))
                    / (torch.sum(target_mask) + 1e-8)
                )
            else:
                l_rec = (
                    torch.sum(target_mask * (x_mb - x_hat) ** 2)
                    / (torch.sum(target_mask) + 1e-8)
                )

            loss = l_rec

            # ── 层次 2：pseudo-label loss（渐进式，激活后才加）─────────────
            # L_pseudo = MSE(x_hat[missing], fill_values[missing])
            # 直觉：模型在 missing 位置的预测应该越来越接近不断改善的 fill 估计。
            if use_progressive and fill_initialized:
                fill_mb = torch.tensor(
                    fill_values[batch_idx], dtype=torch.float32
                ).to(device)
                orig_miss = 1.0 - m_mb                          # missing 位置标记
                l_pseudo = (
                    torch.sum(orig_miss * (x_hat - fill_mb) ** 2)
                    / (torch.sum(orig_miss) + 1e-8)
                )
                loss = loss + gamma * l_pseudo

            # ── 层次 4：L_prob（缺失分布预测 BCE loss）──────────────────────
            # L_prob = BCE(missing_prob, 1 - m_mb)
            # 直觉：模型从补全结果推断哪里会缺失，倒逼主任务学习缺失规律。
            if model.use_prob_head:
                l_prob = F.binary_cross_entropy(missing_prob, 1.0 - m_mb)
                loss = loss + beta * l_prob

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            current_loss = loss.item()

        pbar.set_postfix({"Loss": f"{current_loss:.4f}"})
