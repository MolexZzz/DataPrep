import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def normalization(data):
    """
    Min-max normalization with NaN-aware statistics.

    每一列单独计算 min/max，并忽略 NaN。
    这样不同量纲的特征会被映射到接近 [0, 1] 的范围，
    便于神经网络训练，也匹配 FATEImputerNet 最后的 Sigmoid 输出。
    """
    _min = np.nanmin(data, axis=0)
    _max = np.nanmax(data, axis=0)
    _den = _max - _min
    _den[_den == 0] = 1e-6
    norm_data = (data - _min) / _den
    norm_parameters = {"min": _min, "max": _max, "den": _den}
    return norm_data, norm_parameters


def normalization_with_parameter(data, norm_parameters):
    """Use training-time min/max/den to normalize new input data."""
    return (data - norm_parameters["min"]) / norm_parameters["den"]


def renormalization(norm_data, norm_parameters):
    """Map normalized data back to the original feature scale."""
    return norm_data * norm_parameters["den"] + norm_parameters["min"]


def sample_observed_mask(mask, mask_rate):
    """
    Randomly hide observed entries to create supervised reconstruction targets.

    Args:
        mask: torch.Tensor, 1=observed, 0=missing.
        mask_rate: probability of hiding an observed entry.

    Returns:
        train_mask: entries visible to the model.
        target_mask: entries hidden from the model and used for loss.

    这是 FATE-imputer 训练最关键的一步。

    真实 missing 位置没有 ground truth，不能直接计算误差。
    所以训练时从 observed 位置里随机挑一部分“人为遮住”，
    让模型恢复这些位置，并用原始 observed value 计算 loss。

    例子：
        mask        = [1, 0, 1]
        random_mask = [0, 0, 1]
        target_mask = [0, 0, 1]  # 只批改第 3 个位置
        train_mask  = [1, 0, 0]  # 模型只能看见第 1 个位置
    """
    # random_mask=1 表示“这次训练想遮住这个位置”。
    # 注意它还没有区分该位置本来是不是 observed。
    random_mask = (torch.rand_like(mask) < mask_rate).float()

    # target_mask=1 的位置必须同时满足：
    # 1) 原本 observed，有真实答案；
    # 2) 本轮随机被遮住。
    target_mask = mask * random_mask

    # train_mask=1 的位置表示模型训练时还能看见。
    # 原本 missing 的位置一定看不见；本轮被人为遮住的位置也看不见。
    train_mask = mask * (1.0 - random_mask)
    return train_mask, target_mask


class ContinuousFeatureEmbedding(nn.Module):
    """
    Embed each numerical feature into a token vector.

    Transformer 处理的是 token embedding，而不是原始标量。
    对数值表格来说，一个单元格原本只是一个数，例如 0.7；
    这里会把它映射成 embedding_dim 维向量。

    输入:
        x: [batch_size, num_features]
    输出:
        embeddings: [batch_size, num_features, embedding_dim]
    """

    def __init__(self, num_features, embedding_dim):
        super().__init__()
        self.num_features = num_features
        self.embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim),
            )
            for _ in range(num_features)
        ])
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        outputs = []
        for feature_idx, embedding in enumerate(self.embeddings):
            # 取出第 feature_idx 列，shape: [batch_size, 1]。
            # 每列使用自己的小 MLP，这对应 FATE 论文中
            # “continuous feature uses a fully-connected layer”的思想。
            feature = x[:, feature_idx:feature_idx + 1]

            # embedding(feature): [batch_size, embedding_dim]
            # unsqueeze(1): [batch_size, 1, embedding_dim]
            outputs.append(embedding(feature).unsqueeze(1))

        # 拼回所有列，得到 [batch_size, num_features, embedding_dim]。
        return torch.cat(outputs, dim=1)


class MissingAwareTransformerBlock(nn.Module):
    """
    Transformer block that prevents missing feature tokens from serving as keys.

    PyTorch's key_padding_mask uses True for positions that should be ignored.
    The project mask uses 1=observed and 0=missing, so mask == 0 is ignored.
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
        for module in self.feed_forward:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x, mask):
        # mask 的项目语义是 1=observed，0=missing。
        # PyTorch MultiheadAttention 的 key_padding_mask 语义相反：
        # True 表示这个 key 位置要被忽略。
        # 因此 missing 位置 mask==0 要变成 True。
        key_padding_mask = mask == 0

        # Avoid NaN attention when an entire row has no visible observed token.
        # 如果某一行全是 missing，所有 key 都被屏蔽，softmax 会没有合法位置，
        # 可能产生 NaN。这里遇到全缺失行时，不屏蔽该行的 key，
        # 让模型至少能基于 missing embedding / feature embedding 给出输出。
        all_missing = key_padding_mask.all(dim=1)
        if all_missing.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_missing] = False

        # self-attention:
        # query/key/value 都来自 x。
        # key_padding_mask 会让 missing key 的 attention 权重变为 0，
        # 等价于在 attention score 上给 missing 位置加 -inf。
        attention_output, _ = self.attention(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        # 标准 Transformer block 结构：
        # residual connection + LayerNorm。
        x = self.norm1(x + self.dropout(attention_output))

        # feed-forward network 对每个 token 独立做非线性变换，
        # 再接 residual + LayerNorm。
        feed_forward_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(feed_forward_output))
        return x


class RegularTransformerBlock(nn.Module):
    """
    Standard Transformer block without missing-aware key masking.

    对应原始 FATE 论文 model.py 中第 1 层到最后一层的 Attention：
    原论文只有第 0 层用 First_Attention（带 key_padding_mask），
    其余层用普通 Attention，不屏蔽任何 key。

    这里和 MissingAwareTransformerBlock 结构完全相同，
    唯一区别是 forward 不接收也不使用 mask 参数。
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
        for module in self.feed_forward:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # 标准 self-attention，不传 key_padding_mask，所有 token 都可作为 key。
        attention_output, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attention_output))
        feed_forward_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(feed_forward_output))
        return x


class FATEImputerNet(nn.Module):
    """
    FATE-inspired mask-aware Transformer imputer for numerical tabular data.

    与原始 FATE 论文对齐的三个关键设计：

    1. 缺失位置用可学习的 missing embedding 替换 value embedding（原论文 embed_data_mask）。

    2. mask 矩阵作为额外 token 列输入（对应原论文 run.py:292）：
       原论文把 mask 矩阵当成连续特征拼到 x_cont：
           x_cont = torch.cat([x_cont, con_mask], dim=1)
       这里同样把 mask 的 0/1 值做 embedding 后作为额外 d 个 token 输入 Transformer，
       让模型直接看到哪些位置是 observed/missing 的显式数值信息。

    3. 只有第 0 层用 missing-aware attention（对应原论文 model.py:309-316）：
       原论文 Transformer 只有第 0 层用 First_Attention（带 key_padding_mask），
       其余层用普通 Attention。
    """

    def __init__(self, num_features, embedding_dim=32, depth=3, heads=4, dropout=0.1):
        super().__init__()
        if embedding_dim % heads != 0:
            raise ValueError("embedding_dim must be divisible by heads.")

        self.num_features = num_features
        self.embedding_dim = embedding_dim

        # value token embedding：把每个连续特征值映射成向量。
        # 对应原论文 simple_MLP：每列一个独立的小 MLP。
        self.value_embedding = ContinuousFeatureEmbedding(num_features, embedding_dim)

        # mask token embedding：把 mask 的 0/1 值映射成向量。
        # 对应原论文 run.py:292 把 con_mask 当连续特征拼入 x_cont 的做法。
        # 每列的 mask 值（0.0 或 1.0）经过各自独立的小 MLP 变成 embedding token。
        self.mask_embedding = ContinuousFeatureEmbedding(num_features, embedding_dim)

        # 每列的可学习 missing embedding。
        # 当第 j 列缺失/被遮住时，该 token 不用 value embedding，
        # 而用 missing_embeddings[j] 表示”第 j 列缺失”。
        self.missing_embeddings = nn.Parameter(torch.randn(num_features, embedding_dim) * 0.02)

        # 列身份 embedding，共 2 * num_features 个：
        # 前 d 个给 value token，后 d 个给 mask token。
        # 对应原论文 pos_encodings = nn.Embedding(num_categories + num_continuous, dim)。
        self.feature_embeddings = nn.Parameter(
            torch.randn(num_features * 2, embedding_dim) * 0.02
        )

        # 第 0 层：missing-aware attention（对应原论文 First_Attention）。
        # 只有这一层会用 key_padding_mask 屏蔽缺失 key。
        self.first_block = MissingAwareTransformerBlock(embedding_dim, heads, dropout)

        # 第 1 到 depth-1 层：普通 attention（对应原论文后续层的普通 Attention）。
        self.rest_blocks = nn.ModuleList([
            RegularTransformerBlock(embedding_dim, heads, dropout)
            for _ in range(depth - 1)
        ])

        # reconstruction head：只对前 d 个 value token 输出预测值。
        # Sigmoid 将输出限制到 [0, 1]，匹配 min-max normalized target。
        self.reconstruction_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.reconstruction_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x, mask):
        # x:    [B, d]，缺失/被遮住的位置已临时填 0。
        # mask: [B, d]，1=observed/visible，0=missing/hidden。
        B = x.size(0)

        # --- value tokens（前 d 个 token）---
        # observed 位置用数值 embedding；missing/hidden 位置用该列的 missing embedding。
        value_embs = self.value_embedding(x)                                    # [B, d, emb]
        missing_embs = self.missing_embeddings.unsqueeze(0).expand(B, -1, -1)  # [B, d, emb]
        value_tokens = (
            mask.unsqueeze(-1) * value_embs
            + (1.0 - mask.unsqueeze(-1)) * missing_embs
        )                                                                        # [B, d, emb]

        # --- mask tokens（后 d 个 token）---
        # 把 mask 的 0/1 数值本身也作为 token 输入，
        # 让模型能直接”读到”哪些位置是缺失、哪些是可见。
        # 对应原论文 run.py:292:
        #   x_cont = torch.cat([x_cont, con_mask], dim=1)
        # 这里 mask token 始终是”可见的”（我们永远知道哪里缺失）。
        mask_tokens = self.mask_embedding(mask)                                  # [B, d, emb]

        # --- 拼接成 2d 个 token ---
        all_tokens = torch.cat([value_tokens, mask_tokens], dim=1)              # [B, 2d, emb]
        all_tokens = all_tokens + self.feature_embeddings.unsqueeze(0)

        # --- 构造 2d token 的 attention mask ---
        # value token 的可见性由原始 mask 决定；
        # mask token 永远可见（始终为 1）。
        # 对应原论文 run.py:293:
        #   con_mask = torch.cat([con_mask, con_mask_mask], dim=1)  # con_mask_mask 全为 1
        mask_tokens_visible = torch.ones(B, self.num_features, device=x.device)
        full_mask = torch.cat([mask, mask_tokens_visible], dim=1)               # [B, 2d]

        # --- 第 0 层：missing-aware attention ---
        # 对应原论文 model.py 中 n_layer==0 时使用 First_Attention（带 key_padding_mask）。
        hidden = self.first_block(all_tokens, full_mask)

        # --- 第 1 到 depth-1 层：普通 attention ---
        # 对应原论文后续层使用不带 mask 的普通 Attention。
        for block in self.rest_blocks:
            hidden = block(hidden)

        # --- 只对 value token（前 d 个）做 reconstruction ---
        # mask token 的 hidden 不参与输出，因为我们只需要补全原始 d 列的值。
        value_hidden = hidden[:, :self.num_features, :]                         # [B, d, emb]
        return self.reconstruction_head(value_hidden).squeeze(-1)               # [B, d]


def train_fate_algorithm(model, data_x, mask, params, device):
    """
    Train the FATE-inspired imputer with random observed masking.

    Args:
        model: FATEImputerNet.
        data_x: normalized data with NaN already filled by 0.
        mask: 1=observed, 0=missing.
        params: batch_size, epoch, learning_rate, mask_rate.
    """
    no, _ = data_x.shape
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    batch_size = params["batch_size"]
    mask_rate = params["mask_rate"]

    model.train()
    pbar = tqdm(range(params["epoch"]), desc="FATE Training")

    for _ in pbar:
        # 每个 epoch 打乱样本顺序。
        indices = np.random.permutation(no)
        current_loss = 0.0

        for start in range(0, no, batch_size):
            batch_idx = indices[start:start + batch_size]
            if len(batch_idx) == 0:
                continue

            # x_mb: 当前 batch 的归一化数据，NaN 已经填 0。
            # m_mb: 当前 batch 的原始 mask，1=observed，0=missing。
            x_mb = torch.tensor(data_x[batch_idx], dtype=torch.float32).to(device)
            m_mb = torch.tensor(mask[batch_idx], dtype=torch.float32).to(device)

            # 从 observed 位置中随机遮住一部分。
            # train_mask: 模型输入时可见的位置。
            # target_mask: 用来计算 reconstruction loss 的位置。
            train_mask, target_mask = sample_observed_mask(m_mb, mask_rate)

            # 如果这个 batch 恰好没有任何 target 位置，就没有可监督信号，跳过。
            if torch.sum(target_mask) <= 0:
                continue

            # 被 train_mask 遮住的位置输入为 0。
            # 但模型 forward 时还会接收 train_mask，
            # 因此这些 0 会被视为 hidden/missing，而不是可靠数值。
            x_input = train_mask * x_mb

            # x_hat: 模型对所有位置的预测，shape=[batch_size, num_features]。
            x_hat = model(x_input, train_mask)

            # 只在 target_mask=1 的位置计算 MSE。
            # 这些位置原本 observed，所以 x_mb 中有真实答案。
            # 原本真实 missing 的位置 target_mask 一定是 0，不参与训练 loss。
            loss = torch.sum(target_mask * (x_mb - x_hat) ** 2) / (torch.sum(target_mask) + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            current_loss = loss.item()

        pbar.set_postfix({"Loss": f"{current_loss:.4f}"})
