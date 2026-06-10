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


class FATEImputerNet(nn.Module):
    """
    FATE-inspired mask-aware Transformer imputer for numerical tabular data.

    This network keeps FATE's incomplete data encoding idea:
    observed values use value embeddings, missing values use feature-specific
    missing embeddings, and attention masks missing key positions.
    """

    def __init__(self, num_features, embedding_dim=32, depth=3, heads=4, dropout=0.1):
        super().__init__()
        if embedding_dim % heads != 0:
            raise ValueError("embedding_dim must be divisible by heads.")

        self.num_features = num_features
        self.embedding_dim = embedding_dim

        # observed 数值进入这个模块，变成 feature token embeddings。
        self.value_embedding = ContinuousFeatureEmbedding(num_features, embedding_dim)

        # 每一列都有一个可学习的 missing embedding。
        # 如果第 j 列缺失，不使用填 0 后的 value embedding，
        # 而是使用 missing_embeddings[j] 表示“第 j 列缺失”。
        self.missing_embeddings = nn.Parameter(torch.randn(num_features, embedding_dim) * 0.02)

        # feature embedding 告诉模型“这个 token 属于哪一列”。
        # 否则所有列只是一串 token，模型不容易区分年龄/收入/信用分等列身份。
        self.feature_embeddings = nn.Parameter(torch.randn(num_features, embedding_dim) * 0.02)

        # 堆叠多层 missing-aware Transformer block。
        self.blocks = nn.ModuleList([
            MissingAwareTransformerBlock(embedding_dim, heads, dropout)
            for _ in range(depth)
        ])

        # reconstruction head 把每个 token 的 hidden vector 映射回一个数值。
        # 输入:  [batch_size, num_features, embedding_dim]
        # 输出:  [batch_size, num_features, 1]
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
        # x:    [batch_size, num_features]，缺失位置已经临时填 0。
        # mask: [batch_size, num_features]，1=observed/visible，0=missing/hidden。

        # 先把每个数值标量映射成 token embedding。
        # value_embeddings: [batch_size, num_features, embedding_dim]
        value_embeddings = self.value_embedding(x)

        # 将每列的 missing embedding 扩展到 batch 维度。
        # missing_embeddings: [batch_size, num_features, embedding_dim]
        missing_embeddings = self.missing_embeddings.unsqueeze(0).expand(x.size(0), -1, -1)

        # 根据 mask 选择每个位置使用哪种 embedding：
        # mask=1: 使用真实可见数值的 value embedding；
        # mask=0: 使用该列自己的 missing embedding。
        #
        # 这一步是 FATE incomplete data encoding 的核心：
        # 不把缺失值当作数值 0，而是明确编码成“该特征缺失”。
        token_embeddings = (
            mask.unsqueeze(-1) * value_embeddings
            + (1.0 - mask).unsqueeze(-1) * missing_embeddings
        )

        # 加上列身份 embedding，让模型知道每个 token 对应哪一个特征列。
        token_embeddings = token_embeddings + self.feature_embeddings.unsqueeze(0)

        hidden = token_embeddings
        for block in self.blocks:
            # 每层 block 都会用 mask 防止 missing key 被当成可靠信息。
            hidden = block(hidden, mask)

        # 对每一列都输出一个预测值。
        # squeeze(-1) 后 shape: [batch_size, num_features]
        return self.reconstruction_head(hidden).squeeze(-1)


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
