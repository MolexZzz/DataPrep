# DARN-imputation 实现说明

## 1. 实现目标

本文档说明如何把 DARN 论文中的思想改造成 DataPrep 当前 `tabular/imputation` 模块可用的缺失值补全算法。

新增文件：

```text
tabular/imputation/DARN.py
tabular/imputation/DARN_modules.py
```

当前实现的定位是：

```text
DARN-inspired missing-aware Transformer imputer（四层次渐进增强版）
```

它不是 DARN 原论文联邦学习系统的完整复现，而是针对单机 imputation 任务的工程化改造。

---

## 2. 联邦学习和 DARN 原论文

### 2.1 联邦学习是什么

联邦学习是"多台机器协同训练，不共享原始数据"的机器学习框架：

```text
普通机器学习：一台机器 + 一个完整数据集 → 训练一个模型

联邦学习：
    K 个客户端，每台机器有自己的私有数据 (X_k, Y_k, M_k)
    各客户端本地训练，只把模型参数发送给服务端
    服务端把多个客户端的参数聚合后发回
    各客户端用聚合后的参数继续本地训练
```

DARN 的联邦场景加了一个创新：

```text
不同客户端的缺失模式不同 → 缺失分布有"互补性"
服务端根据"缺失互补分数"加权聚合，让缺失互补的客户端互相补益
```

### 2.2 DARN 原论文的整体目标函数

$$
\min_{\theta}
\sum_k w_k
\left[
L_{sup}(\theta^P_k)
+ \alpha L_{rec}(\theta^I_k)
+ L_{prob}(\theta^M_k)
\right]
$$

其中：

```text
k       = 第 k 个客户端
w_k     = 该客户端的聚合权重（由缺失互补分数决定）
L_sup   = 下游预测任务 loss（分类/回归）
L_rec   = imputation 重构 loss
L_prob  = 缺失分布预测 loss
```

### 2.3 对 DataPrep 单机 imputation 有用的部分

DataPrep 只有一台机器，只需补全数值，所以：

```text
不需要：L_sup、多客户端、联邦聚合、missing complementarity score
不需要：w_k 的计算、个性化参数平均

需要：L_rec 的训练方法（missing-aware Transformer + random masking）
需要：IPS 加权 attention 的思路（论文 4.2 节）
可选：L_prob（单机版仍然有用，让模型学习缺失规律）
```

---

## 3. 统一符号

$$
X \in \mathbb{R}^{n \times d},\quad
M \in \{0, 1\}^{n \times d},\quad
M_{ij} =
\begin{cases}
1, & X_{ij}\ \text{observed} \\
0, & X_{ij}\ \text{missing}
\end{cases}
$$

接口：

```python
model.train(data_missing, missing_mask)
imputed_data = model.predict(data_missing)
```

---

## 4. 四个层次概览

| 层次 | 参数开关 | 核心内容 | Loss 组成 |
|---|---|---|---|
| 1 基础 | `loss_type="mae"` | 与 FATE 相同结构，loss 改 MAE | $L_{rec}$ |
| 2 渐进式 | `use_progressive=True` | fill_values 迭代更新 + pseudo-label | $L_{rec} + \gamma L_{pseudo}$ |
| 3 IPS | `use_ips=True` | IPS 加权 attention key | $L_{rec} + \gamma L_{pseudo}$ |
| 4 L_prob | `use_prob_head=True` | 缺失分布预测辅助任务 | $L_{rec} + \gamma L_{pseudo} + \beta L_{prob}$ |

四个层次可以自由组合，互不依赖。

---

## 5. 所有层次共同的基础结构

### 5.1 模型结构（与 FATE-imputer 对齐，结构完全相同）

DARNImputerNet 和 FATEImputerNet 的网络结构完全相同，包含：

```text
value_embedding    （per-feature MLP，把数值映射成 token）
mask_embedding     （per-feature MLP，把 0/1 mask 值映射成 mask token）
missing_embeddings （每列一个可学习 missing embedding，替换缺失位置）
feature_embeddings （2d 个列身份 embedding）
first_block        （MissingAwareTransformerBlock，第 0 层，带 missing mask）
rest_blocks        （RegularTransformerBlock，第 1 到 depth-1 层，普通 attention）
reconstruction_head（只对前 d 个 value token 输出预测值）
```

区别：
- DARN 的 `MissingAwareTransformerBlock` 额外支持 `ips_weights` 参数
- DARN 可选加 `mask_prediction_head`（`use_prob_head=True` 时启用）

### 5.2 训练基础流程（与 FATE 相同）

```text
1. min-max normalization
2. NaN 填 0
3. 每个 batch 做 random observed masking：T = M ⊙ R，M_train = M ⊙ (1-R)
4. x_input = M_train ⊙ X
5. x_hat = model(x_input, M_train)
6. 在 T=1 的位置计算 L_rec
7. 反向传播
```

### 5.3 层次 1 与 FATE 的唯一区别：MAE loss

FATE 实现用 MSE，DARN 论文用 MAE：

$$
L_{MAE} = \frac{\sum_{ij} T_{ij} |X_{ij} - \hat{X}_{ij}|}{\sum_{ij} T_{ij} + \epsilon}
$$

代码：

```python
# DARN_modules.py: train_darn_algorithm
if loss_type == "mae":
    l_rec = torch.sum(target_mask * torch.abs(x_mb - x_hat)) / (torch.sum(target_mask) + 1e-8)
else:  # mse
    l_rec = torch.sum(target_mask * (x_mb - x_hat) ** 2) / (torch.sum(target_mask) + 1e-8)
```

---

## 6. 层次 2：渐进式自补全训练

### 6.1 问题背景

在基础版（层次 1）里，missing 位置在整个训练过程中都填 0。缺失位置永远对模型提供"0"这个没有意义的值，训练完之后 model 对缺失位置的预测质量受限于"从来没见过合理估计"。

### 6.2 核心思路

```text
第 0 到 K-1 个 epoch：   冷启动，fill_values = 0
第 K 个 epoch：          用当前模型预测 missing 位置 → 更新 fill_values
第 K 到 2K-1 个 epoch：  用 fill_values 作为 pseudo-label 监督 missing 位置
第 2K 个 epoch：         再次更新 fill_values（预测更准了）
…
```

### 6.3 fill_values 的更新

代码（`DARN_modules.py: train_darn_algorithm`）：

```python
if use_progressive and epoch > 0 and epoch % progressive_interval == 0:
    model.eval()
    with torch.no_grad():
        all_x = torch.tensor(data_x, ...)
        all_m = torch.tensor(mask, ...)
        out = model(all_x, all_m, ips_weights=ips_tensor)
        pred = out[0].cpu().numpy() if model.use_prob_head else out.cpu().numpy()
    # 只更新 missing 位置，observed 位置保留原始值
    fill_values = (1.0 - mask) * pred + mask * data_x
    fill_initialized = True
```

用公式写：

$$
X^{fill,(e)}_{ij} =
\begin{cases}
X_{ij}, & M_{ij} = 1 \\
\hat{X}^{(e-1)}_{ij}, & M_{ij} = 0
\end{cases}
$$

用例子理解：假设 2 行 3 列，缺失位置有 [用户1收入, 用户2信用分]：

```text
epoch 0–9：  fill_values = [0, 0]
epoch 10：   模型预测：用户1收入≈0.55，用户2信用分≈0.65 → 更新 fill_values
epoch 11–19：fill_values = [0.55, 0.65]，模型有了更好的参考
epoch 20：   模型再次预测：用户1收入≈0.60，用户2信用分≈0.70 → 再次更新
…
```

### 6.4 pseudo-label loss

fill_values 更新后，会激活 pseudo-label loss：

$$
L_{pseudo} = \frac{\sum_{ij} (1 - M_{ij})(X^{fill}_{ij} - \hat{X}_{ij})^2}{\sum_{ij}(1 - M_{ij}) + \epsilon}
$$

含义：

```text
在 missing 位置（M=0），模型的预测应该越来越接近不断改善的 fill 估计。
fill 越准 → pseudo-label 质量越高 → 监督越可靠。
```

代码：

```python
if use_progressive and fill_initialized:
    fill_mb = torch.tensor(fill_values[batch_idx], ...)
    orig_miss = 1.0 - m_mb                 # missing 位置标记
    l_pseudo = torch.sum(orig_miss * (x_hat - fill_mb) ** 2) / (torch.sum(orig_miss) + 1e-8)
    loss = loss + gamma * l_pseudo
```

注意：`fill_initialized=False` 时（最初几个 epoch）不激活 pseudo-label loss，避免用全 0 作为不合理的伪标签干扰训练。

---

## 7. 层次 3：IPS 加权 attention

### 7.1 什么是 IPS

IPS = Inverse Propensity Score（逆倾向分数）。

**直觉**：某个特征越稀少（缺失率越高），它一旦出现就越有信息量，在 attention 里应该给它更高权重。

### 7.2 简单全局 IPS（`ips_method="simple"`）

整列共用一个 IPS：

$$
p_j = \frac{1}{n}\sum_i M_{ij}, \quad IPS_j = \frac{1}{p_j + \epsilon}
$$

同列所有行的 IPS 相同。

代码（`DARN_modules.py: compute_ips_simple`）：

```python
obs_rate = np.clip(mask.mean(axis=0), 1e-2, 1.0)  # [d]
ips_col = 1.0 / obs_rate                            # [d]
ips = np.tile(ips_col, (len(mask), 1))              # [n, d]
ips = ips * mask                                     # missing 位置 IPS = 0
```

用例子（1000 行，3 列）：

```text
收入列观测率 = 40% → IPS = 2.5（所有 1000 人的收入格子 IPS 都是 2.5）
年龄列观测率 = 95% → IPS = 1.05
信用分列     = 80% → IPS = 1.25
```

### 7.3 进阶逐格 IPS（`ips_method="logistic"`）

每格有自己的 IPS，由逻辑回归估计：

对第 j 列训练逻辑回归：
$$
\text{输入 } X_i\text{（其他列值）} \to \hat{P}(M_{ij}=0 \mid X_i)
$$

$$
IPS_{ij} = \frac{1}{1 - \hat{P}(M_{ij}=0 \mid X_i) + \epsilon}
$$

用例子理解：假设年龄和收入缺失相关：

```text
年轻用户（年龄=25）：预测收入缺失概率=0.75 → IPS=4.0
老年用户（年龄=60）：预测收入缺失概率=0.25 → IPS=1.33
```

同一"收入"列，不同的人的格子有不同的 IPS。

代码（`DARN_modules.py: compute_ips_logistic`）：

```python
for j in range(d):
    y = 1.0 - mask[:, j]           # 1=缺失，0=observed
    lr = LogisticRegression(...)
    lr.fit(data, y)
    p_miss[:, j] = lr.predict_proba(data)[:, 1]
ips = 1.0 / (1.0 - p_miss + 1e-8)
ips = ips * mask                    # missing 位置 IPS = 0
```

**渐进式 + 逻辑回归 IPS 的联动**：

当 `use_progressive=True` 且 `ips_method="logistic"` 时，每次更新 fill_values 后会重新计算 IPS：

```python
# DARN_modules.py: train_darn_algorithm
if ips is not None and ips_method == "logistic":
    x_for_ips = mask * data_x + (1.0 - mask) * fill_values
    ips_updated = compute_ips_logistic(x_for_ips, mask)
    ips_tensor = torch.tensor(ips_updated, ...).to(device)
```

联动链条：

```text
fill_values 更准 → 逻辑回归特征更准 → IPS 估计更准 → attention 加权更合理
```

### 7.4 IPS 如何作用于 attention

在 `MissingAwareAttention.forward` 中（`DARN_modules.py`）：

```python
sim = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale  # [B, h, T, T]

# IPS 乘到 key 维度（最后一维）
if ips_weights is not None:
    sim = sim * ips_weights.unsqueeze(1).unsqueeze(2)   # [B, 1, 1, T]
```

等价公式：

$$
S^{IPS}_{b,h,i,j} = S_{b,h,i,j} \cdot IPS_{b,j}
$$

再接 softmax 和 key_padding_mask（屏蔽缺失位置）。

用数字例子：

```text
query = "信用分" token，看 key = [年龄(1.05), 收入(4.0), 信用分(1.25)] 这三个 token

原始 attention score：[0.5,  0.6,  0.8]
乘 IPS 后：           [0.5×1.05, 0.6×4.0, 0.8×1.25]
                    = [0.525,   2.40,   1.00]

softmax 后，收入的 attention 权重大幅上升。
```

收入虽然稀少（缺失率高），但一旦出现就很重要——IPS 让模型更多地参考这一稀有信息。

### 7.5 IPS 权重的传递路径

在 `DARNImputerNet.forward`（`DARN_modules.py`）中：

```python
# 构造 2d tokens 的 IPS：value tokens 用传入的 ips_weights，mask tokens IPS=1
if ips_weights is not None:
    mask_token_ips = torch.ones(B, self.num_features, device=x.device)
    full_ips = torch.cat([ips_weights, mask_token_ips], dim=1)  # [B, 2d]
else:
    full_ips = None

hidden = self.first_block(all_tokens, full_mask, ips_weights=full_ips)
```

mask tokens 的 IPS 永远设为 1，因为 mask tokens 承载的是确定的 0/1 信息，不需要额外放大。

---

## 8. 层次 4：缺失分布预测 L_prob

### 8.1 什么是 L_prob

在补全 X_hat 之后，模型的 value token hidden state 还会经过一个额外的 MLP，输出每格缺失的概率：

$$
M'_{ij} = \text{MaskPredictionHead}(H^{val}_{ij}) \in [0, 1]
$$

再用 BCE loss 监督：

$$
L_{prob} = -\sum_{ij}
\left[
(1-M_{ij}) \log M'_{ij}
+ M_{ij} \log(1 - M'_{ij})
\right]
$$

其中：

```text
目标 = 1 - M（1 表示缺失，0 表示 observed）
M'  = 模型预测的缺失概率
```

### 8.2 为什么这能帮助补全

L_prob 是一个"附加考试"：

```text
主任务：把 X_hat 补得准（L_rec 监督）
附加题：从 X_hat 推断哪里会缺失（L_prob 监督）
```

做好附加题需要模型理解缺失规律。比如"收入低的人信用分更容易缺失"，模型学会这个规律后，补信用分时会更注意收入的值。

两个任务共用同一套 value_hidden 参数，所以 L_prob 的梯度会影响整个模型，间接改善补全。

### 8.3 代码

`DARNImputerNet.__init__`（`DARN_modules.py`）：

```python
if use_prob_head:
    self.mask_prediction_head = nn.Sequential(
        nn.Linear(embedding_dim, embedding_dim),
        nn.ReLU(),
        nn.Linear(embedding_dim, 1),
        nn.Sigmoid(),            # 输出 [0, 1] 范围的缺失概率
    )
```

`DARNImputerNet.forward`：

```python
value_hidden = hidden[:, :self.num_features, :]      # [B, d, emb]
x_hat = self.reconstruction_head(value_hidden).squeeze(-1)

if self.use_prob_head:
    missing_prob = self.mask_prediction_head(value_hidden).squeeze(-1)
    return x_hat, missing_prob
return x_hat
```

`train_darn_algorithm`（训练循环中）：

```python
if model.use_prob_head:
    l_prob = F.binary_cross_entropy(missing_prob, 1.0 - m_mb)
    loss = loss + beta * l_prob
```

---

## 9. 总 loss 公式

$$
L_{total} = L_{rec} + \gamma \cdot L_{pseudo} + \beta \cdot L_{prob}
$$

各项展开：

$$
L_{rec} = \frac{\sum_{ij} T_{ij} |X_{ij} - \hat{X}_{ij}|}{\sum_{ij} T_{ij} + \epsilon}
\quad (\text{层次 1，MAE})
$$

$$
L_{pseudo} = \frac{\sum_{ij} (1-M_{ij})(\hat{X}_{ij} - X^{fill}_{ij})^2}{\sum_{ij}(1-M_{ij}) + \epsilon}
\quad (\text{层次 2，fill\_initialized 后才激活})
$$

$$
L_{prob} = -\sum_{ij}\left[(1-M_{ij})\log M'_{ij} + M_{ij}\log(1-M'_{ij})\right]
\quad (\text{层次 4，use\_prob\_head=True 时激活})
$$

其中：

```text
T        = M ⊙ R，训练时有答案的监督位置
gamma    = pseudo-label loss 权重（默认 0.1）
beta     = L_prob 权重（默认 0.1）
```

---

## 10. 代码变量对照表

| 代码变量 | 数学符号 | 含义 | shape |
|---|---|---|---|
| `data_x` | `X` | 归一化后、NaN 填 0 的数据 | `[n, d]` |
| `mask` | `M` | 原始 mask，1=observed，0=missing | `[n, d]` |
| `fill_values` | `X^{fill}` | 渐进式训练的填充估计值 | `[n, d]` |
| `fill_initialized` | — | fill_values 是否已完成第一次更新 | bool |
| `ips` | `IPS` | 预计算的 IPS 权重 | `[n, d]` |
| `ips_tensor` | `IPS` | IPS 的 tensor 版本 | `[n, d]` |
| `x_mb` | batch X | 当前 batch 的数据 | `[B, d]` |
| `m_mb` | batch M | 当前 batch 的原始 mask | `[B, d]` |
| `ips_mb` | batch IPS | 当前 batch 的 IPS | `[B, d]` |
| `train_mask` | `M_train` | 训练时可见位置 | `[B, d]` |
| `target_mask` | `T` | 用于算 loss 的位置 | `[B, d]` |
| `x_input` | `X'` | 喂给模型的数据 | `[B, d]` |
| `fill_mb` | batch $X^{fill}$ | 当前 batch 的 fill_values | `[B, d]` |
| `orig_miss` | `1-M` | missing 位置标记（pseudo-label 用） | `[B, d]` |
| `value_tokens` | `Z^{val}` | value token embeddings | `[B, d, emb]` |
| `mask_tokens` | `Z^{mask}` | mask token embeddings | `[B, d, emb]` |
| `all_tokens` | `Z` | 拼接后 2d 个 token | `[B, 2d, emb]` |
| `full_mask` | `M^{full}` | 2d tokens 的可见性 | `[B, 2d]` |
| `full_ips` | `IPS^{full}` | 2d tokens 的 IPS（mask tokens=1） | `[B, 2d]` |
| `hidden` | `H` | Transformer 输出 | `[B, 2d, emb]` |
| `value_hidden` | `H^{val}` | 前 d 个 value token 的输出 | `[B, d, emb]` |
| `x_hat` | `X_hat` | 重建预测值 | `[B, d]` |
| `missing_prob` | `M'` | 预测的缺失概率（层次 4） | `[B, d]` |
| `l_rec` | `L_{rec}` | 重建 loss | scalar |
| `l_pseudo` | `L_{pseudo}` | pseudo-label loss | scalar |
| `l_prob` | `L_{prob}` | 缺失分布预测 loss | scalar |

---

## 11. 代码文件结构

### 11.1 DARN_modules.py 函数/类一览

```text
normalization(data)
normalization_with_parameter(data, norm_parameters)
renormalization(norm_data, norm_parameters)

sample_observed_mask(mask, mask_rate)

compute_ips_simple(mask)              ← 全局 IPS，每列一个值
compute_ips_logistic(data, mask)      ← 逐格 IPS，逻辑回归估计

ContinuousFeatureEmbedding            ← per-feature MLP（与 FATE 相同）
MissingAwareAttention                 ← 支持 IPS 加权的手动实现 attention
MissingAwareTransformerBlock          ← 第 0 层，使用 MissingAwareAttention
RegularTransformerBlock               ← 第 1..depth-1 层，普通 attention

DARNImputerNet                        ← 主模型
    .forward(x, mask, ips_weights)
    返回 x_hat 或 (x_hat, missing_prob)

train_darn_algorithm(model, data_x, mask, params, device, ips, ips_method)
```

### 11.2 DARN.py 类结构

```python
class DARN(BaseImputer):
    # 参数
    batch_size, epoch, learning_rate
    embedding_dim, depth, heads, mask_rate, dropout
    loss_type                    # 层次 1
    use_progressive, progressive_interval, gamma  # 层次 2
    use_ips, ips_method          # 层次 3
    use_prob_head, beta          # 层次 4
    
    def train(data, missing_mask=None):
        # 1. normalization + nan_to_num
        # 2. compute_ips（如果 use_ips=True）
        # 3. 初始化 DARNImputerNet
        # 4. 调用 train_darn_algorithm
    
    def predict(data):
        # 1. normalization_with_parameter + nan_to_num
        # 2. compute_ips_simple（如果 use_ips=True，预测阶段用简单版）
        # 3. model.forward
        # 4. 保留 observed，替换 missing
        # 5. renormalization
```

---

## 12. 与 FATE-imputer 的对比

| 维度 | FATE-imputer | DARN 层次 1 | DARN 层次 2 | DARN 层次 3 | DARN 层次 4 |
|---|---|---|---|---|---|
| 网络结构 | FATEImputerNet | 相同（改名） | 相同 | 相同 | 加 mask_prediction_head |
| Loss | MSE | MAE | MAE + L_pseudo | MAE + L_pseudo | MAE + L_pseudo + L_prob |
| Missing 位置填充 | 永远 0 | 永远 0 | 渐进更新 fill_values | 渐进更新 + 重新算 IPS | 同层次 3 |
| Attention | key_padding_mask | 同 FATE | 同 FATE | + IPS 加权 key | 同层次 3 |
| 额外输出 | 无 | 无 | 无 | 无 | missing_prob |

---

## 13. 使用示例

```python
from tabular.imputation.DARN import DARN

# 层次 1：基础版，只改 loss 为 MAE
model = DARN(epoch=100, loss_type="mae")

# 层次 1+2：加渐进式自补全
model = DARN(epoch=100, use_progressive=True, progressive_interval=10, gamma=0.1)

# 层次 1+2+3：加 IPS（简单版）
model = DARN(epoch=100, use_progressive=True, use_ips=True, ips_method="simple")

# 层次 1+2+3：加 IPS（逻辑回归版，会随 fill 更新重新计算）
model = DARN(epoch=100, use_progressive=True, use_ips=True, ips_method="logistic")

# 全部四个层次
model = DARN(
    epoch=100,
    use_progressive=True, progressive_interval=10, gamma=0.1,
    use_ips=True, ips_method="logistic",
    use_prob_head=True, beta=0.1,
)

# 统一接口
model.train(data_missing, missing_mask)
imputed = model.predict(data_missing)
metrics = model.estimate(ground_truth, imputed, missing_mask)
```

---

## 14. 实现顺序建议

```text
步骤 1：层次 1 单独跑通，与 FATE-imputer 对比 MSE/RMSE/MAE

步骤 2：加渐进式自补全（use_progressive=True），观察是否比步骤 1 更好

步骤 3：加简单 IPS（use_ips=True, ips_method="simple"），观察效果

步骤 4：改用逻辑回归 IPS（ips_method="logistic"），对比简单 IPS

步骤 5（可选）：加 L_prob（use_prob_head=True），对比步骤 4
```

---

## 15. 一句话总结

```text
DARN-imputation 实现 = FATE 结构
                     + MAE loss（层次 1，忠于 DARN 论文）
                     + 渐进式自补全 + pseudo-label（层次 2，fill_values 迭代提升）
                     + IPS 加权 attention（层次 3，稀少特征在 attention 中权重更高）
                     + 缺失分布预测辅助任务（层次 4，让模型同时理解缺失规律）
```
