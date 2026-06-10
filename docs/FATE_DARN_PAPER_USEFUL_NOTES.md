# FATE / DARN 论文中对 imputation 有用的内容

## 1. 阅读目标

这份文档只服务一个工程目标：

```text
把 FATE 和 DARN 改造成 DataPrep 的 tabular/imputation 算法。
```

所以本文不会完整复述论文实验，也不会展开所有公平性、联邦学习指标。这里只保留对下面任务有用的内容：

```text
输入：data_missing, missing_mask
输出：imputed_data
接口：继承 BaseImputer，提供 train(data, missing_mask) 和 predict(data)
```

最重要的结论：

```text
FATE 适合改造成 mask-aware Transformer imputer。
DARN 适合改造成 missing-aware Transformer reconstruction imputer。
两者原论文都不是纯 imputation 论文，不能直接复制主训练流程。
```

## 2. 统一符号

为了避免论文符号、原始代码和 DataPrep 项目之间混乱，本文统一使用 DataPrep 当前约定。

给定带缺失的表格：

$$
X \in \mathbb{R}^{n \times d}
$$

其中：

```text
n = 样本数
d = 特征数
X_{ij} = 第 i 行第 j 列的特征值
```

缺失掩码：

$$
M \in \{0, 1\}^{n \times d}
$$

统一语义：

$$
M_{ij} =
\begin{cases}
1, & X_{ij}\ \text{是观测值 observed} \\
0, & X_{ij}\ \text{是缺失值 missing}
\end{cases}
$$

这一点和当前 `DataPrep` 的 `GAIN / VAEGAIN / SCIS` 一致。

需要注意：

```text
FATE 论文正文里的 mask 符号写法和本项目代码/ DataPrep 约定不完全一致。
FATE 原始代码实际使用的是 1=observed, 0=missing。
DARN 论文和代码也使用 1=observed, 0=missing。
```

因此工程实现时不要照搬论文里的 mask 文本定义，而应统一成：

```text
1 = observed
0 = missing
```

## 3. 为什么 FATE / DARN 不能直接搬主程序

FATE 原始任务：

```text
Fairness-aware classification over incomplete data
```

也就是：

```text
带缺失数据上的公平分类
```

原始输入包含：

```text
X = 非敏感特征
S = 敏感属性
Y = 标签
M = 缺失掩码
```

原始输出是：

```text
预测标签 Y_hat
公平性指标
分类准确率
```

DARN 原始任务：

```text
Federated incomplete tabular data prediction with missing complementarity
```

也就是：

```text
联邦环境下的带缺失表格预测
```

原始输入包含：

```text
多个客户端的数据 D_1, D_2, ..., D_K
每个客户端自己的 X_k, Y_k, M_k
```

原始输出是：

```text
每个客户端的个性化预测模型
分类或回归任务指标
```

当前 DataPrep imputation 需要的是：

```text
只输入一张带缺失表格
只输出补全后的表格
不需要标签 Y
不需要敏感属性 S
不需要联邦客户端 K
```

所以：

```text
FATE 要抽取 IDE / missing-aware Transformer / reconstruction head。
DARN 要抽取 missing-aware transformer-based imputation model / reconstruction loss。
```

## 4. FATE 有用内容详解

## 4.1 FATE 的论文主线

FATE 论文想解决的问题是：

```text
在敏感属性和非敏感属性都可能缺失的情况下，直接用 incomplete data 做公平分类。
```

论文认为传统流程：

```text
先 impute 缺失值
再做 fairness-aware classification
```

会有两个问题：

```text
1. imputation 可能引入偏差
2. imputation 的误差会传播到下游分类和公平性评估
```

因此 FATE 的思想是：

```text
不要先补全再分类，而是让模型直接看见缺失状态。
```

对 imputation 有用的是这句话背后的结构：

```text
missing state information 本身是有信息量的。
模型应该把 X 和 M 一起编码。
```

## 4.2 FATE 模型结构

论文中 FATE 包含两个模块：

```text
1. IDE: Incomplete Data Encoding
2. DRL: Debiased Representation Learning
```

对 imputation 有用的是 IDE。

DRL 主要服务公平性，第一版 imputation 可以不实现。

IDE 包含：

```text
1. incomplete tabular embedding
2. missingness-aware Transformer block
```

原始代码中对应位置：

```text
papers/FATE_SIGIR25/code/model.py
papers/FATE_SIGIR25/code/run.py
```

关键代码位置：

```text
FATE 类:
D:/DataPrep/papers/FATE_SIGIR25/code/model.py

embed_data_mask:
D:/DataPrep/papers/FATE_SIGIR25/code/run.py

First_Attention:
D:/DataPrep/papers/FATE_SIGIR25/code/model.py
```

## 4.3 FATE 的 incomplete tabular embedding

FATE 把特征分成两类：

```text
categorical features
continuous features
```

### 4.3.1 类别特征 embedding

假设第 j 个类别特征的取值是：

$$
X_{ij}^{cat} \in \{0, 1, \dots, C_j - 1\}
$$

FATE 使用 embedding table：

$$
E_j^{cat} \in \mathbb{R}^{C_j \times r}
$$

把类别值映射成 r 维向量：

$$
e_{ij}^{cat} = E_j^{cat}[X_{ij}^{cat}]
$$

其中：

```text
r = embedding dimension
```

代码中是：

```python
self.embeds = nn.Embedding(self.total_tokens, self.dim)
x_categ_enc = model.embeds(x_categ)
```

为了让多个类别列共用一个 embedding table，代码会给每个类别列加 offset：

```python
x_categ = x_categ + model.categories_offset.type_as(x_categ)
```

直观理解：

```text
第 1 个类别列的取值 0、1、2 和第 2 个类别列的取值 0、1、2 不能混在同一组 embedding id 里。
offset 用来把不同列的类别 id 错开。
```

### 4.3.2 连续特征 embedding

连续特征不能直接查表，所以 FATE 给每个连续特征一个小 MLP：

```text
f_j: R -> R^r
```

对第 i 行第 j 个连续特征：

$$
e_{ij}^{cont} = f_j(X_{ij}^{cont})
$$

代码中是：

```python
self.simple_MLP = nn.ModuleList([
    simple_MLP([1, 100, self.dim])
    for _ in range(self.num_continuous)
])
```

对应 forward：

```python
x_cont_enc[:, i, :] = model.simple_MLP[i](x_cont[:, i])
```

直观理解：

```text
每个连续值先被映射成一个 r 维 token。
这样表格的一列就可以像 Transformer 里的一个 token 一样参与 attention。
```

### 4.3.3 缺失位置的 mask embedding

这是 FATE 对 imputation 最有价值的地方。

对于缺失值，FATE 不是简单填 0，也不是直接均值填充，而是用一个可学习的 mask embedding 替代原始值 embedding。

代码中：

```python
self.mask_embeds_cat = nn.Embedding(self.num_categories * 2, self.dim)
self.mask_embeds_cont = nn.Embedding(self.num_continuous * 2, self.dim)
```

在 `embed_data_mask` 中：

```python
x_categ_enc[cat_mask == 0] = cat_mask_temp[cat_mask == 0]
x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]
```

含义是：

```text
如果某个位置缺失，就不用该位置的真实值 embedding，
而是换成这个特征对应的 missing-state embedding。
```

用公式写：

$$
z_{ij} =
\begin{cases}
e_{ij}(X_{ij}), & M_{ij} = 1 \\
e_j^{miss}, & M_{ij} = 0
\end{cases}
$$

其中：

```text
z_{ij}      = 送入 Transformer 的 token embedding
e_{ij}      = 类别或连续特征的正常 embedding
e_j^{miss}  = 第 j 个特征自己的缺失 embedding
```

这一点和简单填 0 的区别很大：

```text
填 0:
  模型看到的是一个数值 0，可能误以为 0 是真实值。

mask embedding:
  模型看到的是“这个位置缺失”这个状态本身。
```

## 4.4 FATE 的 missingness-aware attention

FATE 的 Transformer 把每个特征当成一个 token。

输入 embedding 矩阵：

$$
Z_i = [z_{i1}, z_{i2}, \dots, z_{id}] \in \mathbb{R}^{d \times r}
$$

对一个 batch：

$$
Z \in \mathbb{R}^{b \times d \times r}
$$

标准 self-attention 先计算：

$$
Q = ZW_Q,\quad K = ZW_K,\quad V = ZW_V
$$

然后：

$$
A = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

FATE 的 missingness-aware attention 在 attention score 上加一个 mask 矩阵。

为了和 DataPrep 语义一致，定义：

$$
B_{ij} =
\begin{cases}
0, & M_{ij} = 1 \\
-\infty, & M_{ij} = 0
\end{cases}
$$

对第 i 个样本，attention score 变成：

$$
S_i = \frac{Q_iK_i^\top}{\sqrt{d_k}} + B_i
$$

其中 `B_i` 会 broadcast 到所有 query token 对 key token 的打分上。

最终：

$$
A_i = \operatorname{softmax}(S_i)V_i
$$

如果某个 key 位置缺失：

$$
B_{ij} = -\infty
$$

那么：

$$
\exp(\text{score} + (-\infty)) = 0
$$

softmax 后这个位置的 attention weight 就是 0。

也就是说：

```text
其他特征不会从缺失 token 里取信息。
```

这个机制的作用：

```text
1. 缺失位置有自己的 mask embedding，模型知道它缺失。
2. attention 计算时又避免过度依赖缺失位置的 value。
3. 模型可以利用 observed features 和 missing pattern 共同形成表示。
```

代码中对应：

```python
sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

key_padding_mask = key_padding_mask.bool()
key_padding_mask = ~key_padding_mask

sim = sim.masked_fill(
    key_padding_mask.unsqueeze(1).unsqueeze(2),
    float("-inf"),
)

attn = sim.softmax(dim=-1)
```

这里 `key_padding_mask` 传入的是 `cont_mask/cat_mask` 拼接后的 mask。

由于项目语义是：

```text
1 = observed
0 = missing
```

代码里取反后：

```text
missing 位置变成 True
```

然后填成 `-inf`。

## 4.5 FATE 原始分类 loss

FATE 原论文和代码的主任务是分类。

训练时取 Transformer 输出的第一个 token 作为样本表示：

```python
y_reps = reps[:, 0, :]
y_outs = model.mlpfory(y_reps)
```

分类 loss：

$$
L_{cls} = CE(\hat{Y}, Y)
$$

二分类时：

$$
CE(\hat{Y}, Y) = -\left[Y\log p + (1-Y)\log(1-p)\right]
$$

多分类时：

$$
CE(\hat{Y}, Y) = -\sum_c \mathbf{1}[Y=c]\log p_c
$$

但是：

```text
DataPrep imputation 不需要标签 Y。
```

所以 `mlpfory` 和分类 loss 第一版可以不要。

## 4.6 FATE 代码中已有 reconstruction heads

FATE 的 `model.py` 虽然主流程用于分类，但类里面已经定义了重构头：

```python
self.mlp1 = simple_MLP([dim, (self.total_tokens) * 2, self.total_tokens])
self.mlp2 = simple_MLP([dim, (self.num_continuous), 1])
```

forward 返回：

```python
cat_outs = self.mlp1(x[:, :self.num_categories, :])
con_outs = self.mlp2(x[:, self.num_categories:, :])
return cat_outs, con_outs
```

也就是说，Transformer 输出每个特征的表示后：

```text
类别特征 token -> mlp1 -> 类别 logits
连续特征 token -> mlp2 -> 连续值预测
```

这正是改造成 imputer 最有用的部分。

## 4.7 如何把 FATE 改成 DataPrep imputer

因为当前 imputation 模块主要处理数值矩阵，第一版可以只做连续特征版本。

### 4.7.1 输入

```text
data_missing ∈ R^{n × d}
missing_mask ∈ {0, 1}^{n × d}
```

其中：

```text
missing_mask = 1 表示 observed
missing_mask = 0 表示 missing
```

### 4.7.2 归一化

和 GAIN 一样：

```text
X_norm = (X - min) / (max - min)
```

缺失位置先临时填 0：

```text
X_input = nan_to_num(X_norm, 0)
```

但注意：FATE 的 mask embedding 会标记缺失位置，所以这个 0 不应该被当成真实值。

### 4.7.3 连续特征 embedding

对每个特征：

$$
e_{ij} = f_j(X^{input}_{ij})
$$

如果缺失：

$$
z_{ij} = e_j^{miss}
$$

如果观测：

$$
z_{ij} = e_{ij}
$$

### 4.7.4 Transformer 编码

$$
H = Transformer(Z, M)
$$

其中第一层用 missing-aware attention：

$$
Attention(Q,K,V,M)
= \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + B\right)V
$$

### 4.7.5 重构输出

每个特征 token 输出一个预测值：

$$
\hat{X}_{ij} = g_j(H_{ij})
$$

如果用共享 MLP：

$$
\hat{X} = MLP(H)
$$

如果用每列单独 MLP：

$$
\hat{X}_{:j} = MLP_j(H_{:j})
$$

第一版建议：

```text
每列一个小 MLP，或者一个共享 Linear(dim, 1)。
```

### 4.7.6 训练 loss

最自然的 imputation 训练方式是 masked reconstruction。

因为训练数据本身有真实观测位置和缺失位置。缺失位置没有 ground truth，所以不能直接监督。

可以只在 observed 位置训练：

$$
L_{obs}
= \frac{
\sum_i \sum_j M_{ij}\left(X^{norm}_{ij} - \hat{X}_{ij}\right)^2
}{
\sum_i \sum_j M_{ij} + \epsilon
}
$$

这和 GAIN 的 observed reconstruction loss 类似。

但是只在 observed 位置训练，有一个问题：

```text
模型训练时从来没有被要求恢复“被遮住但原本知道答案”的值。
```

因此更推荐 DARN 式 random masking：

在 observed 位置中随机再遮一部分：

$$
R_{ij} \sim \operatorname{Bernoulli}(\rho)
$$

其中：

```text
R_{ij} = 1 表示这个 observed 位置被人为遮住，用来做重构训练
R_{ij} = 0 表示不遮
```

有效训练位置：

$$
T_{ij} = M_{ij} \cdot R_{ij}
$$

训练输入 mask：

$$
M^{train} = M \odot (1 - R)
$$

训练输入值：

```text
X_train_input = X_norm
X_train_input[T == 1] = 0
```

loss：

$$
L_{rec}
= \frac{
\sum_i \sum_j T_{ij}\left(X^{norm}_{ij} - \hat{X}_{ij}\right)^2
}{
\sum_i \sum_j T_{ij} + \epsilon
}
$$

推导如下：

1. 只有原本 observed 的位置才有真实值，所以必须乘 `M_{ij}`。
2. 为了模拟 missing，需要从 observed 位置中随机挑一部分遮住，所以乘 `R_{ij}`。
3. 因此监督位置是 `M_{ij} R_{ij}`。
4. 对这些位置计算重构误差。
5. 除以监督位置数量，避免 batch 中遮住位置多少影响 loss 尺度。

MSE 版本：

$$
L_{rec}^{MSE}
= \frac{
\sum_i\sum_j M_{ij}R_{ij}(X_{ij}-\hat{X}_{ij})^2
}{
\sum_i\sum_j M_{ij}R_{ij} + \epsilon
}
$$

MAE 版本：

$$
L_{rec}^{MAE}
= \frac{
\sum_i\sum_j M_{ij}R_{ij}|X_{ij}-\hat{X}_{ij}|
}{
\sum_i\sum_j M_{ij}R_{ij} + \epsilon
}
$$

第一版建议用 MSE，因为当前 `DataPrep` 的评估以 MSE/RMSE 为主。

### 4.7.7 predict

预测时：

$$
\hat{X} = model(X^{input}, M)
$$

最终输出必须保留观测值，只替换缺失值：

$$
X^{imputed}_{norm}
= M \odot X^{norm} + (1-M)\odot \hat{X}
$$

反归一化：

$$
X^{imputed} = X^{imputed}_{norm}\odot den + min
$$

这和 GAIN / VAEGAIN / SCIS 的 predict 逻辑一致。

## 4.8 FATE 第一版需要保留和删除的东西

保留：

```text
1. continuous feature embedding
2. mask embedding
3. missing-aware attention
4. Transformer encoder
5. continuous reconstruction head
6. masked reconstruction loss
```

第一版删除：

```text
1. sensitive attribute S
2. fairness metrics
3. debiased representation learning DRL
4. classification head mlpfory
5. categorical feature复杂处理
```

可选增强：

```text
1. 加 categorical support
2. 加 debiased attention
3. 用 MAE + MSE 混合 loss
4. 加 positional/feature id embedding
```

## 5. DARN 有用内容详解

## 5.1 DARN 的论文主线

DARN 论文想解决的问题是：

```text
联邦场景下，多客户端都有不完整表格数据，如何直接训练预测模型，而不是先补全再预测。
```

论文认为“先补全再联邦预测”有问题：

```text
1. 补全会带来误差
2. 误差会传递到下游预测
3. 客户端之间 missing distribution 不同，简单平均模型会忽略互补性
```

DARN 的整体目标函数是：

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
L_sup  = 预测任务监督损失
L_rec  = imputation model 重构损失
L_prob = missing distribution learning loss
```

对 DataPrep imputation 有用的是：

```text
L_rec 和 missing-aware transformer-based imputation model
```

`L_sup`、联邦加权、missing distribution model 第一版都可以不实现。

## 5.2 DARN 的 imputation model

DARN 明确提出：

```text
missing-aware transformer-based imputation model
```

流程是：

```text
1. 从 incomplete data X 出发
2. 用随机 mask R 在 observed 位置上再遮一部分，得到 X'
3. 根据原始 mask M 生成 transformed mask M_hat
4. 用 missing-aware Transformer 编码 X'
5. 用 MLP 输出 reconstructed data X_hat
6. 在被随机遮住且原本 observed 的位置上计算 reconstruction loss
```

这套流程非常适合直接改造成 `DARNImputer`。

## 5.3 DARN 的 mask 定义

DARN 论文定义：

```text
M_{ij} = 1, if X_{ij} is observed
M_{ij} = 0, if X_{ij} is missing
```

这和 DataPrep 一致。

随机 mask：

$$
R \in \{0,1\}^{n \times d}
$$

论文中：

```text
R_{ij} = 1 表示第 j 个特征被随机 mask
R_{ij} = 0 表示不 mask
```

注意，训练时真正可监督的位置是：

$$
M_{ij}\cdot R_{ij}
$$

因为：

```text
M_{ij}=1 才说明原本有真实值
R_{ij}=1 才说明这个位置被人为遮住，需要模型恢复
```

## 5.4 DARN 的 missing-aware attention

DARN 和 FATE 类似，也在 attention score 里加入 mask。

标准 attention：

$$
Attention(Q,K,V)
= \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

DARN missing-aware attention：

$$
A
= \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + \hat{M}\right)V
$$

其中：

$$
\hat{M}_{ij} =
\begin{cases}
1, & M_{ij}=1 \\
-\infty, & M_{ij}=0
\end{cases}
$$

严格实现时，给 observed 位置加 `1` 不是必须的，核心是：

```text
missing 位置加 -∞
```

为了和 FATE / PyTorch 实现保持一致，可以实现成：

$$
B_{ij} =
\begin{cases}
0, & M_{ij}=1 \\
-\infty, & M_{ij}=0
\end{cases}
$$

因为：

```text
softmax(score + 1) 和 softmax(score) 的整体比例会变化；
但对当前工程来说，最重要的是 missing 位置不能参与 attention。
```

如果想更忠于论文，可以用：

```text
observed bias = 1
missing bias = -∞
```

第一版建议：

```text
observed bias = 0
missing bias = -∞
```

这样更稳定，也和 FATE 原代码更接近。

## 5.5 DARN 的 reconstruction loss 数学推导

论文中重构损失：

$$
L_{rec}(\theta^I_k)
=
\frac{
\sum_i\sum_j M_{ij}R_{ij}\cdot \ell(X_{ij}, \hat{X}_{ij})
}{
\sum_i\sum_j M_{ij}R_{ij} + \epsilon
}
$$

其中：

```text
θ^I_k      = 第 k 个客户端 imputation model 参数
M_{ij}     = 原始缺失 mask，1 observed，0 missing
R_{ij}     = 随机 mask，1 表示人为遮住
X_hat_{ij} = imputation model 预测值
ℓ          = 单点误差，论文使用 absolute error
```

如果使用 MAE：

$$
\ell(X_{ij}, \hat{X}_{ij}) = |X_{ij} - \hat{X}_{ij}|
$$

则：

$$
L_{rec}^{MAE}
=
\frac{
\sum_i\sum_j M_{ij}R_{ij}|X_{ij}-\hat{X}_{ij}|
}{
\sum_i\sum_j M_{ij}R_{ij} + \epsilon
}
$$

如果使用 MSE：

$$
\ell(X_{ij}, \hat{X}_{ij}) = (X_{ij} - \hat{X}_{ij})^2
$$

则：

$$
L_{rec}^{MSE}
=
\frac{
\sum_i\sum_j M_{ij}R_{ij}(X_{ij}-\hat{X}_{ij})^2
}{
\sum_i\sum_j M_{ij}R_{ij} + \epsilon
}
$$

这个 loss 的核心思想：

```text
用已知答案的位置制造训练题。
```

比如一行数据：

```text
X = [0.2, 0.5, NaN, 0.9]
M = [1,   1,   0,   1]
```

随机 mask：

```text
R = [0, 1, 0, 1]
```

有效监督位置：

```text
T = M · R = [0, 1, 0, 1]
```

训练输入中第 2 和第 4 个位置被遮住：

```text
X' = [0.2, 0, NaN/0, 0]
```

模型输出：

```text
X_hat = [0.21, 0.48, 0.61, 0.88]
```

loss 只在第 2 和第 4 个位置算：

$$
L = \frac{(0.5 - 0.48)^2 + (0.9 - 0.88)^2}{2}
$$

原本真实缺失的第 3 个位置不参与 loss，因为没有 ground truth。

但 predict 时会用它的输出补第 3 个位置。

## 5.6 DARN 的 missing distribution learning

DARN 还有一个模块：

```text
missing distribution learning model
```

论文中：

$$
M' = \mathcal{M}(\hat{X}\mid \theta^M)
$$

也就是从 reconstructed data 预测 missing probability matrix。

loss 是 BCE：

$$
L_{prob}(\theta^M_k)
=
-\sum_i\sum_j
\left[
M_{ij}\log M'_{ij}
+ (1-M_{ij})\log(1-M'_{ij})
\right]
$$

这个模块的作用：

```text
学习每个客户端自己的 missing distribution。
```

它主要服务后面的 missing complementarity score 和 personalized federated averaging。

对单机 DataPrep imputation 第一版来说：

```text
可以不实现。
```

原因：

```text
1. DataPrep 当前没有多客户端。
2. 不需要根据客户端 missing distribution 做聚合。
3. imputation 输出只需要 X_hat。
```

但可以作为第二版增强：

```text
增加一个 mask predictor，让模型同时学习 missing mechanism。
```

第二版总 loss 可以写成：

$$
L = L_{rec} + \beta L_{prob}
$$

其中：

```text
L_prob = BCE(M_pred, M)
```

这样模型不仅补值，还学习缺失模式。

## 5.7 DARN 的 personalized federated averaging

DARN 论文还有服务端聚合部分。

observed sample size score：

$$
C_k = \frac{O_k}{\max\{O_1, O_2, \dots, O_K\}}
$$

其中：

```text
O_k = 第 k 个客户端 observed cells 总数
```

missing complementary score：

$$
S_{ij}
= \frac{1}{2}\left(1-\cos(\theta^M_i,\theta^M_j)\right)
$$

展开：

$$
S_{ij}
= \frac{1}{2}
\left(
1 -
\frac{
\sum_l \theta^M_{il}\theta^M_{jl}
}{
\sqrt{\sum_l(\theta^M_{il})^2}
\sqrt{\sum_l(\theta^M_{jl})^2}
}
\right)
$$

含义：

```text
两个客户端 missing distribution 越不同，互补性越强。
```

这部分对 DataPrep imputation 第一版不需要。

但理解它有帮助：

```text
DARN 真正的论文创新不是“单机补全器”，而是“利用多客户端缺失分布互补性做个性化联邦预测”。
```

## 5.8 DARN 原始代码和论文对应关系

关键代码：

```text
D:/DataPrep/papers/DARN_VLDB25/DARN-main/models/pretrainmodel.py
D:/DataPrep/papers/DARN_VLDB25/DARN-main/models/model.py
D:/DataPrep/papers/DARN_VLDB25/DARN-main/utils/data_utils.py
D:/DataPrep/papers/DARN_VLDB25/DARN-main/train_DARN.py
```

`embed_data_mask`：

```python
x_categ_enc = model.embeds(x_categ)
x_cont_enc[:, i, :] = model.simple_MLP[i](x_cont[:, i])

cat_mask_temp = model.mask_embeds_cat(cat_mask_temp)
con_mask_temp = model.mask_embeds_cont(con_mask_temp)

x_categ_enc[cat_mask == 0] = cat_mask_temp[cat_mask == 0]
x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]
```

含义和 FATE 很像：

```text
缺失位置用 mask embedding 替换。
```

`SAINT.forward`：

```python
x = self.transformer(x_categ, x_cont, cont_mask, cat_mask, ...)
cat_outs = self.mlp1(x[:, :self.num_categories, :])
con_outs = self.mlp2(x[:, self.num_categories:, :])
return cat_outs, con_outs
```

含义：

```text
Transformer 输出每列表示，再用 reconstruction head 恢复类别和连续特征。
```

训练中连续特征重构：

```python
con_outs = torch.cat(con_outs, dim=1)
l2 = criterion2(con_outs, x_cont)
```

类别特征重构：

```python
log_x = criterion3(cat_outs[j])
log_x = log_x[range(cat_outs[j].shape[0]), x_categ[:, j]]
l1 += abs(sum(log_x) / cat_outs[j].shape[0])
```

总 loss 中加入：

```python
loss += 0.1 * l1 + 1 * l2
```

这说明原始代码确实把重构误差放进训练目标里。

## 5.9 如何把 DARN 改成 DataPrep imputer

第一版只保留：

```text
1. mask embedding
2. missing-aware Transformer
3. reconstruction head
4. random masking reconstruction loss
```

删除：

```text
1. federated clients
2. prediction head
3. missing distribution model
4. personalized weight averaging
5. IPS / client aggregation
```

### 5.9.1 训练流程

输入：

```text
X_missing, M
```

归一化：

```text
X_norm = normalize(X_missing)
```

将缺失位置临时填 0：

```text
X_base = nan_to_num(X_norm, 0)
```

每个 batch：

```text
1. 从 observed 位置随机采样 R
2. 构造 T = M · R
3. 构造训练输入 X'
4. X' 中 T=1 的位置被遮住
5. 输入模型得到 X_hat
6. 在 T=1 的位置计算 L_rec
7. 反向传播更新模型
```

公式：

$$
R_{ij} \sim \operatorname{Bernoulli}(\rho),\quad
T_{ij}=M_{ij}R_{ij},\quad
M^{train}=M\odot(1-R)
$$

训练输入：

$$
X'_{ij} =
\begin{cases}
X_{ij}, & M^{train}_{ij}=1 \\
0, & M^{train}_{ij}=0
\end{cases}
$$

模型：

$$
\hat{X} = DARNImputerNet(X', M^{train})
$$

loss：

$$
L_{rec}
= \frac{
\sum_i\sum_j T_{ij}(X_{ij}-\hat{X}_{ij})^2
}{
\sum_i\sum_j T_{ij} + \epsilon
}
$$

### 5.9.2 预测流程

预测时不做 random mask，只用真实 mask：

$$
\hat{X} = model(X^{base}, M)
$$

组合：

$$
X^{imputed}_{norm}
= M\odot X^{norm} + (1-M)\odot\hat{X}
$$

反归一化：

$$
X^{imputed} = renormalize(X^{imputed}_{norm})
$$

## 6. FATE 和 DARN 对 imputation 的区别

## 6.1 共同点

两者共同点：

```text
1. 都不主张简单先 impute 再做下游任务。
2. 都认为 missing pattern 本身有信息。
3. 都使用 mask-aware / missing-aware Transformer。
4. 都把特征列当成 token。
5. 都有 mask embedding 思想。
6. 都有 reconstruction head 可以抽出来做补全。
```

共同数学骨架：

$$
Z = Embed(X,M)
$$

$$
H = Transformer(Z,M)
$$

$$
\hat{X} = ReconstructionHead(H)
$$

$$
X^{imputed}=M\odot X + (1-M)\odot\hat{X}
$$

## 6.2 不同点

```text
FATE:
  原任务是公平分类。
  重点是 IDE + DRL。
  对 imputation 最有用的是 IDE。
  DRL 第一版可以删。

DARN:
  原任务是联邦表格预测。
  重点是 missing complementarity。
  对 imputation 最有用的是 missing-aware imputation model。
  联邦聚合第一版可以删。
```

从实现难度看：

```text
FATE 第一版更适合先做，因为它可以直接变成单机 mask-aware Transformer imputer。
DARN 第一版也能做，但要从联邦训练里剥离更多东西。
```

从数学训练目标看：

```text
FATE 原论文核心 loss 是 classification CE。
DARN 原论文明确给出了 reconstruction loss L_rec。
```

所以落地时：

```text
FATE 需要我们把训练目标改造成 reconstruction。
DARN 可以直接借它的 L_rec。
```

## 7. 建议最终工程设计

## 7.1 FATE.py

职责：

```text
1. 继承 BaseImputer
2. 保存超参数
3. 做 normalization
4. 初始化 FATENet
5. 调用 train_fate_algorithm
6. 保存 checkpoint
7. predict 时反归一化并保留 observed values
```

建议参数：

```python
batch_size=128
epoch=100
learning_rate=1e-3
embedding_dim=32
depth=3
heads=4
mask_rate=0.2
device=None
```

## 7.2 FATE_modules.py

建议包含：

```text
normalization
normalization_with_parameter
renormalization
ContinuousFeatureEmbedding
MaskAwareAttention
TransformerBlock
FATENet
train_fate_algorithm
```

第一版 `FATENet` 可以是：

```text
X, M
  -> continuous feature embedding
  -> missing embedding replacement
  -> feature id embedding 可选
  -> missing-aware Transformer
  -> Linear(dim, 1)
  -> X_hat
```

## 7.3 DARN.py

职责和 FATE.py 类似。

区别是文档和命名上更强调：

```text
random masking reconstruction
```

建议参数：

```python
batch_size=128
epoch=100
learning_rate=1e-3
embedding_dim=32
depth=3
heads=4
mask_rate=0.2
loss_type="mse"
device=None
```

## 7.4 DARN_modules.py

建议包含：

```text
normalization
normalization_with_parameter
renormalization
random_observed_mask
MissingAwareAttention
DARNReconstructionNet
train_darn_algorithm
```

第一版 DARN 和 FATE 可以共享很多代码，但建议先分文件写，避免一开始抽象过度。

## 8. 最小可运行版本的数学形式

FATE 和 DARN 第一版都可以统一成下面的训练目标。

给定：

$$
X \in \mathbb{R}^{n\times d},\quad
M \in \{0,1\}^{n\times d}
$$

随机遮盖：

$$
R_{ij} \sim \operatorname{Bernoulli}(\rho)
$$

$$
T = M \odot R
$$

$$
M^{train} = M \odot (1-R)
$$

输入：

$$
X' = M^{train}\odot X
$$

模型：

$$
\hat{X}=f_{\theta}(X',M^{train})
$$

损失：

$$
L(\theta)
=
\frac{
\sum_{i=1}^{n}\sum_{j=1}^{d}
T_{ij}(X_{ij}-\hat{X}_{ij})^2
}{
\sum_{i=1}^{n}\sum_{j=1}^{d}T_{ij}+\epsilon
}
$$

预测：

$$
\hat{X}=f_{\theta}(M\odot X,M)
$$

$$
X^{imp}=M\odot X+(1-M)\odot\hat{X}
$$

这就是把论文核心变成 DataPrep imputation 的最小闭环。

## 9. 当前实现时最容易踩的坑

### 9.1 mask 语义混乱

必须统一：

```text
DataPrep: 1 observed, 0 missing
```

如果从 FATE 论文文字抄公式，容易反过来。

工程里所有组合都应该是：

```text
imputed = M * original + (1 - M) * generated
```

### 9.2 不要直接训练真实缺失位置

真实缺失位置没有 ground truth。

不能写：

```text
loss = mean((X_missing - X_hat)^2 on missing positions)
```

因为 missing position 是 NaN 或填充值，不是真实答案。

应该用 observed random masking：

```text
从 observed 位置随机遮住一部分，再让模型恢复。
```

### 9.3 不要第一版实现全部论文任务

FATE 不要第一版实现：

```text
fairness metric
sensitive attribute
DRL theorem
classification head
```

DARN 不要第一版实现：

```text
federated learning
client aggregation
personalized averaging
differential privacy
missing distribution model
```

第一版只要：

```text
train
predict
estimate
unit test
script test
```

## 10. 推荐阅读和实现顺序

建议：

```text
1. 先实现 FATE 最小版
2. 再实现 DARN 最小版
```

原因：

```text
FATE 的工程噪声少，先做更容易形成结果。
DARN 的 reconstruction loss 更清楚，但联邦代码噪声更多。
```

实现 FATE 时用：

```text
D:/DataPrep/papers/FATE_SIGIR25/code/model.py
D:/DataPrep/papers/FATE_SIGIR25/code/run.py
```

实现 DARN 时用：

```text
D:/DataPrep/papers/DARN_VLDB25/DARN-main/models/pretrainmodel.py
D:/DataPrep/papers/DARN_VLDB25/DARN-main/models/model.py
D:/DataPrep/papers/DARN_VLDB25/DARN-main/utils/data_utils.py
D:/DataPrep/papers/DARN_VLDB25/DARN-main/train_DARN.py
```

## 11. 一句话总结

```text
FATE 给我们的有用东西是：用 mask embedding 和 missing-aware attention 表示 incomplete tabular data。

DARN 给我们的有用东西是：用 random masking reconstruction loss 训练 missing-aware Transformer imputation model。

把两者接进 DataPrep 时，第一版统一成：

X_hat = TransformerImputer(X_missing, missing_mask)
X_imputed = M * X_original + (1 - M) * X_hat
```
