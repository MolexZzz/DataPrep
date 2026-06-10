# FATE-imputation 实现说明

## 1. 实现目标

本实现把 FATE 论文中的 `Incomplete Data Encoding` 思想改造成 DataPrep 当前 `tabular/imputation` 模块可用的缺失值补全算法。

新增文件：

```text
tabular/imputation/FATE.py
tabular/imputation/FATE_modules.py
tabular/test/unit_fate.py
tabular/test/test_fate.py
```

当前实现的定位是：

```text
FATE-inspired mask-aware Transformer imputer
```

它不是 FATE 原论文公平分类任务的完整复现，而是为了当前项目需求做的工程化改造。

## 2. 原论文 FATE 做什么

FATE 原论文的任务是：

```text
Fairness-aware classification over incomplete data
```

也就是：

```text
在有缺失值的表格数据上直接做公平分类。
```

原论文流程大致是：

```text
X_missing + missing state M
        ↓
Incomplete Data Encoding
        ↓
Missingness-aware Transformer
        ↓
Debiased Representation Learning
        ↓
Classifier
        ↓
Y_hat
```

原论文输出是：

```text
分类标签 Y_hat
```

不是：

```text
补全后的表格 X_imputed
```

因此，原始 FATE 本身不是一个显式 imputation 算法。

## 3. 为什么需要改造

DataPrep 当前 `imputation` 模块要求统一接口：

```python
model.train(data_missing, missing_mask)
imputed_data = model.predict(data_missing)
```

其中：

```text
data_missing: 带 NaN 的数值矩阵
missing_mask: 1=observed, 0=missing
imputed_data: 与输入同 shape 的完整矩阵
```

所以如果要把 FATE 接入 `imputation`，不能直接使用原论文的分类头，而要把它变成：

```text
X_missing + M
        ↓
FATE-style incomplete data encoding
        ↓
Missing-aware Transformer
        ↓
Reconstruction head
        ↓
X_hat
        ↓
X_imputed = M * X + (1 - M) * X_hat
```

## 4. 当前实现的核心逻辑

当前实现只处理数值型表格。整体流程是：

```text
1. 对输入数据做 min-max normalization
2. 缺失位置临时填 0
3. 训练时从 observed 位置随机遮住一部分
4. 用 FATE-style embedding 编码数值和缺失状态
5. 用 missing-aware Transformer 学习特征之间关系
6. 用 reconstruction head 输出 X_hat
7. 只在被随机遮住的 observed 位置计算 MSE loss
8. predict 时只替换原始 missing 位置
```

下面不要先把公式当公式看，而是先把它当成一套“造训练题、批改答案、最后补全”的流程。

### 4.1 先看输入矩阵 X 和缺失矩阵 M

假设有一个很小的表格，2 行 3 列：

```text
        年龄     收入     信用分
用户1   30      NaN      700
用户2   40      8000     NaN
```

在代码里，归一化并把 NaN 临时填 0 后，可以想象成：

$$
X =
\begin{bmatrix}
0.3 & 0   & 0.7 \\
0.4 & 0.8 & 0
\end{bmatrix}
$$

这里的 0 只是临时占位，不代表真实值。真正告诉模型哪里有值、哪里缺失的是 mask：

$$
M =
\begin{bmatrix}
1 & 0 & 1 \\
1 & 1 & 0
\end{bmatrix}
$$

也就是：

```text
M = 1 表示 observed，原本有真实值
M = 0 表示 missing，原本缺失
```

所以：

$$
X \in \mathbb{R}^{n \times d},\quad M \in \{0,1\}^{n \times d}
$$

只是说：

```text
X 是 n 行 d 列的数值表格
M 是 n 行 d 列的 0/1 缺失标记表格
```

其中：

$$
M_{ij} =
\begin{cases}
1, & X_{ij}\ \text{observed}\\
0, & X_{ij}\ \text{missing}
\end{cases}
$$

### 4.2 为什么要随机遮住 observed 位置

真实缺失位置没有答案。

比如用户1的收入是 NaN，我们不知道真实收入是多少，所以不能用它算：

```text
(真实收入 - 模型预测收入)^2
```

因为“真实收入”不存在。

但是 observed 位置有答案。比如用户1的信用分是 0.7，用户2的收入是 0.8。训练时我们可以故意把这些已知位置遮住，让模型恢复它们。这样就有标准答案可以批改。

这就是随机矩阵 `R` 的作用：

$$
R_{ij} \sim \operatorname{Bernoulli}(\rho)
$$

意思是：

```text
每个位置都以 rho 的概率被选中，作为“人为遮住的位置”。
```

例如假设这次随机出来：

$$
R =
\begin{bmatrix}
0 & 0 & 1 \\
0 & 1 & 0
\end{bmatrix}
$$

意思是：

```text
用户1的信用分被人为遮住
用户2的收入被人为遮住
其他位置不人为遮住
```

注意：`R` 本身只是“随机抽中了哪些格子”。但如果它抽中了本来就是 missing 的格子，也不能用来训练，因为没有答案。所以还要和原始 mask `M` 相乘。

### 4.3 T = M ⊙ R：哪些位置用来算 loss

$$
T = M \odot R
$$

这里的 `⊙` 表示逐元素相乘。也就是两个矩阵同一个位置相乘。

用上面的例子：

$$
M =
\begin{bmatrix}
1 & 0 & 1 \\
1 & 1 & 0
\end{bmatrix},
\quad
R =
\begin{bmatrix}
0 & 0 & 1 \\
0 & 1 & 0
\end{bmatrix}
$$

所以：

$$
T = M \odot R =
\begin{bmatrix}
1\cdot0 & 0\cdot0 & 1\cdot1 \\
1\cdot0 & 1\cdot1 & 0\cdot0
\end{bmatrix}
=
\begin{bmatrix}
0 & 0 & 1 \\
0 & 1 & 0
\end{bmatrix}
$$

`T` 的意思是：

```text
T = 1 的位置，是“原本有答案，并且这次被人为遮住”的位置。
这些位置用于计算训练 loss。
```

所以这次训练只批改两个位置：

```text
用户1的信用分
用户2的收入
```

用户1的收入虽然也是缺失，但它没有答案，所以不参与 loss。

### 4.4 M_train = M ⊙ (1 - R)：训练时模型能看见什么

$$
M^{train} = M \odot (1-R)
$$

这一步是在构造“训练时给模型看的 mask”。

继续用例子：

$$
1-R =
\begin{bmatrix}
1 & 1 & 0 \\
1 & 0 & 1
\end{bmatrix}
$$

所以：

$$
M^{train}
= M \odot (1-R)
=
\begin{bmatrix}
1 & 0 & 1 \\
1 & 1 & 0
\end{bmatrix}
\odot
\begin{bmatrix}
1 & 1 & 0 \\
1 & 0 & 1
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 & 0 \\
1 & 0 & 0
\end{bmatrix}
$$

这个矩阵表示：

```text
用户1：模型只能看见年龄；收入真实缺失，信用分被人为遮住
用户2：模型只能看见年龄；收入被人为遮住，信用分真实缺失
```

也就是说，训练时模型被迫学习：

```text
怎么根据还看得见的特征，恢复被遮住但有答案的特征。
```

### 4.5 X' = M_train ⊙ X：训练时真正喂给模型的数据

$$
X' = M^{train} \odot X
$$

代入例子：

$$
X =
\begin{bmatrix}
0.3 & 0   & 0.7 \\
0.4 & 0.8 & 0
\end{bmatrix},
\quad
M^{train} =
\begin{bmatrix}
1 & 0 & 0 \\
1 & 0 & 0
\end{bmatrix}
$$

所以：

$$
X' =
\begin{bmatrix}
0.3 & 0 & 0 \\
0.4 & 0 & 0
\end{bmatrix}
$$

这就是模型训练时看到的输入。

但请注意：模型不仅看到 `X'`，还看到 `M_train`。所以它知道：

```text
0.3 / 0.4 是真实可见值
其他 0 是缺失或被人为遮住，不是可靠真实值
```

### 4.6 X_hat = f_theta(X', M_train)：模型预测完整表格

$$
\hat{X} = f_{\theta}(X', M^{train})
$$

这里：

```text
f_theta 就是我们实现的 FATEImputerNet
theta 表示模型所有可学习参数
X_hat 是模型对所有位置的预测
```

例如模型可能输出：

$$
\hat{X} =
\begin{bmatrix}
0.31 & 0.62 & 0.68 \\
0.41 & 0.76 & 0.69
\end{bmatrix}
$$

它对所有位置都预测了值，包括 observed、missing、人为遮住的位置。

但是训练时不是所有位置都算 loss，只算 `T = 1` 的位置。

### 4.7 loss：只批改 T = 1 的位置

$$
L(\theta)
=
\frac{
\sum_{i=1}^{n}\sum_{j=1}^{d}T_{ij}(X_{ij}-\hat{X}_{ij})^2
}{
\sum_{i=1}^{n}\sum_{j=1}^{d}T_{ij}+\epsilon
}
$$

看起来复杂，其实就是：

```text
只在 T=1 的位置计算平方误差，然后取平均。
```

在这个例子里：

```text
T=1 的位置有两个：
用户1的信用分，真实值 0.7，预测值 0.68
用户2的收入，真实值 0.8，预测值 0.76
```

所以 loss 等价于：

$$
L
=
\frac{(0.7-0.68)^2 + (0.8-0.76)^2}{2}
$$

分母：

$$
\sum_{i=1}^{n}\sum_{j=1}^{d}T_{ij}+\epsilon
$$

就是：

```text
这次一共批改了多少个位置。
```

这里是 2。`\epsilon` 是一个很小的数，防止极端情况下分母为 0。

### 4.8 预测阶段和训练阶段有什么不同

训练阶段会随机遮住 observed 位置，是为了制造有答案的训练题。

预测阶段不再随机遮。预测阶段只使用真实 mask：

$$
\hat{X} = f_{\theta}(M\odot X, M)
$$

也就是：

```text
模型看到所有原本 observed 的位置；
模型看不到原本 missing 的位置；
然后输出整张表的预测 X_hat。
```

### 4.9 最终补全：只替换原本 missing 的位置

$$
X^{imputed} = M\odot X + (1-M)\odot\hat{X}
$$

这个公式的意思是：

```text
M=1 的位置：保留原始值 X
M=0 的位置：使用模型预测值 X_hat
```

继续用例子，预测阶段假设：

$$
\hat{X} =
\begin{bmatrix}
0.32 & 0.61 & 0.69 \\
0.42 & 0.79 & 0.71
\end{bmatrix}
$$

原始：

$$
X =
\begin{bmatrix}
0.3 & 0   & 0.7 \\
0.4 & 0.8 & 0
\end{bmatrix},
\quad
M =
\begin{bmatrix}
1 & 0 & 1 \\
1 & 1 & 0
\end{bmatrix}
$$

最终：

$$
X^{imputed}
=
\begin{bmatrix}
0.3 & 0.61 & 0.7 \\
0.4 & 0.8  & 0.71
\end{bmatrix}
$$

可以看到：

```text
用户1年龄 0.3 保留
用户1信用分 0.7 保留
用户1收入用预测值 0.61

用户2年龄 0.4 保留
用户2收入 0.8 保留
用户2信用分用预测值 0.71
```

这就是 FATE-imputation 的完整训练和预测逻辑。

### 4.10 和当前代码怎么对应

上面的数学流程最终落在两个文件里：

```text
tabular/imputation/FATE.py
tabular/imputation/FATE_modules.py
```

你读代码时建议按这个顺序读：

```text
1. FATE.py 的 train()
2. FATE_modules.py 的 train_fate_algorithm()
3. FATE_modules.py 的 FATEImputerNet.forward()
4. FATE_modules.py 的 MissingAwareTransformerBlock.forward()
5. FATE.py 的 predict()
```

不要一上来从 `FATEImputerNet` 细节读，会容易迷路。先看数据从外层怎么进来，再看内部怎么处理。

## 5. 按代码执行顺序理解 FATE-imputer

这一节按真实代码执行顺序讲。每个代码片段都对应前面的数学符号。

### 5.1 FATE.py: train() 接收原始数据和 mask

文件：

```text
tabular/imputation/FATE.py
```

训练入口是：

```python
def train(self, data: np.ndarray, missing_mask: np.ndarray = None) -> None:
```

这里的 `data` 就是带 NaN 的原始缺失矩阵，比如：

```text
data =
[
    [30, NaN, 700],
    [40, 8000, NaN],
]
```

这里的 `missing_mask` 就是：

```text
missing_mask =
[
    [1, 0, 1],
    [1, 1, 0],
]
```

如果调用方没有传 mask，代码会自动根据 NaN 生成：

```python
if missing_mask is None:
    missing_mask = 1.0 - np.isnan(data)
else:
    missing_mask = np.array(missing_mask)
```

这一段对应数学里的：

$$
M_{ij} =
\begin{cases}
1, & X_{ij}\ \text{observed}\\
0, & X_{ij}\ \text{missing}
\end{cases}
$$

也就是说，代码里的 `missing_mask` 就是公式里的 `M`。

### 5.2 FATE.py: normalization 和 NaN 临时填 0

代码：

```python
norm_data, self.norm_parameters = fm.normalization(data)
norm_data_x = np.nan_to_num(norm_data, 0)
```

这两行做了两件事。

第一，`normalization(data)` 把每一列归一化到 `[0, 1]` 附近。

比如：

```text
年龄 30/40 -> 0.3/0.4
收入 8000 -> 0.8
信用分 700 -> 0.7
```

第二，`np.nan_to_num(norm_data, 0)` 把 NaN 临时填成 0。

这里最容易误解：这个 0 不是模型认为的真实值。它只是为了让神经网络输入矩阵里没有 NaN。

真正告诉模型“这个 0 是不是可靠值”的，是后面一起传入的 mask。

所以训练时模型收到的是：

```python
norm_data_x
missing_mask
```

而不是只收到 `norm_data_x`。

这对应数学里的：

$$
X
$$

以及：

$$
M
$$

### 5.3 FATE.py: 初始化 FATEImputerNet

代码：

```python
self.model = fm.FATEImputerNet(
    num_features=dim,
    embedding_dim=self.embedding_dim,
    depth=self.depth,
    heads=self.heads,
    dropout=self.dropout,
).to(self.device)
```

这里 `dim` 是表格列数。

如果数据有 3 列：

```text
年龄、收入、信用分
```

那么：

```text
num_features = 3
```

在这个模型里，每一列会被当成一个 token。也就是说，一行 3 列数据会变成 3 个 token：

```text
年龄 token
收入 token
信用分 token
```

`embedding_dim` 表示每个 token 变成多少维向量。

如果：

```text
embedding_dim = 32
```

那么每个单元格不再只是一个数，而是一个 32 维向量。

### 5.4 FATE.py: 调用真正的训练循环

代码：

```python
fm.train_fate_algorithm(
    self.model,
    norm_data_x,
    missing_mask,
    params,
    self.device,
)
```

这里把外层准备好的东西交给 `FATE_modules.py`：

```text
self.model   -> f_theta
norm_data_x  -> X
missing_mask -> M
params       -> batch_size / epoch / learning_rate / mask_rate
```

所以真正的训练逻辑在：

```text
tabular/imputation/FATE_modules.py
train_fate_algorithm(...)
```

### 5.5 FATE_modules.py: 每个 batch 取出 X 和 M

训练循环里有：

```python
x_mb = torch.tensor(data_x[batch_idx], dtype=torch.float32).to(device)
m_mb = torch.tensor(mask[batch_idx], dtype=torch.float32).to(device)
```

这里：

```text
x_mb = 当前 batch 的 X
m_mb = 当前 batch 的 M
```

比如当前 batch 是前面那个 2 行 3 列例子：

$$
x\_mb =
\begin{bmatrix}
0.3 & 0 & 0.7\\
0.4 & 0.8 & 0
\end{bmatrix}
$$

$$
m\_mb =
\begin{bmatrix}
1 & 0 & 1\\
1 & 1 & 0
\end{bmatrix}
$$

### 5.6 FATE_modules.py: sample_observed_mask 制造训练题

代码：

```python
train_mask, target_mask = sample_observed_mask(m_mb, mask_rate)
```

这行对应前面的两个公式：

$$
T = M \odot R
$$

$$
M^{train}=M\odot(1-R)
$$

代码里的名字和公式对应关系是：

```text
m_mb        = M
target_mask = T
train_mask  = M_train
```

`sample_observed_mask` 内部代码是：

```python
random_mask = (torch.rand_like(mask) < mask_rate).float()
target_mask = mask * random_mask
train_mask = mask * (1.0 - random_mask)
```

逐行解释：

```python
random_mask = (torch.rand_like(mask) < mask_rate).float()
```

这行随机生成 `R`。`random_mask=1` 表示“这个位置本轮想人为遮住”。

```python
target_mask = mask * random_mask
```

这行得到 `T`。只有“原本 observed”并且“本轮被随机遮住”的位置才是 1。

```python
train_mask = mask * (1.0 - random_mask)
```

这行得到 `M_train`。被随机遮住的位置，在训练输入中不再可见。

所以这一行：

```python
train_mask, target_mask = sample_observed_mask(m_mb, mask_rate)
```

就是整个训练方法的核心：

```text
拿已知位置制造缺失题，让模型补，再用标准答案批改。
```

### 5.7 FATE_modules.py: 构造模型输入 X'

代码：

```python
x_input = train_mask * x_mb
```

这对应：

$$
X' = M^{train}\odot X
$$

如果：

$$
x\_mb =
\begin{bmatrix}
0.3 & 0 & 0.7\\
0.4 & 0.8 & 0
\end{bmatrix},
\quad
train\_mask =
\begin{bmatrix}
1 & 0 & 0\\
1 & 0 & 0
\end{bmatrix}
$$

那么：

$$
x\_input =
\begin{bmatrix}
0.3 & 0 & 0\\
0.4 & 0 & 0
\end{bmatrix}
$$

这就是训练时真正喂给模型的数值矩阵。

但模型还会同时收到 `train_mask`：

```python
x_hat = model(x_input, train_mask)
```

所以模型知道哪些 0 是可靠的，哪些 0 是 missing/hidden。

### 5.8 FATE_modules.py: FATEImputerNet.forward 先做 value embedding

进入模型内部：

```python
value_embeddings = self.value_embedding(x)
```

这里的 `x` 就是 `x_input`。

`value_embedding` 的作用是把每个数值变成向量。

如果输入 shape 是：

```text
x: [batch_size, num_features]
```

比如：

```text
[2, 3]
```

表示 2 行 3 列。

那么输出 shape 是：

```text
value_embeddings: [batch_size, num_features, embedding_dim]
```

比如：

```text
[2, 3, 32]
```

意思是：

```text
2 行样本
每行 3 个特征 token
每个 token 是 32 维向量
```

对应 FATE 论文里的：

```text
continuous feature -> fully-connected layer + ReLU -> embedding
```

### 5.9 FATE_modules.py: 缺失位置改用 missing embedding

代码：

```python
missing_embeddings = self.missing_embeddings.unsqueeze(0).expand(x.size(0), -1, -1)

token_embeddings = (
    mask.unsqueeze(-1) * value_embeddings
    + (1.0 - mask).unsqueeze(-1) * missing_embeddings
)
```

这段非常关键。

`value_embeddings` 是根据数值算出来的 embedding。

`missing_embeddings` 是每一列自己学习出来的“缺失状态向量”。

如果某个位置：

```text
mask = 1
```

则：

```text
token = value_embedding
```

如果某个位置：

```text
mask = 0
```

则：

```text
token = missing_embedding
```

公式就是：

$$
z_{ij}
=
M^{train}_{ij}\cdot value\_emb_{ij}
+
(1-M^{train}_{ij})\cdot missing\_emb_j
$$

这就是 FATE 的 incomplete data encoding 在我们代码里的实现。

它解决的问题是：

```text
不要把缺失值当成数值 0；
要把它表示成“第 j 列缺失了”。
```

### 5.10 FATE_modules.py: 加 feature embedding

代码：

```python
token_embeddings = token_embeddings + self.feature_embeddings.unsqueeze(0)
```

这一步是告诉模型：

```text
这个 token 是年龄列
这个 token 是收入列
这个 token 是信用分列
```

如果不加 feature embedding，模型只看到一串 token，不容易知道每个 token 对应哪一个表格列。

所以当前 token 实际包含三部分信息：

```text
1. 数值信息，来自 value_embedding
2. 缺失状态信息，来自 missing_embedding
3. 列身份信息，来自 feature_embedding
```

### 5.11 FATE_modules.py: MissingAwareTransformerBlock 屏蔽 missing key

模型接着执行：

```python
for block in self.blocks:
    hidden = block(hidden, mask)
```

每个 block 内部最关键的是：

```python
key_padding_mask = mask == 0
```

项目 mask 语义是：

```text
mask=1 observed
mask=0 missing/hidden
```

但是 PyTorch 的 `key_padding_mask` 语义是：

```text
True 表示这个位置要被 attention 忽略
```

所以：

```python
key_padding_mask = mask == 0
```

意思是：

```text
缺失/被遮住的位置，不作为可靠 key 被其他 token 关注。
```

后面调用：

```python
attention_output, _ = self.attention(
    x,
    x,
    x,
    key_padding_mask=key_padding_mask,
    need_weights=False,
)
```

这里 `x, x, x` 分别是 query、key、value。

这就是 self-attention：

```text
每个特征 token 去看同一行里的其他特征 token。
```

`key_padding_mask` 做的事情等价于：

$$
Attention(Q,K,V,M)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}} + B
\right)V
$$

其中：

$$
B_j=
\begin{cases}
0, & M_j=1\\
-\infty, & M_j=0
\end{cases}
$$

换成人话：

```text
缺失位置的 attention 权重会变成 0。
```

### 5.12 FATE_modules.py: reconstruction head 输出 X_hat

Transformer 结束后：

```python
return self.reconstruction_head(hidden).squeeze(-1)
```

这里 `hidden` 的 shape 是：

```text
[batch_size, num_features, embedding_dim]
```

`reconstruction_head` 会把每个 token 的向量变回一个数：

```text
[batch_size, num_features, embedding_dim]
        ↓
[batch_size, num_features, 1]
        ↓ squeeze(-1)
[batch_size, num_features]
```

这个输出就是：

```text
x_hat
```

也就是模型预测出来的整张表。

### 5.13 FATE_modules.py: loss 只在 target_mask 上算

回到训练循环：

```python
loss = torch.sum(target_mask * (x_mb - x_hat) ** 2) / (torch.sum(target_mask) + 1e-8)
```

这一行对应：

$$
L(\theta)
=
\frac{
\sum_{i=1}^{n}\sum_{j=1}^{d}T_{ij}(X_{ij}-\hat{X}_{ij})^2
}{
\sum_{i=1}^{n}\sum_{j=1}^{d}T_{ij}+\epsilon
}
$$

代码和公式对应：

```text
target_mask = T
x_mb        = X
x_hat       = X_hat
1e-8        = epsilon
```

为什么要乘 `target_mask`？

因为：

```text
target_mask=1 的位置才是“原本有答案、这轮被人为遮住”的位置。
target_mask=0 的位置不参与 loss。
```

也就是说：

```text
真实 missing 位置不算 loss；
没有被遮住的 observed 位置也不算 loss；
只批改本轮被人为遮住的 observed 位置。
```

### 5.14 FATE.py: predict 和 train 的区别

训练时会随机遮住 observed 位置，是为了制造训练题。

预测时不再随机遮。

`FATE.py` 的预测代码是：

```python
missing_mask = 1.0 - np.isnan(data)
norm_data = fm.normalization_with_parameter(data, self.norm_parameters)
norm_data_x = np.nan_to_num(norm_data, 0)
```

这里得到真实输入：

```text
X
M
```

然后：

```python
imputed_norm_pred = self.model(x_torch, m_torch).cpu().numpy()
```

这对应：

$$
\hat{X}=f_{\theta}(M\odot X,M)
$$

最后：

```python
imputed_data_norm = missing_mask * norm_data_x + (1.0 - missing_mask) * imputed_norm_pred
```

对应：

$$
X^{imputed}
=
M\odot X
+
(1-M)\odot\hat{X}
$$

这一步保证：

```text
原本 observed 的位置永远保留原始值；
原本 missing 的位置才使用模型预测值。
```

## 6. 代码里每个核心变量是什么意思

为了方便你读代码，下面把最重要的变量列成表。

| 代码变量 | 数学符号 | 含义 | shape |
|---|---|---|---|
| `data` | 原始 X | 带 NaN 的输入数据 | `[n, d]` |
| `missing_mask` | `M` | 原始 mask，1=observed，0=missing | `[n, d]` |
| `norm_data_x` | `X` | 归一化后、NaN 临时填 0 的数据 | `[n, d]` |
| `x_mb` | batch X | 当前 batch 的归一化数据 | `[batch, d]` |
| `m_mb` | batch M | 当前 batch 的原始 mask | `[batch, d]` |
| `random_mask` | `R` | 本轮随机想遮哪些位置 | `[batch, d]` |
| `target_mask` | `T` | 真正用于算 loss 的位置 | `[batch, d]` |
| `train_mask` | `M_train` | 训练时模型可见的位置 | `[batch, d]` |
| `x_input` | `X'` | 训练时喂给模型的数据 | `[batch, d]` |
| `value_embeddings` | value emb | 数值 embedding | `[batch, d, emb]` |
| `missing_embeddings` | missing emb | 缺失状态 embedding | `[batch, d, emb]` |
| `token_embeddings` | `Z` | 输入 Transformer 的 token | `[batch, d, emb]` |
| `hidden` | `H` | Transformer 输出表示 | `[batch, d, emb]` |
| `x_hat` | `X_hat` | 模型预测的完整表格 | `[batch, d]` |

## 7. FATE_modules.py 结构

文件：

```text
tabular/imputation/FATE_modules.py
```

包含以下部分。

### 7.1 normalization 工具

```python
normalization(data)
normalization_with_parameter(data, norm_parameters)
renormalization(norm_data, norm_parameters)
```

作用和 GAIN / VAEGAIN / SCIS 中一致：

```text
训练时保存 min/max/den
预测时复用同一套 normalization 参数
最后反归一化回原始尺度
```

### 7.2 sample_observed_mask

```python
sample_observed_mask(mask, mask_rate)
```

作用：

```text
从 observed 位置随机遮住一部分，制造有答案的补全训练题。
```

输入：

```text
mask: 1=observed, 0=missing
```

输出：

```text
train_mask: 模型训练时可见的位置
target_mask: 被人为遮住并用于计算 loss 的位置
```

这个函数是当前 FATE-imputer 能训练的关键，因为真实 missing 位置没有 ground truth，不能直接算 loss。

### 7.3 ContinuousFeatureEmbedding

```python
ContinuousFeatureEmbedding(num_features, embedding_dim)
```

作用：

```text
把每个连续特征值映射成 embedding token。
```

输入 shape：

```text
[batch_size, num_features]
```

输出 shape：

```text
[batch_size, num_features, embedding_dim]
```

对应 FATE 论文中：

```text
continuous feature -> fully-connected layer + ReLU -> embedding
```

### 7.4 MissingAwareTransformerBlock

```python
MissingAwareTransformerBlock(embedding_dim, heads, dropout)
```

作用：

```text
用 key_padding_mask 屏蔽缺失位置，让缺失 token 不作为可靠 key 被其他 token 关注。
```

代码中：

```python
key_padding_mask = mask == 0
```

因为 PyTorch `MultiheadAttention` 的 `key_padding_mask=True` 表示该位置被忽略。

数学上对应：

$$
Attention(Q,K,V,M)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}} + B
\right)V
$$

其中：

$$
B_j =
\begin{cases}
0, & M_j=1\\
-\infty, & M_j=0
\end{cases}
$$

### 7.5 FATEImputerNet

```python
FATEImputerNet(
    num_features,
    embedding_dim=32,
    depth=3,
    heads=4,
    dropout=0.1,
)
```

核心结构：

```text
value embedding
missing embeddings
feature embeddings
missing-aware Transformer blocks
reconstruction head
```

forward 输入：

```text
x: [batch_size, num_features]
mask: [batch_size, num_features]
```

forward 输出：

```text
x_hat: [batch_size, num_features]
```

输出经过 `Sigmoid`，因为训练数据被 min-max normalization 到 `[0, 1]`。

### 7.6 train_fate_algorithm

训练循环：

```python
train_fate_algorithm(model, data_x, mask, params, device)
```

主要步骤：

```text
1. 取 batch
2. sample_observed_mask 得到 train_mask 和 target_mask
3. x_input = train_mask * x
4. x_hat = model(x_input, train_mask)
5. loss = sum(target_mask * (x - x_hat)^2) / sum(target_mask)
6. 反向传播
```

## 8. FATE.py 结构

文件：

```text
tabular/imputation/FATE.py
```

类：

```python
class FATE(BaseImputer)
```

主要参数：

```text
batch_size
epoch
learning_rate
embedding_dim
depth
heads
mask_rate
dropout
device
```

### 8.1 train

```python
train(data, missing_mask=None)
```

流程：

```text
1. 创建临时 checkpoint 目录
2. 自动或手动获得 missing_mask
3. normalization
4. NaN 填 0
5. 初始化 FATEImputerNet
6. 调用 train_fate_algorithm
7. 保存 fate_imputer_complete.pkl
```

### 8.2 predict

```python
predict(data)
```

流程：

```text
1. 检查模型是否训练过
2. 用训练时保存的 normalization 参数归一化
3. NaN 填 0
4. 调用 FATEImputerNet 得到 X_hat
5. 保留 observed 位置，只替换 missing 位置
6. 反归一化
7. 返回 imputed_data
```

核心组合：

$$
X^{imputed}_{norm}
=
M\odot X_{norm}+(1-M)\odot\hat{X}
$$

## 9. 和 FATE 原论文的区别

### 9.1 原论文做分类，当前实现做补全

原论文：

```text
输出 Y_hat
```

当前实现：

```text
输出 X_imputed
```

所以当前实现删除了原论文的分类头。

### 9.2 原论文关注公平性，当前实现不实现 DRL

原论文包含：

```text
Debiased Representation Learning
debiased self-attention
fairness metrics
sensitive attribute handling
```

当前实现不包含这些内容。

原因：

```text
当前项目需求是整合到 imputation 模块，主要评估 MSE / RMSE / MAE。
```

### 9.3 原论文支持类别/连续混合表格，当前第一版只支持数值矩阵

原论文：

```text
categorical embedding
continuous embedding
```

当前实现：

```text
continuous numerical features only
```

原因：

```text
当前 DataPrep imputation 示例和现有 GAIN / VAEGAIN / SCIS 都以数值矩阵为主。
```

后续可以扩展类别特征支持。

### 9.4 原论文没有显式 imputation loss，当前实现新增 masked reconstruction loss

原论文 FATE 主要优化分类 loss。

当前实现为了补全任务，引入：

```text
random observed masking + reconstruction MSE loss
```

这是从 FATE 的缺失感知编码思想到 DataPrep imputation 接口的必要改造。

## 10. 当前实现是否满足项目需求

根据 AGENTS.md 中的任务：

```text
把 FATE 和 DARN 整合到 imputation 里
按照模板实现
```

当前 FATE 实现已经满足第一版 imputation 接入需求：

```text
1. 有 tabular/imputation/FATE.py
2. 有 tabular/imputation/FATE_modules.py
3. 继承 BaseImputer
4. 支持 train(data, missing_mask)
5. 支持 predict(data)
6. 输入 mask 语义为 1=observed, 0=missing
7. 输出与输入同 shape 的 np.ndarray
8. predict 保留 observed 位置，只替换 missing 位置
9. 支持 checkpoint 保存和加载
10. 有 unit_fate.py 单元测试
11. 有 test_fate.py 脚本式测试
```

但需要明确当前限制：

```text
1. 不是 FATE 原论文的完整公平分类复现
2. 暂不支持敏感属性和公平性指标
3. 暂不支持类别特征
4. 暂未接入 examples/imputation.py 和 Web 控制台
5. 当前是最小可运行工程版本，后续可继续增强
```

因此，准确表述应该是：

```text
当前实现满足“把 FATE 思想整合到 DataPrep imputation 模块”的第一版需求；
不满足“完整复现 FATE 原论文公平分类系统”的需求。
```

## 11. 后续可扩展方向

可以继续增强：

```text
1. 支持 categorical feature embedding
2. 支持 mixed-type imputation
3. 增加 MAE loss 或 MSE/MAE 可选
4. 增加 feature-wise reconstruction head
5. 接入 examples/imputation.py
6. 接入 main.py / index.html
7. 对比 GAIN / VAEGAIN / SCIS / FATE 的 MSE/RMSE/MAE
```

## 12. 一句话总结

```text
当前 FATE-imputation 实现保留了 FATE 论文中最有用的 incomplete data encoding 和 missing-aware attention 思想，
删除了原论文的公平分类任务，
新增 reconstruction head 和 masked reconstruction loss，
从而把 FATE 改造成符合 DataPrep BaseImputer 接口的数值型缺失值补全算法。
```
