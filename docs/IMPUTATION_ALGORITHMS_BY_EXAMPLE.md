# GAIN / VAEGAIN / SCIS 直观计算例子

## 1. 这份文档解决什么问题

这份文档不从论文公式开始讲，而是用一个很小的表格例子，模拟 `GAIN`、`VAEGAIN`、`SCIS` 三种补全算法到底在算什么。

你可以先记住一句话：

```text
三种算法的输入输出都一样：

输入：带缺失值的数据 data_missing + 缺失掩码 missing_mask
输出：补全后的完整表格 imputed_data
```

它们的区别不是输入输出，而是中间“怎么猜缺失值、怎么训练模型”的方法不同。

## 2. 一个共同的小例子

假设我们有一个 3 行 3 列的数据表：

```text
        年龄     收入     消费
用户1   20      4000    1000
用户2   30      NaN     2000
用户3   40      8000    3000
```

第 2 个用户的 `收入` 缺失了。我们希望模型补出这个值。

在项目里，数据和 mask 可以写成：

```python
data_missing = [
    [20, 4000, 1000],
    [30,  NaN, 2000],
    [40, 8000, 3000],
]

missing_mask = [
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
]
```

mask 的语义一定要记住：

```text
1 = observed，原本有值
0 = missing，原本缺失
```

也就是说：

```text
用户2的收入位置 mask = 0
```

这就是我们要补的位置。

## 3. 先归一化

GAIN、VAEGAIN、SCIS 都会先做 min-max 归一化，把不同量纲的数据变到接近 `[0, 1]` 的范围。

原始数据：

```text
年龄范围：20 到 40
收入范围：4000 到 8000
消费范围：1000 到 3000
```

归一化公式：

```text
归一化值 = (原始值 - 最小值) / (最大值 - 最小值)
```

所以：

```text
年龄 20 -> (20 - 20) / (40 - 20) = 0
年龄 30 -> (30 - 20) / (40 - 20) = 0.5
年龄 40 -> (40 - 20) / (40 - 20) = 1

收入 4000 -> (4000 - 4000) / (8000 - 4000) = 0
收入 8000 -> (8000 - 4000) / (8000 - 4000) = 1

消费 1000 -> (1000 - 1000) / (3000 - 1000) = 0
消费 2000 -> (2000 - 1000) / (3000 - 1000) = 0.5
消费 3000 -> (3000 - 1000) / (3000 - 1000) = 1
```

归一化后：

```text
        年龄     收入     消费
用户1   0       0       0
用户2   0.5     NaN     0.5
用户3   1       1       1
```

训练时，缺失位置的 `NaN` 通常会先临时填成 0，或者填成一个很小的随机噪声。

假设这次随机噪声是 `0.03`，那么用户2会变成：

```text
用户2输入 x = [0.5, 0.03, 0.5]
用户2掩码 m = [1,   0,    1]
```

注意：

```text
0.03 不是真实补全值，只是为了让神经网络能接收一个完整向量。
真正的补全值要由模型生成。
```

## 4. GAIN 怎么算

## 4.1 GAIN 的核心角色

GAIN 有两个网络：

```text
Generator      生成器
Discriminator 判别器
```

它们可以类比成：

```text
Generator      负责填空
Discriminator 负责检查哪些格子是真的，哪些格子是填出来的
```

在当前项目代码里，二者都是简单 MLP。

GAIN 的生成器输入是：

```text
[x, m]
```

也就是把数据和 mask 拼在一起。

对于用户2：

```text
x = [0.5, 0.03, 0.5]
m = [1,   0,    1]
```

拼接后：

```text
[x, m] = [0.5, 0.03, 0.5, 1, 0, 1]
```

这个向量送进 Generator。

## 4.2 用一个简化版 Generator 模拟

真实代码里 Generator 是三层 MLP：

```text
Linear -> ReLU -> Linear -> ReLU -> Linear -> Sigmoid
```

为了讲清楚，我们不用完整矩阵乘法，只假设 Generator 看到了这条规律：

```text
年龄越大，收入越高；
消费越高，收入也越高。
```

用户2的年龄和消费都是中间值：

```text
年龄 = 0.5
消费 = 0.5
```

那么 Generator 可能输出：

```text
G_sample = [0.52, 0.48, 0.51]
```

这里三个值表示模型对三个字段都给出了预测。

但我们不会把所有字段都替换掉。因为年龄和消费本来就有真实值，只需要替换缺失的收入。

组合公式是：

```text
补全后 = m * 原始输入 + (1 - m) * 生成结果
```

代入用户2：

```text
m              = [1,   0,    1]
原始输入 x      = [0.5, 0.03, 0.5]
生成结果 G      = [0.52, 0.48, 0.51]

m * x          = [0.5, 0,    0.5]
(1 - m) * G    = [0,   0.48, 0]

补全后          = [0.5, 0.48, 0.5]
```

所以 GAIN 给出的归一化收入补全值是：

```text
0.48
```

再反归一化：

```text
收入 = 0.48 * (8000 - 4000) + 4000
     = 0.48 * 4000 + 4000
     = 5920
```

所以最后用户2变成：

```text
用户2 = [30, 5920, 2000]
```

## 4.3 Discriminator 在干什么

Generator 填完之后，Discriminator 会看到：

```text
Hat_X = [0.5, 0.48, 0.5]
```

然后它输出每个位置“是真实观测值”的概率，例如：

```text
D_prob = [0.90, 0.35, 0.88]
```

意思是：

```text
年龄位置：我觉得 90% 像真实值
收入位置：我觉得 35% 像真实值
消费位置：我觉得 88% 像真实值
```

因为收入位置其实是生成出来的，所以 Discriminator 觉得它不像真实值。

训练时：

```text
Discriminator 想把观测位置判成真，把缺失生成位置判成假。
Generator 想让缺失生成位置也被判成真。
```

二者互相竞争，Generator 就会被迫生成越来越像真实数据的补充值。

## 4.4 GAIN 的重点

GAIN 的重点是：

```text
用 GAN 的对抗训练，让生成器学会补缺失值。
```

它的主要约束有两个：

```text
1. 缺失位置生成得要像真的，骗过 Discriminator
2. 观测位置不能乱改，要能重构原来的观测值
```

## 5. VAEGAIN 怎么算

## 5.1 VAEGAIN 和 GAIN 最大区别

GAIN 的 Generator 是：

```text
输入 [x, m] -> MLP -> 直接输出补全值
```

VAEGAIN 的 Generator 不是一个普通 MLP，而是：

```text
Encoder -> latent z -> Decoder
```

也就是说：

```text
GAIN    是直接猜答案
VAEGAIN 是先把这行数据压缩成一个潜在表示，再从潜在表示还原数据
```

## 5.2 同一个用户2，VAEGAIN 怎么走

用户2输入仍然是：

```text
x = [0.5, 0.03, 0.5]
m = [1,   0,    1]
```

VAEGAIN 先把它送进 Encoder。

Encoder 不直接输出补全值，而是输出一个潜变量分布：

```text
z_mean    = [0.10, 0.60]
z_log_var = [-1.00, -0.50]
```

你可以先不用管 `log_var` 的数学细节。直观理解：

```text
z_mean    表示这条用户数据大概位于潜在空间的哪里
z_log_var 表示这个位置的不确定性有多大
```

然后 VAEGAIN 会采样一个 latent 向量：

```text
z = z_mean + 随机扰动
```

假设采样后得到：

```text
z = [0.05, 0.72]
```

这个 `z` 再送进 Decoder。

Decoder 输出：

```text
x_hat_mean      = [0.49, 0.50, 0.52]
x_hat_log_sigma = [-2.0, -1.5, -2.2]
```

其中：

```text
x_hat_mean 是预测的平均值
x_hat_log_sigma 是不确定性
```

真正用于补全的通常是：

```text
x_hat_mean
```

所以生成结果是：

```text
G_sample = [0.49, 0.50, 0.52]
```

再用同样的组合公式：

```text
补全后 = m * 原始输入 + (1 - m) * 生成结果
```

代入：

```text
m              = [1,   0,    1]
原始输入 x      = [0.5, 0.03, 0.5]
生成结果 G      = [0.49, 0.50, 0.52]

补全后          = [0.5, 0.50, 0.5]
```

VAEGAIN 给出的归一化收入补全值是：

```text
0.50
```

反归一化：

```text
收入 = 0.50 * 4000 + 4000
     = 6000
```

所以用户2变成：

```text
用户2 = [30, 6000, 2000]
```

## 5.3 VAEGAIN 的 Discriminator 仍然存在

VAEGAIN 不是纯 VAE，它仍然有 Discriminator。

也就是说它同时做两件事：

```text
1. 像 VAE 一样学习数据的潜在分布
2. 像 GAIN 一样用 Discriminator 逼生成值更像真实值
```

训练 Generator，也就是 Encoder + Decoder 时，loss 主要有三部分：

```text
1. adversarial loss
   让生成出来的缺失值骗过 Discriminator

2. reconstruction loss
   让观测位置能被重构出来

3. KL loss
   让 latent z 的分布不要乱跑，尽量接近标准正态分布
```

如果只用很口语的方式理解：

```text
adversarial loss   让补充值看起来像真的
reconstruction loss 让已知值别被学坏
KL loss             让潜在空间更规整
```

## 5.4 VAEGAIN 的重点

VAEGAIN 的重点是：

```text
不要直接从 x 猜缺失值，而是先学习数据背后的潜在结构。
```

比如用户数据里可能有这种潜在模式：

```text
年轻低收入低消费
中年中收入中消费
高收入高消费
```

VAEGAIN 希望先把用户映射到某个潜在模式，再从这个模式里生成合理的完整数据。

## 6. SCIS 怎么算

## 6.1 SCIS 和 GAIN 的关系

SCIS 的网络结构很像 GAIN：

```text
Generator      仍然是 MLP
Discriminator 仍然是 MLP
```

所以单次补全时，它和 GAIN 的流程非常像。

用户2输入：

```text
x = [0.5, 0.03, 0.5]
m = [1,   0,    1]
```

Generator 可能输出：

```text
G_sample = [0.51, 0.47, 0.50]
```

组合后：

```text
补全后 = [0.5, 0.47, 0.5]
```

反归一化：

```text
收入 = 0.47 * 4000 + 4000
     = 5880
```

到这里看起来和 GAIN 没多大区别。

真正区别在训练。

## 6.2 SCIS 多了 Sinkhorn loss

GAIN 训练 Generator 时，主要看：

```text
1. 能不能骗过 Discriminator
2. 观测位置重构得准不准
```

SCIS 在此基础上多看一个东西：

```text
生成数据整体分布像不像真实数据整体分布
```

这个整体分布距离用 `Sinkhorn loss` 表示。

假设一个 batch 里真实数据大概是：

```text
真实 batch:
用户1 [0.0, 0.0, 0.0]
用户2 [0.5, ?,   0.5]
用户3 [1.0, 1.0, 1.0]
```

Generator 生成后：

```text
生成 batch:
用户1 [0.1, 0.2, 0.1]
用户2 [0.5, 0.47, 0.5]
用户3 [0.8, 0.7, 0.9]
```

如果只看用户2的收入，可能还不错。

但从整体看：

```text
生成 batch 的分布可能偏低、偏散、或者不像真实数据。
```

Sinkhorn loss 就是在惩罚这种整体分布差异。

所以 SCIS 的 Generator loss 可以理解为：

```text
SCIS_G_loss =
    对抗损失
  + alpha * 观测位置重构误差
  + value * Sinkhorn分布距离
```

对比 GAIN：

```text
GAIN_G_loss =
    对抗损失
  + alpha * 观测位置重构误差
```

区别就在第三项。

## 6.3 SCIS 的三阶段训练

SCIS 还有一个比 GAIN 复杂的地方：它不是简单从头到尾训练一遍。

代码里是三阶段：

```text
Phase 1: Initial training
Phase 2: SCIS Search
Phase 3: Retraining
```

### Phase 1：先用一部分数据训练

假设我们有 10000 行数据，SCIS 可能先拿 500 行训练一个初始模型。

目的不是得到最终模型，而是先观察：

```text
模型对数据有多敏感？
还需要多少样本才能稳定？
```

### Phase 2：估计需要多少样本

SCIS 会用 Hessian 近似做一个搜索。

不用被 Hessian 吓到，可以先这样理解：

```text
它在估计：如果我多给一些训练样本，模型参数和误差会变化多少？
```

如果从 500 行增加到 2000 行，模型变化很大，说明 500 行不够。

如果从 5000 行增加到 10000 行，模型变化很小，说明 5000 行可能已经够了。

SCIS 就是在找一个比较合适的训练样本量：

```text
estimated_n
```

### Phase 3：用 estimated_n 重训练

假设搜索后得到：

```text
estimated_n = 3000
```

那它就从全量数据里抽 3000 行，重新训练最终模型。

这就是 SCIS 的完整训练思路：

```text
先粗训 -> 估计样本量 -> 再重训
```

## 6.4 SCIS 的重点

SCIS 的重点不是换了一个很新奇的网络，而是：

```text
1. 在 GAIN 的 loss 上加入 Sinkhorn 分布约束
2. 用三阶段训练估计合适样本量
```

所以你可以记成：

```text
SCIS = GAIN 的网络 + 更复杂的训练策略
```

## 7. 三种算法放在同一个例子里对比

同一个缺失用户：

```text
用户2 = [年龄 30, 收入 NaN, 消费 2000]
```

归一化后：

```text
用户2 = [0.5, NaN, 0.5]
```

临时填噪声：

```text
x = [0.5, 0.03, 0.5]
m = [1,   0,    1]
```

三种算法可能得到：

```text
GAIN:
Generator 直接输出 G = [0.52, 0.48, 0.51]
补全后 = [0.5, 0.48, 0.5]
收入 = 5920

VAEGAIN:
Encoder 先得到 z_mean / z_log_var
采样 z
Decoder 输出 G = [0.49, 0.50, 0.52]
补全后 = [0.5, 0.50, 0.5]
收入 = 6000

SCIS:
Generator 直接输出 G = [0.51, 0.47, 0.50]
补全后 = [0.5, 0.47, 0.5]
收入 = 5880
训练时额外考虑 Sinkhorn 分布距离和样本量搜索
```

注意：这里的 `5920 / 6000 / 5880` 是为了讲解流程随手模拟的数字，不是代码真实跑出来的固定结果。

真实结果取决于：

```text
数据集
缺失率
随机种子
训练轮数
网络参数
学习率
```

## 8. 最核心区别

如果只记一张表，记这张：

```text
算法      生成方式                    训练重点
GAIN      MLP 直接生成缺失值           GAN 对抗 + 观测值重构
VAEGAIN   Encoder-Decoder 生成缺失值   潜变量分布 + GAN 对抗 + VAE loss
SCIS      MLP 直接生成缺失值           GAIN loss + Sinkhorn loss + 三阶段训练
```

再换成更口语的说法：

```text
GAIN:
我直接看这行数据，然后猜缺失值。

VAEGAIN:
我先判断这行数据属于哪种潜在类型，再从这个类型里生成完整数据。

SCIS:
我还是直接猜缺失值，但我不仅要求单个值猜得准，还要求一批生成数据整体分布像真实数据，并且我要估计用多少训练样本比较合适。
```

## 9. 从代码角度看差异

对应到项目代码：

```text
GAIN.py
GAIN_modules.py
```

特点：

```text
self.generator
self.discriminator
train_gain_algorithm(...)
```

VAEGAIN：

```text
VAEGAIN.py
VAEGAIN_modules.py
```

特点：

```text
self.encoder
self.decoder
self.discriminator
train_vaegain(...)
```

SCIS：

```text
SCIS.py
SCIS_modules.py
```

特点：

```text
self.generator
self.discriminator
train_scis_algorithm(...)
sinkhorn_loss_torch(...)
compute_hessian_diag(...)
```

所以后面实现 FATE / DARN 时，我们要判断它们更像哪一种：

```text
如果只是普通神经网络直接补全：
  像 GAIN

如果有 encoder / decoder / latent representation：
  像 VAEGAIN

如果训练流程分阶段，或者 loss 很复杂：
  像 SCIS
```

## 10. 你现在需要真正理解到什么程度

现在不需要背公式，也不需要自己手推反向传播。

你需要理解到这个程度就够了：

```text
1. 三者都接收 data_missing 和 missing_mask。
2. 三者都会只替换 mask = 0 的缺失位置。
3. GAIN 用 MLP Generator 直接补。
4. VAEGAIN 用 Encoder + Decoder 通过潜变量补。
5. SCIS 网络像 GAIN，但训练时多了 Sinkhorn 分布约束和三阶段搜索。
6. 最后都要反归一化，返回原始尺度的数据。
```

如果你能用自己的话讲出下面这句话，就说明已经过关：

```text
GAIN 是基础 GAN 补全；VAEGAIN 把生成器换成了 VAE；SCIS 还是 GAIN 风格网络，但加了分布距离和更复杂的训练流程。
```

