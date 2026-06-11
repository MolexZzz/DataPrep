# FATE / DARN Imputation 实验测试计划

## 1. 实验目标

本实验计划用于后续系统评估 FATE 和 DARN 在 DataPrep `imputation` 任务中的表现。

当前项目已经完成：

1. FATE / DARN 核心算法接入 `tabular/imputation`。
2. FATE / DARN 接入 DataPrep Console UI。
3. FATE / DARN 可以通过网页运行并输出补全结果。
4. 页面可以显示 `MSE / RMSE / MAE`。

下一阶段实验目标是：

1. 比较 FATE / DARN 与传统 baseline 的效果。
2. 比较 FATE / DARN 与已有深度学习方法 `GAIN / VAEGAIN / SCIS` 的效果。
3. 找到一组相对合理的 FATE / DARN 超参数。
4. 验证 DARN 中 progressive / IPS / prob head 等模块是否有帮助。
5. 最终能够向老师解释：
   - 输入是什么。
   - 输出是什么。
   - 指标怎么算。
   - 哪个算法效果更好。
   - DARN 的各个增强模块有没有带来收益。

## 2. 数据集

第一阶段先使用项目自带 weather imputation 数据集：

```text
Data Path:
datasets/imputation/weather_raw.csv

Missing Mask Path:
datasets/imputation/weather_missing_mask.csv

Ground Truth Path:
datasets/imputation/weather_ground_truth.csv
```

含义：

1. `weather_raw.csv`：带缺失值的数据，算法真正要补全的输入。
2. `weather_missing_mask.csv`：缺失位置标记，`1=observed`，`0=missing`。
3. `weather_ground_truth.csv`：完整真实数据，只用于评估。

## 3. 评估指标

所有实验统一记录：

```text
MSE
RMSE
MAE
```

三个指标都是：

```text
越小越好
```

当前项目只在原本缺失的位置计算误差：

```text
missing_mask == 0
```

指标解释：

| 指标 | 全称 | 含义 | 备注 |
|---|---|---|---|
| MSE | Mean Squared Error | 平均平方误差 | 对大错误更敏感 |
| RMSE | Root Mean Squared Error | 均方根误差 | 和原始数据单位一致 |
| MAE | Mean Absolute Error | 平均绝对误差 | 更直观，表示平均差多少 |

## 4. 运行方式

### 4.1 启动后端

```powershell
cd D:\DataPrep
conda activate dataprep
uvicorn main:app --host 127.0.0.1 --port 8088
```

如果 8088 被占用：

```powershell
Get-NetTCPConnection -LocalPort 8088
Stop-Process -Id <PID>
```

然后重新启动后端。

### 4.2 打开前端

直接用浏览器打开：

```text
D:\DataPrep\index.html
```

在页面中选择：

```text
Imputation
```

然后在 `Method` 下拉框选择要测试的算法。

## 5. 实验记录模板

每次实验都需要记录：

1. 算法名称。
2. 参数配置。
3. MSE。
4. RMSE。
5. MAE。
6. 备注，例如是否快速测试、是否开启 DARN progressive、是否开启 IPS。

推荐记录表：

```text
| ID | Algorithm | Config Name | MSE | RMSE | MAE | Notes |
|---|---|---|---|---|---|---|
| 1 | FATE | fast-check | | | | epoch=1，仅验证能跑 |
| 2 | DARN | fast-check | | | | epoch=1，仅验证能跑 |
```

## 6. 阶段 0：快速验收配置

这一阶段只用于确认 UI 和后端能跑，不用于正式比较效果。

### 6.1 FATE-fast-check

```text
Method: FATE

batch_size      = 256
epoch           = 1
learning_rate   = 0.001
embedding_dim   = 8
depth           = 1
heads           = 2
mask_rate       = 0.2
dropout         = 0
```

预期：

1. 后端显示 `Running FATE...`。
2. 后端显示 `FATE Training: 100%`。
3. 页面显示 MSE / RMSE / MAE。
4. 页面显示 FATE 补全后数据预览。

注意：

```text
该配置只用于验收，不用于正式效果比较。
```

### 6.2 DARN-fast-check

```text
Method: DARN

batch_size              = 256
epoch                   = 1
learning_rate           = 0.001
embedding_dim           = 8
depth                   = 1
heads                   = 2
mask_rate               = 0.2
dropout                 = 0
loss_type               = mae
use_progressive         = false
progressive_interval    = 10
gamma                   = 0.1
use_ips                 = false
ips_method              = simple
use_prob_head           = false
beta                    = 0.1
```

预期：

1. 后端显示 `Running DARN...`。
2. 后端显示 `Extensions: [none]`。
3. 后端显示 `DARN Training: 100%`。
4. 页面显示 MSE / RMSE / MAE。
5. 页面显示 DARN 补全后数据预览。

注意：

```text
该配置只用于验收，不用于正式效果比较。
```

## 7. 阶段 1：正式基础对比实验

这一阶段目标是比较：

```text
BayesianRidge
Random Forest
GAIN
VAEGAIN
SCIS
FATE
DARN-basic
```

其中 `BayesianRidge` 和 `Random Forest` 是 UI 后端自动运行的 baseline。

### 7.1 FATE-basic-20

```text
Method: FATE

batch_size      = 128
epoch           = 20
learning_rate   = 0.001
embedding_dim   = 16
depth           = 2
heads           = 4
mask_rate       = 0.2
dropout         = 0.1
```

### 7.2 FATE-basic-50

```text
Method: FATE

batch_size      = 128
epoch           = 50
learning_rate   = 0.001
embedding_dim   = 32
depth           = 3
heads           = 4
mask_rate       = 0.2
dropout         = 0.1
```

### 7.3 DARN-basic-20

```text
Method: DARN

batch_size              = 128
epoch                   = 20
learning_rate           = 0.001
embedding_dim           = 16
depth                   = 2
heads                   = 4
mask_rate               = 0.2
dropout                 = 0.1
loss_type               = mae
use_progressive         = false
progressive_interval    = 10
gamma                   = 0.1
use_ips                 = false
ips_method              = simple
use_prob_head           = false
beta                    = 0.1
```

### 7.4 DARN-basic-50

```text
Method: DARN

batch_size              = 128
epoch                   = 50
learning_rate           = 0.001
embedding_dim           = 32
depth                   = 3
heads                   = 4
mask_rate               = 0.2
dropout                 = 0.1
loss_type               = mae
use_progressive         = false
progressive_interval    = 10
gamma                   = 0.1
use_ips                 = false
ips_method              = simple
use_prob_head           = false
beta                    = 0.1
```

### 7.5 阶段 1 记录表

```text
| Algorithm | Config | MSE | RMSE | MAE | Notes |
|---|---|---|---|---|---|
| Bayes | UI baseline | | | | 自动运行 |
| Random Forest | UI baseline | | | | 自动运行 |
| FATE | FATE-basic-20 | | | | |
| FATE | FATE-basic-50 | | | | |
| DARN | DARN-basic-20 | | | | no extensions |
| DARN | DARN-basic-50 | | | | no extensions |
```

观察重点：

1. `epoch=20` 到 `epoch=50` 是否让 FATE / DARN 变好。
2. FATE 是否接近或超过 Bayes / Random Forest。
3. DARN-basic 是否接近或超过 FATE。
4. 如果 FATE/DARN 仍然比 baseline 差，需要判断是训练不足、参数不合适，还是当前数据集 baseline 本身很强。

## 8. 阶段 2：FATE 调参实验

这一阶段只调 FATE，目标是找到 FATE 在 weather 数据集上的相对合理配置。

固定：

```text
batch_size    = 128
depth         = 3
heads         = 4
mask_rate     = 0.2
dropout       = 0.1
```

### 8.1 调 epoch

```text
| Config | epoch | embedding_dim | learning_rate | MSE | RMSE | MAE |
|---|---|---|---|---|---|---|
| FATE-epoch-20 | 20 | 32 | 0.001 | | | |
| FATE-epoch-50 | 50 | 32 | 0.001 | | | |
| FATE-epoch-100 | 100 | 32 | 0.001 | | | |
```

### 8.2 调 learning_rate

固定上一小节表现较好的 epoch。

```text
| Config | epoch | embedding_dim | learning_rate | MSE | RMSE | MAE |
|---|---|---|---|---|---|---|
| FATE-lr-1e-3 | best | 32 | 0.001 | | | |
| FATE-lr-5e-4 | best | 32 | 0.0005 | | | |
| FATE-lr-1e-4 | best | 32 | 0.0001 | | | |
```

### 8.3 调 embedding_dim

固定表现较好的 epoch 和 learning_rate。

```text
| Config | epoch | embedding_dim | learning_rate | MSE | RMSE | MAE |
|---|---|---|---|---|---|---|
| FATE-emb-16 | best | 16 | best | | | |
| FATE-emb-32 | best | 32 | best | | | |
| FATE-emb-64 | best | 64 | best | | | |
```

### 8.4 调 mask_rate

固定前面表现较好的配置。

```text
| Config | mask_rate | MSE | RMSE | MAE |
|---|---|---|---|---|
| FATE-mask-0.1 | 0.1 | | | |
| FATE-mask-0.2 | 0.2 | | | |
| FATE-mask-0.3 | 0.3 | | | |
```

## 9. 阶段 3：DARN 消融实验

这一阶段重点验证 DARN 的增强模块是否有帮助。

建议先固定一组基础配置：

```text
batch_size      = 128
epoch           = 50
learning_rate   = 0.001
embedding_dim   = 32
depth           = 3
heads           = 4
mask_rate       = 0.2
dropout         = 0.1
loss_type       = mae
```

### 9.1 DARN-basic

```text
use_progressive         = false
progressive_interval    = 10
gamma                   = 0.1
use_ips                 = false
ips_method              = simple
use_prob_head           = false
beta                    = 0.1
```

### 9.2 DARN-progressive

```text
use_progressive         = true
progressive_interval    = 5
gamma                   = 0.1
use_ips                 = false
ips_method              = simple
use_prob_head           = false
beta                    = 0.1
```

### 9.3 DARN-progressive-IPS

```text
use_progressive         = true
progressive_interval    = 5
gamma                   = 0.1
use_ips                 = true
ips_method              = simple
use_prob_head           = false
beta                    = 0.1
```

### 9.4 DARN-full

```text
use_progressive         = true
progressive_interval    = 5
gamma                   = 0.1
use_ips                 = true
ips_method              = simple
use_prob_head           = true
beta                    = 0.1
```

### 9.5 DARN 消融记录表

```text
| Variant | progressive | IPS | prob head | MSE | RMSE | MAE | Notes |
|---|---|---|---|---|---|---|---|
| DARN-basic | no | no | no | | | | |
| DARN-progressive | yes | no | no | | | | |
| DARN-progressive-IPS | yes | yes | no | | | | ips_method=simple |
| DARN-full | yes | yes | yes | | | | ips_method=simple |
```

观察重点：

1. progressive 是否降低 MSE / RMSE / MAE。
2. IPS 是否进一步改善结果。
3. prob head 是否有帮助。
4. 如果某个模块让效果变差，也要记录，这同样是有效实验结果。

## 10. 阶段 4：DARN IPS 方法对比

只有在 `ips_method=simple` 能稳定跑通后，再尝试 `logistic`。

固定：

```text
use_progressive = true
progressive_interval = 5
use_ips = true
use_prob_head = false
```

对比：

```text
| Config | ips_method | MSE | RMSE | MAE | Runtime |
|---|---|---|---|---|---|
| DARN-IPS-simple | simple | | | | |
| DARN-IPS-logistic | logistic | | | | |
```

注意：

```text
logistic IPS 会更慢，不建议一开始就跑。
```

## 11. 结果判断原则

### 11.1 指标判断

所有指标：

```text
越小越好
```

优先看：

```text
RMSE
MAE
```

同时记录 MSE，因为 MSE 对大误差更敏感。

### 11.2 如果 FATE / DARN 比 baseline 好

可以说明：

```text
FATE / DARN 的 mask-aware Transformer 思路在当前数据集上有效。
```

如果 DARN 的增强版本优于 DARN-basic，可以进一步说明：

```text
progressive / IPS / prob head 对补全任务有帮助。
```

### 11.3 如果 FATE / DARN 没有比 baseline 好

不能马上说明实现失败。

可能原因：

1. weather 数据集特征数较少，传统方法已经很强。
2. 训练 epoch 不够。
3. 超参数还没有调好。
4. 深度模型对数据规模更敏感。
5. FATE / DARN 原论文任务并不是纯 imputation，当前实现是工程化改造版。

可以在报告中写：

```text
当前工作重点是完成 FATE/DARN 到 DataPrep imputation 的工程化接入，并在统一 MSE/RMSE/MAE 指标下完成初步评估。
在 weather 数据集上，传统 baseline 仍然具有较强竞争力，FATE/DARN 的效果受训练轮数、模型容量和 DARN 扩展模块影响，后续需要继续进行超参数搜索和多数据集验证。
```

## 12. 推荐优先级

如果时间有限，按这个顺序跑：

```text
1. FATE-fast-check
2. DARN-fast-check
3. FATE-basic-20
4. DARN-basic-20
5. FATE-basic-50
6. DARN-basic-50
7. DARN-progressive
8. DARN-progressive-IPS
9. DARN-full
10. FATE 调 learning_rate / mask_rate
11. DARN IPS simple vs logistic
```

最少需要完成：

```text
FATE-basic-20
DARN-basic-20
DARN-progressive
```

比较理想的完成范围：

```text
FATE-basic-50
DARN-basic-50
DARN-progressive
DARN-progressive-IPS
DARN-full
```

## 13. 实验完成后需要整理的内容

实验完成后整理一份结果表：

```text
| Algorithm | Config | MSE | RMSE | MAE | Conclusion |
|---|---|---|---|---|---|
```

并写 3 个结论：

1. FATE 与 baseline 的对比结论。
2. DARN-basic 与 FATE 的对比结论。
3. DARN 增强模块的消融结论。

最终汇报时要分清：

```text
工程目标：FATE/DARN 已经成功接入 DataPrep imputation 和 UI。
实验目标：在统一指标下比较不同算法和不同配置的补全效果。
```
