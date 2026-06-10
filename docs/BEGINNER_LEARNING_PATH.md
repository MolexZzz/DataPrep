# FATE / DARN 集成学习路线

## 1. 适用对象

这份文档面向没有机器学习和深度学习背景、但需要在 2-4 周内把 FATE 和 DARN 接入 `DataPrep imputation` 的开发者。

目标不是先成为深度学习专家，而是先完成一个可运行、可解释、可汇报的工程版本。

## 2. 最终目标

第一阶段只做一件事：

把 FATE 和 DARN 改造成 `DataPrep` 风格的缺失值补全算法，并评估：

1. `MSE`
2. `RMSE`

下游任务，例如分类、公平性评估、联邦预测，可以作为后续加分项，不放进第一版必交范围。

## 3. 必须先记住的概念

### 3.1 缺失值补全

缺失值补全就是把表格里的空值、`NaN`、缺失位置补成合理数值。

在本项目里，补全算法统一长这样：

```python
imputed_data = model.train_and_predict(data_missing, missing_mask)
```

其中：

1. `data_missing` 是带缺失值的数据
2. `missing_mask` 是同样形状的掩码矩阵
3. `imputed_data` 是补全后的完整数据

### 3.2 mask 语义

本项目的 `missing_mask` 语义是：

1. `1 = observed`，这个位置原本有值
2. `0 = missing`，这个位置原本缺失

这点非常重要。

评估时只在 `missing_mask == 0` 的位置计算误差。

### 3.3 MSE 和 RMSE

`MSE` 是平方误差平均值：

```text
MSE = mean((真实值 - 补全值)^2)
```

`RMSE` 是 MSE 开根号：

```text
RMSE = sqrt(MSE)
```

两者都是越小越好。

第一版汇报时，重点看 `MSE/RMSE` 就够了。

## 4. 推荐阅读顺序

### 第一步：先看项目入口

先读：

1. [DOCUMENT_INDEX.md](D:/DataPrep/docs/DOCUMENT_INDEX.md:1)
2. [REQUIREMENTS_SPEC.md](D:/DataPrep/docs/REQUIREMENTS_SPEC.md:1)
3. [USER_GUIDE.md](D:/DataPrep/docs/USER_GUIDE.md:1)

你需要理解：

1. 这个项目能做什么
2. 缺失值补全输入什么、输出什么
3. Web 控制台和 Python 脚本怎么运行

### 第二步：看现有补全算法怎么接入

再读：

1. [CODE_ANALYSIS.md](D:/DataPrep/docs/CODE_ANALYSIS.md:1)
2. [DESIGN_SPEC.md](D:/DataPrep/docs/DESIGN_SPEC.md:1)
3. [examples/imputation.py](D:/DataPrep/examples/imputation.py:1)
4. [tabular/imputation/base.py](D:/DataPrep/tabular/imputation/base.py:1)
5. [tabular/imputation/GAIN.py](D:/DataPrep/tabular/imputation/GAIN.py:1)

你需要理解：

1. `BaseImputer` 是所有补全算法的统一接口
2. `GAIN.py` 是外层算法类
3. `GAIN_modules.py` 是训练细节和神经网络
4. 新算法最好也按这个结构写

### 第三步：看 FATE/DARN 专项文档

再读：

1. [FATE_DARN_INTEGRATION_PLAN.md](D:/DataPrep/docs/FATE_DARN_INTEGRATION_PLAN.md:1)

你需要理解：

1. FATE 原始任务不是纯补全，而是缺失数据上的公平分类
2. DARN 原始任务不是纯补全，而是联邦不完整表预测
3. 第一版不能照搬原主程序
4. 第一版应该只抽取缺失感知编码和重构能力

### 第四步：最后才看论文原始代码

FATE 优先看：

1. [papers/FATE_SIGIR25/code/model.py](D:/DataPrep/papers/FATE_SIGIR25/code/model.py:1)
2. [papers/FATE_SIGIR25/code/run.py](D:/DataPrep/papers/FATE_SIGIR25/code/run.py:1)
3. [papers/FATE_SIGIR25/code/dataloader.py](D:/DataPrep/papers/FATE_SIGIR25/code/dataloader.py:1)

DARN 优先看：

1. [papers/DARN_VLDB25/DARN-main/models/pretrainmodel.py](D:/DataPrep/papers/DARN_VLDB25/DARN-main/models/pretrainmodel.py:1)
2. [papers/DARN_VLDB25/DARN-main/models/build.py](D:/DataPrep/papers/DARN_VLDB25/DARN-main/models/build.py:1)
3. [papers/DARN_VLDB25/DARN-main/train_DARN.py](D:/DataPrep/papers/DARN_VLDB25/DARN-main/train_DARN.py:1)
4. [papers/DARN_VLDB25/DARN-main/utils/data_utils.py](D:/DataPrep/papers/DARN_VLDB25/DARN-main/utils/data_utils.py:1)

不建议一开始就啃论文 PDF。先看项目怎么跑，再看代码怎么接，最后再回头查论文细节。

## 5. 每周学习和开发安排

### 第 1 周：先跑通现有补全流程

目标：

1. 配好 Python 3.10 环境
2. 解决 `dataprep` 包导入问题
3. 跑通 `examples/imputation.py`
4. 跑出现有 `GAIN` 的 `MSE/RMSE`
5. 明白 `BaseImputer` 的 `train`、`predict`、`estimate`
6. 明白 `missing_mask` 的语义

第一周不应该只看文档。至少要跑出一个补全结果，否则后面实现 FATE/DARN 时不知道问题出在环境、路径、数据还是算法。

本周汇报重点：

1. 当前项目有哪些补全算法
2. 输入输出是什么
3. `missing_mask` 为什么是 `1=observed, 0=missing`
4. 现有 `GAIN` 示例是否跑通
5. 当前 `MSE/RMSE` 结果是多少
6. 为什么 FATE/DARN 不能直接复制进来
7. 第一版为什么只评估 `MSE/RMSE`

### 第 1 周推荐执行顺序

第 1 天：

1. 创建或确认 Python 3.10 环境
2. 安装 `requirements.txt`
3. 验证 `torch`、`numpy`、`pandas`、`sklearn` 能导入

第 2 天：

1. 解决 `dataprep` 包导入路径
2. 跑通一个最小 Python 导入测试
3. 跑通 `examples/imputation.py`

第 3 天：

1. 看懂 `examples/imputation.py`
2. 看懂 `mask == 0` 的位置为什么被设为 `np.nan`
3. 看懂 `estimate()` 为什么只在缺失位置评估

第 4 天：

1. 把 `GAIN` 的结果表整理成 PPT
2. 记录 `MSE/RMSE`
3. 截图或记录终端输出

第 5-7 天：

1. 开始看 FATE 的 `model.py`
2. 找出 mask embedding、Transformer、重构头
3. 准备第 2 周实现 FATE 最小版

### 第 2 周：完成 FATE 最小补全版

目标：

1. 新增 `FATE.py`
2. 新增 `FATE_modules.py`
3. 支持数值型输入
4. 支持 `train_and_predict`
5. 能输出 `MSE/RMSE`

本周汇报重点：

1. FATE 原始任务是什么
2. 我们保留了哪些结构
3. 删除了哪些下游任务逻辑
4. 当前跑通结果和误差指标

### 第 3 周：完成 DARN 最小补全版

目标：

1. 新增 `DARN.py`
2. 新增 `DARN_modules.py`
3. 去掉联邦逻辑
4. 保留 SAINT 风格编码和重构头
5. 能输出 `MSE/RMSE`

本周汇报重点：

1. DARN 原始任务是什么
2. 为什么第一版不做联邦训练
3. 单机补全版怎么适配 `BaseImputer`
4. DARN 和 FATE 的初步结果对比

### 第 4 周：工程接入和整理

目标：

1. 接入 `main.py`
2. 接入 `index.html`
3. 更新 `examples/imputation.py`
4. 更新 README 和用户文档
5. 准备最终汇报

本周汇报重点：

1. Web 控制台演示
2. FATE/DARN/GAIN 的 `MSE/RMSE` 对比
3. 当前限制
4. 后续可扩展下游任务

## 6. 不需要先学太深的内容

第一阶段可以先不深入：

1. 反向传播公式
2. Transformer 数学推导
3. 公平性指标
4. 联邦学习聚合公式
5. 论文完整实验复现

你只需要先理解：

1. 输入是什么
2. 输出是什么
3. 缺失位置在哪里
4. 模型怎么训练出补全值
5. 怎么用 `MSE/RMSE` 评价结果

## 7. 一句话路线

先把 FATE 和 DARN 做成能接入 `BaseImputer` 的数值补全器，能跑通 `MSE/RMSE`，再考虑下游分类、公平性或联邦任务。
