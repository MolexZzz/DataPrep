# 每周进展 PPT 模板

## 1. PPT 总体结构

每周 PPT 建议控制在 6-8 页，不要太长。

推荐结构：

1. 本周目标
2. 已完成工作
3. 关键技术理解
4. 当前结果
5. 遇到的问题
6. 下周计划
7. 风险和需要老师确认的问题

## 2. 第 1 周 PPT：现有流程跑通和方案确认

### 第 1 页：标题

标题示例：

```text
FATE / DARN 集成到 DataPrep Imputation 的第 1 周进展
```

### 第 2 页：本周目标

建议写：

1. 读懂 `DataPrep` 当前补全模块结构
2. 明确 `BaseImputer` 接口
3. 跑通现有 `GAIN` 补全示例
4. 得到第一组 `MSE/RMSE`
5. 确认 FATE/DARN 原始任务和本项目目标的差异
6. 确定第一版评估指标为 `MSE/RMSE`

### 第 3 页：现有项目结构

建议放一张简单图：

```text
data_missing + missing_mask
        |
    BaseImputer
        |
 GAIN / VAEGAIN / SCIS
        |
 imputed_data
        |
 MSE / RMSE
```

### 第 4 页：关键结论

建议写：

1. 当前 `missing_mask` 语义是 `1=observed, 0=missing`
2. 评估只在 `missing_mask == 0` 的位置计算误差
3. 第一周已经跑通现有补全流程
4. FATE 原始任务偏公平分类
5. DARN 原始任务偏联邦预测
6. 第一版应改造成单机缺失值补全器

### 第 5 页：第一周运行结果

建议放：

| 算法 | 数据集 | MSE | RMSE | 说明 |
| --- | --- | ---: | ---: | --- |
| GAIN | synthetic | 待填 | 待填 | 现有补全算法 |

如果还没有跑通，要明确写阻塞原因，例如：

1. Python 环境未配置
2. `dataprep` 包路径未识别
3. PyTorch 未安装
4. 依赖版本不匹配

### 第 6 页：初步集成方案

建议写：

1. 新增 `FATE.py` / `FATE_modules.py`
2. 新增 `DARN.py` / `DARN_modules.py`
3. 遵守 `BaseImputer.train()` / `predict()` / `estimate()`
4. 第一版只支持数值型补全

### 第 7 页：下周计划

建议写：

1. 实现 FATE 最小补全版
2. 跑通合成数据
3. 输出 `MSE/RMSE`
4. 准备和 GAIN 做初步对比

## 3. 第 2 周 PPT：FATE 最小版

建议页面：

1. 本周目标：实现 FATE 补全版
2. FATE 原始代码理解：保留 mask-aware embedding、Transformer、重构头
3. 适配方式：删除分类头、公平性指标、敏感属性逻辑
4. 实现文件：`FATE.py`、`FATE_modules.py`
5. 实验结果：表格展示 `MSE/RMSE`
6. 当前问题：训练慢、参数待调、仅支持数值特征
7. 下周计划：实现 DARN 最小版

## 4. 第 3 周 PPT：DARN 最小版

建议页面：

1. 本周目标：实现 DARN 补全版
2. DARN 原始代码理解：联邦训练 + SAINT 风格编码器
3. 适配方式：去掉联邦客户端和聚合逻辑，只保留重构能力
4. 实现文件：`DARN.py`、`DARN_modules.py`
5. 实验结果：FATE / DARN / GAIN 的 `MSE/RMSE` 对比
6. 当前限制：没有恢复下游任务和联邦学习
7. 下周计划：接入 Web、补文档、准备最终汇报

## 5. 第 4 周 PPT：最终整理

建议页面：

1. 项目目标回顾
2. 最终实现结构
3. FATE 接入结果
4. DARN 接入结果
5. `MSE/RMSE` 对比表
6. Web 控制台演示截图
7. 当前限制和后续工作
8. 总结

## 6. 实验结果表格模板

可以直接使用：

| 算法 | 数据集 | MSE | RMSE | 备注 |
| --- | --- | ---: | ---: | --- |
| GAIN | synthetic | 待填 | 待填 | 现有基线 |
| FATE-Imputer | synthetic | 待填 | 待填 | 新增 |
| DARN-Imputer | synthetic | 待填 | 待填 | 新增 |

如果有真实数据集，可以再加：

| 算法 | 数据集 | MSE | RMSE | 备注 |
| --- | --- | ---: | ---: | --- |
| GAIN | weather | 待填 | 待填 | 现有基线 |
| FATE-Imputer | weather | 待填 | 待填 | 新增 |
| DARN-Imputer | weather | 待填 | 待填 | 新增 |

## 7. 老师可能会问的问题

### 为什么不直接复现论文完整任务？

建议回答：

FATE 原始任务偏公平分类，DARN 原始任务偏联邦预测，和当前 `DataPrep imputation` 的统一接口不完全一致。第一阶段先完成缺失值补全能力和 `MSE/RMSE` 评估，保证工程可运行；下游任务后续再扩展。

### 为什么第一版只做数值特征？

建议回答：

当前项目已有补全接口和评估体系主要面向数值矩阵。分类特征需要额外编码、反编码和分类重构损失，复杂度更高。为了 2-4 周内稳定交付，第一版先支持数值型补全。

### MSE/RMSE 在哪里计算？

建议回答：

在 `BaseImputer.estimate()` 中计算，只在原始缺失位置，也就是 `missing_mask == 0` 的位置计算误差。

### 下游任务能不能加？

建议回答：

可以加，但建议作为第二阶段。当前第一阶段目标是把 FATE/DARN 稳定接入 imputation，并完成 `MSE/RMSE` 评估。下游分类、公平性或联邦预测可以在补全器稳定后继续扩展。

## 8. 每周汇报原则

每周汇报只讲三件事：

1. 这周做了什么
2. 现在跑出了什么结果
3. 下周准备解决什么问题

不要在 PPT 里堆大量代码。代码只放关键片段和结构图。
