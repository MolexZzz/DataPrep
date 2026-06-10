# DataPrep 用户使用说明书

## 1. 软件简介

`DataPrep` 是一个面向表格数据的数据治理工具，支持：

1. 缺失值补全
2. 错误检测
3. 数据修复

你既可以把它当作 Python 算法库来用，也可以用图形界面运行。

## 2. 使用前准备

开始前请确认：

1. Python 环境和依赖已安装
2. 若使用 Web 控制台，后端已经启动
3. 输入数据文件路径正确

如果没有完成安装，请先阅读 [INSTALLATION_GUIDE.md](D:/DataPrep/docs/INSTALLATION_GUIDE.md:1)。

## 3. 两种使用方式

### 3.1 方式一：Python 脚本调用

适合：

1. 开发者
2. 希望在代码里批量运行算法的用户

### 3.2 方式二：Web 控制台

适合：

1. 非开发用户
2. 希望通过图形界面查看日志和结果的用户

## 4. Python 方式使用说明

### 4.1 缺失值补全

最简单的补全使用方式：

```python
import pandas as pd
from dataprep.tabular.imputation.GAIN import GAIN

data_missing = pd.read_csv("datasets/imputation/weather_raw.csv").values
missing_mask = pd.read_csv("datasets/imputation/weather_missing_mask.csv").values

model = GAIN(epoch=100, batch_size=64, device="cpu")
imputed = model.train_and_predict(data_missing, missing_mask)
```

### 4.2 错误检测

```python
import pandas as pd
from dataprep.tabular.detection.ZeroED import ZeroED

df_raw = pd.read_csv("datasets/detection/rayyan_dirty_100.csv")
detector = ZeroED()
error_mask = detector.train_and_predict(df_raw)
```

### 4.3 数据修复

```python
from dataprep.tabular.correction.ZeroEC import ZeroEC

corrector = ZeroEC(
    dirty_data_path="datasets/detection/rayyan_dirty_100.csv",
    clean_data_path="datasets/detection/rayyan_clean_100.csv",
    detection_path="datasets/detection/rayyan_dirty_error_detection_100.csv"
)
fixed_df = corrector.train_and_predict()
```

## 5. Web 控制台使用说明

### 5.1 启动后端

```bash
uvicorn main:app --host 127.0.0.1 --port 8088 --reload
```

### 5.2 打开前端

直接打开根目录下的 `index.html`。

### 5.3 页面结构

页面包含三个页签：

1. `Imputation`
2. `Detection`
3. `Correction`

每个页签都包含：

1. 算法选择
2. 输入路径
3. 超参数配置
4. 运行按钮
5. 实时日志
6. 结果表格和图表

## 6. 缺失值补全操作步骤

### 6.1 需要准备的文件

1. 原始缺失数据 CSV
2. 缺失掩码 CSV
3. 真值 CSV

### 6.2 字段含义

1. `Data Path`：含缺失值的数据
2. `Missing Mask Path`：缺失位置矩阵
3. `Ground Truth Path`：完整真值数据

### 6.3 如何理解掩码

当前项目补全任务中：

1. `1` 表示该位置原本有值
2. `0` 表示该位置原本缺失

### 6.4 运行结果

补全完成后，你会看到：

1. `BayesianRidge` 基线指标
2. `RandomForest` 基线指标
3. 当前算法的 `MSE`、`RMSE` 和 `MAE`
4. 补全后的数据预览

## 7. 错误检测操作步骤

### 7.1 需要准备的文件

1. 脏数据 CSV
2. 真实错误位置矩阵 CSV

### 7.2 字段含义

1. `Data Path`：原始脏数据
2. `Error Detection Path`：真实错误位置矩阵

### 7.3 运行结果

检测完成后，你会看到：

1. `Isolation Forest` 指标
2. `LOF` 指标
3. `ZeroED` 指标
4. 检测出的错误位置预览

## 8. 数据修复操作步骤

### 8.1 需要准备的文件

1. 脏数据 CSV
2. 错误位置矩阵 CSV
3. 干净真值 CSV
4. prompt 模板目录
5. embedding 模型目录
6. 可访问的 LLM 服务

### 8.2 字段含义

1. `Data Path`：脏数据路径
2. `Detection Path`：错误位置矩阵
3. `Clean Data Path`：干净真值路径
4. `Embedding Model Path`：本地 embedding 模型目录
5. `Prompt Dir`：prompt 模板目录

### 8.3 运行结果

修复完成后，你会看到：

1. `Mode` 基线
2. `KNN` 基线
3. `ZeroEC` 的 Precision、Recall、F1、EDR
4. 修复后的数据预览

## 9. 如何选择算法

### 9.1 如果你只想先把流程跑通

建议先选：

1. 补全：`GAIN`
2. 检测：`ZeroED`
3. 修复：`ZeroEC`

但如果你暂时没有 LLM 和 embedding 环境，建议先只跑补全模块。

### 9.2 如果你机器没有 GPU

可以：

1. 把 `device` 设为 `cpu`
2. 把 `epoch` 调小
3. 先用小样本数据测试

## 10. 输出结果怎么理解

### 10.1 补全指标

1. `RMSE`：误差平方后求平均再开根号，越小越好
2. `MAE`：绝对误差平均值，越小越好

### 10.2 检测指标

1. `Precision`：报出来的错误里有多少是真的
2. `Recall`：真实错误里找回来了多少
3. `F1`：Precision 和 Recall 的综合指标

### 10.3 修复指标

1. `Precision`：修过的位置里有多少修对了
2. `Recall`：真实错误位置里有多少被正确修复
3. `F1`：综合指标
4. `EDR`：从整体脏数据角度衡量修复带来的净收益

## 11. 常见使用建议

1. 第一次使用时，先跑 `examples/imputation.py`
2. 再用 Web 控制台试现成数据集
3. 先保证路径和输入格式正确
4. 先在小数据集上调参，再上大数据
5. `ZeroEC` 依赖最多，建议最后再尝试

## 12. 常见错误排查

### 12.1 CSV 路径错误

现象：

1. 运行后直接报文件不存在

处理：

1. 检查路径是否是相对项目根目录
2. Windows 路径中尽量避免手工输错反斜杠

### 12.2 掩码形状不一致

现象：

1. 补全任务报输入 shape 不一致

处理：

1. 检查数据、真值、掩码的行列数是否完全一致

### 12.3 没有 LLM 服务

现象：

1. `ZeroED` 或 `ZeroEC` 无法正常执行

处理：

1. 检查 `base_url` 是否可访问
2. 检查 `api_key` 是否正确
3. 如果只是想先体验框架，先跳过修复模块

## 13. 推荐阅读

如果你还想进一步理解代码，建议接着看：

1. [docs/CODE_ANALYSIS.md](D:/DataPrep/docs/CODE_ANALYSIS.md:1)
2. [docs/DESIGN_SPEC.md](D:/DataPrep/docs/DESIGN_SPEC.md:1)
3. [README.md](D:/DataPrep/README.md:1)


