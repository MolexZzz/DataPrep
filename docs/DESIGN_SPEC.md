# DataPrep 设计文档

## 1. 设计目标

`DataPrep` 的设计目标是把多种数据治理算法统一到一个可复用、可扩展、可视化的框架中。当前重点是：

1. 用统一接口封装不同算法
2. 同时支持脚本调用和图形界面调用
3. 在运行时提供日志、评估和结果导出

## 2. 总体架构

系统可以分为四层：

1. 基础抽象层
2. 算法实现层
3. 服务调度层
4. 前端展示层

对应关系如下：

1. 基础抽象层：`base.py`、`tabular/imputation/base.py`
2. 算法实现层：`tabular/imputation/`、`tabular/detection/`、`tabular/correction/`
3. 服务调度层：`main.py`
4. 前端展示层：`index.html`

## 3. 代码分层设计

### 3.1 基础抽象层

#### `base.py`

文件位置：[base.py](D:/DataPrep/base.py:1)

职责：

1. 提供所有算法共用的基础能力
2. 定义 `train`、`predict`、`train_and_predict` 抽象接口
3. 提供模型保存、加载、临时目录和 checkpoint 工具

这层解决的问题是：

“不同算法实现差异很大，但在项目层面最好统一成相似的调用方式。”

#### `tabular/imputation/base.py`

文件位置：[tabular/imputation/base.py](D:/DataPrep/tabular/imputation/base.py:1)

职责：

1. 定义补全算法的专用接口
2. 统一补全任务的 `train` / `predict`
3. 提供 `estimate` 方法计算 `MSE`、`RMSE`、`MAE`

### 3.2 算法实现层

每类算法基本都采用“两段式结构”：

1. 外层类文件：负责参数、状态和对外接口
2. 内层模块文件：负责神经网络结构、训练循环、工具函数

例如：

1. `tabular/imputation/GAIN.py`
2. `tabular/imputation/GAIN_modules.py`

这样的设计优点是：

1. 外部调用简单
2. 训练细节集中，便于替换和调试
3. 以后新增算法时可复用模式

### 3.3 服务调度层

文件位置：[main.py](D:/DataPrep/main.py:1)

`main.py` 是整个系统的后端入口，负责：

1. 启动 FastAPI 服务
2. 提供数据预览接口
3. 通过 WebSocket 执行长时任务
4. 在同一流程中调度算法、基线、评估、结果组装

关键接口：

1. `/api/preview`：读取 CSV 并返回预览和字段统计
2. `/api/ws/run_task`：接收前端任务请求，执行算法并返回实时日志和结果

### 3.4 前端展示层

文件位置：[index.html](D:/DataPrep/index.html:1)

前端是一个单页控制台，核心职责：

1. 选择任务类型和算法
2. 填写数据路径和超参数
3. 发起 WebSocket 请求
4. 展示训练日志、指标和结果表格

## 4. 三类任务的设计

### 4.1 缺失值补全设计

#### 统一接口

补全算法都遵守：

1. `train(data, missing_mask)`
2. `predict(data)`
3. `train_and_predict(data, missing_mask)`

#### 当前实现

1. `GAIN`
2. `VAEGAIN`
3. `SCIS`

#### 调用流程

1. `main.py` 读取缺失数据、真值、掩码
2. 同时运行两个 sklearn 基线
3. 创建项目算法实例
4. 调用 `train_and_predict`
5. 计算 `MSE`、`RMSE`、`MAE`
6. 生成结果预览

#### 设计原因

这样设计的好处是：

1. 算法差异被封装在类内部
2. Web 层不需要关心训练细节
3. 评估逻辑可以复用

### 4.2 错误检测设计

#### 统一接口

当前检测任务也使用类似模式：

1. `train(dirty_csv)`
2. `predict(dirty_csv)`
3. `train_and_predict(dirty_csv)`

#### 当前实现

1. `ZeroED`

#### 核心思路

`ZeroED` 并不是纯监督分类器，它更像一个按列处理的混合管线：

1. 先分析列与列之间的关系
2. 再针对每一列训练局部检测逻辑
3. 可能同时结合 sklearn 模型、规则和 LLM 生成的函数
4. 最终合成一张错误位置矩阵

### 4.3 数据修复设计

#### 统一接口

`ZeroEC` 暴露：

1. `train()`
2. `predict()`
3. `train_and_predict()`

#### 核心思路

修复任务不是端到端单个网络，而是多阶段流程：

1. 加载 prompt、数据、模型
2. 构建 embedding 检索资源
3. 模拟人工修复少量样本
4. 生成 Auto-CoT、代码规则和依赖
5. 对错误位置执行检索和 LLM 修复
6. 评估修复结果

## 5. 关键模块设计说明

### 5.1 GAIN 设计

相关文件：

1. [tabular/imputation/GAIN.py](D:/DataPrep/tabular/imputation/GAIN.py:1)
2. [tabular/imputation/GAIN_modules.py](D:/DataPrep/tabular/imputation/GAIN_modules.py:1)

核心思想：

1. 先把缺失值位置用随机噪声填起来
2. 生成器尝试预测真实缺失值
3. 判别器尝试判断某个位置原本是观测值还是生成值
4. 两者对抗训练，最终得到更合理的补全值

### 5.2 VAEGAIN 设计

相关文件：

1. [tabular/imputation/VAEGAIN.py](D:/DataPrep/tabular/imputation/VAEGAIN.py:1)
2. [tabular/imputation/VAEGAIN_modules.py](D:/DataPrep/tabular/imputation/VAEGAIN_modules.py:1)

核心思想：

1. 保留 GAIN 的对抗机制
2. 同时引入 VAE 编码器/解码器
3. 让模型学习“数据可能来自什么潜在分布”
4. 再从潜在空间生成缺失值

### 5.3 SCIS 设计

相关文件：

1. [tabular/imputation/SCIS.py](D:/DataPrep/tabular/imputation/SCIS.py:1)
2. [tabular/imputation/SCIS_modules.py](D:/DataPrep/tabular/imputation/SCIS_modules.py:1)

核心思想：

1. 在 GAIN 思路上继续增强
2. 增加 `Sinkhorn` 距离项，使生成数据分布更接近真实数据分布
3. 使用 Hessian 近似做样本量搜索，再进行重训练

这意味着 `SCIS` 比 `GAIN` 更复杂，训练也更重。

### 5.4 ZeroED 设计

相关文件：

1. [tabular/detection/ZeroED.py](D:/DataPrep/tabular/detection/ZeroED.py:1)
2. [tabular/detection/ZeroED_modules.py](D:/DataPrep/tabular/detection/ZeroED_modules.py:1)

设计特点：

1. 主类只负责组织流程
2. 复杂逻辑都下沉到 `ZeroED_modules.py`
3. 检测粒度是“按列建模，再拼成整表掩码”

### 5.5 ZeroEC 设计

相关文件：

1. [tabular/correction/ZeroEC.py](D:/DataPrep/tabular/correction/ZeroEC.py:1)
2. [tabular/correction/ZeroEC_modules.py](D:/DataPrep/tabular/correction/ZeroEC_modules.py:1)

设计特点：

1. 使用大量外部资源：prompt、embedding、LLM、检索器
2. 不是单个模型，而是多阶段管线
3. 适合复杂语义修复，但环境依赖更重

## 6. Web 控制台设计

### 6.1 前后端交互

交互方式：

1. 普通 HTTP：用于数据预览
2. WebSocket：用于长时训练任务

原因：

1. 训练任务会持续较长时间
2. 需要持续把日志推给前端
3. WebSocket 更适合实时进度显示

### 6.2 指标与结果组织

后端在运行完任务后统一返回：

1. 基线指标
2. 本项目算法指标
3. 可视化预览数据
4. 高亮掩码

这样前端不需要理解算法细节，只负责展示。

## 7. 环境设计

### 7.1 计算设备

系统在代码上普遍允许：

1. CPU
2. CUDA GPU

具体做法是运行时自动判断 `torch.cuda.is_available()`。

### 7.2 外部服务依赖

当使用 `ZeroED` / `ZeroEC` 时，系统还依赖：

1. LLM API 或本地兼容服务
2. embedding 模型目录
3. prompt 模板文件

这类依赖和纯 PyTorch 补全算法不同，需要在安装文档中单独说明。

## 8. FATE / DARN 的扩展设计建议

### 8.1 设计原则

后续集成 `FATE` 和 `DARN` 时，建议遵循：

1. 不直接搬运论文主程序
2. 只抽取“缺失感知编码 + 重构/补全能力”
3. 保持 `BaseImputer` 接口一致

### 8.2 推荐结构

建议新增：

1. `tabular/imputation/FATE.py`
2. `tabular/imputation/FATE_modules.py`
3. `tabular/imputation/DARN.py`
4. `tabular/imputation/DARN_modules.py`

### 8.3 原因

原因是论文原始代码的主任务并不等于“单机缺失值补全”：

1. `FATE` 原始任务偏公平分类
2. `DARN` 原始任务偏联邦不完整表预测

如果直接照搬主程序，会和当前 `imputation` 接口不匹配。

## 9. 设计结论

`DataPrep` 当前已经具备比较清晰的可扩展框架：

1. 基类统一接口
2. 算法类封装训练和预测
3. `main.py` 统一调度
4. `index.html` 统一展示

后续工作重点不是推倒重来，而是在现有结构上继续补齐：

1. 新算法接入
2. 文档完善
3. 依赖和安装规范化


