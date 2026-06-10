# DataPrep 安装与部署文档

## 1. 适用范围

本文档说明如何在本地或服务器上安装并运行 `DataPrep`。

文档覆盖两种使用方式：

1. Python 算法调用
2. Web 控制台运行

## 2. 环境概览

## 2.1 推荐软件环境

建议使用：

1. Windows 10/11、Linux 或 macOS
2. Python 3.10
3. Conda 或 Miniconda
4. Git

## 2.2 推荐硬件环境

### 最低配置

1. CPU 4 核
2. 内存 8GB
3. 硬盘可用空间 10GB

### 推荐配置

1. CPU 8 核以上
2. 内存 16GB 以上
3. NVIDIA GPU，显存 8GB 以上
4. 硬盘可用空间 20GB 以上

### 说明

1. `GAIN`、`VAEGAIN`、`SCIS` 可以在 CPU 上跑，但速度可能较慢
2. `ZeroED`、`ZeroEC` 若使用本地 embedding 或本地 LLM，资源要求更高
3. GPU 不是必须，但对训练和嵌入生成更友好

## 3. 获取代码

```bash
git clone <your-repo-url>
cd DataPrep
```

如果仓库已经在本地，可直接进入项目目录。

## 4. 创建 Python 环境

建议使用 Conda：

```bash
conda create -n dataprep python=3.10 -y
conda activate dataprep
```

## 5. 安装 PyTorch

### 5.1 GPU 版本

如果机器有 NVIDIA GPU，并计划使用 CUDA 11.8，可参考：

```bash
conda install pytorch==2.2.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 5.2 CPU 版本

如果不使用 GPU，可以安装 CPU 版本 PyTorch。具体命令建议以 PyTorch 官方安装页为准。

一个常见示例是：

```bash
pip install torch torchvision torchaudio
```

## 6. 安装项目依赖

先安装仓库现有依赖：

```bash
pip install -r requirements.txt
```

再补安装 README 和 Web 手册中实际用到、但 `requirements.txt` 未完全覆盖的基础依赖：

```bash
pip install fastapi uvicorn numpy pandas scikit-learn openpyxl joblib tqdm
```

## 7. 可选依赖说明

### 7.1 仅运行补全算法时

如果你只打算运行：

1. `GAIN`
2. `VAEGAIN`
3. `SCIS`

那么核心依赖主要是：

1. PyTorch
2. NumPy
3. Pandas
4. scikit-learn
5. tqdm

### 7.2 运行 `ZeroED` / `ZeroEC` 时

还需要：

1. LangChain 相关依赖
2. `sentence-transformers` 或 `fast_sentence_transformers`
3. `faiss-cpu`
4. OpenAI 兼容接口的模型服务，或可访问的远程 API

### 7.3 `ZeroEC` 的 embedding 模型

`ZeroEC` 依赖本地 embedding 模型目录，例如：

```text
tabular/correction/all-MiniLM-L6-v2
```

如果没有这个目录，`ZeroEC` 无法完整运行。

## 8. 环境验证

安装后建议先执行以下命令确认核心依赖正常：

```bash
python -c "import torch; print(torch.__version__)"
python -c "import fastapi, uvicorn, pandas, numpy, sklearn; print('ok')"
```

如果启用 GPU，再验证：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

返回 `True` 表示当前环境可以使用 CUDA。

## 9. 启动方式

### 9.1 运行 Python 示例

先跑最简单的补全示例：

```bash
python examples/imputation.py
```

如果你想验证类库接口是否正常，这是第一步。

### 9.2 启动 Web 后端

在项目根目录执行：

```bash
uvicorn main:app --host 127.0.0.1 --port 8088 --reload
```

如果你是通过包路径运行，也可以按代码尾部的方式启动。

### 9.3 打开前端

当前前端是一个静态页面：

1. 直接双击 `index.html`
2. 或在浏览器中打开 `index.html`

前提是后端已经在 `127.0.0.1:8088` 启动。

## 10. 服务器部署说明

如果后端运行在远程服务器，而浏览器在本地机器，需要做端口转发：

```bash
ssh -L 8088:127.0.0.1:8088 your_username@your_server_ip
```

然后本地浏览器再打开 `index.html`。

## 11. 常见问题

### 11.1 `torch.cuda.is_available()` 为 `False`

原因通常有：

1. 没安装 GPU 版 PyTorch
2. CUDA 驱动不匹配
3. 机器本身没有可用 NVIDIA GPU

解决方式：

1. 改用 CPU 运行
2. 或重装匹配的 PyTorch CUDA 版本

### 11.2 `ZeroEC` 启动时报模型或 embedding 路径错误

重点检查：

1. `embedding_model_path` 是否存在
2. `prompt_dir` 是否存在
3. `openai_api_base` 是否可访问

### 11.3 `index.html` 打开后无法运行任务

重点检查：

1. FastAPI 后端是否已经启动
2. 是否占用了 `8088` 端口
3. 前端是否能连到 `127.0.0.1:8088`

### 11.4 训练很慢

常见原因：

1. 在 CPU 上跑深度学习算法
2. 数据量较大
3. `epoch` 设得过高

建议：

1. 先用小数据集验证流程
2. 先降低 `epoch`
3. 尽量使用 GPU

## 12. 建议安装顺序

推荐最稳的安装顺序：

1. 创建 Python 3.10 环境
2. 安装 PyTorch
3. 安装 `requirements.txt`
4. 手工补装 `fastapi`、`uvicorn`、`scikit-learn` 等基础库
5. 先运行 `examples/imputation.py`
6. 再启动 `uvicorn`
7. 最后打开 `index.html`

## 13. 当前已知风险

当前仓库的依赖定义还不完全规范，主要体现在：

1. `requirements.txt` 没有完全覆盖 `main.py` 和各模块的全部运行依赖
2. `README.md`、Web 手册和实际依赖之间存在少量分散
3. `ZeroED` / `ZeroEC` 的外部模型依赖需要用户手工准备

因此，首次部署时建议按本文档逐项核对，而不要只依赖 `pip install -r requirements.txt`。
