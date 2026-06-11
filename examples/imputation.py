import numpy as np
import torch

import sys
import os
import importlib.util
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
package_parent = os.path.abspath(os.path.join(project_root, "../"))
if package_parent not in sys.path:
    sys.path.insert(0, package_parent)

if "dataprep" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "dataprep",
        os.path.join(project_root, "__init__.py"),
        submodule_search_locations=[project_root],
    )
    dataprep_module = importlib.util.module_from_spec(spec)
    sys.modules["dataprep"] = dataprep_module
    spec.loader.exec_module(dataprep_module)

from dataprep.tabular.imputation.GAIN import GAIN
from dataprep.tabular.imputation.SCIS import SCIS
from dataprep.tabular.imputation.VAEGAIN import VAEGAIN
from dataprep.tabular.imputation.FATE import FATE
from dataprep.tabular.imputation.DARN import DARN

def generate_fake_data(N=1000, D=10, missing_rate=0.2):
    """
    生成模拟数据
    Returns:
        data_true: 完整真实数据 (Ground Truth)
        data_missing: 带 NaN 的缺失数据
        mask: 掩码矩阵 (1=观测, 0=缺失)
    """
    # 1. 生成真实数据 (正态分布)
    data_true = np.random.randn(N, D)

    # 2. 生成掩码 (随机缺失 MCAR)
    # np.random.rand 返回 [0,1) 均匀分布，> 0.2 意味着保留 80% 的数据
    mask = (np.random.rand(N, D) > missing_rate).astype(float)

    # 3. 制造缺失
    data_missing = data_true.copy()
    data_missing[mask == 0] = np.nan

    return data_true, data_missing, mask


def imputation():
    print("========================================")
    print("      Imputation            ")
    print("========================================")

    # 1. 准备数据
    N, D = 1000, 10
    data_true, data_missing, mask = generate_fake_data(N, D, missing_rate=0.2)
    print(f"Data Shape: {data_true.shape}, Missing Rate: 0.2")

    # 2. 初始化模型
    #这里以GAIN为例
    imputer = GAIN(
        batch_size=64,
        hint_rate=0.9,
        alpha=100,
        epoch=100,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # 以下为SCIS的使用示例
    # imputer = SCIS(
    #     batch_size=64,
    #     epoch=100,
    #     initial_value=300,  # 初始训练集大小
    #     thre_value=0.2,
    #     device='cuda' if torch.cuda.is_available() else 'cpu'
    # )

    # 以下为VAEGAIN的使用示例
    # imputer = VAEGAIN(
    #     batch_size=32,
    #     epoch=100,
    #     latent_size=5,  # 隐变量维度
    #     device='cuda' if torch.cuda.is_available() else 'cpu'
    # )

    # 以下为FATE的使用示例
    # imputer = FATE(
    #     batch_size=64,
    #     epoch=100,
    #     embedding_dim=32,
    #     depth=3,
    #     heads=4,
    #     mask_rate=0.2,
    #     device='cuda' if torch.cuda.is_available() else 'cpu'
    # )

    # 以下为DARN的使用示例
    # imputer = DARN(
    #     batch_size=64,
    #     epoch=100,
    #     embedding_dim=32,
    #     depth=3,
    #     heads=4,
    #     mask_rate=0.2,
    #     use_progressive=True,
    #     use_ips=True,
    #     ips_method="simple",
    #     device='cuda' if torch.cuda.is_available() else 'cpu'
    # )

    # 3. 训练并预测
    print("\n[Step 1] Training & Predicting...")
    imputed_data = imputer.train_and_predict(data_missing, mask)
    metrics = imputer.estimate(data_true, imputed_data, mask)

if __name__ == "__main__":
    imputation()
