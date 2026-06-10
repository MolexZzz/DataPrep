import numpy as np
import torch

from dataprep.tabular.imputation.base import BaseImputer
import dataprep.tabular.imputation.FATE_modules as fm


class FATE(BaseImputer):
    """
    FATE-inspired imputer for numerical tabular data.

    This class adapts FATE's incomplete data encoding idea to the DataPrep
    imputation interface. It is not a full reproduction of FATE's original
    fairness-aware classifier.

    原始 FATE 是“缺失感知公平分类”模型，输出分类标签。
    这里的 FATE 是“借鉴 FATE 编码思想的补全器”，输出补全后的数值矩阵。

    当前版本只处理数值型矩阵：
    - data: shape = [n_samples, n_features]，缺失位置为 np.nan
    - missing_mask: 同 shape，1=observed，0=missing
    - output: 同 shape，不包含 NaN
    """

    def __init__(
        self,
        batch_size=128,
        epoch=100,
        learning_rate=0.001,
        embedding_dim=32,
        depth=3,
        heads=4,
        mask_rate=0.2,
        dropout=0.1,
        device=None,
    ):
        # 训练循环参数，和 GAIN / VAEGAIN / SCIS 的外层类保持同类风格。
        self.batch_size = batch_size
        self.epoch = epoch
        self.learning_rate = learning_rate

        # Transformer 编码器参数。
        # embedding_dim: 每个表格特征被映射成多少维 token。
        # depth: 堆叠多少层 missing-aware Transformer block。
        # heads: multi-head attention 的 head 数。
        self.embedding_dim = embedding_dim
        self.depth = depth
        self.heads = heads

        # mask_rate 表示训练时从 observed 位置中随机遮住多少比例。
        # 这些人为遮住的位置有 ground truth，因此可以用来计算 reconstruction loss。
        self.mask_rate = mask_rate
        self.dropout = dropout
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # 训练后保存的状态。
        # norm_parameters 用于保证 predict 时和 train 使用完全相同的 min-max scaling。
        self.norm_parameters = None
        self.model = None
        self.dim = None

    def train(self, data: np.ndarray, missing_mask: np.ndarray = None) -> None:
        """
        Train FATE-imputer.

        Args:
            data: numerical array with np.nan in missing positions.
            missing_mask: optional mask with 1=observed and 0=missing.
        """
        # 和现有 imputation 算法一致，训练时创建临时目录保存完整模型对象。
        if hasattr(self, "_create_temp_dir"):
            self._create_temp_dir(prefix="fate_train_")

        data = np.array(data)

        # 如果调用方没有显式传入 mask，就根据 np.nan 自动生成。
        # 注意项目统一语义：1=observed，0=missing。
        if missing_mask is None:
            missing_mask = 1.0 - np.isnan(data)
        else:
            missing_mask = np.array(missing_mask)

        # 先做 min-max normalization。
        # Transformer 输出端用了 Sigmoid，天然更适合预测 [0, 1] 范围内的值。
        norm_data, self.norm_parameters = fm.normalization(data)

        # NaN 不能进入神经网络，所以临时填 0。
        # 这里的 0 不代表真实值；模型还会同时接收 missing_mask，
        # 在内部将 mask=0 的位置替换成 missing embedding。
        norm_data_x = np.nan_to_num(norm_data, 0)
        _, dim = norm_data_x.shape
        self.dim = dim

        # 初始化 FATE 风格的 mask-aware Transformer 补全网络。
        # num_features 就是表格列数，每一列会被看作一个 token。
        self.model = fm.FATEImputerNet(
            num_features=dim,
            embedding_dim=self.embedding_dim,
            depth=self.depth,
            heads=self.heads,
            dropout=self.dropout,
        ).to(self.device)

        params = {
            "batch_size": self.batch_size,
            "epoch": self.epoch,
            "learning_rate": self.learning_rate,
            "mask_rate": self.mask_rate,
        }

        print(f"Starting FATE-imputer training on {self.device}...")
        # 真正的训练细节放在 FATE_modules.py：
        # 训练时会随机遮住 observed entries，并只在这些位置计算 loss。
        fm.train_fate_algorithm(
            self.model,
            norm_data_x,
            missing_mask,
            params,
            self.device,
        )

        if hasattr(self, "_save_checkpoint"):
            self._save_checkpoint("fate_imputer_complete.pkl")

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Impute missing values with the trained FATE-imputer.
        """
        if self.model is None:
            raise RuntimeError("Model needs to be trained first. Call fit/train.")

        self.model.eval()
        data = np.array(data)

        # predict 阶段根据当前输入的 NaN 位置重新生成 mask。
        # 这样可以支持对同 shape 的新缺失矩阵做补全。
        missing_mask = 1.0 - np.isnan(data)

        # 必须使用训练时保存的 min/max/den，而不是重新估计。
        # 否则 train 和 predict 的数值尺度不一致。
        norm_data = fm.normalization_with_parameter(data, self.norm_parameters)
        norm_data_x = np.nan_to_num(norm_data, 0)

        x_torch = torch.tensor(norm_data_x, dtype=torch.float32).to(self.device)
        m_torch = torch.tensor(missing_mask, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            # model 会对所有位置都输出预测值 x_hat。
            # 但最后只使用 missing 位置上的预测值。
            imputed_norm_pred = self.model(x_torch, m_torch).cpu().numpy()

        # 保留 observed 位置的原始值，只替换 missing 位置：
        # mask=1 -> 使用 norm_data_x
        # mask=0 -> 使用模型预测 imputed_norm_pred
        imputed_data_norm = missing_mask * norm_data_x + (1.0 - missing_mask) * imputed_norm_pred

        # 将 [0, 1] 归一化空间的结果还原到原始数据尺度。
        imputed_data = fm.renormalization(imputed_data_norm, self.norm_parameters)
        return imputed_data
