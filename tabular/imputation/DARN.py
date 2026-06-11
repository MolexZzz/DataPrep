import numpy as np
import torch

from dataprep.tabular.imputation.base import BaseImputer
import dataprep.tabular.imputation.DARN_modules as dm


class DARN(BaseImputer):
    """
    DARN-inspired imputer for numerical tabular data.

    在 FATE-imputer 结构基础上增加四个层次：

    层次 1（基础）：
        Missing-aware Transformer + MAE/MSE reconstruction loss。
        与 FATE 结构相同，loss 默认改为 MAE（忠于 DARN 论文）。

    层次 2（渐进式自补全，use_progressive=True）：
        每隔 progressive_interval 个 epoch，用当前模型预测 missing 位置。
        更新 fill_values 后，用 pseudo-label loss 监督 missing 位置的输出：
            L_pseudo = MSE(x_hat[missing], fill_values[missing])
        随着训练 fill_values 越来越准，pseudo 监督也越来越可靠。

    层次 3（IPS 加权 attention，use_ips=True）：
        训练前计算 IPS 权重（简单全局或逻辑回归逐格）。
        IPS 在第 0 层 attention 中乘到 key 维度，提升稀少观测特征的权重。
        当 ips_method="logistic" 且 use_progressive=True 时，
        每次更新 fill_values 后会用新数据重新计算 IPS。

    层次 4（缺失分布预测，use_prob_head=True）：
        模型额外输出每格缺失的概率 missing_prob，用 BCE loss 监督：
            L_prob = BCE(missing_prob, 1 - M)
        倒逼模型在补全的同时学习缺失规律。

    Total loss = L_rec + gamma * L_pseudo + beta * L_prob

    接口与 FATE 相同：
        model.train(data_missing, missing_mask)
        imputed_data = model.predict(data_missing)
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
        # 层次 1
        loss_type="mae",
        # 层次 2
        use_progressive=True,
        progressive_interval=10,
        gamma=0.1,
        # 层次 3
        use_ips=False,
        ips_method="simple",
        # 层次 4
        use_prob_head=False,
        beta=0.1,
        device=None,
    ):
        """
        Args:
            batch_size:            每个训练 batch 的样本数
            epoch:                 训练轮数
            learning_rate:         Adam 学习率
            embedding_dim:         每个 token 的向量维度
            depth:                 Transformer 层数（第 0 层 missing-aware，其余普通）
            heads:                 multi-head attention 的 head 数
            mask_rate:             训练时随机遮住 observed 位置的比例
            dropout:               attention 和 FFN 的 dropout 率
            loss_type:             "mae"（默认，忠于 DARN 论文）或 "mse"
            use_progressive:       是否启用渐进式自补全（层次 2）
            progressive_interval:  每隔多少 epoch 更新一次 fill_values
            gamma:                 pseudo-label loss 的权重
            use_ips:               是否启用 IPS 加权 attention（层次 3）
            ips_method:            "simple"（全局 IPS）或 "logistic"（逐格 IPS）
            use_prob_head:         是否启用缺失分布预测头（层次 4）
            beta:                  L_prob 的权重
            device:                "cuda" 或 "cpu"，None 则自动检测
        """
        self.batch_size = batch_size
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.depth = depth
        self.heads = heads
        self.mask_rate = mask_rate
        self.dropout = dropout
        self.loss_type = loss_type
        self.use_progressive = use_progressive
        self.progressive_interval = progressive_interval
        self.gamma = gamma
        self.use_ips = use_ips
        self.ips_method = ips_method
        self.use_prob_head = use_prob_head
        self.beta = beta
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.norm_parameters = None
        self.model = None
        self.dim = None

    def train(self, data: np.ndarray, missing_mask: np.ndarray = None) -> None:
        """
        训练 DARN-imputer。

        Args:
            data:         带 NaN 的数值矩阵，shape = [n_samples, n_features]
            missing_mask: 可选，1=observed，0=missing。不传则根据 NaN 自动生成。
        """
        if hasattr(self, "_create_temp_dir"):
            self._create_temp_dir(prefix="darn_train_")

        data = np.array(data, dtype=np.float64)

        if missing_mask is None:
            missing_mask = 1.0 - np.isnan(data)
        else:
            missing_mask = np.array(missing_mask, dtype=np.float32)

        # Min-max normalization，NaN 临时填 0
        norm_data, self.norm_parameters = dm.normalization(data)
        norm_data_x = np.nan_to_num(norm_data, nan=0.0).astype(np.float32)
        missing_mask = missing_mask.astype(np.float32)
        _, dim = norm_data_x.shape
        self.dim = dim

        # ── 层次 3：训练前计算 IPS ───────────────────────────────────────────
        ips = None
        if self.use_ips:
            print(f"Computing IPS weights (method={self.ips_method})...")
            if self.ips_method == "logistic":
                ips = dm.compute_ips_logistic(norm_data_x, missing_mask)
            else:
                ips = dm.compute_ips_simple(missing_mask)

        # ── 初始化模型 ───────────────────────────────────────────────────────
        self.model = dm.DARNImputerNet(
            num_features=dim,
            embedding_dim=self.embedding_dim,
            depth=self.depth,
            heads=self.heads,
            dropout=self.dropout,
            use_prob_head=self.use_prob_head,
        ).to(self.device)

        params = {
            "batch_size": self.batch_size,
            "epoch": self.epoch,
            "learning_rate": self.learning_rate,
            "mask_rate": self.mask_rate,
            "loss_type": self.loss_type,
            "use_progressive": self.use_progressive,
            "progressive_interval": self.progressive_interval,
            "gamma": self.gamma,
            "beta": self.beta,
        }

        enabled = []
        if self.use_progressive:
            enabled.append("progressive")
        if self.use_ips:
            enabled.append(f"IPS({self.ips_method})")
        if self.use_prob_head:
            enabled.append("L_prob")
        enabled_str = ", ".join(enabled) if enabled else "none"
        print(f"Starting DARN-imputer training on {self.device}. "
              f"Extensions: [{enabled_str}]")

        dm.train_darn_algorithm(
            self.model,
            norm_data_x,
            missing_mask,
            params,
            self.device,
            ips=ips,
            ips_method=self.ips_method,
        )

        if hasattr(self, "_save_checkpoint"):
            self._save_checkpoint("darn_imputer_complete.pkl")

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        用训练好的模型补全 missing 位置。

        Args:
            data: 带 NaN 的数值矩阵，shape = [n_samples, n_features]

        Returns:
            imputed_data: 与输入同 shape 的完整矩阵，observed 位置保留原值。
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        self.model.eval()
        data = np.array(data, dtype=np.float64)
        missing_mask = (1.0 - np.isnan(data)).astype(np.float32)

        norm_data = dm.normalization_with_parameter(data, self.norm_parameters)
        norm_data_x = np.nan_to_num(norm_data, nan=0.0).astype(np.float32)

        x_torch = torch.tensor(norm_data_x, dtype=torch.float32).to(self.device)
        m_torch = torch.tensor(missing_mask, dtype=torch.float32).to(self.device)

        # 预测时用简单 IPS（predict 阶段不做逻辑回归，避免速度问题）
        ips_torch = None
        if self.use_ips:
            ips_np = dm.compute_ips_simple(missing_mask)
            ips_torch = torch.tensor(ips_np, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            out = self.model(x_torch, m_torch, ips_weights=ips_torch)
            if self.model.use_prob_head:
                imputed_norm_pred = out[0].cpu().numpy()
            else:
                imputed_norm_pred = out.cpu().numpy()

        # observed 位置保留原始值，missing 位置用模型预测
        imputed_data_norm = (
            missing_mask * norm_data_x
            + (1.0 - missing_mask) * imputed_norm_pred
        )

        imputed_data = dm.renormalization(imputed_data_norm, self.norm_parameters)
        return imputed_data
