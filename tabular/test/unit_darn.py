import sys
import os
import unittest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from path_setup import ensure_dataprep_importable

ensure_dataprep_importable(__file__)

try:
    from dataprep.tabular.imputation.DARN import DARN
    import dataprep.tabular.imputation.DARN_modules as dm
except ImportError as e:
    raise ImportError(f"导入失败，请检查文件位置。\n详细错误: {e}")


# ============================================================
# DARN_modules 单元测试
# ============================================================

class TestIPSComputation(unittest.TestCase):
    """测试 IPS 计算函数。"""

    def setUp(self):
        np.random.seed(42)
        self.n, self.d = 50, 4
        self.mask = (np.random.rand(self.n, self.d) > 0.3).astype(np.float32)
        self.data = np.random.randn(self.n, self.d).astype(np.float32)
        self.data[self.mask == 0] = 0.0  # NaN 已填 0

    def test_compute_ips_simple_shape(self):
        ips = dm.compute_ips_simple(self.mask)
        self.assertEqual(ips.shape, (self.n, self.d))

    def test_compute_ips_simple_missing_positions_are_zero(self):
        """missing 位置（M=0）的 IPS 应为 0，不应参与 attention。"""
        ips = dm.compute_ips_simple(self.mask)
        missing_ips = ips[self.mask == 0]
        self.assertTrue((missing_ips == 0).all(),
                        "missing 位置的 IPS 应为 0")

    def test_compute_ips_simple_observed_positions_above_one(self):
        """observed 位置的 IPS 应 ≥ 1（逆概率不小于 1）。"""
        ips = dm.compute_ips_simple(self.mask)
        observed_ips = ips[self.mask == 1]
        self.assertTrue((observed_ips >= 1.0 - 1e-6).all(),
                        "observed 位置的 IPS 应 >= 1")

    def test_compute_ips_simple_rare_column_has_higher_ips(self):
        """缺失率更高的列，IPS 应更大。"""
        mask_rare = np.ones((self.n, 2), dtype=np.float32)
        mask_rare[:, 0] = (np.random.rand(self.n) > 0.8).astype(np.float32)  # 观测率 20%
        mask_rare[:, 1] = (np.random.rand(self.n) > 0.2).astype(np.float32)  # 观测率 80%
        ips = dm.compute_ips_simple(mask_rare)
        # 第 0 列（稀少）的 IPS 应高于第 1 列（常见）
        mean_ips_rare = ips[mask_rare[:, 0] == 1, 0].mean()
        mean_ips_common = ips[mask_rare[:, 1] == 1, 1].mean()
        self.assertGreater(mean_ips_rare, mean_ips_common,
                           "缺失率高的列 IPS 应更大")

    def test_compute_ips_logistic_shape(self):
        """逐格 IPS 的 shape 应与输入相同。"""
        ips = dm.compute_ips_logistic(self.data, self.mask)
        self.assertEqual(ips.shape, (self.n, self.d))

    def test_compute_ips_logistic_missing_positions_are_zero(self):
        """逐格 IPS 在 missing 位置也应为 0。"""
        ips = dm.compute_ips_logistic(self.data, self.mask)
        self.assertTrue((ips[self.mask == 0] == 0).all())

    def test_compute_ips_all_observed_column(self):
        """全部 observed 的列不应报错，IPS 应稳定输出。"""
        mask_all = np.ones((self.n, self.d), dtype=np.float32)
        ips = dm.compute_ips_simple(mask_all)
        self.assertFalse(np.isnan(ips).any())
        self.assertFalse(np.isinf(ips).any())


class TestMissingAwareAttention(unittest.TestCase):
    """测试自定义 attention（支持 IPS 加权）。"""

    def setUp(self):
        self.B, self.T, self.emb = 4, 6, 16  # 6 = 2d，d=3
        self.heads = 2
        self.attn = dm.MissingAwareAttention(self.emb, self.heads, dropout=0.0)

    def test_output_shape(self):
        x = torch.randn(self.B, self.T, self.emb)
        out = self.attn(x)
        self.assertEqual(out.shape, (self.B, self.T, self.emb))

    def test_no_nan_without_mask(self):
        x = torch.randn(self.B, self.T, self.emb)
        out = self.attn(x)
        self.assertFalse(torch.isnan(out).any())

    def test_no_nan_with_key_padding_mask(self):
        x = torch.randn(self.B, self.T, self.emb)
        # 部分行有缺失
        mask = torch.ones(self.B, self.T, dtype=torch.bool)
        mask[0, 2] = False  # 行 0 屏蔽位置 2
        mask[2, :] = False  # 行 2 全部屏蔽（测试全缺失处理）
        out = self.attn(x, key_padding_mask=mask)
        self.assertEqual(out.shape, (self.B, self.T, self.emb))
        self.assertFalse(torch.isnan(out).any(),
                         "全缺失行也不应产生 NaN")

    def test_ips_weights_change_output(self):
        """加 IPS 后输出应不同于不加 IPS。"""
        torch.manual_seed(0)
        x = torch.randn(self.B, self.T, self.emb)
        out_no_ips = self.attn(x).detach()

        ips = torch.ones(self.B, self.T)
        ips[:, 1] = 3.0  # 第 1 个 key 位置 IPS = 3
        out_with_ips = self.attn(x, ips_weights=ips).detach()

        self.assertFalse(torch.allclose(out_no_ips, out_with_ips),
                         "IPS 加权后输出应与无 IPS 时不同")

    def test_ips_ones_equals_no_ips(self):
        """IPS 全为 1 时，结果应与不传 IPS 完全相同。"""
        torch.manual_seed(0)
        x = torch.randn(self.B, self.T, self.emb)
        out_no_ips = self.attn(x).detach()

        ips_ones = torch.ones(self.B, self.T)
        out_ips_ones = self.attn(x, ips_weights=ips_ones).detach()

        torch.testing.assert_close(out_no_ips, out_ips_ones)


class TestTransformerBlocks(unittest.TestCase):
    """测试 MissingAwareTransformerBlock 和 RegularTransformerBlock。"""

    def setUp(self):
        self.B, self.T, self.emb = 4, 6, 16
        self.heads = 2

    def test_missing_aware_block_shape(self):
        block = dm.MissingAwareTransformerBlock(self.emb, self.heads, dropout=0.0)
        x = torch.randn(self.B, self.T, self.emb)
        mask = torch.ones(self.B, self.T)
        mask[1, 2] = 0.0
        out = block(x, mask)
        self.assertEqual(out.shape, x.shape)

    def test_missing_aware_block_no_nan_all_missing_row(self):
        """全缺失行不应产生 NaN。"""
        block = dm.MissingAwareTransformerBlock(self.emb, self.heads, dropout=0.0)
        x = torch.randn(self.B, self.T, self.emb)
        mask = torch.ones(self.B, self.T)
        mask[0] = 0.0  # 第 0 行全缺失
        out = block(x, mask)
        self.assertFalse(torch.isnan(out).any())

    def test_missing_aware_block_with_ips(self):
        """带 IPS 参数时 block 应正常运行。"""
        block = dm.MissingAwareTransformerBlock(self.emb, self.heads, dropout=0.0)
        x = torch.randn(self.B, self.T, self.emb)
        mask = torch.ones(self.B, self.T)
        ips = torch.rand(self.B, self.T) + 1.0  # IPS ∈ [1, 2]
        out = block(x, mask, ips_weights=ips)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_regular_block_shape_and_no_nan(self):
        block = dm.RegularTransformerBlock(self.emb, self.heads, dropout=0.0)
        x = torch.randn(self.B, self.T, self.emb)
        out = block(x)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.isnan(out).any())


class TestDARNImputerNet(unittest.TestCase):
    """测试 DARNImputerNet forward pass。"""

    def setUp(self):
        self.B, self.d, self.emb = 4, 3, 16

    def _make_net(self, use_prob_head=False):
        return dm.DARNImputerNet(
            num_features=self.d,
            embedding_dim=self.emb,
            depth=2,
            heads=2,
            dropout=0.0,
            use_prob_head=use_prob_head,
        )

    def test_output_shape_no_ips_no_prob(self):
        net = self._make_net()
        x = torch.rand(self.B, self.d)
        mask = torch.ones(self.B, self.d)
        mask[1, 1] = 0.0
        out = net(x, mask)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (self.B, self.d))

    def test_output_in_sigmoid_range(self):
        """reconstruction head 用 Sigmoid，输出应在 [0, 1]。"""
        net = self._make_net()
        x = torch.rand(self.B, self.d)
        mask = torch.ones(self.B, self.d)
        out = net(x, mask)
        self.assertTrue((out >= 0).all() and (out <= 1).all())

    def test_no_nan_with_partial_missing(self):
        net = self._make_net()
        x = torch.rand(self.B, self.d)
        mask = torch.tensor([
            [1., 1., 1.],
            [1., 0., 1.],
            [0., 1., 0.],
            [0., 0., 0.],
        ])
        out = net(x, mask)
        self.assertFalse(torch.isnan(out).any())

    def test_output_with_ips_weights(self):
        """传入 ips_weights 应正常运行。"""
        net = self._make_net()
        x = torch.rand(self.B, self.d)
        mask = torch.ones(self.B, self.d)
        ips = torch.ones(self.B, self.d)
        ips[:, 1] = 2.5  # 第 1 列 IPS = 2.5
        out = net(x, mask, ips_weights=ips)
        self.assertEqual(out.shape, (self.B, self.d))
        self.assertFalse(torch.isnan(out).any())

    def test_prob_head_returns_tuple(self):
        """use_prob_head=True 时，forward 应返回 (x_hat, missing_prob)。"""
        net = self._make_net(use_prob_head=True)
        x = torch.rand(self.B, self.d)
        mask = torch.ones(self.B, self.d)
        mask[0, 1] = 0.0
        out = net(x, mask)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        x_hat, missing_prob = out
        self.assertEqual(x_hat.shape, (self.B, self.d))
        self.assertEqual(missing_prob.shape, (self.B, self.d))

    def test_missing_prob_in_sigmoid_range(self):
        """missing_prob 也经过 Sigmoid，应在 [0, 1]。"""
        net = self._make_net(use_prob_head=True)
        x = torch.rand(self.B, self.d)
        mask = torch.ones(self.B, self.d)
        _, missing_prob = net(x, mask)
        self.assertTrue((missing_prob >= 0).all() and (missing_prob <= 1).all())

    def test_depth_one_uses_only_first_block(self):
        """depth=1 时只有 first_block，rest_blocks 应为空。"""
        net = dm.DARNImputerNet(self.d, self.emb, depth=1, heads=2, dropout=0.0)
        self.assertEqual(len(net.rest_blocks), 0)
        x = torch.rand(self.B, self.d)
        mask = torch.ones(self.B, self.d)
        out = net(x, mask)
        self.assertEqual(out.shape, (self.B, self.d))


# ============================================================
# DARN 类（外层接口）单元测试
# ============================================================

class TestDARNMain(unittest.TestCase):
    def setUp(self):
        self.raw_data = np.array([
            [1.0, 10.0, 100.0],
            [2.0, np.nan, 200.0],
            [3.0, 30.0, np.nan],
            [np.nan, 40.0, 400.0],
        ])
        self.mask = (1 - np.isnan(self.raw_data)).astype(np.float32)
        self.imputer = DARN(
            batch_size=2,
            epoch=1,
            embedding_dim=8,
            depth=1,
            heads=2,
            device="cpu",
        )
        self.imputer._create_temp_dir = MagicMock()
        self.imputer._save_checkpoint = MagicMock()

    @patch("dataprep.tabular.imputation.DARN_modules.train_darn_algorithm")
    def test_train_sets_model_and_norm_parameters(self, mock_train):
        self.imputer.train(self.raw_data, self.mask)
        self.assertIsNotNone(self.imputer.model)
        self.assertIsNotNone(self.imputer.norm_parameters)
        self.assertEqual(self.imputer.dim, self.raw_data.shape[1])
        mock_train.assert_called_once()
        self.imputer._save_checkpoint.assert_called_once()

    @patch("dataprep.tabular.imputation.DARN_modules.train_darn_algorithm")
    def test_train_auto_generates_mask_from_nan(self, mock_train):
        """不传 missing_mask 时应自动从 NaN 生成。"""
        self.imputer.train(self.raw_data)
        self.assertIsNotNone(self.imputer.model)

    def test_predict_without_train_raises(self):
        with self.assertRaises(RuntimeError):
            self.imputer.predict(self.raw_data)

    def test_predict_shape_and_no_nan(self):
        dim = self.raw_data.shape[1]
        self.imputer.model = dm.DARNImputerNet(
            num_features=dim,
            embedding_dim=8,
            depth=1,
            heads=2,
            dropout=0.0,
        )
        self.imputer.norm_parameters = {
            "min": np.array([1.0, 10.0, 100.0]),
            "max": np.array([3.0, 40.0, 400.0]),
            "den": np.array([2.0, 30.0, 300.0]),
        }
        imputed = self.imputer.predict(self.raw_data)
        self.assertEqual(imputed.shape, self.raw_data.shape)
        self.assertFalse(np.isnan(imputed).any())

    def test_predict_preserves_observed_values(self):
        """predict 应保留 observed 位置的原始值，只替换 missing 位置。"""
        dim = self.raw_data.shape[1]
        self.imputer.model = dm.DARNImputerNet(
            num_features=dim,
            embedding_dim=8,
            depth=1,
            heads=2,
            dropout=0.0,
        )
        self.imputer.norm_parameters = {
            "min": np.array([1.0, 10.0, 100.0]),
            "max": np.array([3.0, 40.0, 400.0]),
            "den": np.array([2.0, 30.0, 300.0]),
        }
        imputed = self.imputer.predict(self.raw_data)
        observed = self.mask.astype(bool)
        np.testing.assert_array_almost_equal(
            imputed[observed],
            self.raw_data[observed],
            decimal=5,
            err_msg="observed 位置的值应与原始数据完全一致",
        )


class TestDARNWithIPS(unittest.TestCase):
    """测试 use_ips=True 时的 train 流程。"""

    def setUp(self):
        np.random.seed(0)
        self.raw_data = np.array([
            [1.0, 10.0, 100.0],
            [2.0, np.nan, 200.0],
            [3.0, 30.0, np.nan],
            [4.0, 40.0, 400.0],
        ])
        self.mask = (1 - np.isnan(self.raw_data)).astype(np.float32)

    @patch("dataprep.tabular.imputation.DARN_modules.train_darn_algorithm")
    def test_ips_simple_computed_before_training(self, mock_train):
        imputer = DARN(
            batch_size=2, epoch=1, embedding_dim=8, depth=1, heads=2,
            device="cpu", use_ips=True, ips_method="simple",
        )
        imputer._create_temp_dir = MagicMock()
        imputer._save_checkpoint = MagicMock()
        imputer.train(self.raw_data, self.mask)
        # 验证 train_darn_algorithm 被调用时传入了 ips 参数（非 None）
        call_kwargs = mock_train.call_args
        ips_arg = call_kwargs.kwargs.get("ips")
        self.assertIsNotNone(ips_arg, "use_ips=True 时应传入 IPS 权重")
        self.assertEqual(ips_arg.shape, self.mask.shape)

    @patch("dataprep.tabular.imputation.DARN_modules.train_darn_algorithm")
    def test_ips_none_when_use_ips_false(self, mock_train):
        """use_ips=False 时不应计算 IPS。"""
        imputer = DARN(
            batch_size=2, epoch=1, embedding_dim=8, depth=1, heads=2,
            device="cpu", use_ips=False,
        )
        imputer._create_temp_dir = MagicMock()
        imputer._save_checkpoint = MagicMock()
        imputer.train(self.raw_data, self.mask)
        call_kwargs = mock_train.call_args
        ips_arg = call_kwargs.kwargs.get("ips")
        self.assertIsNone(ips_arg, "use_ips=False 时 ips 应为 None")


class TestDARNWithProbHead(unittest.TestCase):
    """测试 use_prob_head=True 时的 train + predict 流程。"""

    def setUp(self):
        self.raw_data = np.array([
            [1.0, 10.0, 100.0],
            [2.0, np.nan, 200.0],
            [3.0, 30.0, np.nan],
            [4.0, 40.0, 400.0],
        ])
        self.mask = (1 - np.isnan(self.raw_data)).astype(np.float32)

    @patch("dataprep.tabular.imputation.DARN_modules.train_darn_algorithm")
    def test_model_has_prob_head_when_enabled(self, mock_train):
        imputer = DARN(
            batch_size=2, epoch=1, embedding_dim=8, depth=1, heads=2,
            device="cpu", use_prob_head=True,
        )
        imputer._create_temp_dir = MagicMock()
        imputer._save_checkpoint = MagicMock()
        imputer.train(self.raw_data, self.mask)
        self.assertTrue(imputer.model.use_prob_head)
        self.assertTrue(hasattr(imputer.model, "mask_prediction_head"))

    @patch("dataprep.tabular.imputation.DARN_modules.train_darn_algorithm")
    def test_predict_still_returns_correct_shape_with_prob_head(self, mock_train):
        """use_prob_head=True 时，predict 只返回 X_hat（不返回 missing_prob）。"""
        imputer = DARN(
            batch_size=2, epoch=1, embedding_dim=8, depth=1, heads=2,
            device="cpu", use_prob_head=True,
        )
        imputer._create_temp_dir = MagicMock()
        imputer._save_checkpoint = MagicMock()
        imputer.train(self.raw_data, self.mask)
        # predict 内部从 out[0] 取 x_hat
        imputed = imputer.predict(self.raw_data)
        self.assertIsInstance(imputed, np.ndarray)
        self.assertEqual(imputed.shape, self.raw_data.shape)
        self.assertFalse(np.isnan(imputed).any())


if __name__ == "__main__":
    unittest.main()
