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
    from dataprep.tabular.imputation.FATE import FATE
    import dataprep.tabular.imputation.FATE_modules as fm
except ImportError as e:
    raise ImportError(f"导入失败，请检查文件位置。\n详细错误: {e}")


class TestFATEModules(unittest.TestCase):
    def setUp(self):
        self.data = np.array([
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
        ])
        self.batch_size = 4
        self.dim = 3
        self.embedding_dim = 8

    def test_normalization_renormalization(self):
        norm_data, params = fm.normalization(self.data)
        self.assertTrue((norm_data >= 0).all() and (norm_data <= 1).all())
        restored = fm.renormalization(norm_data, params)
        np.testing.assert_array_almost_equal(self.data, restored)

    def test_continuous_feature_embedding_shape(self):
        embedding = fm.ContinuousFeatureEmbedding(self.dim, self.embedding_dim)
        x = torch.randn(self.batch_size, self.dim)
        out = embedding(x)
        self.assertEqual(out.shape, (self.batch_size, self.dim, self.embedding_dim))

    def test_sample_observed_mask_only_targets_observed_entries(self):
        mask = torch.tensor([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ])
        train_mask, target_mask = fm.sample_observed_mask(mask, mask_rate=1.0)
        self.assertTrue(torch.equal(target_mask, mask))
        self.assertTrue(torch.equal(train_mask, torch.zeros_like(mask)))

    def test_transformer_block_shape_and_no_nan_for_all_missing_row(self):
        block = fm.MissingAwareTransformerBlock(self.embedding_dim, heads=2, dropout=0.0)
        x = torch.randn(self.batch_size, self.dim, self.embedding_dim)
        mask = torch.tensor([
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        out = block(x, mask)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_fate_imputer_net_forward(self):
        net = fm.FATEImputerNet(
            num_features=self.dim,
            embedding_dim=self.embedding_dim,
            depth=2,
            heads=2,
            dropout=0.0,
        )
        x = torch.rand(self.batch_size, self.dim)
        mask = torch.tensor([
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ])
        out = net(x, mask)
        self.assertEqual(out.shape, (self.batch_size, self.dim))
        self.assertTrue((out >= 0).all() and (out <= 1).all())
        self.assertFalse(torch.isnan(out).any())


class TestFATEMain(unittest.TestCase):
    def setUp(self):
        self.raw_data = np.array([
            [1.0, 10.0, 100.0],
            [2.0, np.nan, 200.0],
            [3.0, 30.0, np.nan],
            [np.nan, 40.0, 400.0],
        ])
        self.mask = 1 - np.isnan(self.raw_data)
        self.imputer = FATE(
            batch_size=2,
            epoch=1,
            embedding_dim=8,
            depth=1,
            heads=2,
            device="cpu",
        )
        self.imputer._create_temp_dir = MagicMock()
        self.imputer._save_checkpoint = MagicMock()

    @patch("dataprep.tabular.imputation.FATE_modules.train_fate_algorithm")
    def test_train_pipeline(self, mock_train_loop):
        self.imputer.train(self.raw_data, self.mask)
        self.assertIsNotNone(self.imputer.model)
        self.assertIsNotNone(self.imputer.norm_parameters)
        self.assertEqual(self.imputer.dim, self.raw_data.shape[1])
        mock_train_loop.assert_called_once()
        self.imputer._save_checkpoint.assert_called_once()

    def test_predict_without_train(self):
        with self.assertRaises(RuntimeError):
            self.imputer.predict(self.raw_data)

    def test_predict_pipeline(self):
        dim = self.raw_data.shape[1]
        self.imputer.model = fm.FATEImputerNet(
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

        imputed_data = self.imputer.predict(self.raw_data)
        self.assertEqual(imputed_data.shape, self.raw_data.shape)
        self.assertFalse(np.isnan(imputed_data).any())

        observed = self.mask.astype(bool)
        np.testing.assert_array_almost_equal(
            imputed_data[observed],
            self.raw_data[observed],
            decimal=5,
        )


if __name__ == "__main__":
    unittest.main()
