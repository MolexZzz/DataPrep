import numpy as np
import torch
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from path_setup import ensure_dataprep_importable

ensure_dataprep_importable(__file__)
from dataprep.tabular.imputation.FATE import FATE


def generate_fake_data(N=300, D=8, missing_rate=0.2):
    data_true = np.random.randn(N, D)
    mask = (np.random.rand(N, D) > missing_rate).astype(float)
    data_missing = data_true.copy()
    data_missing[mask == 0] = np.nan
    return data_true, data_missing, mask


def test_fate():
    print("========================================")
    print("      Testing FATE-imputer Algorithm    ")
    print("========================================")

    data_true, data_missing, mask = generate_fake_data(N=300, D=8, missing_rate=0.2)
    print(f"Data Shape: {data_true.shape}, Missing Rate: 0.2")

    imputer = FATE(
        batch_size=32,
        epoch=3,
        learning_rate=0.001,
        embedding_dim=16,
        depth=2,
        heads=4,
        mask_rate=0.2,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("\n[Step 1] Training & Predicting...")
    imputed_data = imputer.train_and_predict(data_missing, mask)
    imputer.estimate(data_true, imputed_data, mask)

    checkpoint_path = os.path.join(imputer.temp_dir, "fate_imputer_complete.pkl")
    imputer = FATE.load_model(checkpoint_path)
    imputed_data = imputer.predict(data_missing)

    print("\n[Step 2] Evaluating loaded model...")
    imputer.estimate(data_true, imputed_data, mask)

    print("\n[Result] FATE-imputer Test Passed!")


if __name__ == "__main__":
    test_fate()
