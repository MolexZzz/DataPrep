import numpy as np
import torch
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from path_setup import ensure_dataprep_importable

ensure_dataprep_importable(__file__)
from dataprep.tabular.imputation.DARN import DARN


def generate_fake_data(N=300, D=8, missing_rate=0.2, seed=0):
    np.random.seed(seed)
    data_true = np.random.randn(N, D)
    mask = (np.random.rand(N, D) > missing_rate).astype(float)
    data_missing = data_true.copy()
    data_missing[mask == 0] = np.nan
    return data_true, data_missing, mask


def run_and_report(label, imputer, data_true, data_missing, mask):
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    imputed = imputer.train_and_predict(data_missing, mask)
    metrics = imputer.estimate(data_true, imputed, mask)
    print(f"  MSE={metrics['mse']:.4f}  RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}")
    return metrics


def test_darn():
    print("=" * 60)
    print("         Testing DARN-imputer (4 Levels)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data_true, data_missing, mask = generate_fake_data(N=300, D=8, missing_rate=0.2)
    print(f"Data Shape: {data_true.shape}, Missing Rate: 0.2\n")

    common = dict(
        batch_size=32,
        epoch=5,
        learning_rate=0.001,
        embedding_dim=16,
        depth=2,
        heads=4,
        mask_rate=0.2,
        device=device,
    )

    # ── 层次 1：基础版，MAE loss ─────────────────────────────────────────────
    imputer_l1 = DARN(**common, loss_type="mae")
    m1 = run_and_report("Layer 1: Basic (MAE loss)", imputer_l1, data_true, data_missing, mask)

    # ── 层次 2：渐进式自补全 ─────────────────────────────────────────────────
    imputer_l2 = DARN(**common, loss_type="mae",
                      use_progressive=True, progressive_interval=2, gamma=0.1)
    m2 = run_and_report("Layer 2: + Progressive self-imputation",
                        imputer_l2, data_true, data_missing, mask)

    # ── 层次 3a：渐进式 + 简单 IPS ───────────────────────────────────────────
    imputer_l3a = DARN(**common, loss_type="mae",
                       use_progressive=True, progressive_interval=2, gamma=0.1,
                       use_ips=True, ips_method="simple")
    m3a = run_and_report("Layer 3a: + IPS (simple global)",
                         imputer_l3a, data_true, data_missing, mask)

    # ── 层次 3b：渐进式 + 逻辑回归 IPS ──────────────────────────────────────
    imputer_l3b = DARN(**common, loss_type="mae",
                       use_progressive=True, progressive_interval=2, gamma=0.1,
                       use_ips=True, ips_method="logistic")
    m3b = run_and_report("Layer 3b: + IPS (logistic per-cell)",
                         imputer_l3b, data_true, data_missing, mask)

    # ── 层次 4：全部四个层次 ────────────────────────────────────────────────
    imputer_l4 = DARN(**common, loss_type="mae",
                      use_progressive=True, progressive_interval=2, gamma=0.1,
                      use_ips=True, ips_method="simple",
                      use_prob_head=True, beta=0.1)
    m4 = run_and_report("Layer 4: + L_prob (missing distribution predictor)",
                        imputer_l4, data_true, data_missing, mask)

    # ── 汇总 ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Summary (MAE on missing positions)")
    print("=" * 60)
    for label, m in [
        ("Layer 1 (Basic MAE)",          m1),
        ("Layer 2 (+Progressive)",       m2),
        ("Layer 3a (+IPS simple)",       m3a),
        ("Layer 3b (+IPS logistic)",     m3b),
        ("Layer 4 (+L_prob)",            m4),
    ]:
        print(f"  {label:<35}  MAE={m['mae']:.4f}")

    # ── checkpoint 保存 / 加载验证 ───────────────────────────────────────────
    print("\n[Step 2] Testing checkpoint save/load...")
    checkpoint_path = os.path.join(imputer_l1.temp_dir, "darn_imputer_complete.pkl")
    if os.path.exists(checkpoint_path):
        loaded = DARN.load_model(checkpoint_path)
        imputed_loaded = loaded.predict(data_missing)
        assert imputed_loaded.shape == data_missing.shape, "加载后 predict shape 不一致"
        assert not np.isnan(imputed_loaded).any(), "加载后 predict 含 NaN"
        print("  Checkpoint save/load: PASSED")
    else:
        print("  Checkpoint not found (skipped in short run)")

    print("\n[Result] DARN-imputer Test Passed!")


if __name__ == "__main__":
    test_darn()
