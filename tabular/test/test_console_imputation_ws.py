import asyncio
import importlib.util
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if "dataprep" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "dataprep",
        os.path.join(project_root, "__init__.py"),
        submodule_search_locations=[project_root],
    )
    dataprep_module = importlib.util.module_from_spec(spec)
    sys.modules["dataprep"] = dataprep_module
    spec.loader.exec_module(dataprep_module)

from dataprep import main


class TestConsoleImputationWebSocket(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        data_true = np.array(
            [
                [1.0, 10.0, 100.0],
                [2.0, 20.0, 200.0],
                [3.0, 30.0, 300.0],
                [4.0, 40.0, 400.0],
                [5.0, 50.0, 500.0],
                [6.0, 60.0, 600.0],
            ],
            dtype=float,
        )
        mask = np.ones_like(data_true)
        mask[1, 1] = 0.0
        mask[2, 2] = 0.0
        mask[4, 0] = 0.0
        data_missing = data_true.copy()
        data_missing[mask == 0] = np.nan

        self.data_path = os.path.join(self.tmpdir.name, "data.csv")
        self.mask_path = os.path.join(self.tmpdir.name, "mask.csv")
        self.truth_path = os.path.join(self.tmpdir.name, "truth.csv")
        columns = ["a", "b", "c"]
        pd.DataFrame(data_missing, columns=columns).to_csv(self.data_path, index=False)
        pd.DataFrame(mask, columns=columns).to_csv(self.mask_path, index=False)
        pd.DataFrame(data_true, columns=columns).to_csv(self.truth_path, index=False)

    def tearDown(self):
        self.tmpdir.cleanup()

    def run_console_task(self, method, params):
        payload = {
            "method": method,
            "paths": {
                "dataPath": self.data_path,
                "missingMaskPath": self.mask_path,
                "groundTruthPath": self.truth_path,
            },
            "params": params,
        }
        with TestClient(main.app) as client:
            with client.websocket_connect("/api/ws/run_task") as ws:
                ws.send_json(payload)
                while True:
                    message = ws.receive_json()
                    if message.get("status") == "success":
                        return message
                    if message.get("status") == "error":
                        self.fail(message.get("detail"))

    def test_fate_websocket_task(self):
        result = self.run_console_task(
            "FATE",
            {
                "batch_size": 3,
                "epoch": 1,
                "learning_rate": 0.001,
                "embedding_dim": 8,
                "depth": 1,
                "heads": 2,
                "mask_rate": 0.5,
                "dropout": 0.0,
            },
        )
        self.assertIn("metrics", result)
        self.assertIn("result_data", result)
        self.assertIn("mse_ours", result["metrics"])

    def test_darn_websocket_task(self):
        result = self.run_console_task(
            "DARN",
            {
                "batch_size": 3,
                "epoch": 1,
                "learning_rate": 0.001,
                "embedding_dim": 8,
                "depth": 1,
                "heads": 2,
                "mask_rate": 0.5,
                "dropout": 0.0,
                "loss_type": "mae",
                "use_progressive": False,
                "progressive_interval": 2,
                "gamma": 0.1,
                "use_ips": False,
                "ips_method": "simple",
                "use_prob_head": False,
                "beta": 0.1,
            },
        )
        self.assertIn("metrics", result)
        self.assertIn("result_data", result)
        self.assertIn("mae_ours", result["metrics"])


if __name__ == "__main__":
    unittest.main()
