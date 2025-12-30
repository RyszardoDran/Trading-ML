import unittest
import tempfile
import shutil
from pathlib import Path
import importlib.util
import pandas as pd


def load_predict_module():
    # Resolve path to predict_sequence.py relative to this test file
    test_dir = Path(__file__).resolve().parent
    ml_dir = test_dir.parent
    script_path = ml_dir / "src" / "scripts" / "predict_sequence.py"
    spec = importlib.util.spec_from_file_location("predict_sequence", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPredictSequenceLoaders(unittest.TestCase):
    def setUp(self):
        self.module = load_predict_module()

    def test_validate_input_candles_length_mismatch_raises(self):
        # Create minimal valid OHLCV DataFrame with 10 rows
        df = pd.DataFrame({
            "Open": [1.0]*10,
            "High": [2.0]*10,
            "Low": [0.5]*10,
            "Close": [1.5]*10,
            "Volume": [100]*10,
        })
        # Expect ValueError when required_size != len(df)
        with self.assertRaises(ValueError):
            self.module.validate_input_candles(df, required_size=11)

    def test_load_candles_from_csv_requires_either_param(self):
        # Create a temporary CSV file with minimal valid content
        tmp_dir = tempfile.mkdtemp()
        try:
            csv_path = Path(tmp_dir) / "data.csv"
            df = pd.DataFrame({
                "Date": pd.date_range("2025-01-01", periods=10, freq="T"),
                "Open": [1.0]*10,
                "High": [2.0]*10,
                "Low": [0.5]*10,
                "Close": [1.5]*10,
                "Volume": [100]*10,
            })
            df.to_csv(csv_path, sep=";", index=False)
            # Expect ValueError when both n_days and n_candles are None
            with self.assertRaises(ValueError):
                self.module.load_candles_from_csv(csv_path, n_days=None, n_candles=None)
        finally:
            shutil.rmtree(tmp_dir)

    def test_load_latest_candles_from_dir_requires_either_param(self):
        # Create a temporary directory with one CSV file
        tmp_dir = tempfile.mkdtemp()
        try:
            data_dir = Path(tmp_dir)
            csv_path = data_dir / "XAU_1m_data_test.csv"
            df = pd.DataFrame({
                "Date": pd.date_range("2025-01-01", periods=10, freq="T"),
                "Open": [1.0]*10,
                "High": [2.0]*10,
                "Low": [0.5]*10,
                "Close": [1.5]*10,
                "Volume": [100]*10,
            })
            df.to_csv(csv_path, sep=";", index=False)
            # Expect ValueError when both n_days and n_candles are None
            with self.assertRaises(ValueError):
                self.module.load_latest_candles_from_dir(data_dir, n_days=None, n_candles=None)
        finally:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()
