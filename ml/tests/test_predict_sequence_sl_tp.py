from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "scripts"))

import predict_sequence as predict_sequence_module


class _DummyModel:
    def __init__(self, proba: float) -> None:
        self._proba = proba
        self.last_input: np.ndarray | None = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.last_input = X.copy()
        return np.array([[1.0 - self._proba, self._proba]], dtype=np.float32)


def test_predict_applies_saved_scaler_and_returns_sl_tp(monkeypatch: pytest.MonkeyPatch) -> None:
    window_size = 2
    feature_columns = ["dist_sma_200", "adx"]
    feature_index = pd.date_range("2024-02-01 09:00", periods=window_size, freq="5min")

    feature_df = pd.DataFrame(
        {"dist_sma_200": [0.10, 0.20], "adx": [20.0, 25.0]},
        index=feature_index,
    )

    scaler = RobustScaler()
    scaler.fit(np.array([[0.0, 0.0, 0.0, 0.0], [5.0, 5.0, 5.0, 5.0]], dtype=np.float32))

    model = _DummyModel(proba=0.6)

    candles_m5 = pd.DataFrame(
        {
            "Open": [101.0, 102.0],
            "High": [103.0, 104.0],
            "Low": [100.0, 101.0],
            "Close": [102.0, 103.0],
            "Volume": [1_000.0, 1_100.0],
        },
        index=feature_index,
    )

    def fake_aggregate_to_m5(_: pd.DataFrame) -> pd.DataFrame:
        return candles_m5

    def fake_engineer_m5_candle_features(_: pd.DataFrame) -> pd.DataFrame:
        return feature_df

    def fake_load_model_artifacts(_: Path) -> dict[str, object]:
        return {
            "model": model,
            "feature_columns": feature_columns,
            "threshold": 0.5,
            "win_rate": 0.7,
            "window_size": window_size,
            "analysis_window_days": 7,
            "scaler": scaler,
        }

    candles_m1 = candles_m5.copy()

    monkeypatch.setattr(predict_sequence_module, "aggregate_to_m5", fake_aggregate_to_m5)
    monkeypatch.setattr(
        predict_sequence_module,
        "engineer_m5_candle_features",
        fake_engineer_m5_candle_features,
    )
    monkeypatch.setattr(predict_sequence_module, "load_model_artifacts", fake_load_model_artifacts)

    result = predict_sequence_module.predict(candles_m1, Path("unused"))

    raw = feature_df.values.flatten().reshape(1, -1)
    expected = scaler.transform(raw).astype(np.float32)

    assert model.last_input is not None
    assert np.allclose(model.last_input, expected)

    expected_entry = float(candles_m5["Close"].iloc[-1])
    expected_atr = float(
        predict_sequence_module.compute_atr(
            candles_m5["High"],
            candles_m5["Low"],
            candles_m5["Close"],
            period=14,
        ).iloc[-1]
    )

    assert result["probability"] == pytest.approx(0.6)
    assert result["prediction"] == 1
    assert result["entry_price"] == pytest.approx(expected_entry)
    assert result["atr_m5"] == pytest.approx(expected_atr)
    assert result["sl"] == pytest.approx(expected_entry - expected_atr)
    assert result["tp"] == pytest.approx(expected_entry + 2.0 * expected_atr)


def test_predict_trend_filter_includes_sl_tp_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    window_size = 2
    feature_columns = ["dist_sma_200", "adx"]
    feature_index = pd.date_range("2024-02-01 09:00", periods=window_size, freq="5min")

    feature_df = pd.DataFrame(
        {"dist_sma_200": [0.10, -0.01], "adx": [20.0, 25.0]},
        index=feature_index,
    )

    scaler = RobustScaler()
    scaler.fit(np.array([[0.0, 0.0, 0.0, 0.0], [5.0, 5.0, 5.0, 5.0]], dtype=np.float32))

    model = _DummyModel(proba=0.9)

    candles_m5 = pd.DataFrame(
        {
            "Open": [101.0, 102.0],
            "High": [103.0, 104.0],
            "Low": [100.0, 101.0],
            "Close": [102.0, 103.0],
            "Volume": [1_000.0, 1_100.0],
        },
        index=feature_index,
    )

    def fake_aggregate_to_m5(_: pd.DataFrame) -> pd.DataFrame:
        return candles_m5

    def fake_engineer_m5_candle_features(_: pd.DataFrame) -> pd.DataFrame:
        return feature_df

    def fake_load_model_artifacts(_: Path) -> dict[str, object]:
        return {
            "model": model,
            "feature_columns": feature_columns,
            "threshold": 0.5,
            "win_rate": 0.7,
            "window_size": window_size,
            "analysis_window_days": 7,
            "scaler": scaler,
        }

    candles_m1 = candles_m5.copy()

    monkeypatch.setattr(predict_sequence_module, "aggregate_to_m5", fake_aggregate_to_m5)
    monkeypatch.setattr(
        predict_sequence_module,
        "engineer_m5_candle_features",
        fake_engineer_m5_candle_features,
    )
    monkeypatch.setattr(predict_sequence_module, "load_model_artifacts", fake_load_model_artifacts)

    result = predict_sequence_module.predict(candles_m1, Path("unused"))

    assert result["prediction"] == 0
    assert result["probability"] == pytest.approx(0.0)
    assert "trend filter" in result["confidence"]

    assert "entry_price" in result
    assert "atr_m5" in result
    assert "sl" in result
    assert "tp" in result

    assert result["sl"] is None
    assert result["tp"] is None
