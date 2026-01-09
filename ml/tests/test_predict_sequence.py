import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Load module directly so its sys.path modifications run
spec = importlib.util.spec_from_file_location(
    "predict_sequence",
    Path(__file__).parent.parent / "src" / "scripts" / "predict_sequence.py",
)
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)


def make_m1_candles(n):
    idx = pd.date_range(end=pd.Timestamp.now().floor('T'), periods=n, freq='T')
    df = pd.DataFrame({
        'Open': np.linspace(100, 110, n),
        'High': np.linspace(100.5, 110.5, n),
        'Low': np.linspace(99.5, 109.5, n),
        'Close': np.linspace(100, 110, n),
        'Volume': np.ones(n) * 1000,
    }, index=idx)
    return df


class DummyScaler:
    def transform(self, X):
        return X


class DummyModel:
    def __init__(self, prob=0.75):
        self._prob = prob

    def predict_proba(self, X):
        return np.array([[1 - self._prob, self._prob]])


@pytest.fixture
def artifacts_template():
    return {
        'model': DummyModel(prob=0.75),
        'feature_columns': [f'f{i}' for i in range(4)],
        'threshold': 0.5,
        'win_rate': 0.68,
        'window_size': 100,
        'analysis_window_days': 7,
        'scaler': DummyScaler(),
    }


def test_predict_success_trade(monkeypatch, artifacts_template):
    # Prepare M1 candles (enough rows)
    candles = make_m1_candles(artifacts_template['window_size'] * 5 + 200)

    # Mock aggregate_to_m5 to return an M5 frame with enough rows
    n_m5 = artifacts_template['window_size'] + 50
    m5_idx = pd.date_range(end=pd.Timestamp.now().floor('T'), periods=n_m5, freq='5T')
    m5_df = pd.DataFrame({
        'Open': np.linspace(100, 101, n_m5),
        'High': np.linspace(100.5, 101.5, n_m5),
        'Low': np.linspace(99.5, 100.5, n_m5),
        'Close': np.linspace(100, 101, n_m5),
        'Volume': np.ones(n_m5) * 1000,
    }, index=m5_idx)

    monkeypatch.setattr(ps, 'aggregate_to_m5', lambda df: m5_df)

    # Engineer features: produce DataFrame with window_size+50 rows and required columns
    features_df = pd.DataFrame(
        np.ones((n_m5, len(artifacts_template['feature_columns']))),
        columns=artifacts_template['feature_columns'],
        index=m5_df.index,
    )
    # ensure trend indicators are favorable
    features_df['dist_sma_200'] = 1.0
    features_df['adx'] = 20.0
    features_df['rsi_m5'] = 50.0

    monkeypatch.setattr(ps, 'engineer_m5_candle_features', lambda df: features_df)
    monkeypatch.setattr(ps, '_compute_atr_m5', lambda df: 2.0)
    monkeypatch.setattr(ps, 'should_trade', lambda atr, adx, price, sma200, threshold: (True, 'GOOD', 'ok'))

    res = ps.predict(candles, Path("/no/need"), artifacts=artifacts_template)

    assert pytest.approx(res['probability'], rel=1e-3) == 0.75
    assert res['prediction'] == 1
    assert res['regime_allowed'] is True
    assert res['sl'] is not None and res['tp'] is not None
    # SL = entry - ATR*SL_ATR_MULTIPLIER (atr=2.0), TP = entry + ATR*TP_ATR_MULTIPLIER
    # With SL multiplier=1.0 and TP multiplier=2.0, TP - SL = 3 * ATR = 6.0
    assert res['tp'] - res['sl'] == pytest.approx(6.0)


def test_trend_filter_blocks(monkeypatch, artifacts_template):
    candles = make_m1_candles(artifacts_template['window_size'] * 5 + 200)
    n_m5 = artifacts_template['window_size'] + 10
    m5_df = pd.DataFrame({
        'Open': np.linspace(100, 101, n_m5),
        'High': np.linspace(100.5, 101.5, n_m5),
        'Low': np.linspace(99.5, 100.5, n_m5),
        'Close': np.linspace(100, 101, n_m5),
        'Volume': np.ones(n_m5) * 1000,
    }, index=pd.date_range(end=pd.Timestamp.now().floor('T'), periods=n_m5, freq='5T'))

    monkeypatch.setattr(ps, 'aggregate_to_m5', lambda df: m5_df)

    features_df = pd.DataFrame(
        np.ones((n_m5, len(artifacts_template['feature_columns']))),
        columns=artifacts_template['feature_columns'],
        index=m5_df.index,
    )
    # make dist_sma_200 <= TREND_MIN_DIST_SMA200 to trigger trend filter
    # (default is relaxed to -0.5 in risk_config)
    features_df['dist_sma_200'] = -0.6
    features_df['adx'] = 30.0
    features_df['rsi_m5'] = 50.0

    monkeypatch.setattr(ps, 'engineer_m5_candle_features', lambda df: features_df)
    monkeypatch.setattr(ps, '_compute_atr_m5', lambda df: 2.0)

    res = ps.predict(candles, Path("/no/need"), artifacts=artifacts_template)

    assert res['probability'] == 0.0
    assert res['prediction'] == 0
    assert 'trend filter' in res['confidence']


def test_regime_blocks(monkeypatch, artifacts_template):
    candles = make_m1_candles(artifacts_template['window_size'] * 5 + 200)
    n_m5 = artifacts_template['window_size'] + 20
    m5_df = pd.DataFrame({
        'Open': np.linspace(100, 101, n_m5),
        'High': np.linspace(100.5, 101.5, n_m5),
        'Low': np.linspace(99.5, 100.5, n_m5),
        'Close': np.linspace(100, 101, n_m5),
        'Volume': np.ones(n_m5) * 1000,
    }, index=pd.date_range(end=pd.Timestamp.now().floor('T'), periods=n_m5, freq='5T'))

    monkeypatch.setattr(ps, 'aggregate_to_m5', lambda df: m5_df)

    features_df = pd.DataFrame(
        np.ones((n_m5, len(artifacts_template['feature_columns']))),
        columns=artifacts_template['feature_columns'],
        index=m5_df.index,
    )
    features_df['dist_sma_200'] = 1.0
    features_df['adx'] = 20.0
    features_df['rsi_m5'] = 50.0

    monkeypatch.setattr(ps, 'engineer_m5_candle_features', lambda df: features_df)
    monkeypatch.setattr(ps, '_compute_atr_m5', lambda df: 2.0)
    # Should block trade regardless of proba
    monkeypatch.setattr(ps, 'should_trade', lambda atr, adx, price, sma200, threshold: (False, 'BAD', 'regime'))

    res = ps.predict(candles, Path("/no/need"), artifacts=artifacts_template)

    assert res['prediction'] == 0
    assert res['regime_allowed'] is False
    assert res['regime_reason'] == 'regime'


def test_missing_scaler_raises(monkeypatch, artifacts_template):
    artifacts = artifacts_template.copy()
    artifacts['scaler'] = None
    candles = make_m1_candles(artifacts['window_size'] * 5 + 200)

    with pytest.raises(ValueError):
        ps.predict(candles, Path('/no/need'), artifacts=artifacts)


def test_feature_reorder(monkeypatch, artifacts_template):
    # Test that features are reordered when columns are different order than feature_columns
    artifacts = artifacts_template.copy()
    artifacts['feature_columns'] = ['a', 'b', 'c']
    window_size = artifacts['window_size']

    candles = make_m1_candles(window_size * 5 + 200)
    n_m5 = window_size + 10
    m5_df = pd.DataFrame({
        'Open': np.linspace(100, 101, n_m5),
        'High': np.linspace(100.5, 101.5, n_m5),
        'Low': np.linspace(99.5, 100.5, n_m5),
        'Close': np.linspace(100, 101, n_m5),
        'Volume': np.ones(n_m5) * 1000,
    }, index=pd.date_range(end=pd.Timestamp.now().floor('T'), periods=n_m5, freq='5T'))

    monkeypatch.setattr(ps, 'aggregate_to_m5', lambda df: m5_df)

    # Build features with columns in different order
    features_df = pd.DataFrame(
        np.ones((n_m5, len(artifacts['feature_columns']))),
        columns=list(reversed(artifacts['feature_columns'])),
        index=m5_df.index,
    )
    features_df['dist_sma_200'] = 1.0
    features_df['adx'] = 20.0
    features_df['rsi_m5'] = 50.0

    monkeypatch.setattr(ps, 'engineer_m5_candle_features', lambda df: features_df)
    monkeypatch.setattr(ps, '_compute_atr_m5', lambda df: 2.0)
    monkeypatch.setattr(ps, 'should_trade', lambda atr, adx, price, sma200, threshold: (True, 'GOOD', 'ok'))

    res = ps.predict(candles, Path('/no/need'), artifacts=artifacts)
    assert 'probability' in res
