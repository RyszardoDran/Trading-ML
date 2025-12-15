from __future__ import annotations
"""Comprehensive tests for sequence training pipeline.

Test coverage:
- Data loading and validation
- Feature engineering for sequences
- Window creation and alignment
- Model training and calibration
- Win rate calculation and evaluation
- Artifact persistence and loading
- Edge cases and error handling
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import functions from sequence pipeline
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "pipelines"))

from sequence_training_pipeline import (
    _validate_schema,
    engineer_candle_features,
    create_sequences,
    make_target,
    split_sequences,
    train_xgb,
    evaluate,
    save_artifacts,
    filter_by_session,
)


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="1min")
    
    close = 1800 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.abs(np.random.randn(n) * 1000) + 5000
    
    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )
    return df


class TestSchemaValidation:
    """Test OHLCV schema validation."""
    
    def test_valid_schema(self, sample_ohlcv_data):
        """Valid data should pass validation."""
        _validate_schema(sample_ohlcv_data)  # Should not raise
    
    def test_missing_columns(self, sample_ohlcv_data):
        """Missing required columns should raise ValueError."""
        df = sample_ohlcv_data.drop(columns=["Close"])
        with pytest.raises(ValueError, match="Missing required columns"):
            _validate_schema(df)
    
    def test_non_positive_prices(self, sample_ohlcv_data):
        """Non-positive prices should raise ValueError."""
        df = sample_ohlcv_data.copy()
        df.loc[df.index[10], "Close"] = -1.0
        with pytest.raises(ValueError, match="non-positive"):
            _validate_schema(df)
    
    def test_high_less_than_low(self, sample_ohlcv_data):
        """High < Low should raise ValueError."""
        df = sample_ohlcv_data.copy()
        df.loc[df.index[10], "High"] = 1000
        df.loc[df.index[10], "Low"] = 2000
        with pytest.raises(ValueError, match="High < Low"):
            _validate_schema(df)


class TestFeatureEngineering:
    """Test per-candle feature engineering."""
    
    def test_feature_engineering_shape(self, sample_ohlcv_data):
        """Feature engineering should produce expected columns."""
        features = engineer_candle_features(sample_ohlcv_data, window_size=100)
        
        expected_cols = [
            "ret_1", "range_n", "body_ratio", "upper_shadow", "lower_shadow",
            "vol_change", "ema_spread_n", "rsi_14", "vol_20",
            "atr_14", "atr_n",
            "hour_sin", "hour_cos", "minute_sin", "minute_cos",
        ]
        assert list(features.columns) == expected_cols
        assert len(features) > 0
        assert len(features) <= len(sample_ohlcv_data)  # Some rows dropped due to NaN
    
    def test_feature_no_nan_inf(self, sample_ohlcv_data):
        """Features should not contain NaN or inf after cleaning."""
        features = engineer_candle_features(sample_ohlcv_data)
        assert not features.isnull().any().any()
        assert not np.isinf(features.values).any()
    
    def test_feature_deterministic(self, sample_ohlcv_data):
        """Feature engineering should be deterministic."""
        features1 = engineer_candle_features(sample_ohlcv_data)
        features2 = engineer_candle_features(sample_ohlcv_data)
        pd.testing.assert_frame_equal(features1, features2)


class TestTargetCreation:
    """Test binary target creation."""
    
    def test_target_binary(self, sample_ohlcv_data):
        """Target should be binary {0, 1}."""
        target = make_target(sample_ohlcv_data, atr_multiplier_sl=2.0, atr_multiplier_tp=4.0)
        assert set(target.unique()).issubset({0, 1})
    
    def test_target_shape(self, sample_ohlcv_data):
        """Target should have fewer rows than input (due to forward window)."""
        target = make_target(sample_ohlcv_data, atr_multiplier_sl=2.0, atr_multiplier_tp=4.0)
        assert len(target) < len(sample_ohlcv_data)
        assert len(target) > 0
    
    def test_target_invalid_params(self, sample_ohlcv_data):
        """Invalid parameters should raise ValueError."""
        with pytest.raises(ValueError, match="min_hold_minutes must be >=1"):
            make_target(sample_ohlcv_data, min_hold_minutes=0)
        
        with pytest.raises(ValueError, match="ATR multipliers must be positive"):
            make_target(sample_ohlcv_data, atr_multiplier_sl=-1.0)


class TestSequenceCreation:
    """Test sliding window sequence creation."""
    
    def test_sequence_creation_shape(self, sample_ohlcv_data):
        """Sequences should have correct shape."""
        features = engineer_candle_features(sample_ohlcv_data, window_size=50)
        targets = make_target(sample_ohlcv_data.loc[features.index])
        
        window_size = 50
        X, y, timestamps = create_sequences(features, targets, window_size=window_size)
        
        n_features = features.shape[1]
        expected_features = window_size * n_features
        
        assert X.shape[1] == expected_features
        assert len(X) == len(y)
        assert len(X) == len(timestamps)
        assert len(X) > 0
    
    def test_sequence_insufficient_data(self, sample_ohlcv_data):
        """Insufficient data should raise ValueError."""
        features = engineer_candle_features(sample_ohlcv_data, window_size=10)
        targets = make_target(sample_ohlcv_data.loc[features.index])
        
        # Take only first 10 rows (not enough for window_size=100)
        features_small = features.head(10)
        targets_small = targets.head(10)
        
        with pytest.raises(ValueError, match="Need at least"):
            create_sequences(features_small, targets_small, window_size=100)
    
    def test_sequence_alignment(self, sample_ohlcv_data):
        """Sequences should be properly aligned with targets."""
        features = engineer_candle_features(sample_ohlcv_data, window_size=50)
        targets = make_target(sample_ohlcv_data.loc[features.index])
        
        window_size = 50
        X, y, timestamps = create_sequences(features, targets, window_size=window_size)
        
        # First timestamp should be window_size rows after first feature
        assert timestamps[0] == features.index[window_size]


class TestSplitting:
    """Test chronological data splitting."""
    
    def test_split_sequences_sizes(self, sample_ohlcv_data):
        """Split should produce non-empty train/val/test sets."""
        features = engineer_candle_features(sample_ohlcv_data, window_size=50)
        targets = make_target(sample_ohlcv_data.loc[features.index])
        X, y, timestamps = create_sequences(features, targets, window_size=50)
        
        # Use dates within data range
        train_until = "2023-01-01 06:00:00"
        val_until = "2023-01-01 10:00:00"
        test_until = "2023-01-01 16:00:00"
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_sequences(
            X, y, timestamps, train_until, val_until, test_until
        )
        
        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(X_test) > 0
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_test) == len(y_test)
    
    def test_split_chronological(self, sample_ohlcv_data):
        """Split should maintain chronological order."""
        features = engineer_candle_features(sample_ohlcv_data, window_size=50)
        targets = make_target(sample_ohlcv_data.loc[features.index])
        X, y, timestamps = create_sequences(features, targets, window_size=50)
        
        train_until = "2023-01-01 06:00:00"
        val_until = "2023-01-01 10:00:00"
        test_until = "2023-01-01 16:00:00"
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_sequences(
            X, y, timestamps, train_until, val_until, test_until
        )
        
        # Verify no overlap
        train_mask = timestamps <= pd.Timestamp(train_until)
        val_mask = (timestamps > pd.Timestamp(train_until)) & (timestamps <= pd.Timestamp(val_until))
        test_mask = (timestamps > pd.Timestamp(val_until)) & (timestamps <= pd.Timestamp(test_until))
        
        assert len(X_train) == train_mask.sum()
        assert len(X_val) == val_mask.sum()
        assert len(X_test) == test_mask.sum()


class TestModelTraining:
    """Test XGBoost training and calibration."""
    
    def test_train_xgb(self, sample_ohlcv_data):
        """Training should produce a calibrated model."""
        features = engineer_candle_features(sample_ohlcv_data, window_size=50)
        targets = make_target(sample_ohlcv_data.loc[features.index])
        X, y, timestamps = create_sequences(features, targets, window_size=50)
        
        train_until = "2023-01-01 06:00:00"
        val_until = "2023-01-01 10:00:00"
        test_until = "2023-01-01 16:00:00"
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_sequences(
            X, y, timestamps, train_until, val_until, test_until
        )
        
        model = train_xgb(X_train, y_train, X_val, y_val, random_state=42)
        
        assert model is not None
        assert hasattr(model, "predict_proba")
        
        # Test prediction
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        assert np.all((proba >= 0) & (proba <= 1))
    
    def test_train_invalid_labels(self, sample_ohlcv_data):
        """Non-binary labels should raise ValueError."""
        features = engineer_candle_features(sample_ohlcv_data, window_size=50)
        targets = make_target(sample_ohlcv_data.loc[features.index])
        X, y, timestamps = create_sequences(features, targets, window_size=50)
        
        train_until = "2023-01-01 06:00:00"
        val_until = "2023-01-01 10:00:00"
        
        X_train, X_val, _, y_train, y_val, _ = split_sequences(
            X, y, timestamps, train_until, val_until, "2023-01-01 16:00:00"
        )
        
        # Create invalid labels
        y_train_invalid = y_train.copy()
        y_train_invalid[0] = 2  # Invalid label
        
        with pytest.raises(ValueError, match="must be binary"):
            train_xgb(X_train, y_train_invalid, X_val, y_val)


class TestEvaluation:
    """Test model evaluation and win rate calculation."""
    
    def test_evaluate_metrics(self, sample_ohlcv_data):
        """Evaluation should return all expected metrics."""
        features = engineer_candle_features(sample_ohlcv_data, window_size=50)
        targets = make_target(sample_ohlcv_data.loc[features.index])
        X, y, timestamps = create_sequences(features, targets, window_size=50)
        
        train_until = "2023-01-01 06:00:00"
        val_until = "2023-01-01 10:00:00"
        test_until = "2023-01-01 16:00:00"
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_sequences(
            X, y, timestamps, train_until, val_until, test_until
        )
        
        model = train_xgb(X_train, y_train, X_val, y_val, random_state=42)
        metrics = evaluate(model, X_test, y_test)
        
        # Check all expected metrics
        expected_keys = {"threshold", "win_rate", "precision", "recall", "f1", "roc_auc", "pr_auc"}
        assert set(metrics.keys()) == expected_keys
        
        # Check value ranges
        assert 0 <= metrics["threshold"] <= 1
        assert 0 <= metrics["win_rate"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1
        assert 0 <= metrics["pr_auc"] <= 1
        
        # Win rate should equal precision
        assert metrics["win_rate"] == metrics["precision"]
    
    def test_evaluate_empty_input(self):
        """Empty test set should raise ValueError."""
        from sklearn.calibration import CalibratedClassifierCV
        from xgboost import XGBClassifier
        
        # Create dummy model
        base = XGBClassifier(n_estimators=10, random_state=42)
        X_dummy = np.random.randn(100, 10)
        y_dummy = np.random.randint(0, 2, 100)
        base.fit(X_dummy, y_dummy)
        model = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
        model.fit(X_dummy[:20], y_dummy[:20])
        
        # Empty test set
        X_test_empty = np.array([]).reshape(0, 10)
        y_test_empty = np.array([])
        
        with pytest.raises(ValueError, match="empty"):
            evaluate(model, X_test_empty, y_test_empty)


class TestArtifactPersistence:
    """Test model and metadata saving/loading."""
    
    def test_save_artifacts(self, sample_ohlcv_data, tmp_path):
        """Artifacts should be saved correctly."""
        from sklearn.calibration import CalibratedClassifierCV
        from xgboost import XGBClassifier
        
        # Create dummy model
        base = XGBClassifier(n_estimators=10, random_state=42)
        X_dummy = np.random.randn(100, 10)
        y_dummy = np.random.randint(0, 2, 100)
        base.fit(X_dummy, y_dummy)
        model = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
        model.fit(X_dummy[:20], y_dummy[:20])
        
        feature_cols = [f"feature_{i}" for i in range(10)]
        threshold = 0.45
        win_rate = 0.68
        window_size = 100
        
        save_artifacts(model, feature_cols, tmp_path, threshold, win_rate, window_size)
        
        # Check files exist
        assert (tmp_path / "sequence_xgb_model.pkl").exists()
        assert (tmp_path / "sequence_feature_columns.json").exists()
        assert (tmp_path / "sequence_threshold.json").exists()
        
        # Load and verify
        with open(tmp_path / "sequence_xgb_model.pkl", "rb") as f:
            loaded_model = pickle.load(f)
        assert loaded_model is not None
        
        with open(tmp_path / "sequence_feature_columns.json", "r") as f:
            loaded_features = json.load(f)
        assert loaded_features == feature_cols
        
        with open(tmp_path / "sequence_threshold.json", "r") as f:
            loaded_metadata = json.load(f)
        assert loaded_metadata["threshold"] == threshold
        assert loaded_metadata["win_rate"] == win_rate
        assert loaded_metadata["window_size"] == window_size


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_pipeline_runs(self, sample_ohlcv_data):
        """Full pipeline should run without errors."""
        # This is a smoke test - verifies all components work together
        features = engineer_candle_features(sample_ohlcv_data, window_size=50)
        targets = make_target(sample_ohlcv_data.loc[features.index])
        X, y, timestamps = create_sequences(features, targets, window_size=50)
        
        train_until = "2023-01-01 06:00:00"
        val_until = "2023-01-01 10:00:00"
        test_until = "2023-01-01 16:00:00"
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_sequences(
            X, y, timestamps, train_until, val_until, test_until
        )
        
        model = train_xgb(X_train, y_train, X_val, y_val, random_state=42)
        metrics = evaluate(model, X_test, y_test)
        
        # Basic sanity checks
        assert metrics["win_rate"] >= 0
        assert metrics["roc_auc"] >= 0.5  # Better than random
        assert model is not None


class TestSessionFiltering:
    """Test session-based data filtering."""

    def test_filter_london(self):
        """London session should keep 08:00-16:00."""
        # Create dummy data covering 24 hours
        dates = pd.date_range("2023-01-01 00:00", "2023-01-01 23:59", freq="1min")
        n = len(dates)
        X = np.zeros((n, 10))
        y = np.zeros(n)
        timestamps = dates

        X_filt, y_filt, t_filt = filter_by_session(X, y, timestamps, "london")

        assert len(X_filt) > 0
        assert (t_filt.hour >= 8).all()
        assert (t_filt.hour < 16).all()

    def test_filter_ny(self):
        """NY session should keep 13:00-22:00."""
        dates = pd.date_range("2023-01-01 00:00", "2023-01-01 23:59", freq="1min")
        n = len(dates)
        X = np.zeros((n, 10))
        y = np.zeros(n)
        timestamps = dates

        X_filt, y_filt, t_filt = filter_by_session(X, y, timestamps, "ny")

        assert len(X_filt) > 0
        assert (t_filt.hour >= 13).all()
        assert (t_filt.hour < 22).all()

    def test_filter_asian(self):
        """Asian session should keep 00:00-09:00."""
        dates = pd.date_range("2023-01-01 00:00", "2023-01-01 23:59", freq="1min")
        n = len(dates)
        X = np.zeros((n, 10))
        y = np.zeros(n)
        timestamps = dates

        X_filt, y_filt, t_filt = filter_by_session(X, y, timestamps, "asian")

        assert len(X_filt) > 0
        assert (t_filt.hour >= 0).all()
        assert (t_filt.hour < 9).all()

    def test_filter_custom(self):
        """Custom session should respect start/end hours."""
        dates = pd.date_range("2023-01-01 00:00", "2023-01-01 23:59", freq="1min")
        n = len(dates)
        X = np.zeros((n, 10))
        y = np.zeros(n)
        timestamps = dates

        # Case 1: Simple range (10-12)
        X_filt, y_filt, t_filt = filter_by_session(
            X, y, timestamps, "custom", custom_start=10, custom_end=12
        )
        assert (t_filt.hour >= 10).all()
        assert (t_filt.hour < 12).all()

        # Case 2: Cross midnight (22-02)
        X_filt, y_filt, t_filt = filter_by_session(
            X, y, timestamps, "custom", custom_start=22, custom_end=2
        )
        hours = t_filt.hour
        assert ((hours >= 22) | (hours < 2)).all()

    def test_filter_invalid(self):
        """Invalid session should raise ValueError."""
        dates = pd.date_range("2023-01-01 00:00", "2023-01-01 01:00", freq="1min")
        X = np.zeros((len(dates), 1))
        y = np.zeros(len(dates))

        with pytest.raises(ValueError, match="Unknown session"):
            filter_by_session(X, y, dates, "invalid_session")

        with pytest.raises(ValueError, match="Must provide custom_start"):
            filter_by_session(X, y, dates, "custom")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
