import pandas as pd
from ml.src.features.engineer_m5 import aggregate_to_m5, engineer_m5_candle_features
from ml.src.features.indicators import compute_rsi


def test_m15_alignment_no_lookahead():
    # Build synthetic M1 data spanning 1 day (1440 minutes) to get sufficient M5 bars
    idx = pd.date_range('2025-12-01 00:00', periods=1440, freq='1min')
    closes = pd.Series(100.0 + (pd.Series(range(len(idx))) * 0.01).values, index=idx)  # Smaller increment for realistic data
    df = pd.DataFrame({
        'Open': closes,
        'High': closes + 0.05,
        'Low': closes - 0.05,
        'Close': closes,
        'Volume': [100] * len(idx),
    }, index=idx)

    # Aggregate to M5 and compute M15 RSI directly (reference)
    df_m5 = aggregate_to_m5(df)
    df_m15 = df_m5.resample('15min').agg({'Close': 'last'})
    rsi_m15_ref = compute_rsi(df_m15['Close'], period=14)
    # Forward-fill to M5 index (use previous closed M15 bar)
    rsi_m15_expected = rsi_m15_ref.reindex(df_m5.index, method='ffill').fillna(50.0)

    # Engineer features and compare
    features = engineer_m5_candle_features(df)

    # Pick timestamps: one between 10:00 and 10:14 (should match 10:00 M15), one at/after 10:15
    t_mid = pd.Timestamp('2025-12-01 10:10')
    t_after = pd.Timestamp('2025-12-01 10:20')

    assert t_mid in features.index, "Expected M5 timestamp for t_mid"
    assert t_after in features.index, "Expected M5 timestamp for t_after"

    val_mid = features.loc[t_mid, 'rsi_m15']
    val_after = features.loc[t_after, 'rsi_m15']

    expected_mid = rsi_m15_expected.loc[t_mid]
    expected_after = rsi_m15_expected.loc[t_after]

    assert float(val_mid) == float(expected_mid), f"rsi_m15 at {t_mid} should equal previous M15 ({expected_mid} != {val_mid})"
    assert float(val_after) == float(expected_after), f"rsi_m15 at {t_after} should equal previous M15 ({expected_after} != {val_after})"
