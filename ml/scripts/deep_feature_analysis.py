#!/usr/bin/env python3
"""Deep analysis of feature distributions and predictiveness."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd
import numpy as np
from ml.src.data_loading.loaders import load_all_years
from ml.src.features.engineer import engineer_candle_features
from ml.src.targets.target_maker import make_target

# Load data
data_dir = Path("ml/src/data")
df = load_all_years(data_dir, year_filter=[2024])
print(f"Loaded {len(df):,} rows")

# Engineer features with OPTION A (0.2/0.4)
df_feat = engineer_candle_features(df)
print(f"Engineered {df_feat.shape[1]} features")

# Create targets with OPTION A
targets = make_target(df, atr_multiplier_sl=0.2, atr_multiplier_tp=0.4, max_horizon=120)
print(f"Created targets: {(targets==1).sum():,} positive, {(targets==0).sum():,} negative\n")

# Align
valid_idx = targets.index
X_feat = df_feat.loc[valid_idx, :]
y = targets

# Categorize features by type
feature_categories = {
    'M5_Context': ['atr_m5_n', 'rsi_m5', 'dist_sma_20_m5', 'macd_n_m5', 'bb_pos_m5'],
    'Micro_Structure': ['micro_vol_5', 'efficiency_5', 'fractal_dim_5', 'trend_consistency_5', 'slope_20', 'slope_60'],
    'Long_Term': ['dist_sma_200', 'dist_sma_1440', 'roc_60', 'vol_ratio_60_200', 'dist_day_high', 'dist_day_low', 
                  'z_score_20', 'vol_ratio_5_20', 'dist_prev_high', 'dist_prev_low', 'dist_prev_close', 
                  'dist_daily_open', 'dist_london_open'],
    'Candle': ['ret_1', 'range_n', 'body_ratio', 'upper_shadow', 'lower_shadow'],
    'Volume': ['vol_change', 'vol_ratio'],
    'Trend': ['ema_spread_n', 'adx', 'plus_di', 'minus_di', 'macd_line_n', 'macd_hist_n'],
    'Momentum': ['rsi_14', 'stoch_k', 'stoch_d', 'cci', 'williams_r', 'roc_5', 'roc_20', 'price_position'],
    'Volatility': ['vol_20', 'atr_14', 'atr_n', 'bb_width', 'bb_position'],
    'OBV_Market': ['obv_normalized', 'market_structure'],
    'Price_Action': ['distance_from_ma'],
    'Time': ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos'],
}

print("="*120)
print("DETAILED FEATURE ANALYSIS BY CATEGORY")
print("="*120)

all_results = []

for category, feature_list in feature_categories.items():
    print(f"\n{category.upper()}")
    print("-" * 120)
    
    for feat in feature_list:
        if feat not in X_feat.columns:
            continue
            
        X_col = X_feat[feat]
        
        # Basic stats
        mean_val = X_col.mean()
        std_val = X_col.std()
        var_val = X_col.var()
        min_val = X_col.min()
        max_val = X_col.max()
        
        # Correlation with target
        corr = X_col.corr(y)
        abs_corr = abs(corr)
        
        # How many zeros?
        zero_pct = 100 * (X_col == 0).sum() / len(X_col)
        
        # Signal strength: mean difference between pos and neg samples
        mean_pos = X_col[y == 1].mean()
        mean_neg = X_col[y == 0].mean()
        signal_diff = abs(mean_pos - mean_neg)
        
        correlation_strength = "STRONG" if abs_corr > 0.05 else "WEAK" if abs_corr > 0.01 else "NONE"
        
        all_results.append({
            'Feature': feat,
            'Category': category,
            'Correlation': corr,
            'Abs_Corr': abs_corr,
            'Mean_Pos': mean_pos,
            'Mean_Neg': mean_neg,
            'Signal_Diff': signal_diff,
            'Std': std_val,
            'Var': var_val,
            'Zeros_%': zero_pct,
        })
        
        print(f"  {feat:30s} | corr={corr:+.5f} | signal_diff={signal_diff:8.4f} | "
              f"std={std_val:8.4f} | zeros={zero_pct:5.1f}% | {correlation_strength}")

# Summary by category
print("\n" + "="*120)
print("CATEGORY PERFORMANCE")
print("="*120)

results_df = pd.DataFrame(all_results)

category_summary = results_df.groupby('Category').agg({
    'Abs_Corr': ['max', 'mean', 'min'],
    'Signal_Diff': ['max', 'mean'],
    'Feature': 'count',
}).round(5)

print(category_summary)

# Top 15 features overall
print("\n" + "="*120)
print("TOP 15 BEST FEATURES BY ABSOLUTE CORRELATION")
print("="*120)

top_15 = results_df.nlargest(15, 'Abs_Corr')[['Feature', 'Category', 'Abs_Corr', 'Signal_Diff', 'Std', 'Mean_Pos', 'Mean_Neg']]
for idx, row in top_15.iterrows():
    print(f"{row['Feature']:30s} | {row['Category']:15s} | corr={row['Abs_Corr']:.5f} | "
          f"signal={row['Signal_Diff']:.4f} | std={row['Std']:.4f}")

# Bottom 10 - potential candidates for removal
print("\n" + "="*120)
print("BOTTOM 10 WEAKEST FEATURES (CANDIDATES FOR REMOVAL)")
print("="*120)

bottom_10 = results_df.nsmallest(10, 'Abs_Corr')[['Feature', 'Category', 'Abs_Corr', 'Signal_Diff', 'Std', 'Zeros_%']]
for idx, row in bottom_10.iterrows():
    print(f"{row['Feature']:30s} | {row['Category']:15s} | corr={row['Abs_Corr']:.5f} | "
          f"signal={row['Signal_Diff']:.4f} | std={row['Std']:.4f} | zeros={row['Zeros_%']:.1f}%")

# Features with very low variance or high zero percentage
print("\n" + "="*120)
print("PROBLEMATIC FEATURES (Low variance or high zero percentage)")
print("="*120)

problematic = results_df[(results_df['Var'] < 0.001) | (results_df['Zeros_%'] > 50)]
print(f"Found {len(problematic)} problematic features:\n")
for idx, row in problematic.iterrows():
    print(f"  {row['Feature']:30s} | var={row['Var']:10.6f} | zeros={row['Zeros_%']:5.1f}% | corr={row['Abs_Corr']:.5f}")

# Recommendations
print("\n" + "="*120)
print("RECOMMENDATIONS FOR FEATURE ENGINEERING")
print("="*120)
print("""
1. TOP PERFORMERS:
   - bb_pos_m5, dist_sma_20_m5, rsi_m5 are best predictors
   - These are from M5 CONTEXT and MOMENTUM categories
   - Consider: Create more multi-timeframe features?

2. WEAK CATEGORIES:
   - Time features (hour_sin, hour_cos, minute_sin, minute_cos) have almost no correlation
   - OBV_Market features are weak
   - Maybe gold doesn't have strong time-of-day patterns

3. PROBLEMATIC FEATURES:
   - Remove or fix features with <0.001 variance
   - Investigate features with >50% zeros

4. NEW FEATURES TO TRY:
   - More multi-timeframe context (15min, 1hour levels)
   - Volatility mean reversion signals
   - Support/Resistance levels
   - Trend strength measures (ADX seems weak)
   - Order flow imbalance (bid/ask from lower timeframes)
   - Momentum oscillator divergences (RSI, Stochastic divergence)

5. FEATURE ENGINEERING ISSUES TO CHECK:
   - Are M5 features calculated correctly?
   - Why is RSI_M5 better than RSI_14?
   - What about RSI_15min or RSI_1hour?
   - Missing features: Gap from previous day, Overnight moves, Session opens
""")
