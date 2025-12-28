from __future__ import annotations
import pandas as pd
from pathlib import Path
import json
import logging

# Add repo root to path so `src` imports work (same approach as other scripts)
import sys
script_dir = Path(__file__).resolve()
ml_dir = script_dir.parent.parent
sys.path.insert(0, str(ml_dir.parent))

from src.scripts.predict_sequence import predict, load_candles_from_csv, load_latest_candles_from_dir

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('diagnose')

MODELS_DIR = (Path(__file__).parents[2] / 'outputs' / 'models').resolve()
# try multiple candidate locations for the 2025 CSV
CANDIDATES = [
    Path(__file__).resolve().parents[3] / 'backend' / 'TradingML.ModelPrediction' / 'Data' / 'XAU_1m_data_2025.csv',
    Path(__file__).resolve().parents[2] / 'backend' / 'TradingML.ModelPrediction' / 'Data' / 'XAU_1m_data_2025.csv',
    Path(__file__).resolve().parents[2] / 'data' / 'XAU_1m_data_2025.csv',
    Path(__file__).resolve().parents[3] / 'ml' / 'src' / 'data' / 'XAU_1m_data_2025.csv'
]
CSV_PATH = next((p for p in CANDIDATES if p.exists()), None)
if CSV_PATH is None:
    raise FileNotFoundError(f"Could not locate XAU_1m_data_2025.csv in candidates: {CANDIDATES}")

if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH, sep=';', parse_dates=['Date']).set_index('Date')
    total = len(df)
    logger.info(f'Loaded {total} candles from {CSV_PATH}')

    WINDOW_M1 = 7 * 24 * 60  # 10080
    STEP = 60
    sample_windows = 200

    counts = {
        'total_tested': 0,
        'trend_filtered': 0,
        'weak_trend_filtered': 0,
        'overbought_filtered': 0,
        'pred1': 0,
        'prob_ge_threshold': 0,
    }

    probs = []

    # We'll test first `sample_windows` windows
    for i in range(0, min(3000, total - WINDOW_M1), STEP):
        if counts['total_tested'] >= sample_windows:
            break
        window = df.iloc[i:i+WINDOW_M1]
        try:
            res = predict(window, MODELS_DIR)
        except Exception as e:
            logger.exception('predict failed for window starting at %s', window.index.min())
            break

        counts['total_tested'] += 1
        probs.append(res['probability'])
        if res['prediction'] == 1:
            counts['pred1'] += 1
        if res['probability'] >= res['threshold']:
            counts['prob_ge_threshold'] += 1

        conf = res.get('confidence', '')
        reason = res.get('confidence', '')
        # infer filter from confidence string
        if isinstance(conf, str) and 'trend filter' in conf:
            counts['trend_filtered'] += 1
        if isinstance(conf, str) and 'weak trend' in conf:
            counts['weak_trend_filtered'] += 1
        if isinstance(conf, str) and 'overbought' in conf:
            counts['overbought_filtered'] += 1

    out = {
        'counts': counts,
        'mean_prob': float(sum(probs)/len(probs)) if probs else None,
        'min_prob': min(probs) if probs else None,
        'max_prob': max(probs) if probs else None,
        'top_probs': sorted(probs, reverse=True)[:20]
    }

    out_path = Path.cwd() / 'backend' / 'outputs' / 'diagnostics_window_sample.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    logger.info(f'Wrote diagnostics to {out_path}')
    logger.info(json.dumps(out, indent=2))