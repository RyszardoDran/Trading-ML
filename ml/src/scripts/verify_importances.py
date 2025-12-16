import json
import pickle
from pathlib import Path

import numpy as np


def main() -> None:
    models_dir = Path(__file__).resolve().parents[1] / "models"

    model_path = models_dir / "sequence_xgb_model.pkl"
    meta_path = models_dir / "sequence_threshold.json"
    cols_path = models_dir / "sequence_feature_columns.json"
    saved_top_path = models_dir / "sequence_feature_importance.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    base = model.calibrated_classifiers_[0].estimator
    importances = base.feature_importances_.astype(float)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    window_size = int(meta["window_size"])  # required for flattened naming

    with open(cols_path, "r", encoding="utf-8") as f:
        cols = json.load(f)

    names = [
        f"t-{window_size - i - 1}_{feat}"
        for i in range(window_size)
        for feat in cols
    ]

    print("n_features_flat", len(names))
    print("importances_len", len(importances))

    if len(names) != len(importances):
        raise ValueError(
            f"Length mismatch: names({len(names)}) vs importances({len(importances)})"
        )

    imp_sum = float(np.sum(importances))
    imp_min = float(np.min(importances))
    imp_max = float(np.max(importances))

    pairs = list(zip(names, importances))
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)

    print("sum", round(imp_sum, 6), "min", round(imp_min, 6), "max", round(imp_max, 6))
    print("top_30:")
    for i, (n, v) in enumerate(pairs_sorted[:30], 1):
        print(f"{i:2d}. {n:40s} = {v:.6f}")

    if saved_top_path.exists():
        with open(saved_top_path, "r", encoding="utf-8") as f:
            saved_top = json.load(f)
        mismatch = []
        for k, v in list(saved_top.items())[:10]:
            try:
                idx = names.index(k)
            except ValueError:
                mismatch.append((k, v, None))
                continue
            cur = float(importances[idx])
            if abs(cur - float(v)) > 1e-6:
                mismatch.append((k, v, cur))
        print("first10_compare_mismatch_count", len(mismatch))
        if mismatch:
            print("sample_mismatch", mismatch[:3])


if __name__ == "__main__":
    main()
