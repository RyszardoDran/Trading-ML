#!/usr/bin/env python3
"""Diagnostics: compare scaler (training) statistics vs backtest features and report probability distribution.

Outputs a JSON report to `ml/outputs/diagnostics_report.json` and prints a concise summary.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    models_dir = Path("ml/outputs/models")
    feats_path = Path("ml/outputs/backtest_features.pkl")
    out_path = Path("ml/outputs/diagnostics_report.json")

    with open(models_dir / "sequence_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(models_dir / "sequence_feature_columns.json", "r", encoding="utf-8") as f:
        cols = json.load(f)
    with open(models_dir / "sequence_threshold.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(models_dir / "sequence_xgb_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open(feats_path, "rb") as f:
        payload = pickle.load(f)

    features = np.asarray(payload["features"])  # shape: (n_rows, n_features_per_bar)
    prices = np.asarray(payload["prices"])      # aligned to features index
    timestamps = np.asarray(payload["timestamps"], dtype="datetime64[ns]")

    window = int(meta.get("window_size", 60))
    n_perbar = features.shape[1]

    # scaler info
    scaler_n_in = getattr(scaler, "n_features_in_", None)
    scaler_center = getattr(scaler, "center_", None)
    scaler_scale = getattr(scaler, "scale_", None)

    report = {
        "scaler_n_features_in": scaler_n_in,
        "perbar_columns": cols,
        "n_perbar_columns": n_perbar,
        "window_size": window,
        "expected_flattened_features": n_perbar * window,
    }

    # compute per-bar stats on backtest features
    df_feats = pd.DataFrame(features, columns=cols)
    perbar_stats = {}
    for c in cols:
        arr = df_feats[c].astype(float).values
        perbar_stats[c] = {
            "mean_backtest": float(np.nanmean(arr)),
            "std_backtest": float(np.nanstd(arr)),
            "pctile_1": float(np.percentile(arr, 1)),
            "pctile_50": float(np.percentile(arr, 50)),
            "pctile_99": float(np.percentile(arr, 99)),
        }

    # compare to scaler centers/scales (if available)
    comparisons = {}
    if scaler_center is not None and scaler_scale is not None:
        # scaler_center/scale are for flattened sequences; we assume per-bar centers repeated across windows
        # extract per-bar reference by taking first n_perbar values from center_
        ref_center = np.array(scaler_center[:n_perbar]) if len(scaler_center) >= n_perbar else None
        ref_scale = np.array(scaler_scale[:n_perbar]) if len(scaler_scale) >= n_perbar else None
        for i, c in enumerate(cols):
            rc = float(ref_center[i]) if ref_center is not None else None
            rs = float(ref_scale[i]) if ref_scale is not None else None
            mb = perbar_stats[c]["mean_backtest"]
            sd = perbar_stats[c]["std_backtest"]
            z = None
            if rc is not None and rs not in (0, None):
                z = (mb - rc) / rs
            comparisons[c] = {
                "ref_center": rc,
                "ref_scale": rs,
                "mean_backtest": mb,
                "std_backtest": sd,
                "z_shift": float(z) if z is not None else None,
            }

    # build flattened sequences and compute proba distribution
    Xseq = np.lib.stride_tricks.sliding_window_view(features, window, axis=0)
    Xseq = Xseq.reshape((Xseq.shape[0], window * n_perbar))
    Xs = scaler.transform(Xseq)
    proba = model.predict_proba(Xs)[:, 1]

    proba_stats = {
        "count": int(len(proba)),
        "pctiles": [float(x) for x in np.percentile(proba, [0,1,5,10,25,50,75,90,95,99,100])],
        "fraction_ge_threshold": float((proba >= meta.get("threshold", 0.623)).mean()),
        "fraction_ge_0.5": float((proba >= 0.5).mean()),
    }

    report["perbar_stats"] = perbar_stats
    report["comparisons"] = comparisons
    report["proba_stats"] = proba_stats

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Print concise summary
    print("Diagnostics written to:", out_path)
    print("scaler_n_features_in:", scaler_n_in)
    print("per-bar cols:", n_perbar)
    # list columns with |z_shift| > 3
    large_shifts = [c for c,v in comparisons.items() if v.get("z_shift") is not None and abs(v.get("z_shift"))>3]
    print("columns with |z_shift|>3:", large_shifts[:10])
    print("proba percentiles:", report["proba_stats"]["pctiles"]) 


if __name__ == "__main__":
    main()
