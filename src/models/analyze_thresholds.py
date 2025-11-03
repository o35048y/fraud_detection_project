import argparse
import json
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

CATEGORICAL_COLS = ["country", "channel", "device_type", "merchant_category", "currency"]
NUMERIC_COLS = ["amount", "hour", "is_high_risk_country", "is_international", "card_present", "velocity_24h"]
DROP_COLS = ["user_id", "merchant_id"]
TARGET_COL = "label"


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV not found: {path}")
    return pd.read_csv(path)


def sweep_thresholds(y_true: np.ndarray, y_prob: np.ndarray, n_points: int = 201, extra_thresholds: Optional[list] = None) -> pd.DataFrame:
    thresholds = np.linspace(0.0, 1.0, n_points)
    if extra_thresholds:
        thresholds = np.unique(np.concatenate([thresholds, np.array([t for t in extra_thresholds if t is not None], dtype=float)]))
    rows = []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        signals = tp + fp
        precision = (tp / signals) if signals > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        rows.append({
            "threshold": float(thr),
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "signals": int(signals),
            "fp_ratio_of_signals": float(fp / signals) if signals > 0 else 0.0,
            "precision": float(precision),
            "recall": float(recall)
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Analyze FP vs total signals across thresholds")
    ap.add_argument("--input_csv", default="data/synthetic_transactions.csv")
    ap.add_argument("--model_in", default="models/baseline_logreg.pkl")
    ap.add_argument("--metrics_in", default="models/metrics.json")
    ap.add_argument("--metrics_retuned", default="models/metrics_retuned.json")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", default="analysis/threshold_sweep.csv")
    ap.add_argument("--out_png", default="analysis/false_positives_vs_signals.png")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)

    # Load data and bundle
    df = load_data(args.input_csv)
    cols_needed = CATEGORICAL_COLS + NUMERIC_COLS + [TARGET_COL] + DROP_COLS
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")
    df = df.drop(columns=DROP_COLS)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    bundle = joblib.load(args.model_in)
    pipe = bundle["model"]
    saved_thr = float(bundle.get("threshold", 0.5))

    # Load metrics to include exact thresholds in sweep and annotate later
    base_thr = None
    ret_thr = None
    base_row = None
    retuned_row = None
    base = None
    re = None

    if os.path.exists(args.metrics_in):
        with open(args.metrics_in, "r", encoding="utf-8") as f:
            base = json.load(f)
        base_thr = float(base.get("threshold", saved_thr))

    if os.path.exists(args.metrics_retuned):
        with open(args.metrics_retuned, "r", encoding="utf-8") as f:
            re = json.load(f)
        ret_thr = float(re.get("threshold", saved_thr))

    y_prob = pipe.predict_proba(X_test)[:, 1]

    sweep = sweep_thresholds(y_test, y_prob, n_points=201, extra_thresholds=[base_thr, ret_thr])
    sweep.to_csv(args.out_csv, index=False)

    # Helper to find nearest row
    def find_row_for_thr(df, thr: float):
        i = (df["threshold"] - thr).abs().idxmin()
        return df.loc[i]

    if base_thr is not None:
        base_row = find_row_for_thr(sweep, base_thr)
    if ret_thr is not None:
        retuned_row = find_row_for_thr(sweep, ret_thr)

    # Plot FP vs total signals; also overlay FP ratio on secondary axis
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(sweep["signals"], sweep["FP"], label="FP vs Signals", color="tomato")
    ax1.set_xlabel("Total signals (TP + FP)")
    ax1.set_ylabel("False Positives (FP)", color="tomato")
    ax1.tick_params(axis='y', labelcolor='tomato')

    ax2 = ax1.twinx()
    ax2.plot(sweep["signals"], sweep["fp_ratio_of_signals"], label="FP / Signals", color="steelblue", alpha=0.7)
    ax2.set_ylabel("FP share of signals (1 - precision)", color="steelblue")
    ax2.tick_params(axis='y', labelcolor='steelblue')

    def annotate_point(row, label, color):
        if row is None:
            return
        ax1.scatter([row["signals"]], [row["FP"]], color=color, zorder=3)
        ax1.annotate(f"{label}\nthr={row['threshold']:.3f}\nFP={int(row['FP'])}\nsignals={int(row['signals'])}",
                     (row["signals"], row["FP"]), xytext=(10,10), textcoords='offset points', color=color,
                     bbox=dict(boxstyle="round,pad=0.3", fc="w", alpha=0.7))

    annotate_point(base_row, "baseline", "black")
    annotate_point(retuned_row, "+10% FP", "green")

    plt.title("False Positives vs Total Signals across thresholds")
    fig.tight_layout()
    plt.savefig(args.out_png, dpi=160)
    print("Saved sweep:", args.out_csv)
    print("Saved plot:", args.out_png)

    if base_row is not None and retuned_row is not None:
        print("Baseline -> Retuned deltas:")
        print("  FP:", int(base_row["FP"]), "->", int(retuned_row["FP"]))
        print("  Signals:", int(base_row["signals"]), "->", int(retuned_row["signals"]))
        print("  FP/Signals:", f"{base_row['fp_ratio_of_signals']:.3f}", "->", f"{retuned_row['fp_ratio_of_signals']:.3f}")

if __name__ == "__main__":
    main()
