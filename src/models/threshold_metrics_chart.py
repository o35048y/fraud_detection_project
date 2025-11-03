import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
import joblib

# Column definitions to prepare the dataset similarly to training
CATEGORICAL_COLS: List[str] = [
    "country", "channel", "device_type", "merchant_category", "currency"
]
NUMERIC_COLS: List[str] = [
    "amount", "hour", "is_high_risk_country", "is_international", "card_present", "velocity_24h"
]
DROP_COLS: List[str] = ["user_id", "merchant_id"]
TARGET_COL = "label"


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV not found: {path}")
    return pd.read_csv(path)


def compute_metrics_across_thresholds(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    P = int(np.sum(y_true == 1))
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        signals = tp + fp
        recall = tp / P if P > 0 else 0.0
        precision = tp / signals if signals > 0 else 0.0
        rows.append({
            "threshold": float(thr),
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "signals": int(signals),
            "precision": float(precision),
            "recall": float(recall),
            "tp_percent_total": float(recall * 100.0)
        })
    return pd.DataFrame(rows)


def plot_threshold_metrics(df: pd.DataFrame, out_png: str, title: str = "Threshold vs Signals/FP and TP% detected") -> None:
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(df["threshold"], df["signals"], label="Total signals (TP+FP)", color="#2c7fb8")
    ax1.plot(df["threshold"], df["FP"], label="FP (count)", color="#f03b20")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Counts (Signals/FP)")

    ax2 = ax1.twinx()
    ax2.plot(df["threshold"], df["tp_percent_total"], label="TP% of total (Recall %)", color="#31a354", linestyle="--")
    ax2.set_ylabel("TP% detected (Recall %)")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.grid(True, linestyle=":", alpha=0.5)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Sweep thresholds and chart FP, TP, and TP% detected vs threshold")
    parser.add_argument("--input_csv", type=str, default="data/synthetic_transactions.csv")
    parser.add_argument("--model_in", type=str, default="models/baseline_logreg.pkl")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_csv", type=str, default="analysis/threshold_metrics.csv")
    parser.add_argument("--out_png", type=str, default="analysis/threshold_metrics.png")
    parser.add_argument("--num_points", type=int, default=401, help="Number of thresholds to evaluate between 0 and 1")
    args = parser.parse_args()

    df = load_data(args.input_csv)
    missing = [c for c in (CATEGORICAL_COLS + NUMERIC_COLS + [TARGET_COL]) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")

    # Drop non-feature ids
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int).values

    # Use the same split strategy as training scripts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    bundle = joblib.load(args.model_in)
    model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle

    y_prob = model.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.0, 1.0, args.num_points)
    sweep_df = compute_metrics_across_thresholds(y_test, y_prob, thresholds)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    sweep_df.to_csv(args.out_csv, index=False)

    title = f"Threshold metrics: {os.path.basename(args.model_in)}"
    plot_threshold_metrics(sweep_df, args.out_png, title=title)

    # Print brief summary
    print(f"Saved: {args.out_csv}")
    print(f"Saved: {args.out_png}")


if __name__ == "__main__":
    main()
