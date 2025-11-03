import argparse
import json
import math
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Columns consistent with baseline trainer
CATEGORICAL_COLS = ["country", "channel", "device_type", "merchant_category", "currency"]
NUMERIC_COLS = ["amount", "hour", "is_high_risk_country", "is_international", "card_present", "velocity_24h"]
DROP_COLS = ["user_id", "merchant_id"]
TARGET_COL = "label"


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV not found: {path}")
    return pd.read_csv(path)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "threshold": float(thr)
    }
    return metrics


def pick_threshold_by_fp(y_true: np.ndarray, y_prob: np.ndarray, target_fp: int) -> Tuple[float, dict]:
    # Consider thresholds at unique probabilities (descending)
    uniq = np.unique(y_prob)
    thresholds = np.concatenate(([1.0], uniq[::-1], [0.0]))

    best_thr = thresholds[0]
    best_fp_diff = float("inf")
    best_fp = None

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        diff = abs(fp - target_fp)
        if diff < best_fp_diff or (diff == best_fp_diff and thr > best_thr):
            best_thr = float(thr)
            best_fp_diff = diff
            best_fp = int(fp)

    return best_thr, {"strategy": "fp_target", "target_fp": int(target_fp), "achieved_fp": int(best_fp), "fp_diff": int(best_fp_diff)}


def main():
    ap = argparse.ArgumentParser(description="Retune decision threshold to target FP increase factor")
    ap.add_argument("--input_csv", default="data/synthetic_transactions.csv")
    ap.add_argument("--model_in", default="models/baseline_logreg.pkl")
    ap.add_argument("--metrics_in", default="models/metrics.json")
    ap.add_argument("--model_out", default="models/baseline_logreg.pkl")
    ap.add_argument("--metrics_out", default="models/metrics_retuned.json")
    ap.add_argument("--fp_factor", type=float, default=1.10, help="Multiply baseline FP by this factor")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load baseline metrics to get FP
    with open(args.metrics_in, "r", encoding="utf-8") as f:
        base_metrics = json.load(f)
    cm = np.array(base_metrics.get("confusion_matrix", [[0,0],[0,0]]))
    tn, fp, fn, tp = cm.ravel()
    baseline_fp = int(fp)
    target_fp = max(0, int(math.ceil(baseline_fp * args.fp_factor)))

    # Load data and model
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

    y_prob = pipe.predict_proba(X_test)[:, 1]

    new_thr, thr_info = pick_threshold_by_fp(y_test, y_prob, target_fp)

    metrics = evaluate(y_test, y_prob, new_thr)
    metrics.update({"threshold_info": thr_info, "fp_factor": args.fp_factor, "baseline_fp": baseline_fp})

    # Save updated metrics
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Update model threshold in-place
    bundle["threshold"] = float(new_thr)
    joblib.dump(bundle, args.model_out)

    print("Retuned threshold from", base_metrics.get("threshold"), "to", new_thr)
    print("Baseline FP:", baseline_fp, "Target FP:", target_fp, "Achieved FP:", thr_info.get("achieved_fp"))
    print("Saved:", args.model_out)
    print("Metrics:", args.metrics_out)


if __name__ == "__main__":
    main()
