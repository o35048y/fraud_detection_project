import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Shared feature lists
CATEGORICAL_COLS: List[str] = [
    "country", "channel", "device_type", "merchant_category", "currency"
]
NUMERIC_COLS: List[str] = [
    "amount", "hour", "is_high_risk_country", "is_international", "card_present", "velocity_24h"
]
DROP_COLS: List[str] = ["user_id", "merchant_id"]
TARGET_COL = "label"


@dataclass
class MethodSpec:
    name: str
    model_path: str
    metrics_path: str


def load_data(path: str, test_size: float, seed: int):
    df = pd.read_csv(path)
    # Ensure expected columns exist
    missing = [c for c in (CATEGORICAL_COLS + NUMERIC_COLS + [TARGET_COL]) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")
    # Drop IDs if present
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    return X_test, y_test


def model_size_bytes(path: str) -> int:
    return os.path.getsize(path) if os.path.exists(path) else -1


def measure_throughput(model, X_test, iters: int = 30) -> float:
    # Warm-up
    _ = model.predict_proba(X_test)[:, 1]
    n = X_test.shape[0]
    start = time.perf_counter()
    for _ in range(iters):
        _ = model.predict_proba(X_test)[:, 1]
    total = time.perf_counter() - start
    return (n * iters) / total if total > 0 else float("inf")


def gather_stats(methods: List[MethodSpec], X_test, y_test) -> pd.DataFrame:
    rows = []
    for m in methods:
        bundle = joblib.load(m.model_path)
        model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
        sz = model_size_bytes(m.model_path)
        thrpt = measure_throughput(model, X_test, iters=30)
        # Load metrics JSON
        with open(m.metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        f2 = float(metrics.get("f2", np.nan))
        pr_auc = float(metrics.get("pr_auc", np.nan))
        rows.append({
            "method": m.name,
            "model_path": m.model_path,
            "metrics_path": m.metrics_path,
            "model_size_bytes": sz,
            "model_size_mb": sz / (1024 * 1024),
            "throughput_eps": thrpt,  # examples per second
            "f2": f2,
            "pr_auc": pr_auc,
        })
    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, out_png: str):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    # Bubble sizes scaled by model size MB
    sizes = (df["model_size_mb"].values + 0.1) * 50.0

    # Left: F2 vs throughput
    ax = axes[0]
    ax.scatter(df["throughput_eps"], df["f2"], s=sizes, c="#2c7fb8", alpha=0.8, edgecolor="black")
    for _, r in df.iterrows():
        ax.annotate(r["method"], (r["throughput_eps"], r["f2"]), xytext=(5,5), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Inference throughput (examples/sec, log scale)")
    ax.set_ylabel("F2 score")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_title("Performance (F2) vs Efficiency")

    # Right: PR AUC vs throughput
    ax = axes[1]
    ax.scatter(df["throughput_eps"], df["pr_auc"], s=sizes, c="#31a354", alpha=0.8, edgecolor="black")
    for _, r in df.iterrows():
        ax.annotate(r["method"], (r["throughput_eps"], r["pr_auc"]), xytext=(5,5), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Inference throughput (examples/sec, log scale)")
    ax.set_ylabel("PR AUC")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_title("Performance (PR AUC) vs Efficiency")

    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Compare methods by performance (F2, PR AUC) and efficiency (model size, throughput)")
    ap.add_argument("--input_csv", type=str, default="data/synthetic_transactions.csv")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", type=str, default="analysis/methods_comparison.csv")
    ap.add_argument("--out_png", type=str, default="analysis/methods_performance_efficiency.png")
    args = ap.parse_args()

    X_test, y_test = load_data(args.input_csv, args.test_size, args.seed)

    methods = [
        MethodSpec(name="LogReg+SMOTE (F2)", model_path="models/baseline_logreg.pkl", metrics_path="models/metrics.json"),
        MethodSpec(name="RandomForest (F2)", model_path="models/tree_f2.pkl", metrics_path="models/tree_metrics.json"),
    ]

    df = gather_stats(methods, X_test, y_test)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    plot_comparison(df, args.out_png)

    # Print small table
    printable = df[["method", "f2", "pr_auc", "model_size_mb", "throughput_eps"]].copy()
    printable = printable.sort_values(by="f2", ascending=False)
    print("\nMethods comparison (sorted by F2):")
    print(printable.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    print(f"\nSaved CSV: {args.out_csv}")
    print(f"Saved Figure: {args.out_png}")


if __name__ == "__main__":
    main()
