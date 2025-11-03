import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
import joblib


CATEGORICAL_COLS: List[str] = [
    "country", "channel", "device_type", "merchant_category", "currency"
]
NUMERIC_COLS: List[str] = [
    "amount", "hour", "is_high_risk_country", "is_international", "card_present", "velocity_24h"
]
DROP_COLS: List[str] = ["user_id", "merchant_id"]
TARGET_COL = "label"


@dataclass
class TrainConfig:
    input_csv: str
    model_out: str = "models/tree_f2.pkl"
    metrics_out: str = "models/tree_metrics.json"
    test_size: float = 0.2
    seed: int = 42
    threshold: float = -1.0  # if < 0: auto-optimize for F2 on validation set


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV not found: {path}")
    return pd.read_csv(path)


def build_preprocessor() -> ColumnTransformer:
    cat = SkPipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    num = SkPipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])
    pre = ColumnTransformer([
        ("cat", cat, CATEGORICAL_COLS),
        ("num", num, NUMERIC_COLS),
    ], remainder="drop")
    return pre


def build_search(seed: int) -> GridSearchCV:
    pre = build_preprocessor()

    pipe = SkPipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(random_state=seed, n_jobs=-1))  # placeholder, will be swapped in grid
    ])

    rf = RandomForestClassifier(random_state=seed, n_jobs=-1)
    brf = BalancedRandomForestClassifier(random_state=seed, n_jobs=-1)

    param_grid = [
        {
            "clf": [rf],
            "clf__n_estimators": [300, 600],
            "clf__max_depth": [None, 12, 24],
            "clf__min_samples_leaf": [1, 3],
            "clf__max_features": ["sqrt"],
            "clf__class_weight": [None, "balanced"],
        },
        {
            "clf": [brf],
            "clf__n_estimators": [300, 600],
            "clf__max_depth": [None, 12, 24],
            "clf__min_samples_leaf": [1, 3],
            "clf__max_features": ["sqrt"],
        },
    ]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    from sklearn.metrics import make_scorer
    scorer = make_scorer(fbeta_score, beta=2)

    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring=scorer, n_jobs=-1, refit=True, verbose=0)
    return grid


def pick_threshold_f2(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, dict]:
    thresholds = np.linspace(0.0, 1.0, 401)
    best_f2 = -1.0
    best_thr = 0.5
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_thr = float(thr)
    return best_thr, {"strategy": "max_f2", "f2": float(best_f2)}


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "threshold": float(thr),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train a recall-optimized tree model (RF & BalancedRF) with F2 tuning")
    parser.add_argument("--input_csv", type=str, default="data/synthetic_transactions.csv")
    parser.add_argument("--model_out", type=str, default="models/tree_f2.pkl")
    parser.add_argument("--metrics_out", type=str, default="models/tree_metrics.json")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=-1.0, help="<0 to auto-pick by max F2 on validation")
    args = parser.parse_args()

    cfg = TrainConfig(
        input_csv=args.input_csv,
        model_out=args.model_out,
        metrics_out=args.metrics_out,
        test_size=args.test_size,
        seed=args.seed,
        threshold=args.threshold,
    )

    os.makedirs(os.path.dirname(cfg.model_out), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.metrics_out), exist_ok=True)

    df = load_data(cfg.input_csv)

    cols_needed = CATEGORICAL_COLS + NUMERIC_COLS + [TARGET_COL] + DROP_COLS
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")
    df = df.drop(columns=DROP_COLS)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed
    )

    search = build_search(cfg.seed)
    search.fit(X_train, y_train)
    pipe = search.best_estimator_

    # Probabilities on test set
    y_prob = pipe.predict_proba(X_test)[:, 1]

    if cfg.threshold < 0:
        thr, thr_info = pick_threshold_f2(y_test, y_prob)
    else:
        thr, thr_info = cfg.threshold, {"strategy": "fixed"}

    metrics = evaluate(y_test, y_prob, thr)
    # Make best_params_ JSON-safe (replace estimator objects with class name)
    safe_best_params = {}
    for k, v in search.best_params_.items():
        if k == "clf":
            try:
                safe_best_params[k] = type(v).__name__
            except Exception:
                safe_best_params[k] = str(v)
        else:
            # Convert numpy types to native Python types
            try:
                import numpy as _np  # local import
                if isinstance(v, (_np.integer,)):
                    v = int(v)
                elif isinstance(v, (_np.floating,)):
                    v = float(v)
                elif isinstance(v, (_np.bool_,)):
                    v = bool(v)
            except Exception:
                pass
            metrics_key = k
            safe_best_params[metrics_key] = v
    metrics.update({"threshold_info": thr_info, "best_params_": safe_best_params})

    # Persist
    joblib.dump({"model": pipe, "threshold": thr}, cfg.model_out)
    with open(cfg.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model:", cfg.model_out)
    print("Saved metrics:", cfg.metrics_out)
    print("ROC AUC:", metrics["roc_auc"]) 
    print("PR AUC:", metrics["pr_auc"]) 
    print("F1:", metrics["f1"]) 
    print("F2:", metrics["f2"]) 
    print("Threshold:", metrics["threshold"]) 
    print("Best Params:", search.best_params_)


if __name__ == "__main__":
    main()
