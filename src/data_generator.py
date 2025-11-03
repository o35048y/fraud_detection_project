import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class GenerationConfig:
    n: int = 10000
    fraud_ratio: float = 0.02
    seed: Optional[int] = 42


HIGH_RISK_COUNTRIES = [
    "RU", "NG", "UA", "PK", "VN", "RO", "ID", "TR", "BR", "CN"
]
COUNTRIES = [
    "US", "GB", "DE", "FR", "CA", "AU", "IN", "JP", "ES", "IT", "NL", "SE", "CH", "PL", "MX"
]
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD"]
CHANNELS = ["online", "pos", "mobile", "atm"]
DEVICE_TYPES = ["web", "ios", "android", "card"]
MERCHANT_CATEGORIES = [
    "grocery", "electronics", "fashion", "fuel", "travel", "digital", "restaurant", "gaming", "utilities"
]


def _rng(seed: Optional[int]):
    return np.random.default_rng(seed)


def _draw_from(p: np.ndarray, choices: list[str], size: int, rng):
    return rng.choice(choices, size=size, p=p)


def _normalize(p):
    p = np.asarray(p, dtype=float)
    p /= p.sum()
    return p


def generate_transactions(cfg: GenerationConfig) -> pd.DataFrame:
    rng = _rng(cfg.seed)

    n_fraud = int(cfg.n * cfg.fraud_ratio)
    n_legit = cfg.n - n_fraud

    # Country distributions
    base_country_weights = np.array([
        0.40, 0.10, 0.06, 0.05, 0.08, 0.05, 0.08, 0.03, 0.04, 0.03, 0.03, 0.02, 0.015, 0.015, 0.015
    ])
    base_country_p = _normalize(base_country_weights)

    high_risk_weight = 0.30
    fraud_country_weights = base_country_weights.copy()
    # Shift some mass towards high-risk (map unknowns to small mass)
    for i, c in enumerate(COUNTRIES):
        if c in HIGH_RISK_COUNTRIES:
            fraud_country_weights[i] += high_risk_weight / max(1, len(HIGH_RISK_COUNTRIES))
    fraud_country_p = _normalize(fraud_country_weights)

    # Channels: fraud more likely online/mobile
    legit_channel_p = _normalize([0.35, 0.45, 0.18, 0.02])
    fraud_channel_p = _normalize([0.65, 0.10, 0.22, 0.03])

    # Device types
    legit_device_p = _normalize([0.45, 0.20, 0.25, 0.10])
    fraud_device_p = _normalize([0.70, 0.10, 0.15, 0.05])

    # Merchant categories
    legit_mc_p = _normalize([0.22, 0.10, 0.15, 0.10, 0.04, 0.08, 0.18, 0.03, 0.10])
    fraud_mc_p = _normalize([0.05, 0.20, 0.06, 0.03, 0.18, 0.20, 0.05, 0.15, 0.08])

    # Amounts: log-normal; fraud heavier tail
    legit_amount = np.clip(np.exp(rng.normal(3.8, 0.8, n_legit)), 1, 5000)
    fraud_amount = np.clip(np.exp(rng.normal(5.0, 1.1, n_fraud)), 5, 20000)

    # Hours: fraud skews to night
    legit_hours = rng.choice(np.arange(24), size=n_legit, p=_normalize([
        0.01,0.01,0.01,0.01,0.02,0.02,  # 0-5
        0.03,0.04,0.05,0.07,0.07,0.07,  # 6-11
        0.07,0.07,0.07,0.07,0.06,0.05,  # 12-17
        0.07,0.07,0.06,0.05,0.03,0.02   # 18-23
    ]))
    fraud_hours = rng.choice(np.arange(24), size=n_fraud, p=_normalize([
        0.05,0.05,0.05,0.05,0.06,0.06,  # 0-5
        0.04,0.04,0.04,0.04,0.04,0.04,  # 6-11
        0.04,0.04,0.04,0.04,0.04,0.04,  # 12-17
        0.08,0.08,0.07,0.06,0.05,0.03   # 18-23
    ]))

    # Compose categorical draws
    legit_country = _draw_from(base_country_p, COUNTRIES, n_legit, rng)
    fraud_country = _draw_from(fraud_country_p, COUNTRIES, n_fraud, rng)

    legit_channel = _draw_from(legit_channel_p, CHANNELS, n_legit, rng)
    fraud_channel = _draw_from(fraud_channel_p, CHANNELS, n_fraud, rng)

    legit_device = _draw_from(legit_device_p, DEVICE_TYPES, n_legit, rng)
    fraud_device = _draw_from(fraud_device_p, DEVICE_TYPES, n_fraud, rng)

    legit_mc = _draw_from(legit_mc_p, MERCHANT_CATEGORIES, n_legit, rng)
    fraud_mc = _draw_from(fraud_mc_p, MERCHANT_CATEGORIES, n_fraud, rng)

    currencies_p = _normalize([0.55, 0.22, 0.10, 0.05, 0.04, 0.04])
    legit_curr = _draw_from(currencies_p, CURRENCIES, n_legit, rng)
    fraud_curr = _draw_from(currencies_p, CURRENCIES, n_fraud, rng)

    # International and risk flags
    def make_flags(country_arr, channel_arr, rng):
        is_high_risk = np.isin(country_arr, HIGH_RISK_COUNTRIES)
        is_international = rng.random(len(country_arr)) < (0.10 + 0.25 * is_high_risk)
        card_present = (channel_arr == "pos") | (channel_arr == "atm")
        return is_high_risk.astype(int), is_international.astype(int), card_present.astype(int)

    legit_hrisk, legit_intern, legit_card_present = make_flags(legit_country, legit_channel, rng)
    fraud_hrisk, fraud_intern, fraud_card_present = make_flags(fraud_country, fraud_channel, rng)

    # Velocity-like feature: random small counts; fraud slightly higher
    legit_velocity_24h = rng.poisson(lam=1.2, size=n_legit)
    fraud_velocity_24h = rng.poisson(lam=2.5, size=n_fraud)

    # Assemble frames
    def frame(n, amount, hour, country, channel, device, mcc, curr, hrisk, intern, card_pres, velocity, label):
        return pd.DataFrame({
            "user_id": rng.integers(1, 50000, size=n),
            "merchant_id": rng.integers(1, 5000, size=n),
            "amount": amount.round(2),
            "hour": hour,
            "country": country,
            "channel": channel,
            "device_type": device,
            "merchant_category": mcc,
            "currency": curr,
            "is_high_risk_country": hrisk,
            "is_international": intern,
            "card_present": card_pres,
            "velocity_24h": velocity,
            "label": np.full(n, label, dtype=int)
        })

    df_legit = frame(n_legit, legit_amount, legit_hours, legit_country, legit_channel, legit_device, legit_mc, legit_curr, legit_hrisk, legit_intern, legit_card_present, legit_velocity_24h, 0)
    df_fraud = frame(n_fraud, fraud_amount, fraud_hours, fraud_country, fraud_channel, fraud_device, fraud_mc, fraud_curr, fraud_hrisk, fraud_intern, fraud_card_present, fraud_velocity_24h, 1)

    df = pd.concat([df_legit, df_fraud], ignore_index=True)

    # Shuffle
    df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)

    return df


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate synthetic transaction dataset with fraud labels")
    parser.add_argument("--n", type=int, default=10000, help="Total number of transactions to generate")
    parser.add_argument("--fraud_ratio", type=float, default=0.02, help="Proportion of fraud (0..1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default="data/synthetic_transactions.csv", help="Output path (.csv or .parquet)")
    args = parser.parse_args(argv)

    cfg = GenerationConfig(n=args.n, fraud_ratio=args.fraud_ratio, seed=args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df = generate_transactions(cfg)

    if args.out.lower().endswith(".parquet"):
        try:
            df.to_parquet(args.out, index=False)
            print(f"Wrote: {args.out}  rows={len(df)}")
        except Exception as e:
            print(f"Parquet write failed ({e}); falling back to CSV")
            csv_out = os.path.splitext(args.out)[0] + ".csv"
            df.to_csv(csv_out, index=False)
            print(f"Wrote: {csv_out}  rows={len(df)}")
    else:
        df.to_csv(args.out, index=False)
        print(f"Wrote: {args.out}  rows={len(df)}")


if __name__ == "__main__":
    main()
