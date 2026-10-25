"""Data ingestion: generate demo data, load JSON files, clean them, build daily time series.

Run `python ingest.py` to (re)create demo data. Replace data/cs-train with your own JSON
files (same fields) to use real data; every other module reads through fetch_data().
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data", "cs-train")
TEST_DATA_DIR = os.path.join(BASE, "data", "test", "cs-train")
COUNTRIES = ["united_kingdom", "singapore", "france", "germany", "spain"]
REQUIRED = ["country", "invoice", "price", "times_viewed", "year", "month", "day"]


def generate_synthetic(out_dir=DATA_DIR, n_days=720, seed=42, end="2019-07-31"):
    """Write one JSON file per month of simulated invoice records (demo data only)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=end, periods=n_days)
    scale = {"united_kingdom": 20, "singapore": 6, "france": 5, "germany": 5, "spain": 3}
    rows = {}
    inv = 100000
    for c in COUNTRIES:
        for i, d in enumerate(dates):
            season = 1 + 0.3 * np.sin(2 * np.pi * d.dayofyear / 365) + 0.0004 * i
            wk = 0.6 if d.dayofweek >= 5 else 1.0
            for _ in range(rng.poisson(scale[c] * season * wk)):
                inv += 1
                rows.setdefault(d.strftime("%Y-%m"), []).append({
                    "country": c, "customer_id": int(rng.integers(1000, 5000)), "invoice": str(inv),
                    "price": round(float(rng.uniform(5, 15)), 2), "stream_id": str(rng.integers(100, 140)),
                    "times_viewed": int(rng.integers(1, 6)), "year": d.year, "month": d.month, "day": d.day})
    os.makedirs(out_dir, exist_ok=True)
    for ym, recs in rows.items():
        with open(os.path.join(out_dir, ym + ".json"), "w") as f:
            json.dump(recs, f)
    return out_dir


def load_raw(data_dir=DATA_DIR):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".json"))
    if not files:
        raise FileNotFoundError(f"No JSON files in {data_dir}; run `python ingest.py`")
    return pd.concat([pd.read_json(os.path.join(data_dir, f), dtype=False) for f in files], ignore_index=True)


def clean(df):
    """Handle missing values at ingestion: standardise names, drop unusable rows and duplicates."""
    df = df.copy()
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["country"] = df["country"].astype(str).str.lower().str.strip().str.replace(" ", "_")
    for c in ("price", "times_viewed", "year", "month", "day"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=REQUIRED).drop_duplicates()
    df = df[(df.price >= 0) & (df.times_viewed >= 0)]
    df["date"] = pd.to_datetime(dict(year=df.year.astype(int), month=df.month.astype(int), day=df.day.astype(int)))
    df["revenue"] = df.price * df.times_viewed
    return df


def build_timeseries(df):
    """Return {country: daily DataFrame}, plus 'all' for all countries combined."""
    out = {}
    full = pd.date_range(df.date.min(), df.date.max())
    for c, g in list(df.groupby("country")) + [("all", df)]:
        d = g.groupby("date").agg(purchases=("invoice", "size"), unique_invoices=("invoice", "nunique"),
                                  views=("times_viewed", "sum"), revenue=("revenue", "sum"))
        out[c] = d.reindex(full, fill_value=0).rename_axis("date").reset_index()
    return out


def fetch_data(data_dir=DATA_DIR):
    """Single entry point for automation: raw JSON -> cleaned records -> time series dict."""
    df = clean(load_raw(data_dir))
    return build_timeseries(df), df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate demo data")
    ap.add_argument("--days", type=int, default=720)
    a = ap.parse_args()
    print("Wrote demo data to", generate_synthetic(n_days=a.days))
