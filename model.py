"""Model training, comparison against a baseline, persistence and prediction.

Target: total revenue over the 30 days after a given date. Baseline: trailing 28-day revenue scaled to 30 days.
Candidates: Ridge, RandomForest, GradientBoosting. The best one (lowest hold-out RMSE) is saved per country.
test=True isolates everything (data, models, logs) under */test.
"""
import os
import shutil
import sys
import time

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ingest
import logger

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_VERSION = "0.1"
HORIZON = 30


def model_dir(test=False):
    d = os.path.join(BASE, "models", "test") if test else os.path.join(BASE, "models")
    os.makedirs(d, exist_ok=True)
    return d


def model_file(country, test=False):
    return os.path.join(model_dir(test), f"sl-{country}-v{MODEL_VERSION}.joblib")


def engineer_features(ts):
    d = ts.set_index("date")
    X = pd.DataFrame(index=d.index)
    for w in (7, 14, 28, 56, 90):
        X[f"rev_{w}"] = d.revenue.rolling(w, min_periods=w).sum()
    X["purch_28"] = d.purchases.rolling(28, min_periods=28).sum()
    X["inv_28"] = d.unique_invoices.rolling(28, min_periods=28).sum()
    X["month_sin"] = np.sin(2 * np.pi * d.index.month / 12)
    X["month_cos"] = np.cos(2 * np.pi * d.index.month / 12)
    y = d.revenue[::-1].rolling(HORIZON).sum()[::-1].shift(-1)  # revenue of days t+1..t+30
    return X, y


def baseline_pred(X):
    return X["rev_28"].values * HORIZON / 28


def candidates():
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=0, n_jobs=1),
        "gradient_boosting": GradientBoostingRegressor(random_state=0),
    }


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def train_one(ts):
    X, y = engineer_features(ts)
    ok = X.notna().all(axis=1) & y.notna()
    Xt, yt = X[ok], y[ok]
    cut = int(len(Xt) * 0.8)
    Xtr, ytr, Xte, yte = Xt.iloc[:cut - HORIZON], yt.iloc[:cut - HORIZON], Xt.iloc[cut:], yt.iloc[cut:]  # gap avoids leakage
    scores = {"baseline": rmse(yte, baseline_pred(Xte))}
    fitted = {}
    for name, m in candidates().items():
        m.fit(Xtr, ytr)
        fitted[name] = m
        scores[name] = rmse(yte, m.predict(Xte))
    best = min(fitted, key=lambda n: scores[n])
    final = candidates()[best].fit(Xt, yt)  # refit best on all labelled data
    preds = {"actual": yte, "baseline": pd.Series(baseline_pred(Xte), index=Xte.index),
             best: pd.Series(fitted[best].predict(Xte), index=Xte.index)}
    return {"model": final, "name": best, "scores": scores, "X": X, "preds": preds}


def plot_reports(results):
    rep = os.path.join(BASE, "reports")
    os.makedirs(rep, exist_ok=True)
    names = list(results)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, c in zip(axes.ravel(), names):
        s = results[c]["scores"]
        ax.bar(list(s), list(s.values()), color=["#8D99AE"] + ["#6C4AB6"] * (len(s) - 1))
        ax.set_title(f"{c}: best = {results[c]['name']}")
        ax.set_ylabel("Hold-out RMSE (lower is better)")
        ax.tick_params(axis="x", rotation=25)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle("Model comparison against the baseline model, by country", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(rep, "model_comparison.png"), dpi=140)
    plt.close()
    r = results["all"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    p = r["preds"]
    ax.plot(p["actual"], label="Actual", color="black")
    ax.plot(p["baseline"], label="Baseline", color="#8D99AE", linestyle="--")
    ax.plot(p[r["name"]], label=f"Best model ({r['name']})", color="#6C4AB6")
    ax.set_title("All countries: next-30-day revenue on the hold-out period")
    ax.set_ylabel("Revenue"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(rep, "best_vs_baseline_all.png"), dpi=140)
    plt.close()


def train(test=False):
    """Train and compare models for every country and for 'all'. Returns {country: scores}."""
    if test:
        ddir = ingest.TEST_DATA_DIR
        if not os.path.isdir(ddir):
            ingest.generate_synthetic(ddir, n_days=300, seed=7)
    else:
        ddir = ingest.DATA_DIR
        if not os.path.isdir(ddir):
            ingest.generate_synthetic(ddir)
    series, _ = ingest.fetch_data(ddir)
    results = {}
    for country, ts in series.items():
        t0 = time.time()
        r = train_one(ts)
        joblib.dump({"model": r["model"], "name": r["name"], "X": r["X"], "scores": r["scores"],
                     "version": MODEL_VERSION}, model_file(country, test))
        logger.update_train_log(country, MODEL_VERSION, r["name"], r["scores"][r["name"]], r["scores"]["baseline"],
                                time.time() - t0, test)
        results[country] = r
    if not test:
        plot_reports(results)
    return {c: r["scores"] for c, r in results.items()}


def model_load(country, test=False):
    path = model_file(country, test)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model for '{country}'. Train first.")
    return joblib.load(path)


def model_predict(country, date, test=False):
    """Predict revenue for the 30 days after `date` (YYYY-MM-DD). country may be 'all'."""
    t0 = time.time()
    b = model_load(country, test)
    dt = pd.Timestamp(date)
    if dt not in b["X"].index or b["X"].loc[dt].isna().any():
        raise ValueError(f"Date {dt.date()} is outside the range with enough history "
                         f"({b['X'].dropna().index.min().date()} to {b['X'].index.max().date()})")
    row = b["X"].loc[[dt]]
    y_pred, base = float(b["model"].predict(row)[0]), float(baseline_pred(row)[0])
    logger.update_predict_log(country, MODEL_VERSION, str(dt.date()), y_pred, base, time.time() - t0, test)
    return {"country": country, "date": str(dt.date()), "y_pred": y_pred, "baseline": base, "model": b["name"]}


def clean_test_artifacts():
    for p in (os.path.join(BASE, "models", "test"), os.path.join(BASE, "logs", "test"), os.path.join(BASE, "data", "test")):
        shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    print(train(test="test" in sys.argv))
